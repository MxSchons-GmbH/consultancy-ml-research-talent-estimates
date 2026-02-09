"""
Correlated Probit Bootstrap Analysis

This module implements bootstrap analysis for the correlated probit model,
integrating functions from run_nested_bootstrap_modal.py and bootstrap_plot_with_annotators.py.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

from hamilton.function_modifiers import config, save_to, cache, source

logger = logging.getLogger(__name__)


# Output paths are now defined in OUTPUT_DATA_CONFIG in hamilton_dataloaders.py
# and accessed via source() in the @save_to decorators


def _calculate_headcount_mismatch(
    df: pd.DataFrame,
    company_database: pd.DataFrame = None,
    filter_mismatched: bool = False
) -> pd.DataFrame:
    """
    Calculate headcount_ratio and headcount_mismatch columns for a DataFrame.
    
    A headcount mismatch is defined as:
    - Ratio of total_headcount / claude_total_employees > 3 or < 1/3
    - OR claude_total_employees is missing (can't verify headcount match)
    
    Args:
        df: DataFrame. If company_database is None, df must already have 
            'total_headcount' and 'claude_total_employees' columns.
            Otherwise, df must have 'linkedin_id' for joining.
        company_database: Optional company database with 'linkedin_id', 'total_headcount', 
                         and 'claude_total_employees' columns. If None, df must already 
                         have the headcount columns.
        filter_mismatched: If True, remove rows with headcount_mismatch=True
        
    Returns:
        DataFrame with 'headcount_ratio' and 'headcount_mismatch' columns added
    """
    result = df.copy()
    
    # If company_database is provided, join to get headcount columns
    if company_database is not None:
        if 'linkedin_id' not in df.columns:
            raise ValueError("DataFrame must have 'linkedin_id' column for headcount mismatch calculation when company_database is provided")
        
        # Check if company_database has the required columns
        required_cols = ['linkedin_id', 'total_headcount', 'claude_total_employees']
        missing_cols = [col for col in required_cols if col not in company_database.columns]
        if missing_cols:
            logger.warning(f"Company database missing columns for headcount mismatch: {missing_cols}. "
                          f"Setting headcount_mismatch=False for all rows.")
            result['headcount_ratio'] = np.nan
            result['headcount_mismatch'] = False
            return result
        
        # Get headcount info from company database
        company_subset = company_database[required_cols].copy()
        
        # Join with df (use suffixes to handle case where columns already exist)
        result = result.merge(company_subset, on='linkedin_id', how='left', suffixes=('', '_db'))
        
        # Use joined columns if they exist, otherwise use original
        total_headcount_col = 'total_headcount_db' if 'total_headcount_db' in result.columns else 'total_headcount'
        claude_col = 'claude_total_employees_db' if 'claude_total_employees_db' in result.columns else 'claude_total_employees'
    else:
        # Headcount columns must already be in df
        if 'total_headcount' not in result.columns or 'claude_total_employees' not in result.columns:
            missing = [c for c in ['total_headcount', 'claude_total_employees'] if c not in result.columns]
            raise ValueError(
                f"DataFrame missing required columns for headcount mismatch: {missing}. "
                f"Either provide company_database or ensure these columns exist in df."
            )
        total_headcount_col = 'total_headcount'
        claude_col = 'claude_total_employees'
    
    # Calculate ratio
    total_headcount = pd.to_numeric(result[total_headcount_col], errors='coerce')
    claude_total_employees = pd.to_numeric(result[claude_col], errors='coerce')
    
    # Calculate ratio only where both values are valid
    valid_mask = total_headcount.notna() & claude_total_employees.notna() & (claude_total_employees > 0)
    ratio = pd.Series(np.nan, index=result.index)
    ratio[valid_mask] = total_headcount[valid_mask] / claude_total_employees[valid_mask]
    result['headcount_ratio'] = ratio
    
    # headcount_mismatch = True if:
    # 1. Ratio > 3 or < 1/3 (where both values are valid)
    # 2. claude_total_employees is NaN (can't verify headcount match, treat as unreliable)
    headcount_mismatch = (ratio > 3) | (ratio < 1/3) | claude_total_employees.isna()
    result['headcount_mismatch'] = headcount_mismatch
    
    # Clean up temporary columns from merge
    cols_to_drop = [col for col in result.columns if col.endswith('_db')]
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)
    
    if filter_mismatched:
        n_before = len(result)
        result = result[~result['headcount_mismatch'].fillna(False)].copy()
        n_excluded = n_before - len(result)
        if n_excluded > 0:
            logger.info(f"Excluded {n_excluded} companies with headcount mismatch (>3x difference)")
    
    return result




def _run_single_bootstrap_iteration(args):
    """Run a single bootstrap iteration - adapted from run_nested_bootstrap_modal.py"""
    from ml_headcount.probit_dawid_skene_inference import (
        predict_new_item_probit_point_estimate,
        create_simple_point_estimate,
        precompute_all_patterns,
        collect_unique_patterns,
        precompute_observed_patterns
    )
    from ml_headcount.synthetic_data_generation import generate_synthetic_employee_data
    import numpy as np
    import pandas as pd
    
    # Unpack arguments - handle both old and new signatures for backward compatibility
    if len(args) == 5:
        (test_data_dict, validation_data_dict, iteration_idx, prior_alpha, prior_beta) = args
        company_database_dict = None
        companies_needing_synthetic_dict = None
        tetrachoric_corr_list = None
        use_synthetic_only = False
        org_type = "consulting"  # Default for backward compatibility
    elif len(args) == 8:
        (test_data_dict, validation_data_dict, company_database_dict, companies_needing_synthetic_dict, 
         tetrachoric_corr_list, iteration_idx, prior_alpha, prior_beta) = args
        use_synthetic_only = False
        org_type = "consulting"  # Default for backward compatibility
    elif len(args) == 9:
        (test_data_dict, validation_data_dict, company_database_dict, companies_needing_synthetic_dict, 
         tetrachoric_corr_list, iteration_idx, prior_alpha, prior_beta, use_synthetic_only) = args
        org_type = "consulting"  # Default for backward compatibility
    else:
        (test_data_dict, validation_data_dict, company_database_dict, companies_needing_synthetic_dict, 
         tetrachoric_corr_list, iteration_idx, prior_alpha, prior_beta, use_synthetic_only, org_type) = args
    
    # Convert dictionaries back to DataFrames
    test_data = pd.DataFrame(test_data_dict)
    validation_data = pd.DataFrame(validation_data_dict)
    company_database = pd.DataFrame(company_database_dict) if company_database_dict is not None else None
    
    # DEFENSIVE CHECK: Validate required columns in test_data
    if not use_synthetic_only and len(test_data) > 0:
        if 'company_id' not in test_data.columns:
            raise ValueError(
                f"test_data missing required 'company_id' column. "
                f"Available columns: {list(test_data.columns)}. "
                f"This column is needed to match companies with headcount data."
            )
        if 'group' not in test_data.columns:
            raise ValueError(
                f"test_data missing required 'group' column. "
                f"Available columns: {list(test_data.columns)}."
            )
    
    # Build headcount map from company database
    headcount_map = {}
    if company_database is not None and len(company_database) > 0:
        # Identify ID column (fail early if none found)
        id_col = None
        for col in ['linkedin_id', 'company_id', 'id']:
            if col in company_database.columns:
                id_col = col
                break
        if id_col is None:
            raise ValueError(
                f"company_database missing ID column. Expected one of: linkedin_id, company_id, id. "
                f"Available columns: {list(company_database.columns)}"
            )
        
        # Identify headcount column (fail early if none found)
        headcount_col = None
        for col in ['total_headcount', 'Total Headcount', 'employees']:
            if col in company_database.columns:
                headcount_col = col
                break
        if headcount_col is None:
            raise ValueError(
                f"company_database missing headcount column. Expected one of: total_headcount, Total Headcount, employees. "
                f"Available columns: {list(company_database.columns)}"
            )
        
        # Build the map
        for _, row in company_database.iterrows():
            company_id = row[id_col]
            headcount = row[headcount_col]
            headcount_num = pd.to_numeric(headcount, errors="coerce")
            
            # Store in map if both ID and headcount are valid
            if not pd.isna(company_id) and pd.notna(headcount_num) and headcount_num > 0:
                headcount_map[str(company_id)] = float(headcount_num)
        
        # DEFENSIVE CHECK: Verify headcount_map was populated
        if len(headcount_map) == 0:
            raise ValueError(
                f"headcount_map is empty after processing {len(company_database)} rows from company_database. "
                f"ID column '{id_col}' or headcount column '{headcount_col}' may have all NaN/invalid values."
            )
        
        logger.debug(f"Built headcount_map with {len(headcount_map)} entries from column '{id_col}' -> '{headcount_col}'")
    elif not use_synthetic_only and len(test_data) > 0:
        # We have real data but no company database - this is a configuration error
        raise ValueError(
            "company_database is None or empty, but test_data is not empty. "
            "company_database_dict must be passed to enable headcount-based resampling."
        )
    
    annotator_cols = [col for col in test_data.columns 
                     if col not in ['company_id', 'group']]
    
    def resample_test_by_company(test_data, headcount_map, seed=None):
        """
        Resample employees within each company **up to the company's total_headcount** if known, else use observed sample size.
        This ensures that every bootstrap iteration generates company-level samples of size total_headcount (population scale),
        aligning the real data bootstrap approach with the synthetic pipeline and company population logic.
        """
        if seed is not None:
            np.random.seed(seed)

        resampled_dfs = []
        # Use 'company_id' (linkedin_id) for lookup since headcount_map is keyed by linkedin_id
        # Note: 'group' column contains numeric indices (0, 1, 2...) which won't match headcount_map keys
        for company_id in test_data['company_id'].unique():
            company_mask = test_data['company_id'] == company_id
            company_data = test_data[company_mask]

            # Determine upsample target size using linkedin_id (company_id) to look up headcount
            company_id_str = str(company_id)
            if company_id_str in headcount_map and headcount_map[company_id_str] > 0:
                target_n = int(headcount_map[company_id_str])
            else:
                target_n = len(company_data)

            resampled_company = company_data.sample(
                n=target_n,
                replace=True
            )
            resampled_dfs.append(resampled_company)

        return pd.concat(resampled_dfs, ignore_index=True)

    
    # Resample validation data (stratified by category)
    boot_val = validation_data.groupby('category', group_keys=False).apply(
        lambda x: x.sample(n=len(x), replace=True, random_state=iteration_idx*1000)
    )
    
    # Resample test data (first iteration uses original)
    # For synthetic-only mode, test_data is empty, so we skip this
    if use_synthetic_only:
        test_use = pd.DataFrame()  # Empty, will be replaced with synthetic data
    elif iteration_idx == 0:
        test_use = test_data
    else:
        # Real data employee-level upsampling: use company total_headcount where available
        test_use = resample_test_by_company(
            test_data,
            headcount_map,
            seed=iteration_idx*10000
        )
    
    # Generate synthetic data for all companies (if enabled)
    # For real data mode: synthetic data is only used where employee-level data is not available
    # For synthetic-only mode: use synthetic data for ALL companies
    #
    # NOTE: Do NOT re-derive use_synthetic_only from args indexing here.
    # The unpacking logic above already sets it correctly, and the "new" signature ends with org_type.
    if (company_database is not None and 
        tetrachoric_corr_list is not None):
        
        # Compute correlation matrix from resampled test data (Component 3 extension: correlation structure uncertainty)
        # This captures uncertainty in the correlation structure by recomputing from resampled test data
        # The correlation matrix is shared across all companies in this iteration
        from ml_headcount.synthetic_data_generation import estimate_test_keyword_filter_correlation_matrix
        
        # Extract keyword filter columns from resampled test data
        keyword_filter_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
        
        # Use resampled test data if available, otherwise fall back to original
        if len(test_use) > 0:
            # Check if resampled test data has keyword filter columns with sufficient data
            test_keyword_data = test_use[keyword_filter_cols].dropna(how='all')  # Keep rows with at least one non-NaN
            
            if len(test_keyword_data) >= 10:  # Need at least 10 samples for stable correlation estimation
                try:
                    # Compute correlation matrix from resampled test data
                    tetrachoric_corr = estimate_test_keyword_filter_correlation_matrix(test_keyword_data)
                    logger.debug(f"Bootstrap iteration {iteration_idx}: Computed correlation matrix from {len(test_keyword_data)} resampled test employees")
                except Exception as e:
                    # Fallback to original correlation matrix if resampling fails
                    logger.warning(f"Bootstrap iteration {iteration_idx}: Failed to compute correlation from resampled data: {e}, using original")
                    tetrachoric_corr = np.array(tetrachoric_corr_list)
            else:
                # Not enough resampled data, use original
                logger.debug(f"Bootstrap iteration {iteration_idx}: Insufficient resampled data ({len(test_keyword_data)} samples), using original correlation matrix")
                tetrachoric_corr = np.array(tetrachoric_corr_list)
        else:
            # No resampled test data (synthetic-only mode or empty), use original
            tetrachoric_corr = np.array(tetrachoric_corr_list)
        
        # Generate synthetic data for all companies with aggregates
        synthetic_dataframes = []
        companies_with_real_data = set(test_use['company_id'].unique()) if not use_synthetic_only else set()
        
        for _, company_row in company_database.iterrows():
            # Handle NA values properly - can't use 'or' with pandas NA
            linkedin_id = company_row.get('linkedin_id')
            company_id_col = company_row.get('company_id')
            id_col = company_row.get('id')
            if pd.notna(linkedin_id):
                company_id = linkedin_id
            elif pd.notna(company_id_col):
                company_id = company_id_col
            elif pd.notna(id_col):
                company_id = str(id_col)  # Use id column as fallback
            else:
                continue
            
            total_headcount_val = company_row.get('total_headcount', 0)
            try:
                total_headcount = int(float(total_headcount_val))
            except Exception:
                total_headcount = 0
            
            if pd.notna(total_headcount) and total_headcount > 0:
                headcount_map[str(company_id)] = total_headcount
            
            if total_headcount <= 0:
                continue
            
            # Get aggregate values (only keyword filters are available for aggregate-only companies)
            company_aggregates = {
                'filter_broad_yes': company_row.get('filter_broad_yes', 0.0),
                'filter_strict_no': company_row.get('filter_strict_no', 0.0),
                'filter_broad_yes_strict_no': company_row.get('filter_broad_yes_strict_no', 0.0)
            }
            
            # Only generate synthetic data if company has at least one nonzero aggregate
            has_aggregates = any(
                not pd.isna(val) and val > 0 
                for val in company_aggregates.values()
            )
            
            if not has_aggregates:
                continue
            
            # Generate synthetic data with iteration-specific seed
            synthetic_seed = iteration_idx * 100000 + abs(hash(str(company_id))) % 100000
            synthetic_df = generate_synthetic_employee_data(
                company_id=str(company_id),
                company_aggregates=company_aggregates,
                n_employees=total_headcount,
                tetrachoric_corr=tetrachoric_corr,  # 3×3 matrix (keyword filters only)
                annotator_names=None,  # No longer needed - function uses keyword filters from company_aggregates
                seed=synthetic_seed
            )
            
            if len(synthetic_df) > 0:
                synthetic_dataframes.append(synthetic_df)
        
        # Combine synthetic data with real test data
        if use_synthetic_only:
            # Synthetic-only mode: use ONLY synthetic data for all companies
            if synthetic_dataframes:
                test_use = pd.concat(synthetic_dataframes, ignore_index=True)
                logger.debug(f"Bootstrap iteration {iteration_idx} (synthetic-only): Generated synthetic data for {len(synthetic_dataframes)} companies")
        else:
            # Real data mode: real data takes precedence, synthetic only for companies without real data
            if synthetic_dataframes:
                synthetic_data = pd.concat(synthetic_dataframes, ignore_index=True)
                
                # Remove synthetic data for companies that already have real data
                synthetic_data = synthetic_data[
                    ~synthetic_data['company_id'].isin(companies_with_real_data)
                ]
                
                # Combine: real data + synthetic data (for companies without real data)
                if len(synthetic_data) > 0:
                    test_use = pd.concat([test_use, synthetic_data], ignore_index=True)
            
            logger.debug(f"Bootstrap iteration {iteration_idx}: Generated synthetic data for {len(synthetic_dataframes)} companies, "
                        f"added {len(synthetic_data)} synthetic employees (after filtering companies with real data)")
    
    # Estimate parameters from bootstrap validation
    point_est = create_simple_point_estimate(boot_val)
    
    # Build per-company priors based on total headcount (fallback to fixed prior)
    # Priors vary by org_type:
    # - consulting: standard priors, but >10k uses 0.1% (1.0, 999.0)
    # - comparator_ml: same as consulting
    # - comparator_non_ml: 0.01% (1.0, 9999.0) for all sizes
    # Fallback prior (when headcount is unknown): Beta(1.563834, 154.819604) = 1% mean
    FALLBACK_PRIOR_ALPHA = 1.563834
    FALLBACK_PRIOR_BETA = 154.819604
    
    if org_type == "comparator_non_ml":
        # For non-ML comparator orgs, use 0.01% prior for all sizes
        piecewise_priors = [
            (float("inf"), (1.0, 9999.0)),  # 0.01% mean for all sizes
        ]
    else:
        # For consulting and comparator_ml orgs, use standard priors with 0.1% for >10k
        piecewise_priors = [
            (100, (2.415889, 21.742998)),       # <100: 10% mean (preserves std)
            (1000, (3.001591, 57.030237)),      # 100-1000: 5% mean (preserves std)
            (10000, (1.563834, 154.819604)),    # 1000-10_000: 1% mean (preserves std)
            (float("inf"), (1.0, 999.0)),  # >10_000: 0.1% mean (changed from 1.896, 474.957)
        ]

    def prior_params_for_company(company_id: Any):
        headcount = headcount_map.get(str(company_id)) if company_id is not None else None
        if headcount is None or not np.isfinite(headcount) or headcount <= 0:
            return FALLBACK_PRIOR_ALPHA, FALLBACK_PRIOR_BETA
        for upper, params in piecewise_priors:
            if headcount < upper:
                return params
        return FALLBACK_PRIOR_ALPHA, FALLBACK_PRIOR_BETA

    # Use 'company_id' (linkedin_id) for prior lookup since headcount_map is keyed by linkedin_id
    # Note: 'group' column contains numeric indices (0, 1, 2...) which won't match headcount_map keys
    company_ids = test_use['company_id'].apply(lambda x: str(x) if pd.notna(x) else None).values
    unique_company_ids = pd.unique([cid for cid in company_ids if cid is not None])
    company_class_priors = {}
    for cid in unique_company_ids:
        a_param, b_param = prior_params_for_company(cid)
        company_class_priors[cid] = np.random.beta(a_param, b_param)
    
    # Get annotation data
    annotation_data = test_use[annotator_cols].values
    theta = point_est['theta']
    chol_0 = point_est['chol_0']
    chol_1 = point_est['chol_1']
    
    # Collect unique patterns (including those with missing values)
    # This allows us to precompute only patterns that actually occur
    unique_patterns = collect_unique_patterns(annotation_data)
    
    # Precompute probabilities for observed patterns only
    # This handles missing values by marginalizing over missing annotators
    pattern_probs = precompute_observed_patterns(
        point_est['theta'], 
        point_est['chol_0'], 
        point_est['chol_1'],
        unique_patterns
    )
    
    # Compute predictions for all items using pattern lookup
    
    predictions = []
    for y_new, company_id in zip(annotation_data, company_ids):
        class_prior = company_class_priors.get(company_id, np.random.beta(FALLBACK_PRIOR_ALPHA, FALLBACK_PRIOR_BETA))
        prob_positive = predict_new_item_probit_point_estimate(
            y_new, theta, chol_0, chol_1, class_prior=class_prior, pattern_probs=pattern_probs
        )
        predictions.append(prob_positive)
    predictions = np.array(predictions)
    
    # Clear pattern cache to free memory
    del pattern_probs
    
    # Aggregate by company with Bernoulli sampling (Component 4: Realization Uncertainty)
    # Sample true labels from Bernoulli distributions to capture realization uncertainty
    # This ensures the bootstrap distribution reflects actual discrete counts rather than expected values
    company_counts = {}
    
    # Set seed for reproducibility (based on iteration index)
    # This ensures different samples across iterations but deterministic within iteration
    np.random.seed(iteration_idx * 1000000)
    
    for company_id in test_use['group'].unique():
        company_mask = (test_use['group'] == company_id)
        company_predictions = predictions[company_mask]
        
        # Sample true labels from Bernoulli distributions
        # z_i^(b) ~ Bernoulli(p_i^(b)) for each employee i in company
        z_samples = np.random.binomial(1, company_predictions)
        
        # Count actual sampled labels (not expected count)
        # Count_c^(b) = sum of z_i^(b) for employees in company c
        actual_count = z_samples.sum()
        company_counts[company_id] = actual_count
    
    return company_counts


def _run_single_machine_bootstrap(
    test_data_dict: dict,
    validation_data_dict: dict,
    start_iter: int,
    n_samples: int,
    machine_idx: int,
    n_machines: int,
    n_cores_per_machine: int,
    prior_alpha: float,
    prior_beta: float,
    company_database_dict: dict = None,
    companies_needing_synthetic_dict: dict = None,
    tetrachoric_corr_list: list = None,
    use_synthetic_only: bool = False,
    org_type: str = "consulting"
):
    """Run bootstrap on a single machine with parallel iterations"""
    import numpy as np
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    import os
    from ml_headcount.modal_functions import NUM_CORES_PER_MACHINE
    
    print(f"Machine {machine_idx+1}/{n_machines}: Starting {n_samples} bootstrap iterations...")
    print(f"Using {NUM_CORES_PER_MACHINE} cores for parallelization")
    
    # Prepare arguments for parallel execution
    iteration_args = []
    for local_iter in range(n_samples):
        global_iter = start_iter + local_iter
        if company_database_dict is not None:
            # Always pass company_database_dict through when provided.
            # Synthetic inputs may be None; _run_single_bootstrap_iteration gates synthetic generation on them.
            iteration_args.append((
                test_data_dict,
                validation_data_dict,
                company_database_dict,
                companies_needing_synthetic_dict,
                tetrachoric_corr_list,
                global_iter,
                prior_alpha,
                prior_beta,
                use_synthetic_only,
                org_type,
            ))
        else:
            # Backward compatibility: old signature (no company database / no synthetic data)
            iteration_args.append((test_data_dict, validation_data_dict, global_iter, prior_alpha, prior_beta))
    
    # Run iterations in parallel using ProcessPoolExecutor
    all_company_counts = {}
    
    with ProcessPoolExecutor(max_workers=NUM_CORES_PER_MACHINE) as executor:
        # Submit all iterations
        future_to_index = {
            executor.submit(_run_single_bootstrap_iteration, args): i 
            for i, args in enumerate(iteration_args)
        }
        
        # Track completed iterations for progress updates
        completed_count = 0
        failed_iterations = {}  # Track failed iterations for retry: {index: retry_count}
        
        # Collect results as they complete (using as_completed for better progress updates)
        from concurrent.futures import as_completed
        
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            max_retries = 3
            iteration_result = None
            
            try:
                iteration_result = future.result()
            except Exception as e:
                # Check if we should retry
                if i not in failed_iterations:
                    failed_iterations[i] = 0
                failed_iterations[i] += 1
                
                if failed_iterations[i] < max_retries:
                    print(f"Machine {machine_idx+1}: Error in iteration {i+1} (attempt {failed_iterations[i]}): {e}")
                    print(f"Retrying iteration {i+1}...")
                    # Resubmit the iteration
                    new_future = executor.submit(_run_single_bootstrap_iteration, iteration_args[i])
                    future_to_index[new_future] = i
                    continue
                else:
                    print(f"Machine {machine_idx+1}: Failed iteration {i+1} after {max_retries} attempts: {e}")
                    print(f"Skipping iteration {i+1} and continuing...")
                    iteration_result = None
            
            if iteration_result is not None:
                # Initialize company_counts if first iteration
                if not all_company_counts:
                    all_company_counts = {company_id: [] for company_id in iteration_result.keys()}
                
                # Add results to storage
                for company_id, count in iteration_result.items():
                    all_company_counts[company_id].append(count)
                
                completed_count += 1
                
                # Progress update (as iterations complete, not in submission order)
                if completed_count % 10 == 0 or completed_count == n_samples:
                    print(f"Machine {machine_idx+1}: Completed {completed_count}/{n_samples} iterations")
    
    print(f"Machine {machine_idx+1}: Completed {n_samples} iterations!")
    return all_company_counts


@cache()
def correlated_probit_bootstrap_raw_results(
    dawid_skene_validation_data: pd.DataFrame,
    dawid_skene_test_data: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    test_keyword_filter_correlation_matrix: np.ndarray,
    companies_needing_synthetic_data: pd.DataFrame,
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_n_machines: int,
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
) -> Dict[str, Any]:
    """
    Run correlated probit bootstrap analysis and return raw distribution results.
    
    This function performs bootstrap analysis using the correlated probit model,
    similar to the run_nested_bootstrap_modal.py script but integrated into Hamilton.
    
    Args:
        dawid_skene_validation_data: Validation dataset with ground truth labels
        dawid_skene_test_data: Test dataset for bootstrap analysis
        company_database_complete: Complete company database with company information
        correlated_probit_bootstrap_n_samples: Number of bootstrap samples per machine
        correlated_probit_bootstrap_n_machines: Number of machines to use
        
    Returns:
        Dictionary with raw bootstrap distributions for each company
    """
    logger.info("Starting correlated probit bootstrap analysis...")
    
    total_samples = correlated_probit_bootstrap_n_samples * correlated_probit_bootstrap_n_machines
    logger.info(f"Running {total_samples} total bootstrap iterations")
    logger.info(f"Using {correlated_probit_bootstrap_n_machines} machine(s) with {correlated_probit_bootstrap_n_samples} samples each")
    
    # Convert data to dictionaries for transfer
    test_data_dict = dawid_skene_test_data.to_dict('list')
    validation_data_dict = dawid_skene_validation_data.to_dict('list')
    company_database_dict = company_database_complete.to_dict('list')
    companies_needing_synthetic_dict = companies_needing_synthetic_data.to_dict('list')
    tetrachoric_corr_list = test_keyword_filter_correlation_matrix.tolist()  # Convert numpy array to list for serialization (3×3 matrix)
    
    # Run bootstrap analysis using multi-machine logic
    from ml_headcount.modal_functions import run_correlated_probit_bootstrap_modal
    from ml_headcount.modal_functions import NUM_CORES_PER_MACHINE
    
    if correlated_probit_bootstrap_n_machines == 1:
        # Single machine case
        company_distributions = run_correlated_probit_bootstrap_modal.remote(
            test_data_dict=test_data_dict,
            validation_data_dict=validation_data_dict,
            company_database_dict=company_database_dict,
            companies_needing_synthetic_dict=companies_needing_synthetic_dict,
            tetrachoric_corr_list=tetrachoric_corr_list,
            start_iter=0,
            n_samples=correlated_probit_bootstrap_n_samples,
            machine_idx=0,
            n_machines=1,
            n_cores_per_machine=NUM_CORES_PER_MACHINE,
            prior_alpha=correlated_probit_bootstrap_prior_alpha,
            prior_beta=correlated_probit_bootstrap_prior_beta,
            use_synthetic_only=False,
            org_type="consulting"
        )
    else:
        # Multiple machines case - launch all machines in parallel
        from concurrent.futures import ThreadPoolExecutor
        
        def launch_machine(machine_idx):
            """Launch a single machine and return its result"""
            start_iter = machine_idx * correlated_probit_bootstrap_n_samples
            return run_correlated_probit_bootstrap_modal.remote(
                test_data_dict=test_data_dict,
                validation_data_dict=validation_data_dict,
                company_database_dict=company_database_dict,
                companies_needing_synthetic_dict=companies_needing_synthetic_dict,
                tetrachoric_corr_list=tetrachoric_corr_list,
                start_iter=start_iter,
                n_samples=correlated_probit_bootstrap_n_samples,
                machine_idx=machine_idx,
                n_machines=correlated_probit_bootstrap_n_machines,
                n_cores_per_machine=NUM_CORES_PER_MACHINE,
                prior_alpha=correlated_probit_bootstrap_prior_alpha,
                prior_beta=correlated_probit_bootstrap_prior_beta,
                use_synthetic_only=False,
                org_type="consulting"
            )
        
        # Launch all machines in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=correlated_probit_bootstrap_n_machines) as executor:
            # Submit all machines to run in parallel
            future_to_machine = {
                executor.submit(launch_machine, machine_idx): machine_idx 
                for machine_idx in range(correlated_probit_bootstrap_n_machines)
            }
            
            # Collect results as they complete
            machine_results = [None] * correlated_probit_bootstrap_n_machines
            for future in future_to_machine:
                machine_idx = future_to_machine[future]
                try:
                    result = future.result()
                    machine_results[machine_idx] = result
                    logger.info(f"Machine {machine_idx+1} launched successfully")
                except Exception as e:
                    logger.error(f"Machine {machine_idx+1} failed to launch: {e}")
                    raise
            
        # Now collect the actual results (this will block until all complete)
        all_results = []
        for i, task in enumerate(machine_results):
            if task is not None:
                result = task  # This blocks until this specific machine completes
                machine_results[i] = result
                logger.info(f"Machine {i+1} completed")
                all_results.append(result)
        
        # Combine results from all machines
        logger.info("Combining results from all machines...")
        company_distributions = {}
        
        for machine_result in all_results:
            for company_id, counts_list in machine_result.items():
                if company_id not in company_distributions:
                    company_distributions[company_id] = []
                company_distributions[company_id].extend(counts_list)
    
    # Process results
    if company_distributions is None or len(company_distributions) == 0:
        logger.warning("No bootstrap results obtained")
        return {}
    
    logger.info(f"Bootstrap analysis returned {len(company_distributions)} companies")
    for company_id, counts_list in company_distributions.items():
        logger.info(f"Company {company_id}: {len(counts_list)} samples")
    
    # Get company information from test data and company database
    company_names = dawid_skene_test_data.groupby('group')['company_id'].first().to_dict()
    
    # Create mapping from LinkedIn company IDs to company database info
    # Handle duplicates by taking the first occurrence
    company_db_mapping = company_database_complete.groupby('linkedin_id').agg({
        'organization_name': 'first',
        'total_headcount': 'first'
    }).to_dict('index')
    
    # Also create mapping from numeric ID to company info (for companies with numeric linkedin_ids)
    # Check if there are numeric linkedin_ids in the database
    company_db_numeric_mapping = {}
    if 'id' in company_database_complete.columns:
        for _, row in company_database_complete.iterrows():
            numeric_id = row.get('id')
            linkedin_id = row.get('linkedin_id')
            org_name = row.get('organization_name')
            if pd.notna(numeric_id):
                # Map numeric ID to company info
                company_db_numeric_mapping[int(numeric_id)] = {
                    'linkedin_id': linkedin_id if pd.notna(linkedin_id) else str(numeric_id),
                    'organization_name': org_name if pd.notna(org_name) else None,
                    'total_headcount': row.get('total_headcount', 0)
                }
    
    # Also check for numeric linkedin_ids directly
    for _, row in company_database_complete.iterrows():
        linkedin_id = row.get('linkedin_id')
        if pd.notna(linkedin_id) and str(linkedin_id).isdigit():
            numeric_id = int(linkedin_id)
            if numeric_id not in company_db_numeric_mapping:
                company_db_numeric_mapping[numeric_id] = {
                    'linkedin_id': str(linkedin_id),
                    'organization_name': row.get('organization_name'),
                    'total_headcount': row.get('total_headcount', 0)
                }
    
    # Get test data employee counts for comparison
    test_employee_counts = dawid_skene_test_data.groupby('company_id').size().to_dict()
    
    # Process raw distributions with company metadata
    processed_distributions = {}
    
    for company_id, counts_list in company_distributions.items():
        if len(counts_list) > 0:
            # Get LinkedIn company ID from test data
            linkedin_company_id = company_names.get(company_id, None)
            
            # If not found in test data, try to look up by numeric ID
            if linkedin_company_id is None:
                # Check if company_id is numeric and can be looked up
                if isinstance(company_id, (int, float)) or (isinstance(company_id, str) and company_id.isdigit()):
                    numeric_id = int(company_id) if isinstance(company_id, (int, float)) else int(company_id)
                    numeric_info = company_db_numeric_mapping.get(numeric_id)
                    if numeric_info:
                        linkedin_company_id = numeric_info['linkedin_id']
                        # Use organization_name from numeric mapping if available
                        if numeric_info['organization_name']:
                            # We'll use this below
                            pass
                    else:
                        # Last resort: use numeric ID as string (better than "Company_xxx")
                        linkedin_company_id = str(numeric_id)
                else:
                    # Non-numeric ID not in test data - use as-is
                    linkedin_company_id = str(company_id)
            
            # Get company info from database
            company_info = company_db_mapping.get(linkedin_company_id, {})
            
            # If not found, try numeric lookup
            if not company_info and isinstance(company_id, (int, float)) or (isinstance(company_id, str) and company_id.isdigit()):
                numeric_id = int(company_id) if isinstance(company_id, (int, float)) else int(company_id)
                numeric_info = company_db_numeric_mapping.get(numeric_id)
                if numeric_info:
                    company_info = {
                        'organization_name': numeric_info['organization_name'],
                        'total_headcount': numeric_info['total_headcount']
                    }
                    # Update linkedin_company_id to the proper linkedin_id from database
                    if numeric_info['linkedin_id'] and numeric_info['linkedin_id'] != str(numeric_id):
                        linkedin_company_id = numeric_info['linkedin_id']
            
            # Get organization name with better fallbacks
            organization_name = company_info.get('organization_name')
            if not organization_name:
                # Try numeric lookup again
                if isinstance(company_id, (int, float)) or (isinstance(company_id, str) and company_id.isdigit()):
                    numeric_id = int(company_id) if isinstance(company_id, (int, float)) else int(company_id)
                    numeric_info = company_db_numeric_mapping.get(numeric_id)
                    if numeric_info and numeric_info['organization_name']:
                        organization_name = numeric_info['organization_name']
            
            # Final fallback: use linkedin_company_id (which should be readable, not "Company_xxx")
            if not organization_name:
                organization_name = linkedin_company_id
            
            total_headcount = company_info.get('total_headcount', 0)
            
            # Get actual employee count from test data
            actual_employee_count = test_employee_counts.get(linkedin_company_id, 0)
            
            # Use the larger of total_headcount or actual employee count
            effective_company_size = max(total_headcount, actual_employee_count) if total_headcount > 0 else actual_employee_count
            
            # Store processed distribution with metadata
            processed_distributions[linkedin_company_id] = {
                'company_name': organization_name,
                'n_employees': effective_company_size,
                'samples': [float(x) for x in counts_list]
            }
            
            logger.info(f"Added company {linkedin_company_id} (name: {organization_name}) with {len(counts_list)} samples to results")
        else:
            logger.warning(f"Company {company_id} has no samples, skipping")
    
    logger.info(f"Bootstrap analysis complete: {len(processed_distributions)} companies analyzed")
    return processed_distributions


@cache()
def correlated_probit_bootstrap_raw_results_synthetic(
    dawid_skene_validation_data: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    company_database_comparator_ml: pd.DataFrame,
    company_database_comparator_non_ml: pd.DataFrame,
    test_keyword_filter_correlation_matrix: np.ndarray,
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_n_machines: int,
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
) -> Dict[str, Any]:
    """
    Run correlated probit bootstrap analysis using ONLY synthetic data for ALL companies.
    
    This generates synthetic employee-level data for all companies using keyword filter
    aggregates from the company database, then runs bootstrap analysis.
    
    Includes companies from:
    - company_database_complete (main organizations)
    - company_database_comparator_ml (comparator ML organizations)
    - company_database_comparator_non_ml (comparator Non-ML organizations)
    
    Args:
        dawid_skene_validation_data: Validation dataset with ground truth labels
        company_database_complete: Complete company database with company information (main orgs)
        company_database_comparator_ml: Comparator ML company database
        company_database_comparator_non_ml: Comparator Non-ML company database
        test_keyword_filter_correlation_matrix: 3×3 correlation matrix for keyword filters
        correlated_probit_bootstrap_n_samples: Number of bootstrap samples per machine
        correlated_probit_bootstrap_n_machines: Number of machines to use
        
    Returns:
        Dictionary with raw bootstrap distributions for each company
    """
    logger.info("Starting synthetic-only correlated probit bootstrap analysis...")
    
    total_samples = correlated_probit_bootstrap_n_samples * correlated_probit_bootstrap_n_machines
    logger.info(f"Running {total_samples} total bootstrap iterations (synthetic-only mode)")
    logger.info(f"Using {correlated_probit_bootstrap_n_machines} machine(s) with {correlated_probit_bootstrap_n_samples} samples each")
    
    # Combine all company databases for synthetic data generation
    # This includes main orgs, comparator ML, and comparator Non-ML
    all_companies = pd.concat([
        company_database_complete,
        company_database_comparator_ml,
        company_database_comparator_non_ml
    ], ignore_index=True)
    
    logger.info(f"Processing {len(all_companies)} total companies for synthetic data:")
    logger.info(f"  - Main orgs: {len(company_database_complete)}")
    logger.info(f"  - Comparator ML: {len(company_database_comparator_ml)}")
    logger.info(f"  - Comparator Non-ML: {len(company_database_comparator_non_ml)}")
    
    # For synthetic-only mode, we need to track the mapping from group to linkedin_id
    # We'll generate a sample synthetic dataframe to get the structure, then use it to build the mapping
    # But actually, we can't do this here - the mapping happens inside the bootstrap iteration
    # Instead, we'll create a mapping by generating synthetic data for one company and seeing the group
    
    # Create empty test data dict (we'll use only synthetic data)
    test_data_dict = {'company_id': [], 'group': []}
    # Add annotator columns (keyword filters only for synthetic)
    keyword_filter_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    for col in keyword_filter_cols + ['llm_gemini_2_5_flash', 'llm_sonnet_4', 'llm_gpt_5_mini']:
        test_data_dict[col] = []
    
    # Create a mapping from group hash to linkedin_id by pre-computing for all companies
    # This is needed because bootstrap aggregates by 'group', not 'company_id'
    group_to_linkedin_id = {}
    for _, company_row in all_companies.iterrows():
        linkedin_id = company_row.get('linkedin_id')
        company_id_col = company_row.get('company_id')
        id_col = company_row.get('id')
        if pd.notna(linkedin_id):
            company_id_for_hash = str(linkedin_id)
        elif pd.notna(company_id_col):
            company_id_for_hash = str(company_id_col)
        elif pd.notna(id_col):
            company_id_for_hash = str(id_col)
        else:
            continue
        # Compute the same hash used in generate_synthetic_employee_data
        group_id = abs(hash(company_id_for_hash)) % (10**9)
        linkedin_id_str = str(linkedin_id) if pd.notna(linkedin_id) else (str(company_id_col) if pd.notna(company_id_col) else str(id_col))
        group_to_linkedin_id[group_id] = linkedin_id_str
    
    validation_data_dict = dawid_skene_validation_data.to_dict('list')
    company_database_dict = all_companies.to_dict('list')
    companies_needing_synthetic_dict = pd.DataFrame({'company_id': []}).to_dict('list')  # Empty, not used in synthetic-only mode
    tetrachoric_corr_list = test_keyword_filter_correlation_matrix.tolist()
    
    # Run bootstrap analysis using multi-machine logic
    from ml_headcount.modal_functions import run_correlated_probit_bootstrap_modal
    from ml_headcount.modal_functions import NUM_CORES_PER_MACHINE
    
    if correlated_probit_bootstrap_n_machines == 1:
        # Single machine case
        company_distributions = run_correlated_probit_bootstrap_modal.remote(
            test_data_dict=test_data_dict,
            validation_data_dict=validation_data_dict,
            company_database_dict=company_database_dict,
            companies_needing_synthetic_dict=companies_needing_synthetic_dict,
            tetrachoric_corr_list=tetrachoric_corr_list,
            start_iter=0,
            n_samples=correlated_probit_bootstrap_n_samples,
            machine_idx=0,
            n_machines=1,
            n_cores_per_machine=NUM_CORES_PER_MACHINE,
            prior_alpha=correlated_probit_bootstrap_prior_alpha,
            prior_beta=correlated_probit_bootstrap_prior_beta,
            use_synthetic_only=True
        )
    else:
        # Multiple machines case - launch all machines in parallel
        from concurrent.futures import ThreadPoolExecutor
        
        def launch_machine(machine_idx):
            """Launch a single machine and return its result"""
            start_iter = machine_idx * correlated_probit_bootstrap_n_samples
            return run_correlated_probit_bootstrap_modal.remote(
                test_data_dict=test_data_dict,
                validation_data_dict=validation_data_dict,
                company_database_dict=company_database_dict,
                companies_needing_synthetic_dict=companies_needing_synthetic_dict,
                tetrachoric_corr_list=tetrachoric_corr_list,
                start_iter=start_iter,
                n_samples=correlated_probit_bootstrap_n_samples,
                machine_idx=machine_idx,
                n_machines=correlated_probit_bootstrap_n_machines,
                n_cores_per_machine=NUM_CORES_PER_MACHINE,
                prior_alpha=correlated_probit_bootstrap_prior_alpha,
                prior_beta=correlated_probit_bootstrap_prior_beta,
                use_synthetic_only=True
            )
        
        # Launch all machines in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=correlated_probit_bootstrap_n_machines) as executor:
            # Submit all machines to run in parallel
            future_to_machine = {
                executor.submit(launch_machine, machine_idx): machine_idx 
                for machine_idx in range(correlated_probit_bootstrap_n_machines)
            }
            
            # Collect results as they complete
            machine_results = [None] * correlated_probit_bootstrap_n_machines
            for future in future_to_machine:
                machine_idx = future_to_machine[future]
                try:
                    result = future.result()
                    machine_results[machine_idx] = result
                    logger.info(f"Machine {machine_idx+1} launched successfully (synthetic-only)")
                except Exception as e:
                    logger.error(f"Machine {machine_idx+1} failed to launch: {e}")
                    raise
            
        # Now collect the actual results (this will block until all complete)
        all_results = []
        for i, task in enumerate(machine_results):
            if task is not None:
                result = task  # This blocks until this specific machine completes
                machine_results[i] = result
                logger.info(f"Machine {i+1} completed (synthetic-only)")
                all_results.append(result)
        
        # Combine results from all machines
        logger.info("Combining results from all machines (synthetic-only)...")
        company_distributions = {}
        
        for machine_result in all_results:
            for company_id, counts_list in machine_result.items():
                if company_id not in company_distributions:
                    company_distributions[company_id] = []
                company_distributions[company_id].extend(counts_list)
    
    # Process results
    if company_distributions is None or len(company_distributions) == 0:
        logger.warning("No bootstrap results obtained (synthetic-only)")
        return {}
    
    logger.info(f"Synthetic-only bootstrap analysis returned {len(company_distributions)} companies")
    
    # Create mapping from LinkedIn company IDs to company database info
    # Use all_companies to include comparator companies
    company_db_mapping = all_companies.groupby('linkedin_id').agg({
        'organization_name': 'first',
        'total_headcount': 'first'
    }).to_dict('index')
    
    # Create mapping from group (hash) to linkedin_id
    # The bootstrap aggregates by 'group', which is a hash of company_id
    # In generate_synthetic_employee_data, we use company_id (which is the linkedin_id string) to hash
    # IMPORTANT: This logic must match exactly the logic in _run_single_bootstrap_iteration (lines 144-156)
    group_to_linkedin_id = {}
    for _, company_row in all_companies.iterrows():
        linkedin_id = company_row.get('linkedin_id')
        company_id_col = company_row.get('company_id')
        id_col = company_row.get('id')
        
        # Determine which ID to use (same logic as in bootstrap iteration)
        if pd.notna(linkedin_id):
            company_id_for_hash = str(linkedin_id)
            linkedin_id_str = str(linkedin_id)
        elif pd.notna(company_id_col):
            company_id_for_hash = str(company_id_col)
            linkedin_id_str = str(company_id_col)
        elif pd.notna(id_col):
            company_id_for_hash = str(id_col)
            linkedin_id_str = str(id_col)  # Use id column as fallback
        else:
            continue
        
        # Compute the same hash used in generate_synthetic_employee_data
        # Use hashlib for deterministic hashing across processes (same as in synthetic_data_generation.py)
        import hashlib
        hash_obj = hashlib.md5(str(company_id_for_hash).encode('utf-8'))
        group_id = int(hash_obj.hexdigest(), 16) % (10**9)
        group_to_linkedin_id[group_id] = linkedin_id_str
        
        # Also try with the numeric ID if it exists (some companies might use numeric IDs)
        if pd.notna(company_id_col) and str(company_id_col) != company_id_for_hash:
            hash_obj_alt = hashlib.md5(str(company_id_col).encode('utf-8'))
            group_id_alt = int(hash_obj_alt.hexdigest(), 16) % (10**9)
            if group_id_alt not in group_to_linkedin_id:  # Don't overwrite if already set
                group_to_linkedin_id[group_id_alt] = linkedin_id_str if pd.notna(linkedin_id) else str(company_id_col)
    
    # Process raw distributions with company metadata
    processed_distributions = {}
    
    for group_id, counts_list in company_distributions.items():
        if len(counts_list) > 0:
            # Map from group (hash) to linkedin_id
            linkedin_company_id = group_to_linkedin_id.get(group_id)
            if linkedin_company_id is None:
                logger.warning(f"Could not find linkedin_id for group {group_id}, skipping")
                continue
            
            # Get company info from database
            company_info = company_db_mapping.get(linkedin_company_id, {})
            organization_name = company_info.get('organization_name', linkedin_company_id)
            total_headcount = company_info.get('total_headcount', 0)
            
            # Store processed distribution with metadata
            processed_distributions[linkedin_company_id] = {
                'company_name': organization_name,
                'n_employees': total_headcount,
                'samples': [float(x) for x in counts_list]
            }
            
            logger.info(f"Added company {linkedin_company_id} (group {group_id}) with {len(counts_list)} samples to synthetic results")
        else:
            logger.warning(f"Group {group_id} has no samples, skipping")
    
    logger.info(f"Synthetic-only bootstrap analysis complete: {len(processed_distributions)} companies analyzed")
    return processed_distributions


@save_to.csv(path=source("correlated_probit_bootstrap_results_synthetic_output_path"))
def correlated_probit_bootstrap_results_synthetic(
    correlated_probit_bootstrap_raw_results_synthetic: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create summary CSV from synthetic-only bootstrap results.
    
    Args:
        correlated_probit_bootstrap_raw_results_synthetic: Raw synthetic bootstrap distributions with metadata
        
    Returns:
        DataFrame with summary statistics for each company
    """
    logger.info("Creating synthetic bootstrap results summary CSV...")
    
    if not correlated_probit_bootstrap_raw_results_synthetic:
        logger.warning("No raw synthetic bootstrap results to process")
        return pd.DataFrame(columns=['linkedin_id', 'company_name', 'mean', 'std', 'q10', 'q50', 'q90', 'n_employees'])
    
    results_list = []
    
    for linkedin_id, company_data in correlated_probit_bootstrap_raw_results_synthetic.items():
        samples = company_data['samples']
        if len(samples) > 0:
            # Calculate summary statistics
            row = {
                'linkedin_id': linkedin_id,
                'company_name': company_data['company_name'],
                'mean': np.mean(samples),
                'std': np.std(samples),
                'q10': np.percentile(samples, 10),
                'q50': np.percentile(samples, 50),  # Median
                'q90': np.percentile(samples, 90),
                'n_employees': company_data['n_employees']
            }
            results_list.append(row)
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('mean', ascending=False)
    
    logger.info(f"Created synthetic summary CSV with {len(results_df)} companies")
    return results_df


def _generate_synthetic_for_company(company_row, tetrachoric_corr, seed=0):
    """Helper function to generate synthetic data for a single company."""
    from ml_headcount.synthetic_data_generation import generate_synthetic_employee_data
    
    # Extract company ID (same logic as bootstrap)
    linkedin_id = company_row.get('linkedin_id')
    company_id_col = company_row.get('company_id')
    id_col = company_row.get('id')
    
    if pd.notna(linkedin_id):
        company_id = linkedin_id
    elif pd.notna(company_id_col):
        company_id = company_id_col
    elif pd.notna(id_col):
        company_id = str(id_col)
    else:
        return pd.DataFrame()  # Empty DataFrame
    
    # Get total headcount (NOTE: This should use claude_total_employees after bug fix)
    total_headcount = int(company_row.get('total_headcount', 0))
    if total_headcount <= 0:
        return pd.DataFrame()  # Empty DataFrame
    
    # Get keyword filter aggregates
    company_aggregates = {
        'filter_broad_yes': company_row.get('filter_broad_yes', 0.0),
        'filter_strict_no': company_row.get('filter_strict_no', 0.0),
        'filter_broad_yes_strict_no': company_row.get('filter_broad_yes_strict_no', 0.0)
    }
    
    # Check if company has aggregates
    has_aggregates = any(
        not pd.isna(val) and val > 0 
        for val in company_aggregates.values()
    )
    
    if not has_aggregates:
        return pd.DataFrame()  # Empty DataFrame
    
    # Generate synthetic data with FIXED seed (deterministic)
    synthetic_df = generate_synthetic_employee_data(
        company_id=str(company_id),
        company_aggregates=company_aggregates,
        n_employees=total_headcount,  # NOTE: Should be claude_total_employees after fix
        tetrachoric_corr=tetrachoric_corr,
        annotator_names=None,
        seed=seed  # Fixed seed for determinism
    )
    
    return synthetic_df


@save_to.csv(path=source("synthetic_employee_level_data_all_output_path"))
@cache()
def synthetic_employee_level_data_all(
    company_database_complete: pd.DataFrame,
    company_database_comparator_ml: pd.DataFrame,
    company_database_comparator_non_ml: pd.DataFrame,
    test_keyword_filter_correlation_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Generate and save synthetic employee-level data for all companies with dataset_source.
    
    Creates a deterministic snapshot of synthetic employees for all companies (main + comparators).
    Adds dataset_source column to indicate origin. Recomputes group numbers to match real data.
    
    Args:
        company_database_complete: Main company database
        company_database_comparator_ml: Comparator ML company database
        company_database_comparator_non_ml: Comparator Non-ML company database
        test_keyword_filter_correlation_matrix: 3×3 correlation matrix for keyword filters
        
    Returns:
        DataFrame with synthetic employee annotations + dataset_source column
    """
    logger.info("Generating synthetic employee-level data for all companies...")
    
    # Convert correlation matrix
    tetrachoric_corr = np.array(test_keyword_filter_correlation_matrix)
    
    # Generate synthetic data for each dataset separately
    all_synthetic_dataframes = []
    
    # Process main companies
    logger.info(f"Processing {len(company_database_complete)} main companies...")
    for _, company_row in company_database_complete.iterrows():
        synthetic_df = _generate_synthetic_for_company(company_row, tetrachoric_corr, seed=0)
        if len(synthetic_df) > 0:
            synthetic_df['dataset_source'] = 'main'
            all_synthetic_dataframes.append(synthetic_df)
    
    # Process comparator ML companies
    logger.info(f"Processing {len(company_database_comparator_ml)} comparator ML companies...")
    for _, company_row in company_database_comparator_ml.iterrows():
        synthetic_df = _generate_synthetic_for_company(company_row, tetrachoric_corr, seed=0)
        if len(synthetic_df) > 0:
            synthetic_df['dataset_source'] = 'comparator_ml'
            all_synthetic_dataframes.append(synthetic_df)
    
    # Process comparator Non-ML companies
    logger.info(f"Processing {len(company_database_comparator_non_ml)} comparator Non-ML companies...")
    for _, company_row in company_database_comparator_non_ml.iterrows():
        synthetic_df = _generate_synthetic_for_company(company_row, tetrachoric_corr, seed=0)
        if len(synthetic_df) > 0:
            synthetic_df['dataset_source'] = 'comparator_non_ml'
            all_synthetic_dataframes.append(synthetic_df)
    
    # Combine all synthetic data
    if all_synthetic_dataframes:
        result_df = pd.concat(all_synthetic_dataframes, ignore_index=True)
    else:
        # Return empty DataFrame with correct schema
        result_df = pd.DataFrame({
            'company_id': [],
            'group': [],
            'dataset_source': [],
            'llm_gemini_2_5_flash': [],
            'llm_sonnet_4': [],
            'llm_gpt_5_mini': [],
            'filter_broad_yes': [],
            'filter_strict_no': [],
            'filter_broad_yes_strict_no': []
        })
    
    # Recompute group sequentially to match real data
    # This ensures group numbers are consistent between real and synthetic
    unique_companies = sorted(result_df['company_id'].unique())
    company_to_group = {company_id: idx for idx, company_id in enumerate(unique_companies)}
    result_df['group'] = result_df['company_id'].map(company_to_group)
    
    # Ensure all annotator columns are float64 (matching schema)
    annotator_cols = [
        'llm_gemini_2_5_flash', 'llm_sonnet_4', 'llm_gpt_5_mini',
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    for col in annotator_cols:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype('float64')
    
    # Reorder columns: company_id, group, dataset_source, then annotators
    column_order = ['company_id', 'group', 'dataset_source'] + annotator_cols
    result_df = result_df[column_order]
    
    logger.info(f"Generated synthetic employee-level data: {len(result_df):,} employees across {len(unique_companies)} companies")
    logger.info(f"  - Main: {(result_df['dataset_source'] == 'main').sum():,} rows")
    logger.info(f"  - Comparator ML: {(result_df['dataset_source'] == 'comparator_ml').sum():,} rows")
    logger.info(f"  - Comparator Non-ML: {(result_df['dataset_source'] == 'comparator_non_ml').sum():,} rows")
    
    return result_df


@save_to.csv(path=source("correlated_probit_bootstrap_results_output_path"))
def correlated_probit_bootstrap_results(
    correlated_probit_bootstrap_raw_results: Dict[str, Any],
    correlated_probit_bootstrap_results_synthetic: pd.DataFrame,
    company_database_complete: pd.DataFrame
) -> pd.DataFrame:
    """
    Create summary CSV from raw bootstrap results (real employee-level data only).
    
    This function only includes companies with actual employee-level data.
    Companies without employee-level data will have NaN in the "real" columns
    of final_results, and their synthetic estimates will appear in separate columns.
    
    Filters out companies where total_headcount and claude_total_employees differ by more than 3x,
    as employee-level data is not reliable for these companies.
    
    Args:
        correlated_probit_bootstrap_raw_results: Raw bootstrap distributions with metadata (real data)
        correlated_probit_bootstrap_results_synthetic: Not used (kept for API compatibility)
        company_database_complete: Company database with headcount information for filtering
        
    Returns:
        DataFrame with summary statistics for companies with real employee-level data only,
        excluding companies with headcount mismatch
    """
    logger.info("Creating bootstrap results summary CSV (real employee-level data only)...")
    
    # Process real data results only
    results_list = []
    
    if correlated_probit_bootstrap_raw_results:
        for linkedin_id, company_data in correlated_probit_bootstrap_raw_results.items():
            samples = company_data.get('samples', [])
            if len(samples) > 0:
                # Calculate summary statistics
                row = {
                    'linkedin_id': linkedin_id,
                    'company_name': company_data.get('company_name', linkedin_id),
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'q10': np.percentile(samples, 10),
                    'q50': np.percentile(samples, 50),  # Median
                    'q90': np.percentile(samples, 90),
                    'n_employees': company_data.get('n_employees', 0)
                }
                results_list.append(row)
    
    # NO FALLBACK: Synthetic data is handled separately to ensure proper separation
    # of real vs synthetic estimates in final results
    
    # Create DataFrame with expected columns even if empty
    if len(results_list) == 0:
        results_df = pd.DataFrame(columns=['linkedin_id', 'company_name', 'mean', 'std', 'q10', 'q50', 'q90', 'n_employees'])
    else:
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('mean', ascending=False)
    
    # Filter out companies with headcount mismatch (>3x difference)
    if len(results_df) > 0:
        results_df = _calculate_headcount_mismatch(
            results_df, 
            company_database_complete, 
            filter_mismatched=True
        )
        
        # Keep only the columns needed for output
        output_cols = ['linkedin_id', 'company_name', 'mean', 'std', 'q10', 'q50', 'q90', 'n_employees', 'headcount_ratio', 'headcount_mismatch']
        output_cols = [col for col in output_cols if col in results_df.columns]
        results_df = results_df[output_cols].copy()
    
    logger.info(f"Created summary CSV with {len(results_df)} companies (real employee-level data only, excluding headcount mismatch)")
    return results_df


@cache()
def correlated_probit_bootstrap_raw_results_main(
    dawid_skene_validation_data: pd.DataFrame,
    dawid_skene_test_data_main: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_n_machines: int,
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
) -> Dict[str, Any]:
    """Run probit bootstrap for main pipeline."""
    logger.info("Starting correlated probit bootstrap analysis for main pipeline...")
    
    total_samples = correlated_probit_bootstrap_n_samples * correlated_probit_bootstrap_n_machines
    logger.info(f"Running {total_samples} total bootstrap iterations")
    
    test_data_dict = dawid_skene_test_data_main.to_dict('list')
    validation_data_dict = dawid_skene_validation_data.to_dict('list')
    company_database_dict = company_database_complete.to_dict('list')
    
    from ml_headcount.modal_functions import run_correlated_probit_bootstrap_modal
    from ml_headcount.modal_functions import NUM_CORES_PER_MACHINE
    
    if correlated_probit_bootstrap_n_machines == 1:
        company_distributions = run_correlated_probit_bootstrap_modal.remote(
            test_data_dict=test_data_dict,
            validation_data_dict=validation_data_dict,
            start_iter=0,
            n_samples=correlated_probit_bootstrap_n_samples,
            machine_idx=0,
            n_machines=1,
            n_cores_per_machine=NUM_CORES_PER_MACHINE,
            prior_alpha=correlated_probit_bootstrap_prior_alpha,
            prior_beta=correlated_probit_bootstrap_prior_beta,
            company_database_dict=company_database_dict,
            companies_needing_synthetic_dict=None,
            tetrachoric_corr_list=None,
            use_synthetic_only=False,
            org_type="consulting"
        )
    else:
        from concurrent.futures import ThreadPoolExecutor
        
        def launch_machine(machine_idx):
            start_iter = machine_idx * correlated_probit_bootstrap_n_samples
            return run_correlated_probit_bootstrap_modal.remote(
                test_data_dict=test_data_dict,
                validation_data_dict=validation_data_dict,
                start_iter=start_iter,
                n_samples=correlated_probit_bootstrap_n_samples,
                machine_idx=machine_idx,
                n_machines=correlated_probit_bootstrap_n_machines,
                n_cores_per_machine=NUM_CORES_PER_MACHINE,
                prior_alpha=correlated_probit_bootstrap_prior_alpha,
                prior_beta=correlated_probit_bootstrap_prior_beta,
                company_database_dict=company_database_dict,
                companies_needing_synthetic_dict=None,
                tetrachoric_corr_list=None,
                use_synthetic_only=False,
                org_type="consulting"
            )
        
        with ThreadPoolExecutor(max_workers=correlated_probit_bootstrap_n_machines) as executor:
            future_to_machine = {
                executor.submit(launch_machine, machine_idx): machine_idx 
                for machine_idx in range(correlated_probit_bootstrap_n_machines)
            }
            
            machine_results = [None] * correlated_probit_bootstrap_n_machines
            for future in future_to_machine:
                machine_idx = future_to_machine[future]
                try:
                    result = future.result()
                    machine_results[machine_idx] = result
                    logger.info(f"Machine {machine_idx+1} launched successfully")
                except Exception as e:
                    logger.error(f"Machine {machine_idx+1} failed to launch: {e}")
                    raise
            
        all_results = []
        for i, task in enumerate(machine_results):
            if task is not None:
                result = task
                machine_results[i] = result
                logger.info(f"Machine {i+1} completed")
                all_results.append(result)
        
        logger.info("Combining results from all machines...")
        company_distributions = {}
        
        for machine_result in all_results:
            for company_id, counts_list in machine_result.items():
                if company_id not in company_distributions:
                    company_distributions[company_id] = []
                company_distributions[company_id].extend(counts_list)
    
    if company_distributions is None or len(company_distributions) == 0:
        logger.warning("No bootstrap results obtained for main pipeline")
        return {}
    
    logger.info(f"Main pipeline bootstrap analysis returned {len(company_distributions)} companies")
    
    company_names = dawid_skene_test_data_main.groupby('group')['company_id'].first().to_dict()
    company_db_mapping = company_database_complete.groupby('linkedin_id').agg({
        'organization_name': 'first',
        'total_headcount': 'first'
    }).to_dict('index')
    test_employee_counts = dawid_skene_test_data_main.groupby('company_id').size().to_dict()
    
    processed_distributions = {}
    for company_id, counts_list in company_distributions.items():
        if len(counts_list) > 0:
            linkedin_company_id = company_names.get(company_id, f'Company_{company_id}')
            company_info = company_db_mapping.get(linkedin_company_id, {})
            organization_name = company_info.get('organization_name', linkedin_company_id)
            total_headcount = company_info.get('total_headcount', 0)
            actual_employee_count = test_employee_counts.get(linkedin_company_id, 0)
            effective_company_size = max(total_headcount, actual_employee_count) if total_headcount > 0 else actual_employee_count
            
            processed_distributions[linkedin_company_id] = {
                'company_name': organization_name,
                'n_employees': effective_company_size,
                'samples': [float(x) for x in counts_list]
            }
    
    logger.info(f"Main pipeline bootstrap analysis complete: {len(processed_distributions)} companies analyzed")
    return processed_distributions


@cache()
def correlated_probit_bootstrap_raw_results_comparator_ml(
    dawid_skene_validation_data: pd.DataFrame,
    dawid_skene_test_data_comparator_ml: pd.DataFrame,
    company_database_comparator_ml: pd.DataFrame,
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_n_machines: int,
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
) -> Dict[str, Any]:
    """Run probit bootstrap for comparator ML pipeline."""
    if len(dawid_skene_test_data_comparator_ml) == 0:
        logger.warning("No test data for comparator ML pipeline, returning empty results")
        return {}
    
    logger.info("Starting correlated probit bootstrap analysis for comparator ML pipeline...")
    
    total_samples = correlated_probit_bootstrap_n_samples * correlated_probit_bootstrap_n_machines
    test_data_dict = dawid_skene_test_data_comparator_ml.to_dict('list')
    validation_data_dict = dawid_skene_validation_data.to_dict('list')
    company_database_dict = company_database_comparator_ml.to_dict('list')
    
    from ml_headcount.modal_functions import run_correlated_probit_bootstrap_modal
    from ml_headcount.modal_functions import NUM_CORES_PER_MACHINE
    
    if correlated_probit_bootstrap_n_machines == 1:
        company_distributions = run_correlated_probit_bootstrap_modal.remote(
            test_data_dict=test_data_dict,
            validation_data_dict=validation_data_dict,
            start_iter=0,
            n_samples=correlated_probit_bootstrap_n_samples,
            machine_idx=0,
            n_machines=1,
            n_cores_per_machine=NUM_CORES_PER_MACHINE,
            prior_alpha=correlated_probit_bootstrap_prior_alpha,
            prior_beta=correlated_probit_bootstrap_prior_beta,
            company_database_dict=company_database_dict,
            companies_needing_synthetic_dict=None,
            tetrachoric_corr_list=None,
            use_synthetic_only=False,
            org_type="comparator_ml"
        )
    else:
        from concurrent.futures import ThreadPoolExecutor
        
        def launch_machine(machine_idx):
            start_iter = machine_idx * correlated_probit_bootstrap_n_samples
            return run_correlated_probit_bootstrap_modal.remote(
                test_data_dict=test_data_dict,
                validation_data_dict=validation_data_dict,
                start_iter=start_iter,
                n_samples=correlated_probit_bootstrap_n_samples,
                machine_idx=machine_idx,
                n_machines=correlated_probit_bootstrap_n_machines,
                n_cores_per_machine=NUM_CORES_PER_MACHINE,
                prior_alpha=correlated_probit_bootstrap_prior_alpha,
                prior_beta=correlated_probit_bootstrap_prior_beta,
                company_database_dict=company_database_dict,
                companies_needing_synthetic_dict=None,
                tetrachoric_corr_list=None,
                use_synthetic_only=False,
                org_type="comparator_ml"
            )
        
        with ThreadPoolExecutor(max_workers=correlated_probit_bootstrap_n_machines) as executor:
            future_to_machine = {
                executor.submit(launch_machine, machine_idx): machine_idx 
                for machine_idx in range(correlated_probit_bootstrap_n_machines)
            }
            
            machine_results = [None] * correlated_probit_bootstrap_n_machines
            for future in future_to_machine:
                machine_idx = future_to_machine[future]
                try:
                    result = future.result()
                    machine_results[machine_idx] = result
                except Exception as e:
                    logger.error(f"Machine {machine_idx+1} failed: {e}")
                    raise
            
        all_results = []
        for i, task in enumerate(machine_results):
            if task is not None:
                result = task
                all_results.append(result)
        
        company_distributions = {}
        for machine_result in all_results:
            for company_id, counts_list in machine_result.items():
                if company_id not in company_distributions:
                    company_distributions[company_id] = []
                company_distributions[company_id].extend(counts_list)
    
    if company_distributions is None or len(company_distributions) == 0:
        return {}
    
    company_names = dawid_skene_test_data_comparator_ml.groupby('group')['company_id'].first().to_dict()
    company_db_mapping = company_database_comparator_ml.groupby('linkedin_id').agg({
        'organization_name': 'first',
        'total_headcount': 'first'
    }).to_dict('index')
    test_employee_counts = dawid_skene_test_data_comparator_ml.groupby('company_id').size().to_dict()
    
    processed_distributions = {}
    for company_id, counts_list in company_distributions.items():
        if len(counts_list) > 0:
            linkedin_company_id = company_names.get(company_id, f'Company_{company_id}')
            company_info = company_db_mapping.get(linkedin_company_id, {})
            organization_name = company_info.get('organization_name', linkedin_company_id)
            total_headcount = company_info.get('total_headcount', 0)
            actual_employee_count = test_employee_counts.get(linkedin_company_id, 0)
            effective_company_size = max(total_headcount, actual_employee_count) if total_headcount > 0 else actual_employee_count
            
            processed_distributions[linkedin_company_id] = {
                'company_name': organization_name,
                'n_employees': effective_company_size,
                'samples': [float(x) for x in counts_list]
            }
    
    logger.info(f"Comparator ML bootstrap analysis complete: {len(processed_distributions)} companies analyzed")
    return processed_distributions


@cache()
def correlated_probit_bootstrap_raw_results_comparator_non_ml(
    dawid_skene_validation_data: pd.DataFrame,
    dawid_skene_test_data_comparator_non_ml: pd.DataFrame,
    company_database_comparator_non_ml: pd.DataFrame,
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_n_machines: int,
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
) -> Dict[str, Any]:
    """Run probit bootstrap for comparator Non-ML pipeline."""
    if len(dawid_skene_test_data_comparator_non_ml) == 0:
        logger.warning("No test data for comparator Non-ML pipeline, returning empty results")
        return {}
    
    logger.info("Starting correlated probit bootstrap analysis for comparator Non-ML pipeline...")
    
    total_samples = correlated_probit_bootstrap_n_samples * correlated_probit_bootstrap_n_machines
    test_data_dict = dawid_skene_test_data_comparator_non_ml.to_dict('list')
    validation_data_dict = dawid_skene_validation_data.to_dict('list')
    company_database_dict = company_database_comparator_non_ml.to_dict('list')
    
    from ml_headcount.modal_functions import run_correlated_probit_bootstrap_modal
    from ml_headcount.modal_functions import NUM_CORES_PER_MACHINE
    
    if correlated_probit_bootstrap_n_machines == 1:
        company_distributions = run_correlated_probit_bootstrap_modal.remote(
            test_data_dict=test_data_dict,
            validation_data_dict=validation_data_dict,
            start_iter=0,
            n_samples=correlated_probit_bootstrap_n_samples,
            machine_idx=0,
            n_machines=1,
            n_cores_per_machine=NUM_CORES_PER_MACHINE,
            prior_alpha=correlated_probit_bootstrap_prior_alpha,
            prior_beta=correlated_probit_bootstrap_prior_beta,
            company_database_dict=company_database_dict,
            companies_needing_synthetic_dict=None,
            tetrachoric_corr_list=None,
            use_synthetic_only=False,
            org_type="comparator_non_ml"
        )
    else:
        from concurrent.futures import ThreadPoolExecutor
        
        def launch_machine(machine_idx):
            start_iter = machine_idx * correlated_probit_bootstrap_n_samples
            return run_correlated_probit_bootstrap_modal.remote(
                test_data_dict=test_data_dict,
                validation_data_dict=validation_data_dict,
                start_iter=start_iter,
                n_samples=correlated_probit_bootstrap_n_samples,
                machine_idx=machine_idx,
                n_machines=correlated_probit_bootstrap_n_machines,
                n_cores_per_machine=NUM_CORES_PER_MACHINE,
                prior_alpha=correlated_probit_bootstrap_prior_alpha,
                prior_beta=correlated_probit_bootstrap_prior_beta,
                company_database_dict=company_database_dict,
                companies_needing_synthetic_dict=None,
                tetrachoric_corr_list=None,
                use_synthetic_only=False,
                org_type="comparator_non_ml"
            )
        
        with ThreadPoolExecutor(max_workers=correlated_probit_bootstrap_n_machines) as executor:
            future_to_machine = {
                executor.submit(launch_machine, machine_idx): machine_idx 
                for machine_idx in range(correlated_probit_bootstrap_n_machines)
            }
            
            machine_results = [None] * correlated_probit_bootstrap_n_machines
            for future in future_to_machine:
                machine_idx = future_to_machine[future]
                try:
                    result = future.result()
                    machine_results[machine_idx] = result
                except Exception as e:
                    logger.error(f"Machine {machine_idx+1} failed: {e}")
                    raise
            
        all_results = []
        for i, task in enumerate(machine_results):
            if task is not None:
                result = task
                all_results.append(result)
        
        company_distributions = {}
        for machine_result in all_results:
            for company_id, counts_list in machine_result.items():
                if company_id not in company_distributions:
                    company_distributions[company_id] = []
                company_distributions[company_id].extend(counts_list)
    
    if company_distributions is None or len(company_distributions) == 0:
        return {}
    
    company_names = dawid_skene_test_data_comparator_non_ml.groupby('group')['company_id'].first().to_dict()
    company_db_mapping = company_database_comparator_non_ml.groupby('linkedin_id').agg({
        'organization_name': 'first',
        'total_headcount': 'first'
    }).to_dict('index')
    test_employee_counts = dawid_skene_test_data_comparator_non_ml.groupby('company_id').size().to_dict()
    
    processed_distributions = {}
    for company_id, counts_list in company_distributions.items():
        if len(counts_list) > 0:
            linkedin_company_id = company_names.get(company_id, f'Company_{company_id}')
            company_info = company_db_mapping.get(linkedin_company_id, {})
            organization_name = company_info.get('organization_name', linkedin_company_id)
            total_headcount = company_info.get('total_headcount', 0)
            actual_employee_count = test_employee_counts.get(linkedin_company_id, 0)
            effective_company_size = max(total_headcount, actual_employee_count) if total_headcount > 0 else actual_employee_count
            
            processed_distributions[linkedin_company_id] = {
                'company_name': organization_name,
                'n_employees': effective_company_size,
                'samples': [float(x) for x in counts_list]
            }
    
    logger.info(f"Comparator Non-ML bootstrap analysis complete: {len(processed_distributions)} companies analyzed")
    return processed_distributions


def _filter_headcount_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out companies with headcount mismatch (>3x difference between total_headcount and claude_total_employees).
    
    Args:
        df: DataFrame with 'linkedin_id', 'total_headcount', and 'claude_total_employees' columns
        
    Returns:
        Filtered DataFrame excluding companies with headcount mismatch
    """
    if 'headcount_mismatch' in df.columns:
        n_before = len(df)
        df = df[~df['headcount_mismatch'].fillna(False)].copy()
        n_excluded = n_before - len(df)
        if n_excluded > 0:
            logger.info(f"Excluded {n_excluded} companies with headcount mismatch (>3x difference)")
    return df


def _create_probit_bootstrap_results(
    correlated_probit_bootstrap_raw_results: Dict[str, Any],
    pipeline_name: str
) -> pd.DataFrame:
    """
    Helper function to create summary CSV from bootstrap results.
    
    Args:
        correlated_probit_bootstrap_raw_results: Raw bootstrap distributions with metadata
        pipeline_name: Name of the pipeline for logging
        
    Returns:
        DataFrame with summary statistics for each company
    """
    logger.info(f"Creating {pipeline_name} bootstrap results summary CSV...")
    
    if not correlated_probit_bootstrap_raw_results:
        return pd.DataFrame(columns=['linkedin_id', 'company_name', 'mean', 'std', 'q10', 'q50', 'q90', 'n_employees'])
    
    results_list = []
    for linkedin_id, company_data in correlated_probit_bootstrap_raw_results.items():
        samples = company_data['samples']
        if len(samples) > 0:
            results_list.append({
                'linkedin_id': linkedin_id,
                'company_name': company_data['company_name'],
                'mean': np.mean(samples),
                'std': np.std(samples),
                'q10': np.percentile(samples, 10),
                'q50': np.percentile(samples, 50),
                'q90': np.percentile(samples, 90),
                'n_employees': company_data['n_employees']
            })
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('mean', ascending=False)
    
    logger.info(f"Created {pipeline_name} summary CSV with {len(results_df)} companies")
    return results_df


def _create_probit_results_orgs(
    correlated_probit_bootstrap_results: pd.DataFrame,
    pipeline_name: str
) -> pd.DataFrame:
    """
    Helper function to create final output table for probit results.
    
    Args:
        correlated_probit_bootstrap_results: Summary DataFrame with bootstrap results
        pipeline_name: Name of the pipeline for logging
        
    Returns:
        DataFrame with final output table
    """
    logger.info(f"Creating {pipeline_name} probit results output table...")
    
    if len(correlated_probit_bootstrap_results) == 0:
        return pd.DataFrame(columns=['linkedin_id', 'mean', 'q10', 'q50', 'q90'])
    
    result_df = correlated_probit_bootstrap_results[['linkedin_id', 'mean', 'q10', 'q50', 'q90']].copy()
    
    logger.info(f"Created {pipeline_name} probit results table with {len(result_df)} companies")
    return result_df


@save_to.csv(path=source("correlated_probit_bootstrap_results_main_output_path"))
def correlated_probit_bootstrap_results_main(
    correlated_probit_bootstrap_raw_results_main: Dict[str, Any]
) -> pd.DataFrame:
    """Create summary CSV from main pipeline bootstrap results."""
    return _create_probit_bootstrap_results(correlated_probit_bootstrap_raw_results_main, "Main pipeline")


@save_to.csv(path=source("correlated_probit_bootstrap_results_comparator_ml_output_path"))
def correlated_probit_bootstrap_results_comparator_ml(
    correlated_probit_bootstrap_raw_results_comparator_ml: Dict[str, Any]
) -> pd.DataFrame:
    """Create summary CSV from comparator ML pipeline bootstrap results."""
    return _create_probit_bootstrap_results(correlated_probit_bootstrap_raw_results_comparator_ml, "Comparator ML")


@save_to.csv(path=source("correlated_probit_bootstrap_results_comparator_non_ml_output_path"))
def correlated_probit_bootstrap_results_comparator_non_ml(
    correlated_probit_bootstrap_raw_results_comparator_non_ml: Dict[str, Any]
) -> pd.DataFrame:
    """Create summary CSV from comparator Non-ML pipeline bootstrap results."""
    return _create_probit_bootstrap_results(correlated_probit_bootstrap_raw_results_comparator_non_ml, "Comparator Non-ML")


@save_to.csv(path=source("probit_results_main_orgs_output_path"))
def probit_results_main_orgs(
    correlated_probit_bootstrap_results_main: pd.DataFrame
) -> pd.DataFrame:
    """Create final output table for main pipeline probit results."""
    return _create_probit_results_orgs(correlated_probit_bootstrap_results_main, "Main pipeline")


@save_to.csv(path=source("probit_results_comparator_ml_orgs_output_path"))
def probit_results_comparator_ml_orgs(
    correlated_probit_bootstrap_results_comparator_ml: pd.DataFrame
) -> pd.DataFrame:
    """Create final output table for comparator ML pipeline probit results."""
    return _create_probit_results_orgs(correlated_probit_bootstrap_results_comparator_ml, "Comparator ML")


@save_to.csv(path=source("probit_results_comparator_non_ml_orgs_output_path"))
def probit_results_comparator_non_ml_orgs(
    correlated_probit_bootstrap_results_comparator_non_ml: pd.DataFrame
) -> pd.DataFrame:
    """Create final output table for comparator Non-ML pipeline probit results."""
    return _create_probit_results_orgs(correlated_probit_bootstrap_results_comparator_non_ml, "Comparator Non-ML")


# ============================================================================
# FINAL RESULTS: JOIN COMPANY DATABASE WITH PROBIT RESULTS
# ============================================================================

@cache()
def global_synthetic_adjustment_factor(
    company_database_complete: pd.DataFrame,
    company_database_comparator_ml: pd.DataFrame,
    company_database_comparator_non_ml: pd.DataFrame,
    correlated_probit_bootstrap_results_main: pd.DataFrame,
    probit_results_comparator_ml_orgs: pd.DataFrame,
    probit_results_comparator_non_ml_orgs: pd.DataFrame,
    correlated_probit_bootstrap_results_synthetic: pd.DataFrame
) -> float:
    """
    Calculate a single global adjustment factor for synthetic estimates across all company datasets.
    
    This factor is the median ratio of synthetic/real estimates from all companies 
    that have both real and synthetic data (with real mean >= 1.0).
    
    Returns:
        Global adjustment factor (float)
    """
    logger.info("Calculating global synthetic adjustment factor across all datasets...")
    
    # Dictionary to store ratios by linkedin_id (to avoid duplicates)
    # Key: linkedin_id, Value: ratio
    ratio_dict = {}
    
    # Helper function to extract ratios from a dataset
    def extract_ratios(company_db, probit_results, synthetic_results):
        # Normalize inputs
        if not isinstance(probit_results, pd.DataFrame):
            probit_df = pd.DataFrame(columns=['linkedin_id', 'mean'])
        else:
            probit_df = probit_results[['linkedin_id', 'mean']].copy() if 'mean' in probit_results.columns else pd.DataFrame(columns=['linkedin_id', 'mean'])
        
        if not isinstance(synthetic_results, pd.DataFrame):
            synthetic_df = pd.DataFrame(columns=['linkedin_id', 'mean'])
        else:
            synthetic_df = synthetic_results[['linkedin_id', 'mean']].copy() if 'mean' in synthetic_results.columns else pd.DataFrame(columns=['linkedin_id', 'mean'])
        
        # Merge company database with probit and synthetic results
        merged = company_db[['linkedin_id']].merge(
            probit_df.rename(columns={'mean': 'real_mean'}),
            on='linkedin_id',
            how='left'
        ).merge(
            synthetic_df.rename(columns={'mean': 'synthetic_mean'}),
            on='linkedin_id',
            how='left'
        )
        
        # Filter to companies with both real and synthetic estimates (real mean >= 1)
        has_both = (merged['real_mean'].notna()) & (merged['synthetic_mean'] > 0) & (merged['real_mean'] >= 1.0)
        if has_both.sum() > 0:
            ratios = merged.loc[has_both, 'synthetic_mean'] / merged.loc[has_both, 'real_mean']
            linkedin_ids = merged.loc[has_both, 'linkedin_id']
            return ratios.values, linkedin_ids.values
        return np.array([]), np.array([])
    
    # Extract ratios from main orgs (these take precedence in case of overlap)
    ratios_main, ids_main = extract_ratios(
        company_database_complete,
        correlated_probit_bootstrap_results_main,
        correlated_probit_bootstrap_results_synthetic
    )
    for linkedin_id, ratio in zip(ids_main, ratios_main):
        ratio_dict[linkedin_id] = ratio
    logger.info(f"  Main orgs: {len(ratios_main)} companies with both estimates")
    
    # Extract ratios from comparator ML (only add if not already in main)
    ratios_ml, ids_ml = extract_ratios(
        company_database_comparator_ml,
        probit_results_comparator_ml_orgs,
        correlated_probit_bootstrap_results_synthetic
    )
    new_ml = 0
    for linkedin_id, ratio in zip(ids_ml, ratios_ml):
        if linkedin_id not in ratio_dict:
            ratio_dict[linkedin_id] = ratio
            new_ml += 1
    logger.info(f"  Comparator ML: {len(ratios_ml)} companies with both estimates ({new_ml} new, {len(ratios_ml) - new_ml} overlapping with main)")
    
    # Extract ratios from comparator Non-ML (only add if not already present)
    ratios_non_ml, ids_non_ml = extract_ratios(
        company_database_comparator_non_ml,
        probit_results_comparator_non_ml_orgs,
        correlated_probit_bootstrap_results_synthetic
    )
    new_non_ml = 0
    for linkedin_id, ratio in zip(ids_non_ml, ratios_non_ml):
        if linkedin_id not in ratio_dict:
            ratio_dict[linkedin_id] = ratio
            new_non_ml += 1
    logger.info(f"  Comparator Non-ML: {len(ratios_non_ml)} companies with both estimates ({new_non_ml} new, {len(ratios_non_ml) - new_non_ml} overlapping)")
    
    # Calculate global median from unique companies only
    unique_ratios = list(ratio_dict.values())
    if len(unique_ratios) > 0:
        adjustment_factor = float(np.median(unique_ratios))
        logger.info(f"Global adjustment factor: {adjustment_factor:.6f} from {len(unique_ratios)} unique companies")
    else:
        adjustment_factor = 1.0
        logger.warning("No companies with both real and synthetic estimates; using adjustment factor = 1.0")
    
    return adjustment_factor


@save_to.csv(path=source("final_results_main_orgs_output_path"))
@cache()
def final_results_main_orgs(
    company_database_complete: pd.DataFrame,
    correlated_probit_bootstrap_results_main: pd.DataFrame,
    correlated_probit_bootstrap_results_synthetic: pd.DataFrame,
    global_synthetic_adjustment_factor: float
) -> pd.DataFrame:
    """
    Join main company database with probit results (real data) and synthetic results.
    
    Includes all rows from company_database_complete (403 orgs).
    Uses the main bootstrap results to get real estimates.
    Missing probit estimates are left as NaN (companies without employee-level data).
    Synthetic results are added as separate columns (synthetic_mean, synthetic_q10, etc.).
    """
    logger.info("Creating final results for main orgs (403 orgs)...")
    
    # Normalize probit results input
    if not isinstance(correlated_probit_bootstrap_results_main, pd.DataFrame):
        probit_df = pd.DataFrame(columns=['linkedin_id', 'mean', 'q10', 'q50', 'q90'])
    else:
        probit_df = correlated_probit_bootstrap_results_main
    
    # Select probit columns for joining (real data)
    # Include headcount_mismatch and headcount_ratio if available (calculated upstream)
    probit_cols = ['linkedin_id', 'mean', 'q10', 'q50', 'q90', 'headcount_ratio', 'headcount_mismatch']
    # Only include columns that exist in probit_df
    probit_cols = [col for col in probit_cols if col in probit_df.columns]
    probit_subset = probit_df.reindex(columns=probit_cols, fill_value=np.nan).copy()
    
    # Rename probit columns to avoid conflicts (but keep headcount columns as-is)
    probit_subset = probit_subset.rename(columns={
        'mean': 'probit_mean',
        'q10': 'probit_q10',
        'q50': 'probit_q50',
        'q90': 'probit_q90'
    })
    
    # Normalize synthetic results input (can be dict if cache returns empty)
    if not isinstance(correlated_probit_bootstrap_results_synthetic, pd.DataFrame):
        synthetic_df = pd.DataFrame(columns=['linkedin_id', 'mean', 'q10', 'q50', 'q90'])
    else:
        synthetic_df = correlated_probit_bootstrap_results_synthetic

    # Select synthetic columns for joining
    synthetic_cols = ['linkedin_id', 'mean', 'q10', 'q50', 'q90']
    synthetic_subset = synthetic_df.reindex(columns=synthetic_cols, fill_value=np.nan).copy()
    
    # Rename synthetic columns
    synthetic_subset = synthetic_subset.rename(columns={
        'mean': 'synthetic_mean',
        'q10': 'synthetic_q10',
        'q50': 'synthetic_q50',
        'q90': 'synthetic_q90'
    })
    
    # Left join on linkedin_id (real data first)
    result = company_database_complete.merge(
        probit_subset,
        on='linkedin_id',
        how='left'
    )
    
    # Then join synthetic data
    result = result.merge(
        synthetic_subset,
        on='linkedin_id',
        how='left'
    )
    
    # Leave missing probit values as NaN (companies without employee-level data)
    # Don't fill with 0.0 - NaN indicates no employee-level data available
    probit_fill_cols = ['probit_mean', 'probit_q10', 'probit_q50', 'probit_q90']
    for col in probit_fill_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    # Fill missing synthetic values with 0 (companies without keyword filters get 0)
    synthetic_fill_cols = ['synthetic_mean', 'synthetic_q10', 'synthetic_q50', 'synthetic_q90']
    for col in synthetic_fill_cols:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)
    
    # Fill missing total_headcount with 0
    if 'total_headcount' in result.columns:
        result['total_headcount'] = result['total_headcount'].fillna(0.0)
    
    # Rename back to original names for real data
    result = result.rename(columns={
        'probit_mean': 'mean',
        'probit_q10': 'q10',
        'probit_q50': 'q50',
        'probit_q90': 'q90'
    })

    # Use global adjustment factor for synthetic estimates
    adjustment_factor = global_synthetic_adjustment_factor
    logger.info(f"Using global adjustment factor: {adjustment_factor:.6f} for main orgs")
    
    # Apply adjustment to create adjusted_synthetic columns
    result['adjusted_synthetic_mean'] = result['synthetic_mean'] / adjustment_factor
    result['adjusted_synthetic_q10'] = result['synthetic_q10'] / adjustment_factor
    result['adjusted_synthetic_q50'] = result['synthetic_q50'] / adjustment_factor
    result['adjusted_synthetic_q90'] = result['synthetic_q90'] / adjustment_factor
    
    # Store the adjustment factor used
    result['synthetic_adjustment_factor'] = adjustment_factor

    # Calculate headcount_mismatch for all companies using centralized helper
    # (company_database=None since headcount columns are already in result from the merge)
    if 'claude_total_employees' in result.columns and 'total_headcount' in result.columns:
        result = _calculate_headcount_mismatch(result, company_database=None, filter_mismatched=False)
    else:
        # No headcount columns available - set defaults
        result['headcount_ratio'] = np.nan
        result['headcount_mismatch'] = False
    
    # Ensure columns exist for robust logic
    if 'mean' not in result.columns:
        result['mean'] = np.nan
    
    # Determine when to use synthetic: no employee-level data or headcount mismatch
    result['use_synthetic_probit'] = result['mean'].isna() | result['headcount_mismatch']

    
    logger.info(f"Created final results for main orgs: {len(result)} companies")
    logger.info(f"  Real data estimates: {result['mean'].notna().sum()} companies")
    logger.info(f"  Synthetic estimates: {(result['synthetic_mean'] > 0).sum()} companies (non-zero)")
    logger.info(f"  Adjusted synthetic estimates: {(result['adjusted_synthetic_mean'] > 0).sum()} companies (non-zero)")
    return result


@save_to.csv(path=source("final_results_core_orgs_output_path"))
@cache()
def final_results_core_orgs(
    company_database_complete: pd.DataFrame,
    correlated_probit_bootstrap_results_main: pd.DataFrame
) -> pd.DataFrame:
    """
    Join core company database (max_headcount=True AND max_population=True) with probit results.
    
    Filters for 130 core orgs where max_headcount=True AND max_population=True.
    Uses the main bootstrap results to get real estimates.
    Missing probit estimates are left as NaN (companies without employee-level data).
    """
    logger.info("Creating final results for core orgs (130 orgs)...")
    
    # Filter for core orgs: max_headcount=True AND max_population=True
    # Use .fillna(False) to handle NaN values
    core_orgs = company_database_complete[
        (company_database_complete['max_headcount'].fillna(False) == True) &
        (company_database_complete['max_population'].fillna(False) == True)
    ].copy()
    
    logger.info(f"Filtered to {len(core_orgs)} core orgs (max_headcount=True AND max_population=True)")
    
    # Normalize probit results input
    if not isinstance(correlated_probit_bootstrap_results_main, pd.DataFrame):
        probit_df = pd.DataFrame(columns=['linkedin_id', 'mean', 'q10', 'q50', 'q90'])
    else:
        probit_df = correlated_probit_bootstrap_results_main
    
    # Select only probit columns for joining
    probit_cols = ['linkedin_id', 'mean', 'q10', 'q50', 'q90']
    probit_subset = probit_df.reindex(columns=probit_cols, fill_value=np.nan).copy()
    
    # Rename probit columns to avoid conflicts
    probit_subset = probit_subset.rename(columns={
        'mean': 'probit_mean',
        'q10': 'probit_q10',
        'q50': 'probit_q50',
        'q90': 'probit_q90'
    })
    
    # Left join on linkedin_id
    result = core_orgs.merge(
        probit_subset,
        on='linkedin_id',
        how='left'
    )
    
    # Leave missing probit values as NaN (companies without employee-level data)
    # Don't fill with 0.0 - NaN indicates no employee-level data available
    probit_fill_cols = ['probit_mean', 'probit_q10', 'probit_q50', 'probit_q90']
    for col in probit_fill_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    # Rename back to original names
    result = result.rename(columns={
        'probit_mean': 'mean',
        'probit_q10': 'q10',
        'probit_q50': 'q50',
        'probit_q90': 'q90'
    })
    
    logger.info(f"Created final results for core orgs: {len(result)} companies")
    return result


@save_to.csv(path=source("final_results_comparator_ml_output_path"))
@cache()
def final_results_comparator_ml(
    company_database_comparator_ml: pd.DataFrame,
    probit_results_comparator_ml_orgs: pd.DataFrame,
    correlated_probit_bootstrap_results_synthetic: pd.DataFrame,
    global_synthetic_adjustment_factor: float
) -> pd.DataFrame:
    """
    Join comparator ML company database with probit results (real data) and synthetic results.
    
    Includes all rows from company_database_comparator_ml.
    Missing probit estimates are left as NaN (companies without employee-level data).
    Synthetic results are added as separate columns (synthetic_mean, synthetic_q10, etc.).
    """
    logger.info("Creating final results for comparator ML orgs...")
    
    # Normalize probit results input
    if not isinstance(probit_results_comparator_ml_orgs, pd.DataFrame):
        probit_df = pd.DataFrame(columns=['linkedin_id', 'mean', 'q10', 'q50', 'q90'])
    else:
        probit_df = probit_results_comparator_ml_orgs
    
    # Select only probit columns for joining (real data)
    probit_cols = ['linkedin_id', 'mean', 'q10', 'q50', 'q90']
    probit_subset = probit_df.reindex(columns=probit_cols, fill_value=np.nan).copy()
    
    # Rename probit columns to avoid conflicts
    probit_subset = probit_subset.rename(columns={
        'mean': 'probit_mean',
        'q10': 'probit_q10',
        'q50': 'probit_q50',
        'q90': 'probit_q90'
    })
    
    # Normalize synthetic results input
    if not isinstance(correlated_probit_bootstrap_results_synthetic, pd.DataFrame):
        synthetic_df = pd.DataFrame(columns=['linkedin_id', 'mean', 'q10', 'q50', 'q90'])
    else:
        synthetic_df = correlated_probit_bootstrap_results_synthetic

    # Select synthetic columns for joining
    synthetic_cols = ['linkedin_id', 'mean', 'q10', 'q50', 'q90']
    synthetic_subset = synthetic_df.reindex(columns=synthetic_cols, fill_value=np.nan).copy()
    
    # Rename synthetic columns
    synthetic_subset = synthetic_subset.rename(columns={
        'mean': 'synthetic_mean',
        'q10': 'synthetic_q10',
        'q50': 'synthetic_q50',
        'q90': 'synthetic_q90'
    })
    
    # Left join on linkedin_id (real data first)
    result = company_database_comparator_ml.merge(
        probit_subset,
        on='linkedin_id',
        how='left'
    )
    
    # Then join synthetic data
    result = result.merge(
        synthetic_subset,
        on='linkedin_id',
        how='left'
    )
    
    # Leave missing probit values as NaN (companies without employee-level data)
    # Don't fill with 0.0 - NaN indicates no employee-level data available
    probit_fill_cols = ['probit_mean', 'probit_q10', 'probit_q50', 'probit_q90']
    for col in probit_fill_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    # Fill missing synthetic values with 0 (companies without keyword filters get 0)
    synthetic_fill_cols = ['synthetic_mean', 'synthetic_q10', 'synthetic_q50', 'synthetic_q90']
    for col in synthetic_fill_cols:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)
    
    # Fill missing total_headcount with 0
    if 'total_headcount' in result.columns:
        result['total_headcount'] = result['total_headcount'].fillna(0.0)
    
    # Rename back to original names for real data
    result = result.rename(columns={
        'probit_mean': 'mean',
        'probit_q10': 'q10',
        'probit_q50': 'q50',
        'probit_q90': 'q90'
    })
    
    # Calculate adjustment factor for synthetic estimates (Option 1: simple scaling)
    # Use global adjustment factor for synthetic estimates
    adjustment_factor = global_synthetic_adjustment_factor
    logger.info(f"Using global adjustment factor: {adjustment_factor:.6f} for comparator ML orgs")
    
    # Apply adjustment to create adjusted_synthetic columns
    result['adjusted_synthetic_mean'] = result['synthetic_mean'] / adjustment_factor
    result['adjusted_synthetic_q10'] = result['synthetic_q10'] / adjustment_factor
    result['adjusted_synthetic_q50'] = result['synthetic_q50'] / adjustment_factor
    result['adjusted_synthetic_q90'] = result['synthetic_q90'] / adjustment_factor
    
    # Store the adjustment factor used
    result['synthetic_adjustment_factor'] = adjustment_factor
    
    logger.info(f"Created final results for comparator ML orgs: {len(result)} companies")
    logger.info(f"  Real data estimates: {result['mean'].notna().sum()} companies")
    logger.info(f"  Synthetic estimates: {(result['synthetic_mean'] > 0).sum()} companies (non-zero)")
    logger.info(f"  Adjusted synthetic estimates: {(result['adjusted_synthetic_mean'] > 0).sum()} companies (non-zero)")
    return result


@save_to.csv(path=source("final_results_comparator_non_ml_output_path"))
@cache()
def final_results_comparator_non_ml(
    company_database_comparator_non_ml: pd.DataFrame,
    probit_results_comparator_non_ml_orgs: pd.DataFrame,
    correlated_probit_bootstrap_results_synthetic: pd.DataFrame,
    global_synthetic_adjustment_factor: float
) -> pd.DataFrame:
    """
    Join comparator Non-ML company database with probit results (real data) and synthetic results.
    
    Includes all rows from company_database_comparator_non_ml.
    Missing probit estimates are left as NaN (companies without employee-level data).
    Synthetic results are added as separate columns (synthetic_mean, synthetic_q10, etc.).
    """
    logger.info("Creating final results for comparator Non-ML orgs...")
    
    # Normalize probit results input
    if not isinstance(probit_results_comparator_non_ml_orgs, pd.DataFrame):
        probit_df = pd.DataFrame(columns=['linkedin_id', 'mean', 'q10', 'q50', 'q90'])
    else:
        probit_df = probit_results_comparator_non_ml_orgs
    
    # Select only probit columns for joining (real data)
    probit_cols = ['linkedin_id', 'mean', 'q10', 'q50', 'q90']
    probit_subset = probit_df.reindex(columns=probit_cols, fill_value=np.nan).copy()
    
    # Rename probit columns to avoid conflicts
    probit_subset = probit_subset.rename(columns={
        'mean': 'probit_mean',
        'q10': 'probit_q10',
        'q50': 'probit_q50',
        'q90': 'probit_q90'
    })
    
    # Normalize synthetic results input
    if not isinstance(correlated_probit_bootstrap_results_synthetic, pd.DataFrame):
        synthetic_df = pd.DataFrame(columns=['linkedin_id', 'mean', 'q10', 'q50', 'q90'])
    else:
        synthetic_df = correlated_probit_bootstrap_results_synthetic

    # Select synthetic columns for joining
    synthetic_cols = ['linkedin_id', 'mean', 'q10', 'q50', 'q90']
    synthetic_subset = synthetic_df.reindex(columns=synthetic_cols, fill_value=np.nan).copy()
    
    # Rename synthetic columns
    synthetic_subset = synthetic_subset.rename(columns={
        'mean': 'synthetic_mean',
        'q10': 'synthetic_q10',
        'q50': 'synthetic_q50',
        'q90': 'synthetic_q90'
    })
    
    # Left join on linkedin_id (real data first)
    result = company_database_comparator_non_ml.merge(
        probit_subset,
        on='linkedin_id',
        how='left'
    )
    
    # Then join synthetic data
    result = result.merge(
        synthetic_subset,
        on='linkedin_id',
        how='left'
    )
    
    # Leave missing probit values as NaN (companies without employee-level data)
    # Don't fill with 0.0 - NaN indicates no employee-level data available
    probit_fill_cols = ['probit_mean', 'probit_q10', 'probit_q50', 'probit_q90']
    for col in probit_fill_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    # Fill missing synthetic values with 0 (companies without keyword filters get 0)
    synthetic_fill_cols = ['synthetic_mean', 'synthetic_q10', 'synthetic_q50', 'synthetic_q90']
    for col in synthetic_fill_cols:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)
    
    # Fill missing total_headcount with 0
    if 'total_headcount' in result.columns:
        result['total_headcount'] = result['total_headcount'].fillna(0.0)
    
    # Rename back to original names for real data
    result = result.rename(columns={
        'probit_mean': 'mean',
        'probit_q10': 'q10',
        'probit_q50': 'q50',
        'probit_q90': 'q90'
    })
    
    # Use global adjustment factor for synthetic estimates
    adjustment_factor = global_synthetic_adjustment_factor
    logger.info(f"Using global adjustment factor: {adjustment_factor:.6f} for comparator Non-ML orgs")
    
    # Apply adjustment to create adjusted_synthetic columns
    result['adjusted_synthetic_mean'] = result['synthetic_mean'] / adjustment_factor
    result['adjusted_synthetic_q10'] = result['synthetic_q10'] / adjustment_factor
    result['adjusted_synthetic_q50'] = result['synthetic_q50'] / adjustment_factor
    result['adjusted_synthetic_q90'] = result['synthetic_q90'] / adjustment_factor
    
    # Store the adjustment factor used
    result['synthetic_adjustment_factor'] = adjustment_factor
    
    logger.info(f"Created final results for comparator Non-ML orgs: {len(result)} companies")
    logger.info(f"  Real data estimates: {result['mean'].notna().sum()} companies")
    logger.info(f"  Synthetic estimates: {(result['synthetic_mean'] > 0).sum()} companies (non-zero)")
    logger.info(f"  Adjusted synthetic estimates: {(result['adjusted_synthetic_mean'] > 0).sum()} companies (non-zero)")
    return result


# ============================================================================
# COMBINED FINAL RESULTS: ALL COMPANIES
# ============================================================================

@save_to.csv(path=source("final_results_all_output_path"))
@cache()
def final_results_all(
    final_results_main_orgs: pd.DataFrame,
    final_results_comparator_ml: pd.DataFrame,
    final_results_comparator_non_ml: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine all final results into a single CSV with all companies.
    
    Combines:
    - final_results_main_orgs (all companies from company_database_complete)
    - final_results_comparator_ml (all comparator ML companies)
    - final_results_comparator_non_ml (all comparator Non-ML companies)
    
    Removes duplicates based on linkedin_id (if a company appears in multiple sources,
    the main_orgs version takes precedence).
    """
    logger.info("Creating combined final results with all companies...")
    
    # Start with main orgs (these take precedence)
    result = final_results_main_orgs.copy()
    
    # Add comparator ML companies that aren't already in main_orgs
    main_linkedin_ids = set(final_results_main_orgs['linkedin_id'].dropna())
    comparator_ml_new = final_results_comparator_ml[
        ~final_results_comparator_ml['linkedin_id'].isin(main_linkedin_ids)
    ].copy()
    
    # Add comparator Non-ML companies that aren't already in main_orgs
    comparator_non_ml_new = final_results_comparator_non_ml[
        ~final_results_comparator_non_ml['linkedin_id'].isin(main_linkedin_ids)
    ].copy()
    
    # Combine all
    result = pd.concat([result, comparator_ml_new, comparator_non_ml_new], ignore_index=True)
    
    logger.info(f"Created combined final results: {len(result)} total companies")
    logger.info(f"  From main_orgs: {len(final_results_main_orgs)}")
    logger.info(f"  From comparator_ml (new): {len(comparator_ml_new)}")
    logger.info(f"  From comparator_non_ml (new): {len(comparator_non_ml_new)}")
    
    return result


def _run_bootstrap_with_modal(
    test_data_dict: dict,
    validation_data_dict: dict,
    n_samples: int,
    n_machines: int,
    n_cores_per_machine: int,
    timeout: int
) -> Dict[str, List[float]]:
    """Run bootstrap analysis using Modal Labs"""
    from ml_headcount.modal_functions import run_correlated_probit_bootstrap_modal
    
    # Run on Modal using the existing Modal function
    if n_machines == 1:
        # Single machine case
        company_distributions = run_correlated_probit_bootstrap_modal.remote(
            test_data_dict=test_data_dict,
            validation_data_dict=validation_data_dict,
            start_iter=0,
            n_samples=n_samples,
            machine_idx=0,
            n_machines=1,
            n_cores_per_machine=n_cores_per_machine
        )
    else:
        # Multiple machines case - launch all machines in parallel
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def launch_machine(machine_idx):
            """Launch a single machine and return its result"""
            start_iter = machine_idx * n_samples
            return run_correlated_probit_bootstrap_modal.remote(
                test_data_dict=test_data_dict,
                validation_data_dict=validation_data_dict,
                start_iter=start_iter,
                n_samples=n_samples,
                machine_idx=machine_idx,
                n_machines=n_machines,
                n_cores_per_machine=n_cores_per_machine
            )
        
        # Launch all machines in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_machines) as executor:
            # Submit all machines to run in parallel
            future_to_machine = {
                executor.submit(launch_machine, machine_idx): machine_idx 
                for machine_idx in range(n_machines)
            }
            
            # Collect results as they complete
            machine_results = [None] * n_machines
            for future in future_to_machine:
                machine_idx = future_to_machine[future]
                try:
                    result = future.result()
                    machine_results[machine_idx] = result
                    logger.info(f"Machine {machine_idx+1} launched successfully")
                except Exception as e:
                    logger.error(f"Machine {machine_idx+1} failed to launch: {e}")
                    raise
        
        # Now collect the actual results (this will block until all complete)
        for i, task in enumerate(machine_results):
            if task is not None:
                result = task  # This blocks until this specific machine completes
                machine_results[i] = result
                logger.info(f"Machine {i+1} completed")
        
        # Combine results from all machines
        logger.info("Combining results from all machines...")
        company_distributions = {}
        
        for machine_result in machine_results:
            for company_id, counts_list in machine_result.items():
                if company_id not in company_distributions:
                    company_distributions[company_id] = []
                company_distributions[company_id].extend(counts_list)
    
    return company_distributions


def _run_bootstrap_local(
    test_data_dict: dict,
    validation_data_dict: dict,
    n_samples: int,
    n_machines: int,
    n_cores_per_machine: int
) -> Dict[str, List[float]]:
    """Run bootstrap analysis locally"""
    logger.info("Running bootstrap analysis locally...")
    
    # For local processing, we'll run all iterations on a single machine
    # This is a simplified version of the Modal implementation
    company_distributions = {}
    
    for iteration in range(n_samples):
        if iteration % 10 == 0:
            logger.info(f"Bootstrap iteration {iteration+1}/{n_samples}")
        
        # Run single bootstrap iteration
        result = _run_single_bootstrap_iteration((
            test_data_dict, validation_data_dict, iteration
        ))
        
        # Add results to distributions
        for company_id, count in result.items():
            if company_id not in company_distributions:
                company_distributions[company_id] = []
            company_distributions[company_id].append(count)
    
    return company_distributions


def correlated_probit_bootstrap_distributions(
    correlated_probit_bootstrap_raw_results: Dict[str, Any],
    correlated_probit_bootstrap_distributions_output_path: str
) -> Dict[str, List[float]]:
    """
    Save bootstrap distributions as NetCDF.
    
    Args:
        correlated_probit_bootstrap_raw_results: Raw bootstrap distributions with metadata
        correlated_probit_bootstrap_distributions_output_path: Output path for NetCDF file
        
    Returns:
        Dictionary with company distributions
    """
    logger.info("Saving bootstrap distributions as NetCDF...")
    
    if not correlated_probit_bootstrap_raw_results:
        logger.warning("No raw bootstrap results to save")
        return {}
    
    import xarray as xr
    import numpy as np
    
    # Extract data from raw results
    company_ids = []
    company_names = []
    n_employees_list = []
    distributions = {}
    
    for linkedin_id, company_data in correlated_probit_bootstrap_raw_results.items():
        company_ids.append(linkedin_id)
        company_names.append(company_data['company_name'])
        n_employees_list.append(company_data['n_employees'])
        distributions[linkedin_id] = company_data['samples']
    
    # Get maximum number of samples across all companies
    max_samples = max(len(samples) for samples in distributions.values())
    
    # Create distributions array (companies x samples)
    distributions_array = np.full((len(company_ids), max_samples), np.nan)
    for i, company_id in enumerate(company_ids):
        samples = distributions[company_id]
        distributions_array[i, :len(samples)] = samples
    
    # Create xarray Dataset
    ds = xr.Dataset({
        'company_names': (['company'], company_names),
        'ml_headcount_distributions': (['company', 'sample'], distributions_array),
        'n_employees': (['company'], n_employees_list)
    }, coords={
        'company': company_ids,
        'sample': range(max_samples)
    })
    
    # Add metadata
    ds.attrs['description'] = 'ML headcount bootstrap distributions from correlated probit analysis'
    ds.attrs['total_iterations'] = max_samples
    ds.attrs['n_machines'] = 1
    ds.attrs['samples_per_machine'] = max_samples
    
    # Save as NetCDF
    ds.to_netcdf(correlated_probit_bootstrap_distributions_output_path)
    logger.info(f"Saved bootstrap distributions to {correlated_probit_bootstrap_distributions_output_path}")
    logger.info(f"Saved {len(distributions)} companies with up to {max_samples} samples each")
    
    return distributions




def correlated_probit_bootstrap_plots_with_annotators(
    dawid_skene_test_data: pd.DataFrame,
    correlated_probit_bootstrap_raw_results: Dict[str, Any],
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_plots_with_annotators_output_path: str
) -> plt.Figure:
    """
    Create bootstrap plots with individual annotator estimates as scatter points.
    
    This function integrates the visualization from bootstrap_plot_with_annotators.py.
    
    Args:
        dawid_skene_test_data: Test data with annotator columns
        correlated_probit_bootstrap_raw_results: Raw bootstrap distributions with metadata
        correlated_probit_bootstrap_n_samples: Number of bootstrap samples
        
    Returns:
        Dictionary with plot information
    """
    logger.info("Creating bootstrap plots with annotator estimates...")
    
    if not correlated_probit_bootstrap_raw_results:
        logger.warning("No bootstrap results to plot")
        # Return empty figure when no results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No bootstrap results to plot', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Bootstrap Results with Annotators')
        return fig
    
    # Extract annotator columns
    annotator_cols = [col for col in dawid_skene_test_data.columns 
                     if col not in ['company_id', 'group']]
    
    # Compute individual annotator counts by company
    company_annotator_counts = {}
    for company_id in dawid_skene_test_data['company_id'].unique():
        company_mask = dawid_skene_test_data['company_id'] == company_id
        company_data = dawid_skene_test_data[company_mask]
        
        # Count individual annotator estimates
        annotator_counts = {}
        for col in annotator_cols:
            annotator_counts[col] = company_data[col].sum()
        
        company_annotator_counts[company_id] = {
            'company_name': company_id,  # Use company_id as the name for now
            'n_employees': len(company_data),
            **annotator_counts
        }
    
    # Create summary DataFrame from raw results
    results_list = []
    for linkedin_id, company_data in correlated_probit_bootstrap_raw_results.items():
        samples = company_data['samples']
        if len(samples) > 0:
            results_list.append({
                'linkedin_id': linkedin_id,
                'company_name': company_data['company_name'],
                'mean': np.mean(samples),
                'std': np.std(samples),
                'q10': np.percentile(samples, 10),
                'q90': np.percentile(samples, 90),
                'n_employees': company_data['n_employees']
            })
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('mean', ascending=False)
    
    # Get top 30 companies
    top_30 = results_df.head(30)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Company names for x-axis
    display_names = [name[:20] + '...' if len(name) > 20 else name 
                    for name in top_30['company_name']]
    x_pos = np.arange(len(display_names))
    
    # Plot bootstrap bars with error bars
    means = top_30['mean'].values
    # Ensure error bar values are non-negative (can happen with skewed distributions)
    ci_low = np.maximum(0, means - top_30['q10'].values)
    ci_high = np.maximum(0, top_30['q90'].values - means)
    
    bars = ax.bar(x_pos, means, 
                  yerr=[ci_low, ci_high],
                  capsize=5, alpha=0.7, color='steelblue',
                  label=f'Bootstrap Mean ± 80% CI\n({correlated_probit_bootstrap_n_samples} iterations)')
    
    # Plot individual annotator estimates as scatter points
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, col in enumerate(annotator_cols):
        if i < len(colors):
            # Get annotator counts for top companies
            annotator_values = []
            for _, row in top_30.iterrows():
                linkedin_id = row['linkedin_id']  # This is the LinkedIn company ID from bootstrap results
                if linkedin_id in company_annotator_counts:
                    annotator_values.append(company_annotator_counts[linkedin_id][col])
                else:
                    annotator_values.append(0)
            
            ax.scatter(x_pos, annotator_values, 
                      color=colors[i], marker=markers[i], s=60, alpha=0.8,
                      label=col.replace('_', ' ').title())
    
    # Customize the plot
    ax.set_xlabel('Companies (Top 30 by ML Headcount)', fontsize=12)
    ax.set_ylabel('Number of ML Experts', fontsize=12)
    ax.set_title('ML Expert Headcount by Company\nBootstrap Analysis with Individual Annotator Estimates', 
                fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(correlated_probit_bootstrap_plots_with_annotators_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(correlated_probit_bootstrap_plots_with_annotators_output_path, dpi=300, bbox_inches='tight')
    
    # Return the figure directly for Hamilton
    return fig

@cache(behavior="recompute")
def probit_bootstrap_talent_landscape_plot(
    correlated_probit_bootstrap_results_main: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    probit_bootstrap_talent_landscape_plot_output_path: str
) -> plt.Figure:
    """
    Create talent landscape plot for probit bootstrap results.
    
    This function adapts the talent landscape visualization for the probit bootstrap results.
    
    Args:
        correlated_probit_bootstrap_results_main: Summary DataFrame with bootstrap results (main pipeline)
        
    Returns:
        Dictionary with plot information
    """
    logger.info("Creating probit bootstrap talent landscape plot...")
    
    try:
        from .scripts.visualization.talent_landscape import create_ml_talent_landscape_plot
        
        # Normalize input: sometimes cache can return {} instead of a DataFrame
        if not isinstance(correlated_probit_bootstrap_results_main, pd.DataFrame):
            plot_data = pd.DataFrame(
                columns=['linkedin_id', 'company_name', 'mean', 'std', 'q10', 'q50', 'q90', 'n_employees']
            )
            logger.warning("correlated_probit_bootstrap_results_main was not a DataFrame; initialized empty DataFrame.")
        else:
            plot_data = correlated_probit_bootstrap_results_main.copy()
        
        # Check if DataFrame is empty or missing required columns
        if plot_data.empty:
            logger.warning("No bootstrap results to plot, creating empty plot")
            return _create_empty_plot(
                'No bootstrap results available',
                'Talent Landscape Plot',
                probit_bootstrap_talent_landscape_plot_output_path
            )
        
        if 'mean' not in plot_data.columns:
            logger.error(f"Missing 'mean' column in bootstrap results. Available columns: {list(plot_data.columns)}")
            raise ValueError(f"correlated_probit_bootstrap_results_main must have 'mean' column. Available columns: {list(plot_data.columns)}")
        
        # Create columns expected by talent landscape function
        # The talent landscape function expects raw ML staff counts in ml_consensus_round
        # It will calculate percentages internally
        plot_data['ml_consensus_round'] = plot_data['mean']  # Raw ML staff counts
        
        # Log warning for prevalence >100% (data quality issue)
        prevalence = (plot_data['mean'] / plot_data['n_employees'] * 100).fillna(0)
        high_prevalence = prevalence > 100
        if high_prevalence.any():
            n_high = high_prevalence.sum()
            logger.warning(f"Found {n_high} companies with prevalence >100% (data quality issue)")
            high_companies = plot_data[high_prevalence][['company_name', 'mean', 'n_employees']].head(5)
            high_companies['prevalence_pct'] = prevalence[high_prevalence].head(5)
            logger.warning(f"Sample high prevalence companies:\n{high_companies}")
        plot_data['organization_name'] = plot_data['company_name']  # Use the proper company name from database
        plot_data['number_of_employees_numeric'] = plot_data['n_employees']  # Y-axis is effective company size
        
        # Get actual work trial status from company database
        logger.info("Getting work trial status from company database...")
        work_trial_mapping = company_database_complete.set_index('linkedin_id')['Stage Reached'].to_dict()
        plot_data['stage_reached'] = plot_data['linkedin_id'].map(work_trial_mapping).fillna('Unknown')
        
        # Check for work trial companies
        work_trial_companies = plot_data[plot_data['stage_reached'] == '5 - Work Trial']
        logger.info(f"Found {len(work_trial_companies)} companies with '5 - Work Trial' status")
        logger.info(f"Work trial companies: {work_trial_companies['company_name'].tolist()[:5]}...")  # Show first 5
        
        # Debug statistics
        logger.info(f"=== TALENT LANDSCAPE PLOT DEBUG STATISTICS ===")
        logger.info(f"Total companies in plot_data: {len(plot_data)}")
        logger.info(f"Companies with valid organization_name: {plot_data['organization_name'].notna().sum()}")
        logger.info(f"Companies with valid total_headcount: {plot_data['number_of_employees_numeric'].notna().sum()}")
        logger.info(f"Companies with valid ml_consensus_round: {plot_data['ml_consensus_round'].notna().sum()}")
        
        # Check data quality
        logger.info(f"Bootstrap results companies: {len(plot_data)}")
        logger.info(f"Unique linkedin_ids in bootstrap: {plot_data['linkedin_id'].nunique()}")
        
        # Sample of linkedin_ids to debug
        sample_bootstrap_ids = plot_data['linkedin_id'].head(10).tolist()
        logger.info(f"Sample bootstrap linkedin_ids: {sample_bootstrap_ids}")
        
        # Debug the data
        logger.info(f"Sample linkedin_ids in plot_data: {plot_data['linkedin_id'].head(10).tolist()}")
        
        # Check data completeness
        merge_success = plot_data['organization_name'].notna().sum()
        logger.info(f"Companies with valid data: {merge_success} out of {len(plot_data)}")
        
        # X-axis (% prevalence of experts) statistics
        x_values = plot_data['ml_consensus_round'].dropna()
        logger.info(f"X-axis (% prevalence of experts) - Min: {x_values.min():.2f}%, Max: {x_values.max():.2f}%, Mean: {x_values.mean():.2f}%")
        logger.info(f"X-axis (% prevalence of experts) - 25th percentile: {x_values.quantile(0.25):.2f}%, 75th percentile: {x_values.quantile(0.75):.2f}%")
        
        # Y-axis (company size) statistics  
        y_values = plot_data['number_of_employees_numeric'].dropna()
        logger.info(f"Y-axis (company size) - Min: {y_values.min():.0f}, Max: {y_values.max():.0f}, Mean: {y_values.mean():.0f}")
        logger.info(f"Y-axis (company size) - 25th percentile: {y_values.quantile(0.25):.0f}, 75th percentile: {y_values.quantile(0.75):.0f}")
        
        # Check for extreme values that might be out of bounds
        extreme_x = x_values[(x_values < 0) | (x_values > 100)]  # % prevalence should be 0-100%
        extreme_y = y_values[(y_values < 1) | (y_values > 1000000)]
        logger.info(f"Extreme X values (outside 0-100%): {len(extreme_x)} companies")
        logger.info(f"Extreme Y values (outside 1-1M): {len(extreme_y)} companies")
        
        logger.info(f"=== END DEBUG STATISTICS ===")
        
        # Create the talent landscape plot
        result = create_ml_talent_landscape_plot(plot_data)
        
        # Convert tuple result to figure for saving
        if isinstance(result, tuple):
            fig, ax, agg = result
        else:
            fig = result
        
        # Adjust axis limits as requested
        ax.set_xlim(0, 45)  # X-axis: 0 to 45%
        ax.set_ylim(0.5, 1e5)  # Y-axis: 0.5 to 1e5 (log scale)
        logger.info("Set axis limits: xlim(0, 45), ylim(0.5, 1e5)")
        
        # Ensure output directory exists
        Path(probit_bootstrap_talent_landscape_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(probit_bootstrap_talent_landscape_plot_output_path, dpi=300, bbox_inches='tight')
        
        logger.info("Created probit bootstrap talent landscape plot")
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create probit bootstrap talent landscape plot: {str(e)}")
        raise


@cache(behavior="recompute")
def probit_bootstrap_per_organization_estimates_plot(
    final_results_all: pd.DataFrame,
    dawid_skene_test_data: pd.DataFrame,
    probit_bootstrap_per_organization_estimates_plot_output_path: str
) -> plt.Figure:
    """
    Create per-organization estimates plot for probit bootstrap results.
    
    This function uses the original estimate comparison visualization adapted for bootstrap results,
    including individual annotator estimates from the test data.
    
    Args:
        final_results_all: Combined final results DataFrame with all companies (includes organization_name)
        dawid_skene_test_data: Test data with individual annotator estimates
        
    Returns:
        Figure object
    """
    logger.info("Creating probit bootstrap per-organization estimates plot...")
    
    try:
        from .scripts.visualization.estimate_comparison import create_visualization
        import numpy as np
        
        # Prepare data in the format expected by the original function
        plot_data = final_results_all.copy()
        
        # Filter out organizations with mean ML headcount < 1
        # Use real estimate if available, otherwise use adjusted synthetic
        before_filter = len(plot_data)
        effective_mean = plot_data['mean'].fillna(plot_data['adjusted_synthetic_mean'])
        plot_data = plot_data[effective_mean >= 1].copy()
        after_filter = len(plot_data)
        logger.info(f"Filtered out {before_filter - after_filter} organizations with effective mean ML headcount < 1")
        logger.info(f"Remaining organizations for plotting: {after_filter}")
        
        # Map bootstrap results to the expected column names
        # Use real estimates, falling back to adjusted synthetic if real is not available
        plot_data['ml_consensus_round'] = plot_data['mean'].fillna(plot_data['adjusted_synthetic_mean'])
        plot_data['ml_lower80_round'] = plot_data['q10'].fillna(plot_data['adjusted_synthetic_q10'])
        plot_data['ml_upper80_round'] = plot_data['q90'].fillna(plot_data['adjusted_synthetic_q90'])
        
        # Use organization_name from final_results_all (preferred) or fall back to company_name
        if 'organization_name' in plot_data.columns:
            plot_data['Organization Name'] = plot_data['organization_name']
        elif 'company_name' in plot_data.columns:
            plot_data['Organization Name'] = plot_data['company_name']
        else:
            logger.warning("Neither organization_name nor company_name found, using linkedin_id")
            plot_data['Organization Name'] = plot_data['linkedin_id']
        
        # Compute individual annotator estimates by company
        logger.info("Computing individual annotator estimates by company...")
        
        # Determine which companies should use company database totals instead of employee-level data:
        # 1. Companies with no employee-level data
        companies_with_employee_data = set(dawid_skene_test_data['company_id'].unique())
        plot_data['has_employee_data'] = plot_data['linkedin_id'].isin(companies_with_employee_data)
        
        # 2. Companies where total_headcount and claude_total_employees differ by more than 3x
        # Check if claude_total_employees column exists (Claude's estimate of total employees)
        use_db_totals = ~plot_data['has_employee_data'].copy()
        
        if 'claude_total_employees' in plot_data.columns:
            # Check headcount mismatch
            total_headcount = pd.to_numeric(plot_data['total_headcount'], errors='coerce')
            claude_total_employees = pd.to_numeric(plot_data['claude_total_employees'], errors='coerce')
            
            # Calculate ratio (avoid division by zero)
            valid_mask = (total_headcount.notna()) & (claude_total_employees.notna()) & (claude_total_employees > 0)
            headcount_ratio = pd.Series(np.nan, index=plot_data.index)
            headcount_ratio[valid_mask] = total_headcount[valid_mask] / claude_total_employees[valid_mask]
            
            # Flag companies where ratio is > 3 or < 1/3
            headcount_mismatch = (headcount_ratio > 3) | (headcount_ratio < 1/3)
            use_db_totals = use_db_totals | headcount_mismatch.fillna(False)
            
            logger.info(f"Headcount mismatch check: {headcount_mismatch.sum()} companies with >3x difference")
        else:
            logger.info("claude_total_employees column not found, skipping headcount mismatch check")
        
        logger.info(f"Using company database totals for {use_db_totals.sum()} companies (out of {len(plot_data)})")
        
        # Get annotator columns from test data
        annotator_cols = [col for col in dawid_skene_test_data.columns if col not in ['company_id', 'group']]
        logger.info(f"Found annotator columns: {annotator_cols}")
        
        # Compute annotator counts by company from employee-level data
        annotator_counts = {}
        for company_id in dawid_skene_test_data['company_id'].unique():
            company_mask = dawid_skene_test_data['company_id'] == company_id
            company_data = dawid_skene_test_data[company_mask]
            
            # Sum annotations for each annotator
            for col in annotator_cols:
                if col not in annotator_counts:
                    annotator_counts[col] = {}
                annotator_counts[col][company_id] = company_data[col].sum()
        
        # Map annotator columns to the expected names in the original visualization
        annotator_mapping = {
            'filter_broad_yes': 'filter_broad_yes',
            'filter_strict_no': 'filter_strict_no', 
            'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no',
            'llm_gemini_2_5_flash': 'gemini_total_accepted',
            'llm_sonnet_4': 'claude_total_accepted',
            'llm_gpt_5_mini': 'gpt5_total_accepted'
        }
        
        # Map from employee-level data column names to company database column names
        db_column_mapping = {
            'filter_broad_yes': 'filter_broad_yes',
            'filter_strict_no': 'filter_strict_no',
            'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no',
            'gemini_total_accepted': 'gemini_total_accepted',
            'claude_total_accepted': 'claude_total_accepted',
            'gpt5_total_accepted': 'gpt5_total_accepted'
        }
        
        # Add individual annotator estimates to plot data
        for original_col, expected_col in annotator_mapping.items():
            # Track which companies have employee-level data for this specific annotator
            has_employee_data_for_annotator = pd.Series(False, index=plot_data.index)
            
            # Get company database values (they're already in plot_data from final_results_all)
            db_col = db_column_mapping.get(expected_col)
            db_values = plot_data[db_col].copy() if (db_col and db_col in plot_data.columns) else pd.Series(np.nan, index=plot_data.index)
            
            # Start with employee-level data (if available)
            if original_col in annotator_counts:
                mapped_values = plot_data['linkedin_id'].map(annotator_counts[original_col])
                # Mark companies that have employee-level data (even if it's 0 - that's legitimate)
                has_employee_data_for_annotator = mapped_values.notna()
                
                # Use employee-level data where available, but preserve company DB values where employee data is missing
                # Initialize with company DB values (for companies without employee-level data)
                plot_data[expected_col] = db_values.copy()
                
                # Overwrite with employee-level data where available (preserving legitimate zeros)
                employee_mask = has_employee_data_for_annotator & ~use_db_totals
                plot_data.loc[employee_mask, expected_col] = mapped_values[employee_mask]
            else:
                # No employee-level data available, use company database values
                plot_data[expected_col] = db_values.copy()
            
            # For companies flagged to use DB totals, ensure we use DB values (even if employee data exists)
            if db_col and db_col in plot_data.columns:
                db_mask = use_db_totals & db_values.notna()
                plot_data.loc[db_mask, expected_col] = db_values[db_mask]
            
            # Fill remaining NaN with 0 for plotting (but these won't show on log scale)
            plot_data[expected_col] = plot_data[expected_col].fillna(0)
            
            # Count how many companies have values from each source
            # Companies with employee-level data (preserved, including zeros)
            n_from_employee_kept = (has_employee_data_for_annotator & ~use_db_totals).sum()
            # Companies where we used company database values
            n_from_db = 0
            if db_col and db_col in plot_data.columns:
                # Count companies where we used DB value
                # (either because use_db_totals without employee data, or because employee data was NaN)
                db_used_mask = ((use_db_totals & ~has_employee_data_for_annotator) | 
                               (plot_data[expected_col].notna() & ~has_employee_data_for_annotator & 
                                (plot_data[expected_col] == plot_data[db_col])))
                n_from_db = db_used_mask.sum()
            n_zero = (plot_data[expected_col] == 0).sum()
            n_nonzero = (plot_data[expected_col] > 0).sum()
            logger.info(f"Added {expected_col} estimates: {n_from_employee_kept} from employee data (kept), {n_from_db} from company database, {n_nonzero} non-zero, {n_zero} zero")
        
        # Create the visualization using the original function
        fig, ax, df_processed = create_visualization(
            plot_data, 
            y_scale="log",
            org_col_candidates=("Organization Name", "company_name")
        )
        
        # Set Y-axis limits to 5e4 (50000) as requested
        ax.set_ylim(1, 5e4)
        logger.info(f"Set Y-axis limits: 1 to 5e4 (50000)")
        
        # Remove excess whitespace on x-axis (eliminate padding between first/last companies and borders)
        n_companies = len(df_processed)
        ax.set_xlim(-0.5, n_companies - 0.5)
        
        # Update the title to reflect bootstrap analysis
        ax.set_title('Per-organization estimates with consensus estimate (80% CI)', fontsize=14, pad=60)
        
        # Ensure output directory exists
        Path(probit_bootstrap_per_organization_estimates_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(probit_bootstrap_per_organization_estimates_plot_output_path, dpi=300, bbox_inches='tight')
        
        logger.info("Created probit bootstrap per-organization estimates plot")
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create probit bootstrap per-organization estimates plot: {str(e)}")
        raise


@cache(behavior="recompute")
def probit_bootstrap_prior_distribution_plot(
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
    probit_bootstrap_prior_distribution_plot_output_path: str
) -> plt.Figure:
    """
    Create a plot showing the Beta prior distribution used for prevalence.
    
    This function visualizes the prior distribution Beta(alpha, beta) that is used
    as a hyperprior on the class prior (prevalence) in the bootstrap analysis.
    
    Args:
        correlated_probit_bootstrap_prior_alpha: Alpha parameter for Beta prior
        correlated_probit_bootstrap_prior_beta: Beta parameter for Beta prior
        probit_bootstrap_prior_distribution_plot_output_path: Output path for the plot
        
    Returns:
        Figure object
    """
    logger.info("Creating prior distribution plot...")
    
    from scipy.stats import beta
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for the plot
    x = np.linspace(0, 1, 1000)
    
    # Calculate the PDF
    pdf = beta.pdf(x, correlated_probit_bootstrap_prior_alpha, correlated_probit_bootstrap_prior_beta)
    
    # Plot the distribution
    ax.plot(x, pdf, 'b-', linewidth=2, label=f'Beta({correlated_probit_bootstrap_prior_alpha}, {correlated_probit_bootstrap_prior_beta})')
    ax.fill_between(x, pdf, alpha=0.3, color='blue')
    
    # Add vertical lines for mean and mode
    mean_val = beta.mean(correlated_probit_bootstrap_prior_alpha, correlated_probit_bootstrap_prior_beta)
    
    # Calculate mode manually: mode = (alpha-1) / (alpha+beta-2) for alpha>1 and beta>1
    if correlated_probit_bootstrap_prior_alpha > 1 and correlated_probit_bootstrap_prior_beta > 1:
        mode_val = (correlated_probit_bootstrap_prior_alpha - 1) / (correlated_probit_bootstrap_prior_alpha + correlated_probit_bootstrap_prior_beta - 2)
    else:
        # Edge cases (not applicable for typical values)
        mode_val = 0.0
    
    ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_val:.3f}')
    ax.axvline(mode_val, color='orange', linestyle='--', linewidth=2,
               label=f'Mode = {mode_val:.3f}')
    
    # Add text annotations
    ax.text(0.05, 0.85, f'Mean: {mean_val:.3f}\nMode: {mode_val:.3f}',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Customize the plot
    ax.set_xlabel('Prevalence (π)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Beta Prior Distribution for Prevalence\nBeta({correlated_probit_bootstrap_prior_alpha}, {correlated_probit_bootstrap_prior_beta})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(probit_bootstrap_prior_distribution_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(probit_bootstrap_prior_distribution_plot_output_path, dpi=300, bbox_inches='tight')
    
    logger.info("Created prior distribution plot")
    return fig


@cache(behavior="recompute")
def keyword_annotator_prevalence_plot(
    dawid_skene_test_data_main: pd.DataFrame,
    dawid_skene_test_data_comparator_ml: pd.DataFrame,
    dawid_skene_test_data_comparator_non_ml: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    keyword_annotator_prevalence_plot_output_path: str
) -> plt.Figure:
    """
    Create scatter plots comparing LinkedIn Sales Navigator company-level keyword counts
    vs BrightData employee-level positive rates for the three keyword annotators.
    
    Args:
        dawid_skene_test_data_main: Main test dataset with employee-level annotations
        dawid_skene_test_data_comparator_ml: Comparator ML test dataset
        dawid_skene_test_data_comparator_non_ml: Comparator Non-ML test dataset
        company_database_complete: Company database with LinkedIn Sales Navigator aggregates
        keyword_annotator_prevalence_plot_output_path: Output path for the plot
        
    Returns:
        Figure object
    """
    logger.info("Creating keyword annotator prevalence comparison plot...")
    
    # Combine all three test datasets
    main_df = dawid_skene_test_data_main.copy()
    main_df['dataset_source'] = 'main'
    
    comp_ml_df = dawid_skene_test_data_comparator_ml.copy()
    comp_ml_df['dataset_source'] = 'comparator_ml'
    
    comp_non_ml_df = dawid_skene_test_data_comparator_non_ml.copy()
    comp_non_ml_df['dataset_source'] = 'comparator_non_ml'
    
    combined_test_data = pd.concat([main_df, comp_ml_df, comp_non_ml_df], ignore_index=True)
    
    # Define the three keyword annotators
    keyword_annotators = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    
    # Compute BrightData positive rates per company for keyword filters
    brightdata_rates = {}
    for company_id in combined_test_data['company_id'].unique():
        company_mask = combined_test_data['company_id'] == company_id
        company_data = combined_test_data[company_mask]
        
        # Count total employees (non-NaN for at least one annotator)
        # For each keyword filter, count employees with non-NaN values
        for annotator in keyword_annotators:
            if annotator not in brightdata_rates:
                brightdata_rates[annotator] = {}
            
            # Count employees with valid (non-NaN) annotations for this filter
            valid_mask = company_data[annotator].notna()
            n_employees = valid_mask.sum()
            
            if n_employees > 0:
                # Sum of positive annotations (1s)
                n_positives = company_data.loc[valid_mask, annotator].sum()
                brightdata_rates[annotator][company_id] = n_positives / n_employees
            else:
                brightdata_rates[annotator][company_id] = np.nan
    
    # Convert to DataFrame
    brightdata_df = pd.DataFrame(brightdata_rates)
    brightdata_df.index.name = 'company_id'
    brightdata_df = brightdata_df.reset_index()
    
    # Rename brightdata columns to avoid conflict with company_database columns
    brightdata_rename = {annotator: f'{annotator}_brightdata' for annotator in keyword_annotators}
    brightdata_df = brightdata_df.rename(columns=brightdata_rename)
    
    # Merge with company database to get LinkedIn counts and total_headcount
    plot_data = brightdata_df.merge(
        company_database_complete[['linkedin_id', 'total_headcount'] + keyword_annotators],
        left_on='company_id',
        right_on='linkedin_id',
        how='inner'
    )
    
    # Filter to companies with valid total_headcount and employee data
    plot_data = plot_data[
        (plot_data['total_headcount'].notna()) & 
        (plot_data['total_headcount'] > 0)
    ].copy()
    
    # Compute LinkedIn ratios: company_database[filter_X] / total_headcount
    for annotator in keyword_annotators:
        linkedin_col = f'{annotator}_linkedin'
        plot_data[linkedin_col] = plot_data[annotator] / plot_data['total_headcount']
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, annotator in enumerate(keyword_annotators):
        ax = axes[idx]
        
        # Get data for this annotator
        brightdata_col = f'{annotator}_brightdata'
        linkedin_col = f'{annotator}_linkedin'
        
        # Filter out NaN values and zeros (log scale can't handle zeros)
        plot_subset = plot_data[
            (plot_data[brightdata_col].notna()) & 
            (plot_data[linkedin_col].notna()) &
            (plot_data[brightdata_col] > 0) &
            (plot_data[linkedin_col] > 0)
        ].copy()
        
        if len(plot_subset) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{annotator}', fontsize=12)
            continue
        
        # Create scatter plot
        ax.scatter(plot_subset[linkedin_col], plot_subset[brightdata_col], 
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add diagonal reference line (y=x)
        max_val = max(plot_subset[linkedin_col].max(), plot_subset[brightdata_col].max())
        min_val = min(plot_subset[linkedin_col].min(), plot_subset[brightdata_col].min())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, alpha=0.5, label='y=x')
        
        # Calculate linear regression statistics in log-log space
        from scipy.stats import linregress
        x_data = plot_subset[linkedin_col].values
        y_data = plot_subset[brightdata_col].values
        
        # Transform to log space (base 10, matching matplotlib log scale)
        log_x = np.log10(x_data)
        log_y = np.log10(y_data)
        
        # Linear regression in log-log space: log(y) = intercept + slope * log(x)
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        r_squared = r_value ** 2
        
        # Draw regression line
        x_min_log = log_x.min()
        x_max_log = log_x.max()
        x_line_log = np.linspace(x_min_log, x_max_log, 100)
        y_line_log = intercept + slope * x_line_log
        
        # Convert back to original space for plotting
        x_line = 10 ** x_line_log
        y_line = 10 ** y_line_log
        
        ax.plot(x_line, y_line, 'b-', linewidth=2, alpha=0.8, label='Log-log fit')
        
        # Calculate means and standard deviations in original space
        x_mean = x_data.mean()
        x_std = x_data.std()
        y_mean = y_data.mean()
        y_std = y_data.std()
        
        # Add statistics text (slope and intercept are for log-log fit)
        stats_text = (
            f'Log-log fit: log(y) = {intercept:.4f} + {slope:.4f}·log(x)\n'
            f'r² = {r_squared:.3f}, r = {r_value:.3f}\n'
            f'p = {p_value:.3e}\n'
            f'LinkedIn: μ={x_mean:.4f}, σ={x_std:.4f}\n'
            f'BrightData: μ={y_mean:.4f}, σ={y_std:.4f}'
        )
        ax.text(0.05, 0.95, stats_text,
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               family='monospace')
        
        # Labels and title
        ax.set_xlabel('LinkedIn Ratio\n(Company-level count / total_headcount)', fontsize=10)
        ax.set_ylabel('BrightData Ratio\n(Employee positives / total employees)', fontsize=10)
        ax.set_title(f'{annotator}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(keyword_annotator_prevalence_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(keyword_annotator_prevalence_plot_output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Created keyword annotator prevalence plot with {len(plot_data)} companies")
    return fig


@cache(behavior="recompute")
def empirical_keyword_correlations_pairplot(
    dawid_skene_test_data_main: pd.DataFrame,
    dawid_skene_test_data_comparator_ml: pd.DataFrame,
    dawid_skene_test_data_comparator_non_ml: pd.DataFrame,
    empirical_keyword_correlations_pairplot_output_path: str
) -> plt.Figure:
    """
    Create a pairplot showing empirical correlations between all 6 annotators
    at the company level using BrightData employee-level positive rates.
    
    Args:
        dawid_skene_test_data_main: Main test dataset with employee-level annotations
        dawid_skene_test_data_comparator_ml: Comparator ML test dataset
        dawid_skene_test_data_comparator_non_ml: Comparator Non-ML test dataset
        empirical_keyword_correlations_pairplot_output_path: Output path for the plot
        
    Returns:
        Figure object
    """
    logger.info("Creating empirical keyword correlations pairplot...")
    
    # Combine all three test datasets
    main_df = dawid_skene_test_data_main.copy()
    main_df['dataset_source'] = 'main'
    
    comp_ml_df = dawid_skene_test_data_comparator_ml.copy()
    comp_ml_df['dataset_source'] = 'comparator_ml'
    
    comp_non_ml_df = dawid_skene_test_data_comparator_non_ml.copy()
    comp_non_ml_df['dataset_source'] = 'comparator_non_ml'
    
    combined_test_data = pd.concat([main_df, comp_ml_df, comp_non_ml_df], ignore_index=True)
    
    # Define all 6 annotators
    annotator_cols = [col for col in combined_test_data.columns 
                     if col not in ['company_id', 'group', 'dataset_source']]
    
    # Compute positive rates per company for all annotators
    company_rates = []
    
    for company_id in combined_test_data['company_id'].unique():
        company_mask = combined_test_data['company_id'] == company_id
        company_data = combined_test_data[company_mask]
        
        rates = {'company_id': company_id}
        
        for annotator in annotator_cols:
            # Count employees with valid (non-NaN) annotations for this annotator
            valid_mask = company_data[annotator].notna()
            n_employees = valid_mask.sum()
            
            if n_employees > 0:
                # Sum of positive annotations (1s)
                n_positives = company_data.loc[valid_mask, annotator].sum()
                rates[annotator] = n_positives / n_employees
            else:
                rates[annotator] = np.nan
        
        company_rates.append(rates)
    
    # Convert to DataFrame
    rates_df = pd.DataFrame(company_rates)
    
    # Filter to companies with at least some valid data
    # Keep companies that have valid rates for at least one annotator
    valid_companies = rates_df[annotator_cols].notna().any(axis=1)
    rates_df = rates_df[valid_companies].copy()
    
    # For log scale, replace zeros and NaN with small epsilon
    # This ensures all values are positive for log scale
    epsilon = 1e-6
    rates_df[annotator_cols] = rates_df[annotator_cols].fillna(epsilon).replace(0, epsilon)
    
    if len(rates_df) == 0:
        logger.warning("No valid company data for pairplot")
        return _create_empty_plot(
            'No valid company data available',
            'Empirical Keyword Correlations Pairplot',
            empirical_keyword_correlations_pairplot_output_path
        )
    
    # Use seaborn for pairplot
    try:
        import seaborn as sns
    except ImportError:
        logger.error("seaborn is required for pairplot. Install with: pip install seaborn")
        raise
    
    # Create pairplot with lower triangle only (hide upper triangle, keep diagonal)
    # Use PairGrid for more control
    g = sns.PairGrid(rates_df[annotator_cols])
    
    # Hide upper triangle only, keep diagonal and lower triangle
    for i in range(len(annotator_cols)):
        for j in range(len(annotator_cols)):
            if i < j:  # Hide upper triangle
                g.axes[i, j].set_visible(False)
            elif i == j:  # Diagonal: histograms with log-spaced bins
                # Calculate log-spaced bins
                data = rates_df[annotator_cols[i]].values
                data_positive = data[data > 0]
                if len(data_positive) > 0:
                    log_min = np.log10(data_positive.min())
                    log_max = np.log10(data_positive.max())
                    log_bins = np.logspace(log_min, log_max, 20)
                    g.axes[i, j].hist(data_positive, bins=log_bins, alpha=0.7, edgecolor='black')
            else:  # Lower triangle: scatter plots
                g.axes[i, j].scatter(
                    rates_df[annotator_cols[j]], 
                    rates_df[annotator_cols[i]], 
                    alpha=0.6, s=30
                )
    
    fig = g
    
    # Set log scales on all visible axes and add correlation statistics
    from scipy.stats import linregress
    
    n_vars = len(annotator_cols)
    for i in range(n_vars):
        for j in range(n_vars):
            ax = fig.axes[i, j]
            if ax is not None and ax.get_visible():
                ax.set_xscale('log')
                if i != j:  # Only set yscale for scatter plots, not histograms
                    ax.set_yscale('log')
                
                # Add statistics only to scatter plots (lower triangle)
                if i > j:
                    x_var = annotator_cols[j]
                    y_var = annotator_cols[i]
                    
                    # Get data for this pair
                    x_data = rates_df[x_var].values
                    y_data = rates_df[y_var].values
                    
                    # Filter out NaN and zeros (shouldn't have any after preprocessing, but just in case)
                    valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
                    if valid_mask.sum() > 1:  # Need at least 2 points for regression
                        x_valid = x_data[valid_mask]
                        y_valid = y_data[valid_mask]
                        
                        # Transform to log space (base 10, matching matplotlib log scale)
                        log_x = np.log10(x_valid)
                        log_y = np.log10(y_valid)
                        
                        # Linear regression in log-log space: log(y) = intercept + slope * log(x)
                        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
                        r_squared = r_value ** 2
                        
                        # Calculate means and standard deviations in original space
                        x_mean = x_valid.mean()
                        x_std = x_valid.std()
                        y_mean = y_valid.mean()
                        y_std = y_valid.std()
                        
                        # Draw regression line on log-log plot
                        # Get axis limits in log space
                        x_min_log = np.log10(x_valid.min())
                        x_max_log = np.log10(x_valid.max())
                        
                        # Generate line points in log space
                        x_line_log = np.linspace(x_min_log, x_max_log, 100)
                        y_line_log = intercept + slope * x_line_log
                        
                        # Convert back to original space for plotting (matplotlib handles log scale automatically)
                        x_line = 10 ** x_line_log
                        y_line = 10 ** y_line_log
                        
                        # Plot regression line
                        ax.plot(x_line, y_line, 'r-', linewidth=1.5, alpha=0.7, label='log-log fit')
                        
                        # Add statistics text (compact format for pairplot, slope and intercept are for log-log fit)
                        stats_text = (
                            f'm={slope:.2f}, b={intercept:.4f}\n'
                            f'r²={r_squared:.3f}\n'
                            f'μx={x_mean:.3f}, σx={x_std:.3f}\n'
                            f'μy={y_mean:.3f}, σy={y_std:.3f}'
                        )
                        ax.text(0.05, 0.95, stats_text,
                               transform=ax.transAxes, fontsize=7,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                               family='monospace')
    
    # Add title
    fig.fig.suptitle('Empirical Keyword Correlations (Company-Level Positive Rates)', 
                     fontsize=14, fontweight='bold', y=1.02)
    
    # Ensure output directory exists
    Path(empirical_keyword_correlations_pairplot_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(empirical_keyword_correlations_pairplot_output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Created empirical keyword correlations pairplot with {len(rates_df)} companies")
    return fig.fig


@cache(behavior="recompute")
def real_vs_synthetic_scatter_plot(
    final_results_all: pd.DataFrame,
    real_vs_synthetic_scatter_plot_output_path: str
) -> plt.Figure:
    """
    Create scatter plot comparing real vs synthetic mean headcount estimates
    with error bars in both directions, log-log scale, and log-space linear regression.
    
    Args:
        final_results_all: Final results DataFrame with both real and synthetic estimates
        real_vs_synthetic_scatter_plot_output_path: Output path for the plot
        
    Returns:
        Figure object
    """
    logger.info("Creating real vs synthetic scatter plot...")
    
    # Filter to companies with both real and synthetic estimates
    # Filter out companies with real mean < 1 to avoid unstable ratios
    has_real = final_results_all['mean'].notna()
    has_synthetic = final_results_all['synthetic_mean'].notna()
    plot_data = final_results_all[has_real & has_synthetic].copy()
    
    # Filter out real mean < 1 and zeros for log scale
    plot_data = plot_data[
        (plot_data['mean'] >= 1.0) & 
        (plot_data['synthetic_mean'] > 0)
    ].copy()
    
    if len(plot_data) == 0:
        logger.warning("No companies with both real and synthetic estimates found")
        return _create_empty_plot(
            'No companies with both real and synthetic estimates',
            'Real vs Synthetic Mean Headcount Estimates',
            real_vs_synthetic_scatter_plot_output_path
        )
    
    logger.info(f"Plotting {len(plot_data)} companies with both real and synthetic estimates")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate error bars (asymmetric: q10 to mean, mean to q90)
    real_lower = np.maximum(1e-6, plot_data['mean'] - plot_data['q10'])
    real_upper = np.maximum(1e-6, plot_data['q90'] - plot_data['mean'])
    synthetic_lower = np.maximum(1e-6, plot_data['synthetic_mean'] - plot_data['synthetic_q10'])
    synthetic_upper = np.maximum(1e-6, plot_data['synthetic_q90'] - plot_data['synthetic_mean'])
    
    # Create scatter plot with error bars in both directions
    ax.errorbar(
        plot_data['synthetic_mean'],
        plot_data['mean'],
        xerr=[synthetic_lower, synthetic_upper],
        yerr=[real_lower, real_upper],
        fmt='o',
        alpha=0.6,
        markersize=6,
        capsize=3,
        capthick=1.5,
        elinewidth=1.2,
        label='Companies'
    )
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add diagonal reference line (y=x)
    min_val = min(plot_data['mean'].min(), plot_data['synthetic_mean'].min())
    max_val = max(plot_data['mean'].max(), plot_data['synthetic_mean'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', linewidth=2, alpha=0.5, label='y=x')
    
    # Calculate robust linear regression in log-log space
    from sklearn.linear_model import TheilSenRegressor
    from scipy.stats import pearsonr
    
    x_data = plot_data['synthetic_mean'].values
    y_data = plot_data['mean'].values
    
    # Transform to log space (base 10)
    log_x = np.log10(x_data)
    log_y = np.log10(y_data)
    
    # Reshape for sklearn (needs 2D array)
    log_x_2d = log_x.reshape(-1, 1)
    
    # Robust linear regression using Theil-Sen estimator
    robust_reg = TheilSenRegressor(random_state=42)
    robust_reg.fit(log_x_2d, log_y)
    
    slope = robust_reg.coef_[0]
    intercept = robust_reg.intercept_
    
    # Calculate correlation and r² using robust predictions
    y_pred = robust_reg.predict(log_x_2d)
    r_value, _ = pearsonr(log_x, log_y)
    r_squared = r_value ** 2
    
    # Draw robust regression line
    x_min_log = log_x.min()
    x_max_log = log_x.max()
    x_line_log = np.linspace(x_min_log, x_max_log, 100).reshape(-1, 1)
    y_line_log = robust_reg.predict(x_line_log)
    
    # Convert back to original space for plotting
    x_line = 10 ** x_line_log.flatten()
    y_line = 10 ** y_line_log
    
    ax.plot(x_line, y_line, 'b-', linewidth=2, alpha=0.8, label='Robust log-log fit')
    
    # Calculate second fit: log(y) = -log(M) + log(x), where M is median of synthetic/real
    # Calculate M as median of ratio in linear space
    ratio = x_data / y_data
    M = np.median(ratio)
    M_log = np.log10(M)  # This is log(M)
    
    # Fit: log(y) = log(x) - log(M), so intercept = -log(M), slope = 1
    # Calculate r² for this fit by comparing predicted vs actual
    y_pred_median_log = log_x - M_log
    # Calculate r² from correlation between predicted and actual log(y)
    r_value_median, _ = pearsonr(log_y, y_pred_median_log)
    r_squared_median = r_value_median ** 2
    
    # Draw median-based regression line: log(y) = log(x) - log(M)
    # Use same x range as robust fit
    y_line_median_log = x_line_log.flatten() - M_log
    y_line_median = 10 ** y_line_median_log
    
    ax.plot(x_line, y_line_median, 
           'g-', linewidth=2, alpha=0.8, label='Median-based fit')
    
    # Add statistics text (first annotation - robust fit)
    stats_text = (
        f'Robust log-log fit: log(y) = {intercept:.4f} + {slope:.4f}·log(x)\n'
        f'r² = {r_squared:.3f}\n'
        f'n = {len(plot_data)}'
    )
    ax.text(0.05, 0.95, stats_text,
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           family='monospace')
    
    # Add second statistics text (median-based fit, below first)
    stats_text2 = (
        f'Median-based fit: log(y) = -log({M:.2f}) + log(x)\n'
        f'r² = {r_squared_median:.3f}'
    )
    ax.text(0.05, 0.88, stats_text2,
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
           family='monospace')
    
    # Labels and title
    ax.set_xlabel('Synthetic Mean Headcount Estimate', fontsize=12)
    ax.set_ylabel('Real Mean Headcount Estimate', fontsize=12)
    ax.set_title('Real vs Synthetic Mean Headcount Estimates\n(Error bars: q10-q90)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(real_vs_synthetic_scatter_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(real_vs_synthetic_scatter_plot_output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Created real vs synthetic scatter plot with {len(plot_data)} companies")
    return fig


@cache(behavior="recompute")
def synthetic_real_ratio_distributions_plot(
    final_results_all: pd.DataFrame,
    synthetic_real_ratio_distributions_plot_output_path: str
) -> plt.Figure:
    """
    Create distribution plots of per-company synthetic/real ratios:
    one for means and one for CI widths, with mean + std annotated.
    
    Args:
        final_results_all: Final results DataFrame with both real and synthetic estimates
        synthetic_real_ratio_distributions_plot_output_path: Output path for the plot
        
    Returns:
        Figure object
    """
    logger.info("Creating synthetic/real ratio distributions plot...")
    
    # Filter to companies with both real and synthetic estimates
    # Filter out companies with real mean < 1 to avoid unstable ratios
    has_real = final_results_all['mean'].notna()
    has_synthetic = final_results_all['synthetic_mean'].notna()
    plot_data = final_results_all[has_real & has_synthetic].copy()
    
    # Filter out zeros and real mean < 1
    plot_data = plot_data[
        (plot_data['mean'] >= 1.0) & 
        (plot_data['synthetic_mean'] > 0)
    ].copy()
    
    if len(plot_data) == 0:
        logger.warning("No companies with both real and synthetic estimates found")
        return _create_empty_plot(
            'No companies with both real and synthetic estimates',
            'Synthetic/Real Ratio Distributions',
            synthetic_real_ratio_distributions_plot_output_path
        )
    
    # Calculate ratios
    # Mean ratio
    mean_ratio = plot_data['synthetic_mean'] / plot_data['mean']
    
    # CI width ratio (q90 - q10)
    real_ci_width = plot_data['q90'] - plot_data['q10']
    synthetic_ci_width = plot_data['synthetic_q90'] - plot_data['synthetic_q10']
    ci_width_ratio = synthetic_ci_width / real_ci_width
    
    # Filter out invalid ratios (zeros, infinities, NaN)
    valid_mean_ratio = mean_ratio[(mean_ratio > 0) & np.isfinite(mean_ratio)]
    valid_ci_ratio = ci_width_ratio[(ci_width_ratio > 0) & np.isfinite(ci_width_ratio)]
    
    if len(valid_mean_ratio) == 0 or len(valid_ci_ratio) == 0:
        logger.warning("No valid ratios to plot")
        return _create_empty_plot(
            'No valid ratios to plot',
            'Synthetic/Real Ratio Distributions',
            synthetic_real_ratio_distributions_plot_output_path
        )
    
    # Use seaborn for distribution plots
    try:
        import seaborn as sns
    except ImportError:
        logger.error("seaborn is required for distribution plots. Install with: pip install seaborn")
        raise
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean ratio distribution (log scale with log-spaced bins)
    ax1 = axes[0]
    # Calculate log-spaced bins
    log_min = np.log10(valid_mean_ratio.min())
    log_max = np.log10(valid_mean_ratio.max())
    log_bins = np.logspace(log_min, log_max, 30)
    sns.histplot(valid_mean_ratio, bins=log_bins, kde=False, ax=ax1, alpha=0.7)
    ax1.set_xscale('log')
    
    # Calculate and annotate mean and median
    mean_val = valid_mean_ratio.mean()
    median_val = valid_mean_ratio.median()
    std_val = valid_mean_ratio.std()
    
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3f}')
    ax1.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median = {median_val:.3f}')
    
    ax1.set_xlabel('Synthetic/Real Mean Ratio (log scale)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Mean Ratios\n(Synthetic/Real)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text1 = f'μ = {mean_val:.3f}\nMedian = {median_val:.3f}\nσ = {std_val:.3f}\nn = {len(valid_mean_ratio)}'
    ax1.text(0.95, 0.95, stats_text1,
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            family='monospace')
    
    # Plot 2: CI width ratio distribution (log scale with log-spaced bins)
    ax2 = axes[1]
    # Calculate log-spaced bins
    log_min2 = np.log10(valid_ci_ratio.min())
    log_max2 = np.log10(valid_ci_ratio.max())
    log_bins2 = np.logspace(log_min2, log_max2, 30)
    sns.histplot(valid_ci_ratio, bins=log_bins2, kde=False, ax=ax2, alpha=0.7)
    ax2.set_xscale('log')
    
    # Calculate and annotate mean and median
    mean_val2 = valid_ci_ratio.mean()
    median_val2 = valid_ci_ratio.median()
    std_val2 = valid_ci_ratio.std()
    
    ax2.axvline(mean_val2, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val2:.3f}')
    ax2.axvline(median_val2, color='blue', linestyle='--', linewidth=2, label=f'Median = {median_val2:.3f}')
    
    ax2.set_xlabel('Synthetic/Real CI Width Ratio (log scale)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of CI Width Ratios\n(Synthetic/Real)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text2 = f'μ = {mean_val2:.3f}\nMedian = {median_val2:.3f}\nσ = {std_val2:.3f}\nn = {len(valid_ci_ratio)}'
    ax2.text(0.95, 0.95, stats_text2,
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            family='monospace')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(synthetic_real_ratio_distributions_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(synthetic_real_ratio_distributions_plot_output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Created synthetic/real ratio distributions plot with {len(valid_mean_ratio)} mean ratios and {len(valid_ci_ratio)} CI width ratios")
    return fig


# ============================================================================
# HELPER FUNCTIONS FOR REUSABLE PLOT LOGIC
# ============================================================================

def _create_empty_plot(message: str, title: str, output_path: str) -> plt.Figure:
    """Helper function to create an empty plot with a message."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title(title)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig


# Main pipeline plot functions
def correlated_probit_bootstrap_plots_with_annotators_main(
    dawid_skene_test_data_main: pd.DataFrame,
    correlated_probit_bootstrap_raw_results_main: Dict[str, Any],
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_plots_with_annotators_main_output_path: str
) -> plt.Figure:
    """Create bootstrap plots for main pipeline."""
    return correlated_probit_bootstrap_plots_with_annotators(
        dawid_skene_test_data_main,
        correlated_probit_bootstrap_raw_results_main,
        correlated_probit_bootstrap_n_samples,
        correlated_probit_bootstrap_plots_with_annotators_main_output_path
    )


def probit_bootstrap_talent_landscape_plot_main(
    correlated_probit_bootstrap_results_main: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    probit_bootstrap_talent_landscape_plot_main_output_path: str
) -> plt.Figure:
    """Create talent landscape plot for main pipeline."""
    return probit_bootstrap_talent_landscape_plot(
        correlated_probit_bootstrap_results_main,
        company_database_complete,
        probit_bootstrap_talent_landscape_plot_main_output_path
    )


def probit_bootstrap_per_organization_estimates_plot_main(
    final_results_all: pd.DataFrame,
    dawid_skene_test_data_main: pd.DataFrame,
    probit_bootstrap_per_organization_estimates_plot_main_output_path: str
) -> plt.Figure:
    """Create per-organization estimates plot for main pipeline."""
    return probit_bootstrap_per_organization_estimates_plot(
        final_results_all,
        dawid_skene_test_data_main,
        probit_bootstrap_per_organization_estimates_plot_main_output_path
    )


def probit_bootstrap_prior_distribution_plot_main(
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
    probit_bootstrap_prior_distribution_plot_main_output_path: str
) -> plt.Figure:
    """Create prior distribution plot for main pipeline."""
    return probit_bootstrap_prior_distribution_plot(
        correlated_probit_bootstrap_prior_alpha,
        correlated_probit_bootstrap_prior_beta,
        probit_bootstrap_prior_distribution_plot_main_output_path
    )


# Comparator ML pipeline plot functions
def correlated_probit_bootstrap_plots_with_annotators_comparator_ml(
    dawid_skene_test_data_comparator_ml: pd.DataFrame,
    correlated_probit_bootstrap_raw_results_comparator_ml: Dict[str, Any],
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_plots_with_annotators_comparator_ml_output_path: str
) -> plt.Figure:
    """Create bootstrap plots for comparator ML pipeline."""
    if len(dawid_skene_test_data_comparator_ml) == 0 or not correlated_probit_bootstrap_raw_results_comparator_ml:
        return _create_empty_plot(
            'No bootstrap results to plot',
            'Bootstrap Results with Annotators (Comparator ML)',
            correlated_probit_bootstrap_plots_with_annotators_comparator_ml_output_path
        )
    return correlated_probit_bootstrap_plots_with_annotators(
        dawid_skene_test_data_comparator_ml,
        correlated_probit_bootstrap_raw_results_comparator_ml,
        correlated_probit_bootstrap_n_samples,
        correlated_probit_bootstrap_plots_with_annotators_comparator_ml_output_path
    )


def probit_bootstrap_talent_landscape_plot_comparator_ml(
    correlated_probit_bootstrap_results_comparator_ml: pd.DataFrame,
    company_database_comparator_ml: pd.DataFrame,
    probit_bootstrap_talent_landscape_plot_comparator_ml_output_path: str
) -> plt.Figure:
    """Create talent landscape plot for comparator ML pipeline."""
    if len(correlated_probit_bootstrap_results_comparator_ml) == 0:
        return _create_empty_plot(
            'No results to plot',
            'Talent Landscape Plot (Comparator ML)',
            probit_bootstrap_talent_landscape_plot_comparator_ml_output_path
        )
    return probit_bootstrap_talent_landscape_plot(
        correlated_probit_bootstrap_results_comparator_ml,
        company_database_comparator_ml,
        probit_bootstrap_talent_landscape_plot_comparator_ml_output_path
    )


def probit_bootstrap_per_organization_estimates_plot_comparator_ml(
    final_results_all: pd.DataFrame,
    dawid_skene_test_data_comparator_ml: pd.DataFrame,
    probit_bootstrap_per_organization_estimates_plot_comparator_ml_output_path: str
) -> plt.Figure:
    """Create per-organization estimates plot for comparator ML pipeline."""
    # Filter final_results_all to only comparator ML companies
    # Comparator ML companies are those in the comparator_ml test data
    comparator_ml_ids = set(dawid_skene_test_data_comparator_ml['company_id'].unique())
    filtered_results = final_results_all[final_results_all['linkedin_id'].isin(comparator_ml_ids)].copy()
    
    if len(filtered_results) == 0:
        return _create_empty_plot(
            'No results to plot',
            'Per-Organization Estimates Plot (Comparator ML)',
            probit_bootstrap_per_organization_estimates_plot_comparator_ml_output_path
        )
    return probit_bootstrap_per_organization_estimates_plot(
        filtered_results,
        dawid_skene_test_data_comparator_ml,
        probit_bootstrap_per_organization_estimates_plot_comparator_ml_output_path
    )


def probit_bootstrap_prior_distribution_plot_comparator_ml(
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
    probit_bootstrap_prior_distribution_plot_comparator_ml_output_path: str
) -> plt.Figure:
    """Create prior distribution plot for comparator ML pipeline."""
    return probit_bootstrap_prior_distribution_plot(
        correlated_probit_bootstrap_prior_alpha,
        correlated_probit_bootstrap_prior_beta,
        probit_bootstrap_prior_distribution_plot_comparator_ml_output_path
    )


# Comparator Non-ML pipeline plot functions
def correlated_probit_bootstrap_plots_with_annotators_comparator_non_ml(
    dawid_skene_test_data_comparator_non_ml: pd.DataFrame,
    correlated_probit_bootstrap_raw_results_comparator_non_ml: Dict[str, Any],
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_plots_with_annotators_comparator_non_ml_output_path: str
) -> plt.Figure:
    """Create bootstrap plots for comparator Non-ML pipeline."""
    if len(dawid_skene_test_data_comparator_non_ml) == 0 or not correlated_probit_bootstrap_raw_results_comparator_non_ml:
        return _create_empty_plot(
            'No bootstrap results to plot',
            'Bootstrap Results with Annotators (Comparator Non-ML)',
            correlated_probit_bootstrap_plots_with_annotators_comparator_non_ml_output_path
        )
    return correlated_probit_bootstrap_plots_with_annotators(
        dawid_skene_test_data_comparator_non_ml,
        correlated_probit_bootstrap_raw_results_comparator_non_ml,
        correlated_probit_bootstrap_n_samples,
        correlated_probit_bootstrap_plots_with_annotators_comparator_non_ml_output_path
    )


def probit_bootstrap_talent_landscape_plot_comparator_non_ml(
    correlated_probit_bootstrap_results_comparator_non_ml: pd.DataFrame,
    company_database_comparator_non_ml: pd.DataFrame,
    probit_bootstrap_talent_landscape_plot_comparator_non_ml_output_path: str
) -> plt.Figure:
    """Create talent landscape plot for comparator Non-ML pipeline."""
    if len(correlated_probit_bootstrap_results_comparator_non_ml) == 0:
        return _create_empty_plot(
            'No results to plot',
            'Talent Landscape Plot (Comparator Non-ML)',
            probit_bootstrap_talent_landscape_plot_comparator_non_ml_output_path
        )
    return probit_bootstrap_talent_landscape_plot(
        correlated_probit_bootstrap_results_comparator_non_ml,
        company_database_comparator_non_ml,
        probit_bootstrap_talent_landscape_plot_comparator_non_ml_output_path
    )


def probit_bootstrap_per_organization_estimates_plot_comparator_non_ml(
    final_results_all: pd.DataFrame,
    dawid_skene_test_data_comparator_non_ml: pd.DataFrame,
    probit_bootstrap_per_organization_estimates_plot_comparator_non_ml_output_path: str
) -> plt.Figure:
    """Create per-organization estimates plot for comparator Non-ML pipeline."""
    # Filter final_results_all to only comparator Non-ML companies
    # Comparator Non-ML companies are those in the comparator_non_ml test data
    comparator_non_ml_ids = set(dawid_skene_test_data_comparator_non_ml['company_id'].unique())
    filtered_results = final_results_all[final_results_all['linkedin_id'].isin(comparator_non_ml_ids)].copy()
    
    if len(filtered_results) == 0:
        return _create_empty_plot(
            'No results to plot',
            'Per-Organization Estimates Plot (Comparator Non-ML)',
            probit_bootstrap_per_organization_estimates_plot_comparator_non_ml_output_path
        )
    return probit_bootstrap_per_organization_estimates_plot(
        filtered_results,
        dawid_skene_test_data_comparator_non_ml,
        probit_bootstrap_per_organization_estimates_plot_comparator_non_ml_output_path
    )


def probit_bootstrap_prior_distribution_plot_comparator_non_ml(
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
    probit_bootstrap_prior_distribution_plot_comparator_non_ml_output_path: str
) -> plt.Figure:
    """Create prior distribution plot for comparator Non-ML pipeline."""
    return probit_bootstrap_prior_distribution_plot(
        correlated_probit_bootstrap_prior_alpha,
        correlated_probit_bootstrap_prior_beta,
        probit_bootstrap_prior_distribution_plot_comparator_non_ml_output_path
    )


@cache(behavior="recompute")
def real_vs_synthetic_comparison_plot(
    final_results_main_orgs: pd.DataFrame,
    real_vs_synthetic_comparison_plot_output_path: str
) -> plt.Figure:
    """
    Create comparison plot for companies with both real and synthetic data.
    
    Shows scatter plot of means, CI coverage, and identifies discrepancies.
    
    Args:
        final_results_main_orgs: Final results with both real and synthetic estimates
        real_vs_synthetic_comparison_plot_output_path: Output path for the plot
        
    Returns:
        Figure object
    """
    logger.info("Creating real vs synthetic comparison plot...")
    
    # Filter to companies with both real and synthetic estimates
    # Column name is 'mean' (not 'probit_mean') in final_results_main_orgs
    has_real = final_results_main_orgs['mean'].notna()
    has_synthetic = final_results_main_orgs['synthetic_mean'].notna()
    merged = final_results_main_orgs[has_real & has_synthetic].copy()
    
    if len(merged) == 0:
        logger.warning("No companies with both real and synthetic estimates found")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No companies with both real and synthetic estimates', 
                ha='center', va='center', fontsize=14)
        ax.set_title('Real vs Synthetic Comparison')
        plt.tight_layout()
        return fig
    
    logger.info(f"Found {len(merged)} companies with both real and synthetic estimates")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Scatter plot: mean (real) vs synthetic_mean with asymmetric error bars (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate asymmetric error bars (ensure non-negative)
    real_lower = np.maximum(0, merged['mean'] - merged['q10'])
    real_upper = np.maximum(0, merged['q90'] - merged['mean'])
    synthetic_lower = np.maximum(0, merged['synthetic_mean'] - merged['synthetic_q10'])
    synthetic_upper = np.maximum(0, merged['synthetic_q90'] - merged['synthetic_mean'])
    
    # Use log scale
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot with error bars
    ax1.errorbar(merged['mean'], merged['synthetic_mean'],
                 xerr=[real_lower, real_upper],
                 yerr=[synthetic_lower, synthetic_upper],
                 fmt='o', alpha=0.6, markersize=4, capsize=2, capthick=1,
                 elinewidth=0.8, label='Companies')
    
    # Add diagonal line
    max_val = max(merged['mean'].max(), merged['synthetic_mean'].max())
    min_val = min(merged['mean'].min(), merged['synthetic_mean'].min())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    ax1.set_xlabel('Real Data Mean Estimate (log scale)')
    ax1.set_ylabel('Synthetic Data Mean Estimate (log scale)')
    ax1.set_title('Mean Estimates: Real vs Synthetic (with CIs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate correlation
    corr = merged['mean'].corr(merged['synthetic_mean'])
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Difference plot: synthetic_mean - mean (real) with log scale
    ax2 = fig.add_subplot(gs[0, 1])
    diff = merged['synthetic_mean'] - merged['mean']
    ax2.scatter(merged['mean'], diff, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Real Data Mean Estimate (log scale)')
    ax2.set_ylabel('Difference (Synthetic - Real)')
    ax2.set_title('Mean Difference: Synthetic - Real')
    ax2.grid(True, alpha=0.3)
    
    # Add mean difference
    mean_diff = diff.mean()
    ax2.text(0.05, 0.95, f'Mean Difference: {mean_diff:.2f}', transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. CI Coverage: Check if real CI falls within synthetic CI
    # "Covered" means the real data's 80% CI (q10 to q90) is completely contained within 
    # the synthetic data's 80% CI. This indicates that the synthetic approach captures 
    # the uncertainty in the real data estimate.
    ax3 = fig.add_subplot(gs[1, 0])
    coverage = []
    for _, row in merged.iterrows():
        # Real CI is "covered" if both q10 and q90 fall within synthetic CI bounds
        real_in_synthetic = (row['q10'] >= row['synthetic_q10']) and (row['q90'] <= row['synthetic_q90'])
        coverage.append(real_in_synthetic)
    coverage = np.array(coverage)
    coverage_pct = coverage.mean() * 100
    
    ax3.bar(['Covered', 'Not Covered'], [coverage.sum(), (~coverage).sum()], 
            color=['green', 'red'], alpha=0.6)
    ax3.set_ylabel('Number of Companies')
    ax3.set_title(f'Real CI Coverage by Synthetic CI\n({coverage_pct:.1f}% covered)\n"Covered" = real CI fully within synthetic CI')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. CI Width comparison (log scale)
    ax4 = fig.add_subplot(gs[1, 1])
    real_width = merged['q90'] - merged['q10']
    synthetic_width = merged['synthetic_q90'] - merged['synthetic_q10']
    ax4.scatter(real_width, synthetic_width, alpha=0.6, s=50)
    max_width = max(real_width.max(), synthetic_width.max())
    min_width = min(real_width.min(), synthetic_width.min())
    ax4.plot([min_width, max_width], [min_width, max_width], 'r--', alpha=0.5, label='y=x')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Real CI Width (q90 - q10, log scale)')
    ax4.set_ylabel('Synthetic CI Width (q90 - q10, log scale)')
    ax4.set_title('Confidence Interval Width Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Identify significant discrepancies (mean difference > 20% of real mean) with log scale and error bars
    ax5 = fig.add_subplot(gs[2, :])
    rel_diff = np.abs(diff) / (merged['mean'] + 1e-6) * 100  # Percentage difference
    significant = rel_diff > 20
    
    # Plot non-significant with error bars
    non_sig_mask = ~significant
    if non_sig_mask.sum() > 0:
        ax5.errorbar(merged.loc[non_sig_mask, 'mean'], 
                    merged.loc[non_sig_mask, 'synthetic_mean'],
                    xerr=[real_lower[non_sig_mask], real_upper[non_sig_mask]],
                    yerr=[synthetic_lower[non_sig_mask], synthetic_upper[non_sig_mask]],
                    fmt='o', alpha=0.6, markersize=4, capsize=2, capthick=1,
                    elinewidth=0.8, label='Agreement (<20% diff)', color='green')
    
    # Plot significant with error bars
    if significant.sum() > 0:
        ax5.errorbar(merged.loc[significant, 'mean'],
                    merged.loc[significant, 'synthetic_mean'],
                    xerr=[real_lower[significant], real_upper[significant]],
                    yerr=[synthetic_lower[significant], synthetic_upper[significant]],
                    fmt='x', alpha=0.8, markersize=8, capsize=3, capthick=1.5,
                    elinewidth=1.2, label='Discrepancy (>20% diff)', color='red')
    
    max_val = max(merged['mean'].max(), merged['synthetic_mean'].max())
    min_val = min(merged['mean'].min(), merged['synthetic_mean'].min())
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('Real Data Mean Estimate (log scale)')
    ax5.set_ylabel('Synthetic Data Mean Estimate (log scale)')
    ax5.set_title(f'Significant Discrepancies (>20% difference)\n{significant.sum()} companies flagged')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add text summary
    summary_text = (
        f"Total companies: {len(merged)}\n"
        f"Mean correlation: {corr:.3f}\n"
        f"Mean difference: {mean_diff:.2f}\n"
        f"CI coverage: {coverage_pct:.1f}%\n"
        f"Significant discrepancies: {significant.sum()} ({significant.sum()/len(merged)*100:.1f}%)"
    )
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='bottom')
    
    plt.suptitle('Real vs Synthetic Data Comparison', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save the plot
    Path(real_vs_synthetic_comparison_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(real_vs_synthetic_comparison_plot_output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved comparison plot to {real_vs_synthetic_comparison_plot_output_path}")
    logger.info(f"  Companies compared: {len(merged)}")
    logger.info(f"  Mean correlation: {corr:.3f}")
    logger.info(f"  CI coverage: {coverage_pct:.1f}%")
    logger.info(f"  Significant discrepancies: {significant.sum()}")
    
    return fig


@cache(behavior="recompute")
def real_vs_synthetic_per_company_plot(
    final_results_main_orgs: pd.DataFrame,
    real_vs_synthetic_per_company_plot_output_path: str
) -> plt.Figure:
    """
    Create per-company comparison plot showing both real and synthetic estimates side-by-side.
    
    Similar to probit_bootstrap_per_organization_estimates_plot, but shows both real and
    synthetic estimates with error bars to visualize where synthetic CIs envelop real ones.
    
    Args:
        final_results_main_orgs: Final results with both real and synthetic estimates
        real_vs_synthetic_per_company_plot_output_path: Output path for the plot
        
    Returns:
        Figure object
    """
    logger.info("Creating real vs synthetic per-company comparison plot...")
    
    # Filter to companies with both real and synthetic estimates
    has_real = final_results_main_orgs['mean'].notna()
    has_synthetic = final_results_main_orgs['synthetic_mean'].notna()
    merged = final_results_main_orgs[has_real & has_synthetic].copy()
    
    if len(merged) == 0:
        logger.warning("No companies with both real and synthetic estimates found")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No companies with both real and synthetic estimates', 
                ha='center', va='center', fontsize=14)
        ax.set_title('Real vs Synthetic Per-Company Comparison')
        plt.tight_layout()
        Path(real_vs_synthetic_per_company_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(real_vs_synthetic_per_company_plot_output_path, dpi=300, bbox_inches='tight')
        return fig
    
    logger.info(f"Found {len(merged)} companies with both real and synthetic estimates")
    
    # Sort by real mean estimate for better visualization
    merged = merged.sort_values('mean', ascending=False).reset_index(drop=True)
    
    # Limit to top N companies for readability (or all if fewer)
    max_companies = 40
    if len(merged) > max_companies:
        logger.info(f"Limiting to top {max_companies} companies by real mean estimate")
        merged = merged.head(max_companies)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(16, len(merged) * 0.35), 10))
    
    # Prepare data for plotting
    x_pos = np.arange(len(merged))
    company_names = merged['organization_name'].fillna(merged['linkedin_id']).values
    
    # Calculate error bars (asymmetric)
    # Use np.maximum to ensure non-negative error bars (handles edge cases)
    real_lower = np.maximum(0, merged['mean'] - merged['q10'])
    real_upper = np.maximum(0, merged['q90'] - merged['mean'])
    synthetic_lower = np.maximum(0, merged['synthetic_mean'] - merged['synthetic_q10'])
    synthetic_upper = np.maximum(0, merged['synthetic_q90'] - merged['synthetic_mean'])
    
    # Determine which companies have real CI covered by synthetic CI
    coverage = []
    for _, row in merged.iterrows():
        real_in_synthetic = (row['q10'] >= row['synthetic_q10']) and (row['q90'] <= row['synthetic_q90'])
        coverage.append(real_in_synthetic)
    coverage = np.array(coverage)
    
    # Calculate adjusted synthetic error bars
    adjusted_synthetic_lower = np.maximum(0, merged['adjusted_synthetic_mean'] - merged['adjusted_synthetic_q10'])
    adjusted_synthetic_upper = np.maximum(0, merged['adjusted_synthetic_q90'] - merged['adjusted_synthetic_mean'])
    
    # Plot real estimates (offset left) - tighter spacing for better visual separation
    real_offset = -0.15
    ax.errorbar(x_pos + real_offset, merged['mean'],
                yerr=[real_lower, real_upper],
                fmt='o', color='blue', alpha=0.7, markersize=6, capsize=3, capthick=1.5,
                elinewidth=1.2, label='Real Data', zorder=3)
    
    # Plot adjusted synthetic estimates (center)
    adjusted_offset = 0.0
    ax.errorbar(x_pos + adjusted_offset, merged['adjusted_synthetic_mean'],
                yerr=[adjusted_synthetic_lower, adjusted_synthetic_upper],
                fmt='D', color='green', alpha=0.7, markersize=6, capsize=3, capthick=1.5,
                elinewidth=1.2, label='Adjusted Synthetic Data', zorder=2)
    
    # Plot synthetic estimates (offset right)
    synthetic_offset = 0.15
    ax.errorbar(x_pos + synthetic_offset, merged['synthetic_mean'],
                yerr=[synthetic_lower, synthetic_upper],
                fmt='s', color='orange', alpha=0.7, markersize=6, capsize=3, capthick=1.5,
                elinewidth=1.2, label='Synthetic Data', zorder=2)
    
    # Add 3 overlaid scatterplots for keyword filter counts from company database
    keyword_filters = {
        'filter_broad_yes': {'marker': '^', 'color': 'red', 'label': 'filter_broad_yes'},
        'filter_strict_no': {'marker': 'v', 'color': 'green', 'label': 'filter_strict_no'},
        'filter_broad_yes_strict_no': {'marker': 'D', 'color': 'purple', 'label': 'filter_broad_yes_strict_no'}
    }
    
    for filter_name, style in keyword_filters.items():
        if filter_name in merged.columns:
            # Get filter counts, handling NaN and ensuring positive values for log scale
            filter_counts = merged[filter_name].fillna(0)
            # Only plot companies with non-zero counts
            mask = filter_counts > 0
            if mask.sum() > 0:
                # Use the same x_pos for alignment
                ax.scatter(x_pos[mask], 
                          filter_counts[mask],
                          marker=style['marker'], 
                          color=style['color'], 
                          alpha=0.7, 
                          s=60,
                          edgecolors='black',
                          linewidths=0.5,
                          label=style['label'],
                          zorder=5)
    
    # Add 3 overlaid scatterplots for LLM annotator sums from company database
    llm_annotators = {
        'gemini_total_accepted': {'marker': '*', 'color': 'cyan', 'label': 'Gemini Total Accepted'},
        'claude_total_accepted': {'marker': 'X', 'color': 'magenta', 'label': 'Claude Total Accepted'},
        'gpt5_total_accepted': {'marker': 'P', 'color': 'yellow', 'label': 'GPT-5 Total Accepted'}
    }
    
    for annotator_name, style in llm_annotators.items():
        if annotator_name in merged.columns:
            # Get annotator counts, handling NaN and ensuring positive values for log scale
            annotator_counts = merged[annotator_name].fillna(0)
            # Only plot companies with non-zero counts
            mask = annotator_counts > 0
            if mask.sum() > 0:
                # Use the same x_pos for alignment
                ax.scatter(x_pos[mask], 
                          annotator_counts[mask],
                          marker=style['marker'], 
                          color=style['color'], 
                          alpha=0.7, 
                          s=60,
                          edgecolors='black',
                          linewidths=0.5,
                          label=style['label'],
                          zorder=5)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    ax.set_ylabel('ML Headcount Estimate / Annotator Count (log scale)', fontsize=12)
    ax.set_xlabel('Company', fontsize=12)
    ax.set_title('Real vs Synthetic Estimates Per Company + Annotator Counts\n(Companies sorted by real mean estimate)', 
                 fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(company_names, rotation=45, ha='right', fontsize=8)
    
    # Eliminate whitespace on the sides
    ax.set_xlim(-0.5, len(x_pos) - 0.5)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add coverage statistics text
    coverage_pct = coverage.mean() * 100
    stats_text = f"Coverage: {coverage.sum()}/{len(merged)} ({coverage_pct:.1f}%)\nCompanies shown: {len(merged)}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    Path(real_vs_synthetic_per_company_plot_output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(real_vs_synthetic_per_company_plot_output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved per-company comparison plot to {real_vs_synthetic_per_company_plot_output_path}")
    logger.info(f"  Companies shown: {len(merged)}")
    logger.info(f"  Coverage: {coverage.sum()}/{len(merged)} ({coverage_pct:.1f}%)")
    
    return fig
