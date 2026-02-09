"""
Synthetic Employee-Level Data Generation

This module implements synthetic data generation for companies with only
company-level aggregate data, using tetrachoric correlations estimated from
training companies and company-specific prevalences.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def estimate_tetrachoric_correlation_binary(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate tetrachoric correlation between two binary variables.
    
    Uses maximum likelihood estimation from 2×2 contingency table.
    
    Args:
        x: Binary array (0/1)
        y: Binary array (0/1)
        
    Returns:
        Tetrachoric correlation coefficient (in [-1, 1])
    """
    # Remove any NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask].astype(int)
    y_clean = y[valid_mask].astype(int)
    
    if len(x_clean) < 2:
        return 0.0
    
    # Compute 2×2 contingency table
    n_00 = np.sum((x_clean == 0) & (y_clean == 0))
    n_01 = np.sum((x_clean == 0) & (y_clean == 1))
    n_10 = np.sum((x_clean == 1) & (y_clean == 0))
    n_11 = np.sum((x_clean == 1) & (y_clean == 1))
    
    n_total = len(x_clean)
    
    # Marginal probabilities
    p_x1 = (n_10 + n_11) / n_total
    p_y1 = (n_01 + n_11) / n_total
    
    # If either marginal is 0 or 1, correlation is undefined
    if p_x1 <= 0 or p_x1 >= 1 or p_y1 <= 0 or p_y1 >= 1:
        return 0.0
    
    # Thresholds (probit space)
    tau_x = stats.norm.ppf(p_x1)
    tau_y = stats.norm.ppf(p_y1)
    
    # Observed joint probabilities
    p_00 = n_00 / n_total
    p_01 = n_01 / n_total
    p_10 = n_10 / n_total
    p_11 = n_11 / n_total
    
    # Negative log-likelihood function
    def neg_log_likelihood(rho):
        """Negative log-likelihood for tetrachoric correlation rho"""
        # Clamp rho to valid range
        rho = np.clip(rho, -0.999, 0.999)
        
        # Bivariate normal probabilities
        # P(X=0, Y=0) = Φ_2(-τ_x, -τ_y; ρ)
        # P(X=0, Y=1) = Φ(-τ_x) - Φ_2(-τ_x, -τ_y; ρ)
        # P(X=1, Y=0) = Φ(-τ_y) - Φ_2(-τ_x, -τ_y; ρ)
        # P(X=1, Y=1) = 1 - Φ(-τ_x) - Φ(-τ_y) + Φ_2(-τ_x, -τ_y; ρ)
        
        try:
            # Use bivariate normal CDF
            from scipy.stats import multivariate_normal
            
            # Mean vector (both thresholds are negative in probit space)
            mean = np.array([0.0, 0.0])
            cov = np.array([[1.0, rho], [rho, 1.0]])
            
            # P(X=0, Y=0) = P(Z_x < -τ_x, Z_y < -τ_y)
            p_00_model = multivariate_normal.cdf([-tau_x, -tau_y], mean=mean, cov=cov)
            
            # P(X=0) = Φ(-τ_x)
            p_x0 = stats.norm.cdf(-tau_x)
            # P(Y=0) = Φ(-τ_y)
            p_y0 = stats.norm.cdf(-tau_y)
            
            # P(X=0, Y=1) = P(X=0) - P(X=0, Y=0)
            p_01_model = p_x0 - p_00_model
            # P(X=1, Y=0) = P(Y=0) - P(X=0, Y=0)
            p_10_model = p_y0 - p_00_model
            # P(X=1, Y=1) = 1 - P(X=0) - P(Y=0) + P(X=0, Y=0)
            p_11_model = 1 - p_x0 - p_y0 + p_00_model
            
            # Ensure probabilities are valid
            p_00_model = np.clip(p_00_model, 1e-10, 1.0)
            p_01_model = np.clip(p_01_model, 1e-10, 1.0)
            p_10_model = np.clip(p_10_model, 1e-10, 1.0)
            p_11_model = np.clip(p_11_model, 1e-10, 1.0)
            
            # Negative log-likelihood
            nll = -(n_00 * np.log(p_00_model) + 
                   n_01 * np.log(p_01_model) + 
                   n_10 * np.log(p_10_model) + 
                   n_11 * np.log(p_11_model))
            
            return nll
            
        except Exception as e:
            # Return large value if computation fails
            return 1e10
    
    # Optimize to find maximum likelihood estimate
    try:
        result = minimize_scalar(neg_log_likelihood, bounds=(-0.999, 0.999), method='bounded')
        rho_est = result.x
    except Exception as e:
        logger.warning(f"Failed to estimate tetrachoric correlation: {e}, using 0.0")
        rho_est = 0.0
    
    return np.clip(rho_est, -1.0, 1.0)


def estimate_test_keyword_filter_correlation_matrix(dawid_skene_test_data: pd.DataFrame) -> np.ndarray:
    """
    Estimate 3×3 tetrachoric correlation matrix from all test data (keyword filters only).
    
    This represents the unconditional correlation structure of the latent Gaussian variables
    for the 3 keyword filter annotators, estimated from all employee-level test data.
    
    Args:
        dawid_skene_test_data: Test data with employee-level annotations
                               Must contain columns: filter_broad_yes, filter_strict_no, filter_broad_yes_strict_no
        
    Returns:
        3×3 correlation matrix (values in [-1, 1], diagonal = 1)
    """
    import time
    start_time = time.time()
    
    logger.info("Estimating test keyword filter correlation matrix from all test data (unconditional)...")
    
    # Define keyword filter columns (only 3 annotators)
    keyword_filter_cols = [
        'filter_broad_yes',
        'filter_strict_no',
        'filter_broad_yes_strict_no'
    ]
    
    # Check that all columns exist
    missing_cols = [col for col in keyword_filter_cols if col not in dawid_skene_test_data.columns]
    if missing_cols:
        raise ValueError(f"Missing keyword filter columns in test data: {missing_cols}")
    
    n_annotators = len(keyword_filter_cols)
    n_items = len(dawid_skene_test_data)
    
    logger.info(f"Test data: {n_items} items, {n_annotators} keyword filter annotators")
    
    # Extract annotation data (only keyword filters)
    annotations = dawid_skene_test_data[keyword_filter_cols].values
    
    # Initialize correlation matrix
    correlation_matrix = np.eye(n_annotators)
    
    # Estimate tetrachoric correlation for each pair
    logger.info("Computing tetrachoric correlations for keyword filter pairs...")
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            x = annotations[:, i]
            y = annotations[:, j]
            
            rho_ij = estimate_tetrachoric_correlation_binary(x, y)
            
            # Fill symmetric matrix
            correlation_matrix[i, j] = rho_ij
            correlation_matrix[j, i] = rho_ij
    
    tetrachoric_time = time.time() - start_time
    logger.info(f"Tetrachoric correlation estimation completed in {tetrachoric_time:.1f}s")
    
    # Ensure positive definiteness with PD projection
    logger.info("Checking positive definiteness and projecting to PD if needed...")
    pd_start_time = time.time()
    try:
        np.linalg.cholesky(correlation_matrix)
        logger.info("Tetrachoric matrix is already positive definite")
    except np.linalg.LinAlgError:
        logger.info("Tetrachoric matrix not positive definite, projecting to nearest PD...")
        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        correlation_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalize to correlation matrix
        diag_sqrt = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
    
    pd_time = time.time() - pd_start_time
    total_time = time.time() - start_time
    
    logger.info(f"PD projection completed in {pd_time:.3f}s (total: {total_time:.1f}s)")
    logger.info(f"Test keyword filter correlation matrix estimated: shape {correlation_matrix.shape}, "
                f"range [{correlation_matrix.min():.3f}, {correlation_matrix.max():.3f}]")
    
    return correlation_matrix


def estimate_tetrachoric_correlations(validation_data: pd.DataFrame) -> np.ndarray:
    """
    Estimate 6×6 tetrachoric correlation matrix from validation data.
    
    This represents the correlation structure of the latent Gaussian variables
    that generate the binary annotations. It is company-invariant.
    
    Args:
        validation_data: DataFrame with 6 annotator columns and 'category' (true label)
                       Expected columns: llm_gemini_2_5_flash, llm_sonnet_4, llm_gpt_5_mini,
                                        filter_broad_yes, filter_strict_no, filter_broad_yes_strict_no
        
    Returns:
        6×6 correlation matrix (values in [-1, 1], diagonal = 1)
    """
    logger.info("Estimating tetrachoric correlation matrix from validation data...")
    
    # Define annotator columns in order (3 LLMs + 3 keyword filters)
    annotator_cols = [
        'llm_gemini_2_5_flash',
        'llm_sonnet_4', 
        'llm_gpt_5_mini',
        'filter_broad_yes',
        'filter_strict_no',
        'filter_broad_yes_strict_no'
    ]
    
    # Check that all columns exist
    missing_cols = [col for col in annotator_cols if col not in validation_data.columns]
    if missing_cols:
        raise ValueError(f"Missing annotator columns: {missing_cols}")
    
    n_annotators = len(annotator_cols)
    correlation_matrix = np.eye(n_annotators)  # Initialize with identity
    
    # Extract annotation data
    annotations = validation_data[annotator_cols].values
    
    # Estimate tetrachoric correlation for each pair
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            x = annotations[:, i]
            y = annotations[:, j]
            
            rho_ij = estimate_tetrachoric_correlation_binary(x, y)
            
            # Fill symmetric matrix
            correlation_matrix[i, j] = rho_ij
            correlation_matrix[j, i] = rho_ij
    
    logger.info(f"Estimated tetrachoric correlation matrix (shape: {correlation_matrix.shape})")
    logger.info(f"Correlation range: [{correlation_matrix.min():.3f}, {correlation_matrix.max():.3f}]")
    
    return correlation_matrix


def generate_synthetic_employee_data(
    company_id: str,
    company_aggregates: Dict[str, float],
    n_employees: int,
    tetrachoric_corr: np.ndarray,
    annotator_names: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic employee-level binary annotations for a company.
    
    Uses Gaussian copula approach with tetrachoric correlations and
    company-specific prevalences from company database aggregates.
    
    Args:
        company_id: Company identifier
        company_aggregates: Dictionary mapping annotator names to aggregate counts
                           (e.g., {'filter_broad_yes': 3500, 'filter_strict_no': 1200, ...})
                           Should contain keyword filter counts from company_database_complete
        n_employees: Number of synthetic employees to generate (typically total_headcount)
        tetrachoric_corr: 3×3 tetrachoric correlation matrix (keyword filters only)
        annotator_names: Optional list of all annotator names (for backward compatibility)
                        If None, uses only keyword filters from company_aggregates
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic annotations matching DawidSkeneTestDataSchema
        Columns: company_id, group, and annotator columns (keyword filters only, LLMs as NaN)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define keyword filter annotators (only these are used for synthetic data)
    keyword_filter_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    
    # Determine available annotators from company aggregates
    # Only use keyword filters that have aggregate data
    available_annotators = []
    available_indices = []
    
    for idx, annotator_name in enumerate(keyword_filter_cols):
        if annotator_name in company_aggregates:
            val = company_aggregates[annotator_name]
            if val is not None and not np.isnan(val) and val >= 0:
                available_annotators.append(annotator_name)
                available_indices.append(idx)
    
    if len(available_annotators) == 0:
        logger.warning(f"No available keyword filter annotators for company {company_id}, returning empty DataFrame")
        # Return empty DataFrame with correct schema
        result = pd.DataFrame({
            'company_id': [],
            'group': []
        })
        # Include all possible annotator columns (matching DawidSkeneTestDataSchema)
        all_annotators = keyword_filter_cols + ['llm_gemini_2_5_flash', 'llm_sonnet_4', 'llm_gpt_5_mini']
        for annotator_name in all_annotators:
            result[annotator_name] = []
        return result
    
    # Extract submatrix Σ_A from 3×3 correlation matrix
    # Since we're using only keyword filters, available_indices should be [0, 1, 2] or subset
    available_indices = np.array(available_indices)
    corr_A = tetrachoric_corr[np.ix_(available_indices, available_indices)]
    
    # Ensure positive definiteness (add small regularization if needed)
    try:
        # Try Cholesky decomposition to check positive definiteness
        chol_A = np.linalg.cholesky(corr_A)
    except np.linalg.LinAlgError:
        # If not positive definite, regularize
        logger.warning(f"Correlation matrix not positive definite for company {company_id}, regularizing...")
        corr_A = corr_A + np.eye(len(available_indices)) * 1e-6
        # Re-normalize to ensure it's still a correlation matrix
        diag_sqrt = np.sqrt(np.diag(corr_A))
        corr_A = corr_A / np.outer(diag_sqrt, diag_sqrt)
        chol_A = np.linalg.cholesky(corr_A)
    
    # Compute thresholds for each available annotator
    thresholds = []
    prevalences = []
    
    for annotator_name in available_annotators:
        aggregate_count = company_aggregates[annotator_name]
        prevalence = aggregate_count / n_employees if n_employees > 0 else 0.0
        prevalences.append(prevalence)
        
        # Clamp prevalence to valid range
        prevalence = np.clip(prevalence, 1e-6, 1.0 - 1e-6)
        
        # Compute probit threshold: τ_j = Φ^(-1)(1 - p_j)
        # This ensures P(Z_j > τ_j) = p_j
        threshold = stats.norm.ppf(1.0 - prevalence)
        thresholds.append(threshold)
    
    thresholds = np.array(thresholds)
    
    # Generate latent vectors for all employees at once (vectorized)
    # Z_A ~ N(0, Σ_A) where Σ_A is the correlation matrix
    # Since it's a correlation matrix, we can use it directly
    # Generate standard normal and transform
    z_standard = np.random.randn(n_employees, len(available_annotators))
    z_latent = z_standard @ chol_A.T  # Transform to have correlation Σ_A
    
    # Apply thresholds to get binary annotations
    # Y_j = 1[Z_j > τ_j]
    annotations_binary = (z_latent > thresholds).astype(float)
    
    # Create DataFrame with all annotators
    # Use a deterministic hash of company_id to create a consistent group number
    # This ensures each company gets a unique but deterministic group ID
    # Use hashlib for deterministic hashing across processes
    import hashlib
    hash_obj = hashlib.md5(str(company_id).encode('utf-8'))
    group_id = int(hash_obj.hexdigest(), 16) % (10**9)  # Large positive integer
    
    result_data = {
        'company_id': [company_id] * n_employees,
        'group': [group_id] * n_employees
    }
    
    # Initialize all annotator columns (keyword filters + LLMs, matching DawidSkeneTestDataSchema)
    all_annotator_cols = keyword_filter_cols + ['llm_gemini_2_5_flash', 'llm_sonnet_4', 'llm_gpt_5_mini']
    for annotator_name in all_annotator_cols:
        result_data[annotator_name] = [np.nan] * n_employees
    
    # Fill in available keyword filter annotators
    for idx, annotator_name in enumerate(available_annotators):
        result_data[annotator_name] = annotations_binary[:, idx].tolist()
    
    result_df = pd.DataFrame(result_data)
    
    logger.debug(f"Generated {n_employees} synthetic employees for company {company_id} "
                f"with {len(available_annotators)} available keyword filter annotators")
    
    return result_df


def identify_companies_needing_synthetic_data(
    company_database_complete: pd.DataFrame,
    dawid_skene_test_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify companies that need synthetic data generation.
    
    A company needs synthetic data if:
    - It exists in company_database_complete
    - It has nonzero keyword filter aggregates
    - It has no (or very few) employee-level records in test data
    
    Args:
        company_database_complete: Company database with aggregate data
        dawid_skene_test_data: Test data with employee-level annotations
        
    Returns:
        DataFrame with company_id, aggregate values, and metadata
    """
    logger.info("Identifying companies needing synthetic data...")
    
    # Get companies in test data
    companies_in_test = set(dawid_skene_test_data['company_id'].unique())
    
    # Keyword filter columns
    keyword_filter_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    
    # Find companies with aggregates but missing from test data
    companies_needing_synthetic = []
    
    for _, company_row in company_database_complete.iterrows():
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
        
        # Check if company has keyword filter aggregates
        has_aggregates = False
        aggregate_values = {}
        
        for col in keyword_filter_cols:
            if col in company_row:
                val = company_row[col]
                if not pd.isna(val) and val > 0:
                    has_aggregates = True
                    aggregate_values[col] = float(val)
                else:
                    aggregate_values[col] = 0.0
        
        # Check if company is missing from test data or has very few records
        in_test_data = company_id in companies_in_test
        
        if in_test_data:
            # Count employees in test data
            n_test_employees = len(dawid_skene_test_data[dawid_skene_test_data['company_id'] == company_id])
            # If very few employees, we might still want to generate synthetic data
            # But for now, only generate for companies completely missing
            if n_test_employees > 0:
                continue
        
        # If company has aggregates but no test data, it needs synthetic data
        if has_aggregates:
            companies_needing_synthetic.append({
                'company_id': company_id,
                'organization_name': company_row.get('organization_name', ''),
                'total_headcount': company_row.get('total_headcount', 0),
                'filter_broad_yes': aggregate_values.get('filter_broad_yes', 0.0),
                'filter_strict_no': aggregate_values.get('filter_strict_no', 0.0),
                'filter_broad_yes_strict_no': aggregate_values.get('filter_broad_yes_strict_no', 0.0),
                'has_test_data': in_test_data,
                'n_test_employees': len(dawid_skene_test_data[dawid_skene_test_data['company_id'] == company_id]) if in_test_data else 0
            })
    
    result_df = pd.DataFrame(companies_needing_synthetic)
    
    logger.info(f"Identified {len(result_df)} companies needing synthetic data")
    
    return result_df
