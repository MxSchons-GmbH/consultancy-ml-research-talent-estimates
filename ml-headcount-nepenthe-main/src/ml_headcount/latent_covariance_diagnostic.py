"""
Latent Covariance Diagnostic Module

This module implements Stage 0 of the probit upgrade plan: validating that companies
share similar latent correlation structures as the validation set.
"""

from hamilton.function_modifiers import source, save_to, config, check_output, cache
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from ml_headcount.probit_dawid_skene_inference import (
    estimate_class_covariance
)
from ml_headcount.synthetic_data_generation import estimate_tetrachoric_correlation_binary

logger = logging.getLogger(__name__)


@cache()
def validation_covariance(
    dawid_skene_validation_data: pd.DataFrame
) -> np.ndarray:
    """
    Estimate unconditional covariance matrix from all validation data (not split by true label).
    
    Uses tetrachoric correlations + PD projection (no EM/QMC needed).
    
    Args:
        dawid_skene_validation_data: Validation data with annotations
        
    Returns:
        6×6 correlation matrix (diagonal = 1)
    """
    import time
    start_time = time.time()
    
    logger.info("Estimating validation covariance from all data (unconditional)...")
    
    # Get annotator columns
    annotator_cols = [col for col in dawid_skene_validation_data.columns 
                     if col not in ['cv_text', 'category']]
    n_annotators = len(annotator_cols)
    n_items = len(dawid_skene_validation_data)
    
    logger.info(f"Validation data: {n_items} items, {n_annotators} annotators")
    
    # Get annotations (handle missing values)
    annotations = dawid_skene_validation_data[annotator_cols].values
    
    # Compute mean vector from marginal probabilities (unconditional)
    logger.info("Computing marginal probabilities for mu...")
    from scipy import stats
    mu = np.zeros(n_annotators)
    for j in range(n_annotators):
        p_j = np.nanmean(annotations[:, j])
        p_j = np.clip(p_j, 1e-6, 1.0 - 1e-6)
        mu[j] = stats.norm.ppf(p_j)
    logger.info(f"Computed mu: {mu}")
    
    # Initialize from tetrachoric
    logger.info("Initializing covariance from tetrachoric correlations...")
    n_pairs = n_annotators * (n_annotators - 1) // 2
    init_Sigma = np.eye(n_annotators)
    pair_idx = 0
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            pair_idx += 1
            if pair_idx % 5 == 0 or pair_idx == n_pairs:
                logger.info(f"  Computing tetrachoric pair {pair_idx}/{n_pairs}: ({i}, {j})")
            x = annotations[:, i]
            y = annotations[:, j]
            rho_ij = estimate_tetrachoric_correlation_binary(x, y)
            init_Sigma[i, j] = rho_ij
            init_Sigma[j, i] = rho_ij
    
    tetrachoric_time = time.time() - start_time
    logger.info(f"Tetrachoric correlation estimation completed in {tetrachoric_time:.1f}s")
    
    # Ensure positive definiteness with PD projection
    logger.info("Checking positive definiteness and projecting to PD if needed...")
    pd_start_time = time.time()
    try:
        np.linalg.cholesky(init_Sigma)
        logger.info("Tetrachoric matrix is already positive definite")
        Sigma_validation = init_Sigma
    except np.linalg.LinAlgError:
        logger.info("Tetrachoric matrix not positive definite, projecting to nearest PD...")
        eigvals, eigvecs = np.linalg.eigh(init_Sigma)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma_validation = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalize to correlation matrix
        diag_sqrt = np.sqrt(np.diag(Sigma_validation))
        Sigma_validation = Sigma_validation / np.outer(diag_sqrt, diag_sqrt)
    
    pd_time = time.time() - pd_start_time
    total_time = time.time() - start_time
    
    logger.info(f"PD projection completed in {pd_time:.3f}s (total: {total_time:.1f}s)")
    logger.info(f"Validation covariance estimated: shape {Sigma_validation.shape}, range [{Sigma_validation.min():.3f}, {Sigma_validation.max():.3f}]")
    return Sigma_validation


@cache()
def company_covariances(
    dawid_skene_test_data: pd.DataFrame,
    latent_covariance_diagnostic_min_employees: int = 10
) -> Dict[str, np.ndarray]:
    """
    Estimate unconditional covariance matrix for each company with sufficient data.
    
    Args:
        dawid_skene_test_data: Test data with company_id and annotations
        latent_covariance_diagnostic_min_employees: Minimum employees per company to estimate covariance
        
    Returns:
        Dictionary mapping company_id to 6×6 correlation matrix
    """
    logger.info(f"Estimating per-company covariances (min employees: {latent_covariance_diagnostic_min_employees})...")
    
    # Get annotator columns
    annotator_cols = [col for col in dawid_skene_test_data.columns 
                     if col not in ['company_id', 'group']]
    n_annotators = len(annotator_cols)
    
    # Filter companies with enough employees
    company_counts = dawid_skene_test_data['company_id'].value_counts()
    valid_companies = company_counts[company_counts >= latent_covariance_diagnostic_min_employees].index.tolist()
    
    logger.info(f"Found {len(valid_companies)} companies with ≥{latent_covariance_diagnostic_min_employees} employees")
    
    import time
    total_start_time = time.time()
    company_covariances_dict = {}
    
    for idx, company_id in enumerate(valid_companies):
        company_start_time = time.time()
        logger.info(f"Processing company {idx + 1}/{len(valid_companies)}: {company_id}")
        
        company_data = dawid_skene_test_data[dawid_skene_test_data['company_id'] == company_id].copy()
        company_annotations = company_data[annotator_cols].values
        n_company_items = len(company_data)
        
        logger.info(f"  Company {company_id}: {n_company_items} employees")
        
        # Compute mean vector from marginal probabilities (unconditional)
        from scipy import stats
        mu = np.zeros(n_annotators)
        for j in range(n_annotators):
            p_j = np.nanmean(company_annotations[:, j])
            p_j = np.clip(p_j, 1e-6, 1.0 - 1e-6)
            mu[j] = stats.norm.ppf(p_j)
        
        # Estimate using tetrachoric correlations
        logger.info(f"  Computing tetrachoric correlations...")
        n_pairs = n_annotators * (n_annotators - 1) // 2
        Sigma_company = np.eye(n_annotators)
        pair_idx = 0
        for i in range(n_annotators):
            for j in range(i + 1, n_annotators):
                pair_idx += 1
                x = company_annotations[:, i]
                y = company_annotations[:, j]
                rho_ij = estimate_tetrachoric_correlation_binary(x, y)
                Sigma_company[i, j] = rho_ij
                Sigma_company[j, i] = rho_ij
        
        # Ensure positive definiteness with PD projection
        logger.info(f"  Checking positive definiteness...")
        try:
            np.linalg.cholesky(Sigma_company)
        except np.linalg.LinAlgError:
            logger.info(f"  Projecting to nearest PD matrix...")
            eigvals, eigvecs = np.linalg.eigh(Sigma_company)
            eigvals = np.maximum(eigvals, 1e-6)
            Sigma_company = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Renormalize to correlation matrix
            diag_sqrt = np.sqrt(np.diag(Sigma_company))
            Sigma_company = Sigma_company / np.outer(diag_sqrt, diag_sqrt)
        
        company_time = time.time() - company_start_time
        logger.info(f"  Company {company_id} completed in {company_time:.1f}s")
        
        company_covariances_dict[company_id] = Sigma_company
    
    total_time = time.time() - total_start_time
    logger.info(f"Estimated covariances for {len(company_covariances_dict)} companies in {total_time:.1f}s (avg {total_time/len(company_covariances_dict):.1f}s per company)")
    return company_covariances_dict


@cache()
def company_sizes_for_diagnostic(
    dawid_skene_test_data: pd.DataFrame,
    latent_covariance_diagnostic_min_employees: int = 10
) -> Dict[str, int]:
    """
    Get company sizes for companies included in covariance estimation.
    
    Args:
        dawid_skene_test_data: Test data with company_id
        latent_covariance_diagnostic_min_employees: Minimum employees threshold
        
    Returns:
        Dictionary mapping company_id to employee count
    """
    company_counts = dawid_skene_test_data['company_id'].value_counts()
    valid_companies = company_counts[company_counts >= latent_covariance_diagnostic_min_employees].index
    
    return {company_id: int(company_counts[company_id]) for company_id in valid_companies}


def flatten_correlation_matrix(Sigma: np.ndarray) -> np.ndarray:
    """
    Extract upper triangle (excluding diagonal) as a vector.
    
    Args:
        Sigma: Correlation matrix (n×n)
        
    Returns:
        Vector of length n*(n-1)/2 with unique correlations
    """
    upper_tri_indices = np.triu_indices_from(Sigma, k=1)
    return Sigma[upper_tri_indices]


@save_to.csv(
    path=source('latent_covariance_parallel_coordinates_data_output_path')
)
@cache()
def latent_covariance_parallel_coordinates_data(
    validation_covariance: np.ndarray,
    company_covariances: Dict[str, np.ndarray],
    company_sizes_for_diagnostic: Dict[str, int],
    dawid_skene_validation_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Build dataframe for parallel coordinates plot with flattened correlation matrices.
    
    Args:
        validation_covariance: Unconditional validation covariance (6×6)
        company_covariances: Dict mapping company_id to 6×6 correlation matrix
        company_sizes_for_diagnostic: Dict mapping company_id to size
        dawid_skene_validation_data: Validation data for computing total size
        
    Returns:
        DataFrame with columns: company_id, source, company_size, and 15 correlation columns
    """
    logger.info("Building parallel coordinates dataframe...")
    
    # Get annotator columns to create correlation labels
    annotator_cols = [col for col in dawid_skene_validation_data.columns 
                     if col not in ['cv_text', 'category']]
    n_annotators = len(annotator_cols)
    
    # Create correlation labels (use full names to avoid collisions)
    corr_labels = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            # Use full annotator names to avoid duplicate column names from truncation
            corr_labels.append(f"{annotator_cols[i]}_{annotator_cols[j]}")
    
    records = []
    
    # Add validation covariance (all data, unconditional)
    corr_validation = flatten_correlation_matrix(validation_covariance)
    records.append({
        'company_id': 'validation',
        'source': 'validation',
        'company_size': len(dawid_skene_validation_data),
        **{corr_labels[i]: corr_validation[i] for i in range(len(corr_labels))}
    })
    
    # Add company covariances
    for company_id, Sigma_company in company_covariances.items():
        corr_company = flatten_correlation_matrix(Sigma_company)
        records.append({
            'company_id': company_id,
            'source': 'company',
            'company_size': company_sizes_for_diagnostic.get(company_id, 0),
            **{corr_labels[i]: corr_company[i] for i in range(len(corr_labels))}
        })
    
    df_parallel = pd.DataFrame(records)
    logger.info(f"Created parallel coordinates dataframe: {len(df_parallel)} rows, {len(df_parallel.columns)} columns")
    
    return df_parallel


@check_output(importance="fail")
def latent_covariance_parallel_coordinates_plot(
    latent_covariance_parallel_coordinates_data: pd.DataFrame,
    latent_covariance_parallel_plot_output_path: str
) -> str:
    """
    Create parallel coordinates plot comparing company vs validation covariances.
    
    Uses matplotlib to create a static PNG plot.
    
    Args:
        latent_covariance_parallel_coordinates_data: DataFrame with flattened correlations
        latent_covariance_parallel_plot_output_path: Output path for PNG plot
        
    Returns:
        Path to saved plot file
    """
    logger.info("Creating parallel coordinates plot (matplotlib)...")
    
    # Get correlation column names (exclude metadata columns)
    corr_labels = [col for col in latent_covariance_parallel_coordinates_data.columns 
                   if col not in ['company_id', 'source', 'company_size']]
    
    # Separate companies and validation
    company_data = latent_covariance_parallel_coordinates_data[
        latent_covariance_parallel_coordinates_data['source'] == 'company'
    ].copy()
    validation_row = latent_covariance_parallel_coordinates_data[
        latent_covariance_parallel_coordinates_data['source'] == 'validation'
    ].iloc[0]
    
    n_dims = len(corr_labels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize correlation values to [0, 1] for each dimension
    company_values = company_data[corr_labels].values
    validation_values = validation_row[corr_labels].values
    
    # Normalize across all data (companies + validation)
    all_values = np.vstack([company_values, validation_values.reshape(1, -1)])
    mins = all_values.min(axis=0)
    maxs = all_values.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    
    company_normalized = (company_values - mins) / ranges
    validation_normalized = (validation_values - mins) / ranges
    
    # Create x positions for each dimension
    x_positions = np.linspace(0, 1, n_dims)
    
    # Plot companies (colored by log size)
    if len(company_data) > 0:
        company_data['log_size'] = np.log10(company_data['company_size'] + 1)
        log_sizes = company_data['log_size'].values
        norm_log_sizes = (log_sizes - log_sizes.min()) / (log_sizes.max() - log_sizes.min() + 1e-10)
        
        # Use colormap
        cmap = plt.cm.get_cmap('viridis')
        
        for i in range(len(company_data)):
            y_values = company_normalized[i]
            color = cmap(norm_log_sizes[i])
            ax.plot(x_positions, y_values, color=color, alpha=0.3, linewidth=0.5)
    
    # Overlay validation (bold black)
    ax.plot(x_positions, validation_normalized, color='black', linewidth=3, 
            label='Validation (all data)', zorder=10)
    
    # Set labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Normalized Correlation', fontsize=10)
    ax.set_title('Latent Correlation Structure: Companies vs Validation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add colorbar for company sizes
    if len(company_data) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=log_sizes.min(), vmax=log_sizes.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Log10(Company Size)', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(latent_covariance_parallel_plot_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved parallel coordinates plot to {output_path}")
    
    return str(output_path)


@save_to.csv(
    path=source('latent_covariance_diagnostic_summary_output_path')
)
@cache()
def latent_covariance_diagnostic_summary(
    validation_covariance: np.ndarray,
    company_covariances: Dict[str, np.ndarray],
    latent_covariance_parallel_coordinates_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute summary statistics comparing company vs validation covariances.
    
    Returns as a DataFrame for CSV export.
    
    Args:
        validation_covariance: Unconditional validation covariance
        company_covariances: Dict of company covariances
        latent_covariance_parallel_coordinates_data: DataFrame with flattened correlations
        
    Returns:
        DataFrame with summary statistics (one row per metric)
    """
    logger.info("Computing diagnostic summary statistics...")
    
    # Get correlation column names
    corr_labels = [col for col in latent_covariance_parallel_coordinates_data.columns 
                   if col not in ['company_id', 'source', 'company_size']]
    
    # Flatten validation matrix
    corr_validation = flatten_correlation_matrix(validation_covariance)
    
    # Get company correlations
    company_data = latent_covariance_parallel_coordinates_data[
        latent_covariance_parallel_coordinates_data['source'] == 'company'
    ]
    
    # Build summary as DataFrame
    summary_rows = []
    
    # Validation statistics
    summary_rows.append({
        'category': 'validation',
        'metric': 'mean_correlation',
        'value': float(corr_validation.mean())
    })
    summary_rows.append({
        'category': 'validation',
        'metric': 'std_correlation',
        'value': float(corr_validation.std())
    })
    summary_rows.append({
        'category': 'validation',
        'metric': 'min_correlation',
        'value': float(corr_validation.min())
    })
    summary_rows.append({
        'category': 'validation',
        'metric': 'max_correlation',
        'value': float(corr_validation.max())
    })
    
    # Company statistics
    if len(company_data) > 0:
        company_corrs = company_data[corr_labels].values
        summary_rows.append({
            'category': 'companies',
            'metric': 'n_companies',
            'value': len(company_data)
        })
        summary_rows.append({
            'category': 'companies',
            'metric': 'mean_correlation',
            'value': float(company_corrs.mean())
        })
        summary_rows.append({
            'category': 'companies',
            'metric': 'std_correlation',
            'value': float(company_corrs.std())
        })
        summary_rows.append({
            'category': 'companies',
            'metric': 'min_correlation',
            'value': float(company_corrs.min())
        })
        summary_rows.append({
            'category': 'companies',
            'metric': 'max_correlation',
            'value': float(company_corrs.max())
        })
        
        # Distance from validation
        # Ensure shapes match: corr_validation should be broadcastable to company_corrs
        if len(corr_validation) != company_corrs.shape[1]:
            logger.warning(f"Mismatch: corr_validation has {len(corr_validation)} elements, but company_corrs has {company_corrs.shape[1]} columns. Using only matching columns.")
            n_match = min(len(corr_validation), company_corrs.shape[1])
            corr_validation_matched = corr_validation[:n_match]
            company_corrs_matched = company_corrs[:, :n_match]
        else:
            corr_validation_matched = corr_validation
            company_corrs_matched = company_corrs
        
        dist_to_validation = np.abs(company_corrs_matched - corr_validation_matched).mean(axis=1)
        summary_rows.append({
            'category': 'distance_analysis',
            'metric': 'mean_distance_to_validation',
            'value': float(dist_to_validation.mean())
        })
        summary_rows.append({
            'category': 'distance_analysis',
            'metric': 'std_distance_to_validation',
            'value': float(dist_to_validation.std())
        })
        summary_rows.append({
            'category': 'distance_analysis',
            'metric': 'min_distance_to_validation',
            'value': float(dist_to_validation.min())
        })
        summary_rows.append({
            'category': 'distance_analysis',
            'metric': 'max_distance_to_validation',
            'value': float(dist_to_validation.max())
        })
    else:
        summary_rows.append({
            'category': 'companies',
            'metric': 'n_companies',
            'value': 0
        })
    
    summary_df = pd.DataFrame(summary_rows)
    logger.info("Diagnostic summary computed")
    return summary_df


@save_to.csv(
    path=source('validation_covariance_matrix_output_path'),
    index=True
)
@cache()
def validation_covariance_matrix(
    validation_covariance: np.ndarray,
    dawid_skene_validation_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Save full validation correlation matrix to CSV.
    
    Args:
        validation_covariance: 6×6 correlation matrix
        dawid_skene_validation_data: Validation data to get annotator names
        
    Returns:
        DataFrame with correlation matrix (annotator names as index/columns)
    """
    # Get annotator column names
    annotator_cols = [col for col in dawid_skene_validation_data.columns 
                     if col not in ['cv_text', 'category']]
    
    # Create DataFrame with annotator names as index and columns
    df = pd.DataFrame(
        validation_covariance,
        index=annotator_cols,
        columns=annotator_cols
    )
    
    logger.info(f"Saved validation correlation matrix: shape {df.shape}")
    return df


@save_to.csv(
    path=source('company_covariance_matrices_output_path')
)
@cache()
def company_covariance_matrices(
    company_covariances: Dict[str, np.ndarray],
    dawid_skene_validation_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Save all company correlation matrices to a single CSV file.
    
    Each row represents one company, with columns for:
    - company_id
    - All correlation values (flattened: row_annotator_col_annotator format)
    
    Args:
        company_covariances: Dict mapping company_id to 6×6 correlation matrix
        dawid_skene_validation_data: Validation data to get annotator names
        
    Returns:
        DataFrame with one row per company and correlation values as columns
    """
    # Get annotator column names
    annotator_cols = [col for col in dawid_skene_validation_data.columns 
                     if col not in ['cv_text', 'category']]
    
    records = []
    for company_id, Sigma in company_covariances.items():
        record = {'company_id': company_id}
        # Add all correlation values (including diagonal and both triangles for completeness)
        for i, annot_i in enumerate(annotator_cols):
            for j, annot_j in enumerate(annotator_cols):
                record[f"{annot_i}_{annot_j}"] = float(Sigma[i, j])
        records.append(record)
    
    df = pd.DataFrame(records)
    logger.info(f"Saved {len(records)} company correlation matrices: {len(df.columns)-1} correlation columns")
    return df
