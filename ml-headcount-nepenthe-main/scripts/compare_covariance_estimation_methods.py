#!/usr/bin/env python3
"""
Compare three approaches for estimating latent covariance matrix:
1. Previous +/- tau approach (sign-flipped probits)
2. Bivariate tetrachoric correlations + projection to PD
3. Multivariate EM approach

This script loads validation CVs and computes the covariance matrix
for the full dataset (both true labels together) using all three methods.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import time
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ml_headcount.probit_dawid_skene_inference import (
    estimate_covariance_multivariate_probit_em
)
from ml_headcount.synthetic_data_generation import estimate_tetrachoric_correlation_binary
from ml_headcount.hamilton_processors import validation_cvs_with_paper_filters


def load_validation_data(data_dir: str = "data") -> pd.DataFrame:
    """Load and prepare validation data."""
    print("Loading validation CVs from Excel...")
    
    # Load from Excel
    excel_path = Path(data_dir) / "raw_data_cvs" / "Validation Set" / "2025-08-06_validation_cvs_rated.xlsx"
    if not excel_path.exists():
        # Try alternative path
        excel_path = Path(data_dir) / "validation_cvs.xlsx"
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Validation CVs file not found. Tried: {excel_path}")
    
    print(f"  Loading from: {excel_path}")
    validation_cvs_raw = pd.read_excel(excel_path, sheet_name="Validation full")
    
    # Clean up unnamed columns
    validation_cvs_raw = validation_cvs_raw.drop(
        columns=[col for col in validation_cvs_raw.columns if col.startswith('Unnamed:')]
    )
    
    # Apply paper filters to get 6 annotators
    print("  Applying paper filters...")
    validation_data = validation_cvs_with_paper_filters(validation_cvs_raw)
    
    # Filter to rows with all annotator scores
    annotator_cols = [
        'llm_gemini-2.5-flash', 'llm_sonnet-4', 'llm_gpt-5-mini',
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    validation_data = validation_data.dropna(subset=annotator_cols, how='any')
    
    print(f"  Loaded {len(validation_data)} validation items with all 6 annotators")
    print(f"  True label distribution: {validation_data['category'].value_counts().to_dict()}")
    
    return validation_data


def method1_sign_flipped_probit(annotations: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Method 1: Previous +/- tau approach (sign-flipped probits).
    
    For each annotator j:
    - Compute p_j = P(annotation_j = 1) from data
    - Transform: if annotation = 1, use +Φ^(-1)(p_j); if 0, use -Φ^(-1)(p_j)
    - Compute empirical covariance of transformed values
    """
    start_time = time.time()
    
    n_items, n_annotators = annotations.shape
    
    # Compute marginal probabilities for each annotator
    p_j = np.nanmean(annotations, axis=0)
    p_j = np.clip(p_j, 1e-10, 1.0 - 1e-10)
    
    # Transform to probit space
    probit_values = stats.norm.ppf(p_j)
    
    # Apply sign-flipped transformation
    probit_annotations = np.zeros_like(annotations, dtype=float)
    for j in range(n_annotators):
        probit_annotations[:, j] = np.where(
            annotations[:, j] == 1,
            probit_values[j],      # If Y_j = 1: use +Φ^(-1)(p_j)
            -probit_values[j]      # If Y_j = 0: use -Φ^(-1)(p_j)
        )
    
    # Compute empirical covariance
    Sigma = np.cov(probit_annotations.T)
    
    # Add small regularization to ensure positive definiteness
    Sigma = Sigma + np.eye(n_annotators) * 1e-6
    
    # Convert to correlation matrix
    diag_sqrt = np.sqrt(np.diag(Sigma))
    diag_sqrt = np.maximum(diag_sqrt, 1e-6)
    Sigma_corr = Sigma / np.outer(diag_sqrt, diag_sqrt)
    
    # Ensure positive definiteness via eigendecomposition
    try:
        np.linalg.cholesky(Sigma_corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma_corr)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalize to correlation matrix
        diag_sqrt = np.sqrt(np.diag(Sigma_corr))
        Sigma_corr = Sigma_corr / np.outer(diag_sqrt, diag_sqrt)
    
    elapsed = time.time() - start_time
    return Sigma_corr, elapsed


def method2_tetrachoric_pairwise_no_projection(annotations: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Method 2: Bivariate tetrachoric correlations WITHOUT PD projection.
    
    For each pair of annotators (i, j):
    - Estimate tetrachoric correlation rho_ij
    - Fill correlation matrix
    - Do NOT project to PD (may not be positive definite)
    """
    start_time = time.time()
    
    n_items, n_annotators = annotations.shape
    
    # Initialize correlation matrix
    Sigma = np.eye(n_annotators)
    
    # Compute pairwise tetrachoric correlations
    n_pairs = n_annotators * (n_annotators - 1) // 2
    pair_count = 0
    
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            pair_count += 1
            if pair_count % 5 == 0:
                print(f"    Computing tetrachoric pair {pair_count}/{n_pairs}...")
            
            x = annotations[:, i]
            y = annotations[:, j]
            rho_ij = estimate_tetrachoric_correlation_binary(x, y)
            Sigma[i, j] = rho_ij
            Sigma[j, i] = rho_ij
    
    # Check if PD (but don't project)
    is_pd = True
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        is_pd = False
        print("    WARNING: Tetrachoric matrix is NOT positive definite (no projection applied)")
    
    elapsed = time.time() - start_time
    return Sigma, elapsed, is_pd


def method3_tetrachoric_pairwise_with_projection(annotations: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Method 3: Bivariate tetrachoric correlations + projection to PD.
    
    For each pair of annotators (i, j):
    - Estimate tetrachoric correlation rho_ij
    - Fill correlation matrix
    - Project to nearest positive definite matrix
    """
    start_time = time.time()
    
    n_items, n_annotators = annotations.shape
    
    # Initialize correlation matrix
    Sigma = np.eye(n_annotators)
    
    # Compute pairwise tetrachoric correlations
    n_pairs = n_annotators * (n_annotators - 1) // 2
    pair_count = 0
    
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            pair_count += 1
            if pair_count % 5 == 0:
                print(f"    Computing tetrachoric pair {pair_count}/{n_pairs}...")
            
            x = annotations[:, i]
            y = annotations[:, j]
            rho_ij = estimate_tetrachoric_correlation_binary(x, y)
            Sigma[i, j] = rho_ij
            Sigma[j, i] = rho_ij
    
    # Ensure positive definiteness
    projection_applied = False
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        projection_applied = True
        print("    Tetrachoric matrix not positive definite, projecting to nearest PD...")
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalize to correlation matrix
        diag_sqrt = np.sqrt(np.diag(Sigma))
        Sigma = Sigma / np.outer(diag_sqrt, diag_sqrt)
    
    elapsed = time.time() - start_time
    return Sigma, elapsed, projection_applied


def method4_multivariate_em(annotations: np.ndarray, n_samples: int = 2048) -> Tuple[np.ndarray, float]:
    """
    Method 3: Multivariate EM approach.
    
    Uses EM algorithm to estimate full covariance matrix from multivariate probit model.
    """
    start_time = time.time()
    
    n_items, n_annotators = annotations.shape
    
    # Compute mean vector from marginal probabilities
    mu = np.zeros(n_annotators)
    for j in range(n_annotators):
        p_j = np.nanmean(annotations[:, j])
        p_j = np.clip(p_j, 1e-6, 1.0 - 1e-6)
        mu[j] = stats.norm.ppf(p_j)
    
    # Initialize from tetrachoric for faster convergence
    print("    Initializing from tetrachoric correlations...")
    init_Sigma = np.eye(n_annotators)
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            x = annotations[:, i]
            y = annotations[:, j]
            rho_ij = estimate_tetrachoric_correlation_binary(x, y)
            init_Sigma[i, j] = rho_ij
            init_Sigma[j, i] = rho_ij
    
    # Run EM
    print(f"    Running EM algorithm (n_samples={n_samples})...")
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    Sigma = estimate_covariance_multivariate_probit_em(
        annotations, mu, init_Sigma=init_Sigma, n_samples=n_samples, logger=logger
    )
    
    elapsed = time.time() - start_time
    return Sigma, elapsed


def compare_matrices(Sigma1: np.ndarray, Sigma2: np.ndarray, name1: str, name2: str):
    """Compare two covariance matrices."""
    diff = Sigma1 - Sigma2
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    frobenius_diff = np.linalg.norm(diff, 'fro')
    
    print(f"\n  Comparison: {name1} vs {name2}")
    print(f"    Max absolute difference: {max_diff:.6f}")
    print(f"    Mean absolute difference: {mean_diff:.6f}")
    print(f"    Frobenius norm difference: {frobenius_diff:.6f}")
    
    # Compare eigenvalues
    eigvals1 = np.linalg.eigvalsh(Sigma1)
    eigvals2 = np.linalg.eigvalsh(Sigma2)
    print(f"    Eigenvalue range: {name1} [{eigvals1.min():.4f}, {eigvals1.max():.4f}]")
    print(f"    Eigenvalue range: {name2} [{eigvals2.min():.4f}, {eigvals2.max():.4f}]")


def print_matrix_summary(Sigma: np.ndarray, name: str):
    """Print summary statistics for a covariance matrix."""
    print(f"\n{name}:")
    print(f"  Shape: {Sigma.shape}")
    print(f"  Diagonal (should be 1.0): {np.diag(Sigma)}")
    print(f"  Off-diagonal range: [{Sigma[~np.eye(Sigma.shape[0], dtype=bool)].min():.4f}, "
          f"{Sigma[~np.eye(Sigma.shape[0], dtype=bool)].max():.4f}]")
    print(f"  Eigenvalue range: [{np.linalg.eigvalsh(Sigma).min():.4f}, "
          f"{np.linalg.eigvalsh(Sigma).max():.4f}]")
    print(f"  Is positive definite: {np.all(np.linalg.eigvalsh(Sigma) > 0)}")
    
    # Print correlation matrix (upper triangle)
    print(f"  Correlation matrix (upper triangle):")
    for i in range(Sigma.shape[0]):
        row_str = "    " + " ".join([f"{Sigma[i, j]:6.3f}" for j in range(i+1)])
        print(row_str)


def main():
    """Main comparison script."""
    print("=" * 80)
    print("Covariance Estimation Methods Comparison")
    print("=" * 80)
    
    # Load data
    validation_data = load_validation_data()
    
    # Extract annotations (all data together, not split by true label)
    annotator_cols = [
        'llm_gemini-2.5-flash', 'llm_sonnet-4', 'llm_gpt-5-mini',
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    annotations = validation_data[annotator_cols].values.astype(int)
    
    n_items, n_annotators = annotations.shape
    print(f"\nComputing covariance matrix for {n_items} items, {n_annotators} annotators")
    print(f"Annotator columns: {annotator_cols}")
    
    # Method 1: Sign-flipped probit
    print("\n" + "=" * 80)
    print("Method 1: Sign-Flipped Probit (+/- tau approach)")
    print("=" * 80)
    Sigma1, time1 = method1_sign_flipped_probit(annotations)
    print(f"\nCompleted in {time1:.3f} seconds")
    print_matrix_summary(Sigma1, "Result")
    
    # Method 2: Tetrachoric pairwise (NO projection)
    print("\n" + "=" * 80)
    print("Method 2: Bivariate Tetrachoric Correlations (NO PD Projection)")
    print("=" * 80)
    Sigma2, time2, is_pd2 = method2_tetrachoric_pairwise_no_projection(annotations)
    print(f"\nCompleted in {time2:.3f} seconds")
    print(f"Matrix is positive definite: {is_pd2}")
    print_matrix_summary(Sigma2, "Result")
    
    # Method 3: Tetrachoric pairwise (WITH projection)
    print("\n" + "=" * 80)
    print("Method 3: Bivariate Tetrachoric Correlations + PD Projection")
    print("=" * 80)
    Sigma3, time3, projection_applied3 = method3_tetrachoric_pairwise_with_projection(annotations)
    print(f"\nCompleted in {time3:.3f} seconds")
    if projection_applied3:
        print("PD projection was applied")
    print_matrix_summary(Sigma3, "Result")
    
    # Method 4: Multivariate EM
    print("\n" + "=" * 80)
    print("Method 4: Multivariate EM Algorithm")
    print("=" * 80)
    Sigma4, time4 = method4_multivariate_em(annotations, n_samples=2048)
    print(f"\nCompleted in {time4:.3f} seconds")
    print_matrix_summary(Sigma4, "Result")
    
    # Runtime comparison
    print("\n" + "=" * 80)
    print("Runtime Comparison")
    print("=" * 80)
    print(f"Method 1 (Sign-flipped probit): {time1:.3f} seconds")
    print(f"Method 2 (Tetrachoric, no projection): {time2:.3f} seconds")
    print(f"Method 3 (Tetrachoric, with projection): {time3:.3f} seconds")
    print(f"Method 4 (Multivariate EM): {time4:.3f} seconds")
    print(f"\nSpeedup: Method 2 is {time1/time2:.2f}x {'faster' if time2 < time1 else 'slower'} than Method 1")
    print(f"Speedup: Method 3 is {time1/time3:.2f}x {'faster' if time3 < time1 else 'slower'} than Method 1")
    print(f"Speedup: Method 4 is {time1/time4:.2f}x {'faster' if time4 < time1 else 'slower'} than Method 1")
    print(f"Speedup: Method 3 is {time2/time3:.2f}x {'faster' if time3 < time2 else 'slower'} than Method 2")
    print(f"Speedup: Method 4 is {time2/time4:.2f}x {'faster' if time4 < time2 else 'slower'} than Method 2")
    
    # Results comparison
    print("\n" + "=" * 80)
    print("Results Comparison")
    print("=" * 80)
    compare_matrices(Sigma1, Sigma2, "Method 1", "Method 2 (no projection)")
    compare_matrices(Sigma1, Sigma3, "Method 1", "Method 3 (with projection)")
    compare_matrices(Sigma1, Sigma4, "Method 1", "Method 4 (EM)")
    compare_matrices(Sigma2, Sigma3, "Method 2 (no projection)", "Method 3 (with projection)")
    compare_matrices(Sigma2, Sigma4, "Method 2 (no projection)", "Method 4 (EM)")
    compare_matrices(Sigma3, Sigma4, "Method 3 (with projection)", "Method 4 (EM)")
    
    # Detailed correlation comparison
    print("\n" + "=" * 80)
    print("Detailed Correlation Comparison")
    print("=" * 80)
    print("\nPairwise correlations (upper triangle):")
    print("  (i, j) | Method 1 | Method 2 | Method 3 | Method 4 | Diff (2-3) | Diff (3-4)")
    print("  " + "-" * 85)
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            rho1 = Sigma1[i, j]
            rho2 = Sigma2[i, j]
            rho3 = Sigma3[i, j]
            rho4 = Sigma4[i, j]
            diff23 = rho2 - rho3
            diff34 = rho3 - rho4
            print(f"  ({i}, {j}) | {rho1:7.4f} | {rho2:7.4f} | {rho3:7.4f} | {rho4:7.4f} | "
                  f"{diff23:10.4f} | {diff34:10.4f}")
    
    # Impact of PD projection
    if not is_pd2:
        print("\n" + "=" * 80)
        print("Impact of PD Projection")
        print("=" * 80)
        print("Method 2 (no projection) is NOT positive definite")
        print("Comparing Method 2 vs Method 3 shows the effect of PD projection:")
        diff_projection = Sigma2 - Sigma3
        print(f"  Max absolute change from projection: {np.max(np.abs(diff_projection)):.6f}")
        print(f"  Mean absolute change from projection: {np.mean(np.abs(diff_projection)):.6f}")
        print(f"  Frobenius norm change: {np.linalg.norm(diff_projection, 'fro'):.6f}")
        
        # Show eigenvalue comparison
        eigvals2 = np.linalg.eigvalsh(Sigma2)
        eigvals3 = np.linalg.eigvalsh(Sigma3)
        print(f"\n  Eigenvalues before projection: min={eigvals2.min():.6f}, max={eigvals2.max():.6f}")
        print(f"  Eigenvalues after projection:  min={eigvals3.min():.6f}, max={eigvals3.max():.6f}")
        print(f"  Negative eigenvalues before projection: {(eigvals2 < 0).sum()}")
        print(f"  Negative eigenvalues after projection:  {(eigvals3 < 0).sum()}")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Method 1 (Sign-flipped probit): {time1:.3f}s - Fastest but heuristic")
    print(f"Method 2 (Tetrachoric, no projection): {time2:.3f}s - Fast, may not be PD")
    print(f"Method 3 (Tetrachoric, with projection): {time3:.3f}s - Fast, guaranteed PD")
    print(f"Method 4 (Multivariate EM): {time4:.3f}s - Slowest but most principled")
    print(f"\nKey finding: PD projection impact (Method 2 vs 3):")
    if not is_pd2:
        print(f"  - Method 2 is NOT positive definite (cannot be used for Cholesky/sampling)")
        print(f"  - Method 3 fixes this with PD projection")
        print(f"  - Mean change from projection: {np.mean(np.abs(Sigma2 - Sigma3)):.6f}")
    else:
        print(f"  - Method 2 is already positive definite (no projection needed)")
    print(f"\nRecommendation: Use Method 3 (Tetrachoric + PD projection) for production")
    print(f"  - Fast enough for bootstrap iterations")
    print(f"  - Results very similar to Method 4 (EM)")
    print(f"  - Guaranteed to be positive definite")


if __name__ == "__main__":
    main()
