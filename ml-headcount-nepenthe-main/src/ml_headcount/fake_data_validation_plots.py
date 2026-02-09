"""
Validation plots for fake data evaluation of Dawid-Skene model.

This module contains plotting functions that compare model predictions
against known ground truth from synthetically generated data.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_prevalence_recovery(
    dawid_skene_inference: Dict[str, Any],
    fake_data_ground_truth: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Plot comparison of true vs estimated company prevalences.
    
    Args:
        dawid_skene_inference: Dawid-Skene inference results with posterior samples
        fake_data_ground_truth: Ground truth parameters
        output_dir: Directory to save plots
        
    Returns:
        Path to saved plot
    """
    logger.info("Plotting prevalence recovery...")
    
    # Extract true prevalences from nested structure
    true_prevalences = fake_data_ground_truth['test_data']['company_prevalences']
    G = len(true_prevalences)
    
    # Extract posterior prevalence estimates
    pi_draws = dawid_skene_inference['pi_draws']  # Shape: (n_samples, G)
    
    # Calculate posterior mean and credible intervals for each company
    pi_mean = np.mean(pi_draws, axis=0)
    pi_lower = np.percentile(pi_draws, 5, axis=0)
    pi_upper = np.percentile(pi_draws, 95, axis=0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot estimated vs true
    company_indices = np.arange(G)
    
    # Scatter plot with error bars
    ax.errorbar(
        true_prevalences,
        pi_mean,
        yerr=[pi_mean - pi_lower, pi_upper - pi_mean],
        fmt='o',
        alpha=0.6,
        capsize=3,
        markersize=4,
        label='Estimated (mean ± 90% CI)'
    )
    
    # Add diagonal line (perfect recovery)
    min_val = min(true_prevalences.min(), pi_mean.min())
    max_val = max(true_prevalences.max(), pi_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect recovery')
    
    # Calculate R² and RMSE
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(true_prevalences, pi_mean)
    rmse = np.sqrt(mean_squared_error(true_prevalences, pi_mean))
    
    # Coverage: proportion of true values within 90% CI
    coverage = np.mean((true_prevalences >= pi_lower) & (true_prevalences <= pi_upper))
    
    # Set log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('True Prevalence (log scale)', fontsize=12)
    ax.set_ylabel('Estimated Prevalence (log scale)', fontsize=12)
    ax.set_title(
        f'Company Prevalence Recovery (Log-Log Scale)\nR² = {r2:.3f}, RMSE = {rmse:.3f}, 90% CI Coverage = {coverage:.1%}',
        fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Make it square
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "fake_data_plots" / "validation_prevalence_recovery.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Prevalence recovery plot saved: {output_path}")
    logger.info(f"  R² = {r2:.3f}, RMSE = {rmse:.3f}, Coverage = {coverage:.1%}")
    
    return str(output_path)


def plot_confusion_matrix_recovery(
    dawid_skene_inference: Dict[str, Any],
    fake_data_ground_truth: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Plot comparison of true vs estimated annotator confusion matrices.
    
    Args:
        dawid_skene_inference: Dawid-Skene inference results with posterior samples
        fake_data_ground_truth: Ground truth parameters
        output_dir: Directory to save plots
        
    Returns:
        Path to saved plot
    """
    logger.info("Plotting confusion matrix recovery...")
    
    # Extract true sensitivities and specificities
    true_sensitivities = fake_data_ground_truth['test_data']['sensitivities']
    true_specificities = fake_data_ground_truth['test_data']['specificities']
    
    # Annotator names (in order)
    annotator_names = list(true_sensitivities.keys())
    J = len(annotator_names)
    
    # Extract posterior estimates
    s_draws = dawid_skene_inference['s_draws']  # Shape: (n_samples, J)
    c_draws = dawid_skene_inference['c_draws']  # Shape: (n_samples, J)
    
    # Calculate posterior means and credible intervals
    s_mean = np.mean(s_draws, axis=0)
    s_lower = np.percentile(s_draws, 5, axis=0)
    s_upper = np.percentile(s_draws, 95, axis=0)
    
    c_mean = np.mean(c_draws, axis=0)
    c_lower = np.percentile(c_draws, 5, axis=0)
    c_upper = np.percentile(c_draws, 95, axis=0)
    
    # Create true values arrays
    true_s = np.array([true_sensitivities[name] for name in annotator_names])
    true_c = np.array([true_specificities[name] for name in annotator_names])
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Sensitivity
    ax = axes[0]
    x = np.arange(J)
    width = 0.35
    
    ax.bar(x - width/2, true_s, width, label='True', alpha=0.7, color='steelblue')
    ax.bar(x + width/2, s_mean, width, label='Estimated', alpha=0.7, color='coral')
    ax.errorbar(
        x + width/2,
        s_mean,
        yerr=[s_mean - s_lower, s_upper - s_mean],
        fmt='none',
        ecolor='black',
        capsize=3,
        alpha=0.5
    )
    
    ax.set_xlabel('Annotator', fontsize=12)
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
    ax.set_title('Sensitivity Recovery', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in annotator_names], rotation=0, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Calculate metrics for sensitivity
    from sklearn.metrics import r2_score, mean_squared_error
    r2_s = r2_score(true_s, s_mean)
    rmse_s = np.sqrt(mean_squared_error(true_s, s_mean))
    coverage_s = np.mean((true_s >= s_lower) & (true_s <= s_upper))
    
    ax.text(
        0.02, 0.98,
        f'R² = {r2_s:.3f}\nRMSE = {rmse_s:.3f}\n90% Coverage = {coverage_s:.1%}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    # Plot 2: Specificity
    ax = axes[1]
    
    ax.bar(x - width/2, true_c, width, label='True', alpha=0.7, color='steelblue')
    ax.bar(x + width/2, c_mean, width, label='Estimated', alpha=0.7, color='coral')
    ax.errorbar(
        x + width/2,
        c_mean,
        yerr=[c_mean - c_lower, c_upper - c_mean],
        fmt='none',
        ecolor='black',
        capsize=3,
        alpha=0.5
    )
    
    ax.set_xlabel('Annotator', fontsize=12)
    ax.set_ylabel('Specificity (True Negative Rate)', fontsize=12)
    ax.set_title('Specificity Recovery', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace('_', '\n') for name in annotator_names], rotation=0, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Calculate metrics for specificity
    r2_c = r2_score(true_c, c_mean)
    rmse_c = np.sqrt(mean_squared_error(true_c, c_mean))
    coverage_c = np.mean((true_c >= c_lower) & (true_c <= c_upper))
    
    ax.text(
        0.02, 0.98,
        f'R² = {r2_c:.3f}\nRMSE = {rmse_c:.3f}\n90% Coverage = {coverage_c:.1%}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "fake_data_plots" / "validation_confusion_matrix_recovery.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Confusion matrix recovery plot saved: {output_path}")
    logger.info(f"  Sensitivity: R² = {r2_s:.3f}, RMSE = {rmse_s:.3f}, Coverage = {coverage_s:.1%}")
    logger.info(f"  Specificity: R² = {r2_c:.3f}, RMSE = {rmse_c:.3f}, Coverage = {coverage_c:.1%}")
    
    return str(output_path)


def generate_fake_data_validation_plots(
    dawid_skene_inference: Dict[str, Any],
    fake_data_ground_truth: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Generate all fake data validation plots.
    
    Args:
        dawid_skene_inference: Dawid-Skene inference results
        fake_data_ground_truth: Ground truth parameters
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with plot file paths
    """
    logger.info("Generating fake data validation plots...")
    
    plot_paths = {}
    
    # Plot 1: Prevalence recovery
    prevalence_path = plot_prevalence_recovery(
        dawid_skene_inference,
        fake_data_ground_truth,
        output_dir
    )
    if prevalence_path:
        plot_paths['prevalence_recovery'] = prevalence_path
    
    # Plot 2: Confusion matrix recovery
    confusion_path = plot_confusion_matrix_recovery(
        dawid_skene_inference,
        fake_data_ground_truth,
        output_dir
    )
    if confusion_path:
        plot_paths['confusion_matrix_recovery'] = confusion_path
    
    logger.info(f"Generated {len(plot_paths)} validation plots")
    
    return {
        'status': 'completed',
        'plot_paths': plot_paths,
        'n_plots': len(plot_paths)
    }

