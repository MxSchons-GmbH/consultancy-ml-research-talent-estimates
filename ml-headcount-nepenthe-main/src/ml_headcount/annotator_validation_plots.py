"""
Annotator Validation Plots

This module contains Hamilton functions for generating annotator validation plots
including sensitivity vs specificity, bias analysis, and empirical correlations.
"""

from hamilton.function_modifiers import source, save_to, config, check_output, cache
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@cache(behavior="recompute")
def annotator_metrics(
    dawid_skene_validation_data: pd.DataFrame,
    correlated_probit_bootstrap_prior_alpha: float,
    correlated_probit_bootstrap_prior_beta: float,
    correlated_probit_bootstrap_n_samples: int,
    correlated_probit_bootstrap_n_machines: int,
    plot_ci_width: float,
) -> Dict[str, Dict[str, float]]:
    """Compute performance metrics for each annotator."""
    logger.info("Computing annotator metrics...")
    
    # Extract annotator columns
    annotator_cols = [col for col in dawid_skene_validation_data.columns 
                     if col not in ['cv_text', 'category']]
    
    true_labels = dawid_skene_validation_data['category'].values
    annotations = dawid_skene_validation_data[annotator_cols].values
    
    annotator_metrics_dict = {}
    
    for j, col in enumerate(annotator_cols):
        annotator_pred = annotations[:, j]
        
        # Confusion matrix
        true_pos = np.sum((true_labels == 1) & (annotator_pred == 1))
        false_pos = np.sum((true_labels == 0) & (annotator_pred == 1))
        true_neg = np.sum((true_labels == 0) & (annotator_pred == 0))
        false_neg = np.sum((true_labels == 1) & (annotator_pred == 0))
        
        # Calculate metrics
        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
        accuracy = (true_pos + true_neg) / len(true_labels)
        
        annotator_metrics_dict[col] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'n_positive': true_pos + false_neg,
            'n_negative': true_neg + false_pos
        }
    
    # Add probit point-estimate metrics with fixed prior and 0.5 threshold
    try:
        from .probit_dawid_skene_inference import (
            create_simple_point_estimate,
            predict_new_item_probit_point_estimate,
            collect_unique_patterns,
            precompute_observed_patterns,
        )

        class_prior = 0.4
        n_pattern_samples = 4096

        point_estimates = create_simple_point_estimate(dawid_skene_validation_data)

        unique_patterns = collect_unique_patterns(annotations)
        pattern_probs = precompute_observed_patterns(
            point_estimates['theta'],
            point_estimates['chol_0'],
            point_estimates['chol_1'],
            unique_patterns,
            n_samples=n_pattern_samples,
        )

        probit_predictions = np.zeros(len(dawid_skene_validation_data))
        for i in range(len(dawid_skene_validation_data)):
            y_new = annotations[i]
            prob_positive = predict_new_item_probit_point_estimate(
                y_new,
                point_estimates['theta'],
                point_estimates['chol_0'],
                point_estimates['chol_1'],
                class_prior=class_prior,
                pattern_probs=pattern_probs,
            )
            probit_predictions[i] = prob_positive

        probit_binary = (probit_predictions >= 0.5).astype(int)

        true_pos = np.sum((true_labels == 1) & (probit_binary == 1))
        false_pos = np.sum((true_labels == 0) & (probit_binary == 1))
        true_neg = np.sum((true_labels == 0) & (probit_binary == 0))
        false_neg = np.sum((true_labels == 1) & (probit_binary == 0))

        probit_sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        probit_specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
        probit_accuracy = (true_pos + true_neg) / len(true_labels)

        annotator_metrics_dict['correlated_probit'] = {
            'sensitivity': probit_sensitivity,
            'specificity': probit_specificity,
            'accuracy': probit_accuracy,
            'n_positive': true_pos + false_neg,
            'n_negative': true_neg + false_pos
        }

        logger.info("Added correlated probit point-estimate metrics")
    except Exception as e:
        logger.warning(f"Could not compute probit metrics: {e}")
    
    logger.info(f"Computed metrics for {len(annotator_metrics_dict)} annotators")
    return annotator_metrics_dict


@cache(behavior="recompute")
@save_to.csv(path=source("annotator_metrics_output_path"))
def annotator_metrics_table(
    annotator_metrics: Dict[str, Dict[str, float]],
    dawid_skene_validation_data: pd.DataFrame
) -> pd.DataFrame:
    """Convert annotator metrics dictionary to DataFrame with bias included."""
    logger.info("Converting annotator metrics to DataFrame...")
    
    # Calculate true prevalence for bias calculation
    true_prevalence = dawid_skene_validation_data['category'].mean()
    
    # Build list of records
    records = []
    for annotator, metrics in annotator_metrics.items():
        # Calculate bias (positive rate - true prevalence) unless precomputed
        if 'bias' in metrics:
            bias = metrics['bias']
        else:
            positive_rate = (metrics['n_positive'] * metrics['sensitivity'] + 
                            metrics['n_negative'] * (1 - metrics['specificity'])) / (metrics['n_positive'] + metrics['n_negative'])
            bias = positive_rate - true_prevalence
        
        records.append({
            'annotator': annotator,
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'accuracy': metrics['accuracy'],
            'bias': bias,
            'n_positive': int(metrics['n_positive']),
            'n_negative': int(metrics['n_negative']),
            'n_total': int(metrics['n_positive'] + metrics['n_negative'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by annotator name (put correlated_probit last)
    df['sort_key'] = df['annotator'].apply(lambda x: 1 if x == 'correlated_probit' else 0)
    df = df.sort_values(['sort_key', 'annotator']).drop(columns=['sort_key']).reset_index(drop=True)
    
    logger.info(f"Created annotator metrics table with {len(df)} annotators")
    return df


@cache(behavior="recompute")
@check_output(importance="fail")
def annotator_performance_sens_vs_spec(
    dawid_skene_validation_data: pd.DataFrame,
    annotator_metrics: Dict[str, Dict[str, float]],
    annotator_performance_sens_vs_spec_output_path: str
) -> str:
    """Create sensitivity vs specificity scatter plot."""
    logger.info("Creating annotator performance sens vs spec plot...")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot each annotator
    for col, metrics in annotator_metrics.items():
        if col == 'correlated_probit':
            ax.scatter(metrics['specificity'], metrics['sensitivity'], 
                      s=300, label=col, alpha=0.9, edgecolors='red', linewidth=1,
                      marker='*', color='gold', zorder=5)
        else:
            ax.scatter(metrics['specificity'], metrics['sensitivity'], 
                      s=150, label=col, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add diagonal line for random performance
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.5, linewidth=2, label='Random Classifier')
    
    # Styling
    ax.set_xlabel('Specificity (True Negative Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Annotator Performance: Sensitivity vs Specificity', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add performance zones
    ax.axhspan(0.8, 1, alpha=0.1, color='green', label='High Sensitivity Zone')
    ax.axvspan(0.8, 1, alpha=0.1, color='blue', label='High Specificity Zone')
    
    plt.tight_layout()
    output_path = Path(annotator_performance_sens_vs_spec_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved annotator performance plot to {output_path}")
    return str(output_path)


@cache(behavior="recompute")
@check_output(importance="fail")
def bias_vs_accuracy(
    dawid_skene_validation_data: pd.DataFrame,
    annotator_metrics: Dict[str, Dict[str, float]],
    bias_vs_accuracy_output_path: str
) -> str:
    """Create bias vs accuracy scatter plot."""
    logger.info("Creating bias vs accuracy plot...")
    
    # Calculate bias for each annotator
    true_prevalence = dawid_skene_validation_data['category'].mean()
    
    biases = []
    accuracies = []
    annotators = []
    
    for col, metrics in annotator_metrics.items():
        if 'bias' in metrics:
            bias = metrics['bias']
        else:
            positive_rate = (metrics['n_positive'] * metrics['sensitivity'] + 
                            metrics['n_negative'] * (1 - metrics['specificity'])) / (metrics['n_positive'] + metrics['n_negative'])
            bias = positive_rate - true_prevalence
        biases.append(bias)
        accuracies.append(metrics['accuracy'])
        annotators.append(col)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot each annotator
    for i, (col, bias, acc) in enumerate(zip(annotators, biases, accuracies)):
        if col == 'correlated_probit':
            ax.scatter(bias, acc, s=300, label=col, alpha=0.9, edgecolors='red', linewidth=1,
                      marker='*', color='gold', zorder=5)
        else:
            ax.scatter(bias, acc, s=150, label=col, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add reference lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7, label='No Bias')
    ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='Overestimation Threshold')
    ax.axvline(x=-0.05, color='blue', linestyle='--', alpha=0.5, label='Underestimation Threshold')
    
    # Add performance zones
    ax.axhspan(0.8, 1, alpha=0.1, color='green', label='High Accuracy Zone')
    
    # Styling
    ax.set_xlabel('Bias (Positive Rate - True Prevalence)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Annotator Performance: Bias vs Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add correlation info
    correlation = np.corrcoef(biases, accuracies)[0, 1]
    ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.tight_layout()
    output_path = Path(bias_vs_accuracy_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved bias vs accuracy plot to {output_path}")
    return str(output_path)


@cache(behavior="recompute")
@check_output(importance="fail")
def annotator_bias_analysis(
    dawid_skene_validation_data: pd.DataFrame,
    annotator_metrics: Dict[str, Dict[str, float]],
    annotator_bias_analysis_output_path: str
) -> str:
    """Create annotator bias analysis bar chart."""
    logger.info("Creating annotator bias analysis plot...")
    
    # Calculate bias for each annotator
    true_prevalence = dawid_skene_validation_data['category'].mean()
    
    annotators = []
    biases = []
    
    for col, metrics in annotator_metrics.items():
        if 'bias' in metrics:
            bias = metrics['bias']
        else:
            positive_rate = (metrics['n_positive'] * metrics['sensitivity'] + 
                            metrics['n_negative'] * (1 - metrics['specificity'])) / (metrics['n_positive'] + metrics['n_negative'])
            bias = positive_rate - true_prevalence
        biases.append(bias)
        annotators.append(col)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Color code by bias direction, with special treatment for correlated probit
    colors = []
    for i, (col, b) in enumerate(zip(annotators, biases)):
        if col == 'correlated_probit':
            colors.append('gold')
    for i, (col, b) in enumerate(zip(annotators, biases)):
        if col == 'correlated_probit':
            colors.append('gold')
        elif b > 0.05:
            colors.append('red')
        elif b < -0.05:
            colors.append('blue')
        else:
            colors.append('green')
    
    bars = ax.bar(annotators, biases, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Highlight correlated probit bar
    for i, (bar, col) in enumerate(zip(bars, annotators)):
        if col == 'correlated_probit':
            bar.set_edgecolor('red')
            bar.set_linewidth(1)
    
    # Add bias values on bars
    for bar, bias in zip(bars, biases):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.005),
                f'{bias:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Overestimation Threshold')
    ax.axhline(y=-0.05, color='blue', linestyle='--', alpha=0.5, label='Underestimation Threshold')
    
    # Styling
    ax.set_ylabel('Bias (Positive Rate - True Prevalence)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Annotator', fontsize=12, fontweight='bold')
    ax.set_title('Annotator Bias: Systematic Over/Under-estimation', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Add summary text
    overestimators = sum(1 for b in biases if b > 0.05)
    underestimators = sum(1 for b in biases if b < -0.05)
    balanced = sum(1 for b in biases if abs(b) <= 0.05)
    
    ax.text(0.02, 0.98, f'Overestimators: {overestimators}\nUnderestimators: {underestimators}\nBalanced: {balanced}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    output_path = Path(annotator_bias_analysis_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved annotator bias analysis plot to {output_path}")
    return str(output_path)


@cache(behavior="recompute")
@check_output(importance="fail")
def empirical_correlations_by_class(
    dawid_skene_validation_data: pd.DataFrame,
    empirical_correlations_by_class_output_path: str
) -> str:
    """Create empirical correlations heatmap by true label class."""
    logger.info("Creating empirical correlations by class plot...")
    
    # Extract annotator columns
    annotator_cols = [col for col in dawid_skene_validation_data.columns 
                     if col not in ['cv_text', 'category']]
    n_annotators = len(annotator_cols)
    
    true_labels = dawid_skene_validation_data['category'].values
    annotations = dawid_skene_validation_data[annotator_cols].values
    
    # Compute empirical correlations for each true label class
    empirical_corr_0 = np.eye(n_annotators)
    empirical_corr_1 = np.eye(n_annotators)
    
    # For true_label = 0
    mask_0 = (true_labels == 0)
    if mask_0.sum() > 1:
        annotations_0 = annotations[mask_0]
        empirical_corr_0 = np.corrcoef(annotations_0.T)
        empirical_corr_0 = np.nan_to_num(empirical_corr_0, nan=0.0)
    
    # For true_label = 1
    mask_1 = (true_labels == 1)
    if mask_1.sum() > 1:
        annotations_1 = annotations[mask_1]
        empirical_corr_1 = np.corrcoef(annotations_1.T)
        empirical_corr_1 = np.nan_to_num(empirical_corr_1, nan=0.0)
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot correlations for true_label = 0
    im1 = axes[0].imshow(empirical_corr_0, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Empirical Correlations (True Label = 0)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(n_annotators))
    axes[0].set_yticks(range(n_annotators))
    axes[0].set_xticklabels(annotator_cols, rotation=45, ha='right')
    axes[0].set_yticklabels(annotator_cols)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Correlation Coefficient', fontsize=10)
    
    # Add correlation values as text
    for i in range(n_annotators):
        for j in range(n_annotators):
            text = axes[0].text(j, i, f'{empirical_corr_0[i, j]:.2f}',
                              ha="center", va="center", color="white" if abs(empirical_corr_0[i, j]) > 0.5 else "black",
                              fontsize=8, fontweight='bold')
    
    # Plot correlations for true_label = 1
    im2 = axes[1].imshow(empirical_corr_1, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Empirical Correlations (True Label = 1)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(n_annotators))
    axes[1].set_yticks(range(n_annotators))
    axes[1].set_xticklabels(annotator_cols, rotation=45, ha='right')
    axes[1].set_yticklabels(annotator_cols)
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Correlation Coefficient', fontsize=10)
    
    # Add correlation values as text
    for i in range(n_annotators):
        for j in range(n_annotators):
            text = axes[1].text(j, i, f'{empirical_corr_1[i, j]:.2f}',
                              ha="center", va="center", color="white" if abs(empirical_corr_1[i, j]) > 0.5 else "black",
                              fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(empirical_correlations_by_class_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved empirical correlations plot to {output_path}")
    return str(output_path)
