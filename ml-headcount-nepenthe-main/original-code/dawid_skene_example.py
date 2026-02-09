"""
Dawid-Skene Model Implementation with Population Extrapolation

This script implements a marginalized Dawid-Skene model using PyMC v5 for
crowdsourcing annotation tasks with group-specific prevalence estimation
and population extrapolation capabilities.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, Union, List
from traditional_dawid_skene import traditional_dawid_skene


def simulate_and_split_data(n_total: int = 500, train_fraction: float = 0.2, n_annotators: int = 5, 
                          n_groups: int = 3, balanced_groups: bool = True, 
                          seed: int = 123) -> Dict:
    """
    Simulate crowdsourcing annotation data and split into training and test sets.
    
    Parameters:
    -----------
    n_total : int
        Total number of items to simulate
    train_fraction : float
        Fraction of data to use for training (with ground truth)
    n_annotators : int  
        Number of annotators
    n_groups : int
        Number of groups
    balanced_groups : bool
        Whether to ensure balanced representation across groups
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing train/test datasets and true parameters
    """
    np.random.seed(seed)
    
    # True prevalence per group
    true_pi = [0.2, 0.5, 0.8]
    if n_groups != 3:
        # Generate prevalences if not using default
        true_pi = np.linspace(0.2, 0.8, n_groups).tolist()
    
    # Annotator sensitivities and specificities
    default_s = [0.9, 0.7, 0.8, 0.6, 0.85]  # sensitivity
    default_c = [0.9, 0.8, 0.85, 0.7, 0.9]  # specificity
    
    # Use only as many as we need, or generate new ones if more are needed
    s_true = default_s[:n_annotators]
    c_true = default_c[:n_annotators]
    
    if n_annotators > len(default_s):
        for j in range(len(default_s), n_annotators):
            s_true.append(np.random.uniform(0.6, 0.9))  # Random sensitivities
            c_true.append(np.random.uniform(0.6, 0.9))  # Random specificities
    
    # Create group assignments
    if balanced_groups:
        # Create balanced groups
        items_per_group = n_total // n_groups
        remainder = n_total % n_groups
        
        all_groups = []
        for g in range(n_groups):
            count = items_per_group + (1 if g < remainder else 0)
            all_groups.extend([g] * count)
        
        # Shuffle the group assignments
        np.random.shuffle(all_groups)
        all_groups = np.array(all_groups)
    else:
        # Create random group assignments
        all_groups = np.random.choice(n_groups, size=n_total)
    
    # Generate true labels
    z_true = np.array([np.random.binomial(1, true_pi[g]) for g in all_groups])
    
    # Generate all annotations
    data = {
        'item_id': np.arange(n_total),
        'group': all_groups,
        'true_label': z_true
    }
    
    for j in range(n_annotators):
        annotations = []
        for i in range(n_total):
            p = s_true[j] if z_true[i] == 1 else 1 - c_true[j]
            y = np.random.binomial(1, p)
            annotations.append(y)
        data[f'annotator_{j}'] = annotations
    
    # Create full dataset
    full_df = pd.DataFrame(data)
    
    # Split into train and test
    n_train = int(n_total * train_fraction)
    
    # Stratify by group to maintain group distribution
    train_indices = []
    for g in range(n_groups):
        group_indices = np.where(all_groups == g)[0]
        n_group_train = int(len(group_indices) * train_fraction)
        train_indices.extend(np.random.choice(group_indices, size=n_group_train, replace=False))
    
    # Create train/test masks
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[train_indices] = True
    test_mask = ~train_mask
    
    # Create train and test datasets
    train_df = full_df[train_mask].reset_index(drop=True)
    test_df_with_labels = full_df[test_mask].reset_index(drop=True)
    
    # Create a test set without labels for inference
    test_df = test_df_with_labels.drop(columns=['true_label'])
    
    # Calculate true counts
    train_group_counts = []
    test_group_counts = []
    
    for g in range(n_groups):
        train_group_mask = train_df['group'] == g
        test_group_mask = test_df['group'] == g
        
        train_positive = np.sum(train_df[train_group_mask]['true_label'] == 1)
        train_total = np.sum(train_group_mask)
        
        test_positive = np.sum(test_df_with_labels[test_group_mask]['true_label'] == 1)
        test_total = np.sum(test_group_mask)
        
        train_group_counts.append({
            'group': g, 
            'total': train_total,
            'positive': train_positive,
            'observed_prevalence': train_positive / train_total if train_total > 0 else 0
        })
        
        test_group_counts.append({
            'group': g, 
            'total': test_total,
            'positive': test_positive,
            'observed_prevalence': test_positive / test_total if test_total > 0 else 0
        })
    
    # Prepare and return results
    results = {
        'train_df': train_df,                       # Training data with ground truth
        'test_df': test_df,                         # Test data without ground truth
        'test_df_with_labels': test_df_with_labels,  # Test data with ground truth for evaluation
        'true_params': {
            'pi': true_pi,                           # True prevalence by group
            's': s_true,                             # True sensitivity by annotator
            'c': c_true,                             # True specificity by annotator
        },
        'true_counts': {
            'train': pd.DataFrame(train_group_counts),  # True group counts in training data
            'test': pd.DataFrame(test_group_counts)     # True group counts in test data
        },
        'split_info': {
            'n_total': n_total,
            'n_train': n_train,
            'n_test': n_total - n_train,
            'train_fraction': train_fraction
        }
    }
    
    return results


def ds_marginalized_population(
    df: pd.DataFrame, J=None, G=None, N_group=None,
    draws=1000, tune=2000, random_seed=42
) -> Dict:
    """
    Fit marginalized Dawid-Skene model with population extrapolation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Wide format dataframe with item_id, group, and annotator columns
    J : int, optional
        Number of annotators (inferred if None)
    G : int, optional
        Number of groups (inferred if None)
    N_group : list, optional
        Population sizes for each group
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing model results, summaries, and posterior draws
    """
    # Get annotator columns
    annotator_cols = [col for col in df.columns if col.startswith('annotator_')]
    group_of_item = df['group'].values
    n_items = len(df)

    if J is None:
        J = len(annotator_cols)
    if G is None:
        G = len(np.unique(group_of_item))

    # Create mask for non-missing annotations
    annotation_mask = ~df[annotator_cols].isna()
    y_obs = df[annotator_cols].values
    
    with pm.Model() as model:
        mu_pi = pm.Normal("mu_pi", 0, 2)
        sigma_pi = pm.HalfNormal("sigma_pi", 2)
        logit_pi = pm.Normal("logit_pi", mu=mu_pi, sigma=sigma_pi, shape=G)
        pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi))

        s = pm.Beta("s", 8, 2, shape=J)
        c = pm.Beta("c", 8, 2, shape=J)

        # Calculate probabilities for each item-annotator pair
        pi_per_item = pi[group_of_item]
        
        # Create probability matrix: n_items x J
        p_y1 = pi_per_item[:, None] * s[None, :] + (1 - pi_per_item[:, None]) * (1 - c[None, :])
        
        # Only use non-missing annotations
        y_obs_masked = pm.math.where(annotation_mask, y_obs, 0)
        pm.Bernoulli("y", p=p_y1, observed=y_obs_masked)

        idata = pm.sample(draws=draws, tune=tune, target_accept=0.9,
                          random_seed=random_seed)

    # Posterior draws for pi_g
    pi_draws = idata.posterior["pi"].stack(sample=("chain","draw")).values.T

    # Sample counts for observed items
    G_counts = np.zeros((pi_draws.shape[0], G), dtype=int)
    for g in range(G):
        n_g = np.sum(group_of_item == g)
        G_counts[:, g] = np.random.binomial(n=n_g, p=pi_draws[:, g])

    # Population counts
    pop_counts = None
    if N_group is not None:
        pop_counts = np.zeros((pi_draws.shape[0], G), dtype=int)
        for g in range(G):
            pop_counts[:, g] = np.random.binomial(N_group[g], p=pi_draws[:, g])

    # Summaries
    group_summary = []
    for g in range(G):
        arr = G_counts[:, g]
        group_summary.append({
            "group": g,
            "mean_sample_count": arr.mean(),
            "ci_2.5": np.percentile(arr, 2.5),
            "ci_97.5": np.percentile(arr, 97.5)
        })

    pop_summary = None
    if pop_counts is not None:
        pop_summary = []
        for g in range(G):
            arr = pop_counts[:, g]
            pop_summary.append({
                "group": g,
                "mean_pop_count": arr.mean(),
                "ci_2.5": np.percentile(arr, 2.5),
                "ci_97.5": np.percentile(arr, 97.5)
            })

    return {
        "idata": idata,
        "group_sample_counts_draws": G_counts,
        "group_summary": pd.DataFrame(group_summary),
        "pop_summary": pd.DataFrame(pop_summary) if pop_counts is not None else None
    }


def create_sample_count_plots(result: Dict, G: int, save_path: str = 'plots/') -> None:
    """Create improved sample count visualization plots."""
    plt.figure(figsize=(12, 8))
    
    # Create subplots for better comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Posterior Distribution of Sample Counts per Group', fontsize=16)

    for g in range(G):
        counts = result['group_sample_counts_draws'][:, g]
        
        # Create histogram with proper binning
        bins = np.arange(0, counts.max() + 2) - 0.5
        axes[g].hist(counts, bins=bins, alpha=0.7, color=f'C{g}', edgecolor='black', linewidth=0.5)
        axes[g].set_title(f'Group {g}\nMean: {counts.mean():.2f}, Std: {counts.std():.2f}')
        axes[g].set_xlabel('Number of True Items')
        axes[g].set_ylabel('Frequency')
        axes[g].grid(True, alpha=0.3)
        
        # Add summary statistics
        mean_val = counts.mean()
        ci_low = np.percentile(counts, 2.5)
        ci_high = np.percentile(counts, 97.5)
        axes[g].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
        axes[g].axvline(ci_low, color='orange', linestyle=':', alpha=0.8, label=f'95% CI: [{ci_low:.0f}, {ci_high:.0f}]')
        axes[g].axvline(ci_high, color='orange', linestyle=':', alpha=0.8)
        axes[g].legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}sample_counts_posterior_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved sample_counts_posterior_improved.png")


def create_population_count_plots(result: Dict, groups: np.ndarray, N_group: list, 
                                G: int, save_path: str = 'plots/') -> None:
    """Create improved population count visualization plots."""
    # Get the actual population count draws (not just summary)
    pop_counts = result['group_sample_counts_draws']  # This contains the actual draws
    # Scale up to population sizes
    pop_counts_scaled = np.zeros_like(pop_counts)
    for g in range(G):
        n_g = np.sum(groups == g)  # sample size for group g
        N_g = N_group[g]  # population size for group g
        # Scale the sample counts to population counts
        pop_counts_scaled[:, g] = (pop_counts[:, g] / n_g) * N_g

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Posterior Distribution of Population Counts per Group', fontsize=16)

    for g in range(G):
        counts = pop_counts_scaled[:, g]
        
        # Create histogram with proper binning
        bins = np.linspace(0, counts.max() + 10, 30)
        axes[g].hist(counts, bins=bins, alpha=0.7, color=f'C{g}', edgecolor='black', linewidth=0.5)
        axes[g].set_title(f'Group {g}\nMean: {counts.mean():.1f}, Std: {counts.std():.1f}')
        axes[g].set_xlabel('Number of True Items in Population')
        axes[g].set_ylabel('Frequency')
        axes[g].grid(True, alpha=0.3)
        
        # Add summary statistics
        mean_val = counts.mean()
        ci_low = np.percentile(counts, 2.5)
        ci_high = np.percentile(counts, 97.5)
        axes[g].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
        axes[g].axvline(ci_low, color='orange', linestyle=':', alpha=0.8, label=f'95% CI: [{ci_low:.0f}, {ci_high:.0f}]')
        axes[g].axvline(ci_high, color='orange', linestyle=':', alpha=0.8)
        axes[g].legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}population_counts_posterior_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved population_counts_posterior_improved.png")


def create_comparison_plots(result: Dict, groups: np.ndarray, N_group: list, 
                          G: int, save_path: str = 'plots/') -> None:
    """Create combined comparison plots for sample vs population counts."""
    # Scale up to population sizes
    pop_counts = result['group_sample_counts_draws']
    pop_counts_scaled = np.zeros_like(pop_counts)
    for g in range(G):
        n_g = np.sum(groups == g)
        N_g = N_group[g]
        pop_counts_scaled[:, g] = (pop_counts[:, g] / n_g) * N_g

    plt.figure(figsize=(14, 6))

    # Sample counts - normalized for comparison
    plt.subplot(1, 2, 1)
    for g in range(G):
        counts = result['group_sample_counts_draws'][:, g]
        # Normalize to density for better comparison
        plt.hist(counts, bins=np.arange(0, counts.max() + 2) - 0.5, 
                 alpha=0.6, label=f'Group {g}', density=True, histtype='stepfilled')
    plt.xlabel('Number of True Items (Sample)')
    plt.ylabel('Density')
    plt.title('Sample Counts (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Population counts - normalized for comparison  
    plt.subplot(1, 2, 2)
    for g in range(G):
        counts = pop_counts_scaled[:, g]
        bins = np.linspace(0, counts.max() + 10, 30)
        plt.hist(counts, bins=bins, alpha=0.6, label=f'Group {g}', density=True, histtype='stepfilled')
    plt.xlabel('Number of True Items (Population)')
    plt.ylabel('Density')
    plt.title('Population Counts (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}counts_comparison_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved counts_comparison_improved.png")


def run_model_diagnostics(result: Dict) -> None:
    """Run comprehensive model diagnostics using ArviZ."""
    print("\n" + "="*60)
    print("ARVIZ EXPLORATION AND DIAGNOSTICS")
    print("="*60)

    # Basic summary statistics
    print("\n1. POSTERIOR SUMMARY STATISTICS")
    print("-" * 40)
    print(az.summary(result['idata']))

    # Model diagnostics
    print("\n2. MODEL DIAGNOSTICS")
    print("-" * 40)

    # Check for divergences
    divergences = result['idata'].sample_stats.divergences
    divergence_count = divergences.sum().values
    divergence_rate = divergence_count / divergences.size
    print(f"Number of divergences: {divergence_count}")
    print(f"Divergence rate: {divergence_rate:.3f}")

    # Get diagnostics from summary table
    summary_df = az.summary(result['idata'])
    max_rhat = summary_df['r_hat'].max()
    min_ess = summary_df['ess_bulk'].min()

    print(f"Max R-hat: {max_rhat:.3f}")
    print(f"Min effective sample size: {min_ess:.0f}")

    # Check if diagnostics are good
    print(f"R-hat < 1.01: {'✓' if max_rhat < 1.01 else '✗'}")
    print(f"Min ESS > 100: {'✓' if min_ess > 100 else '✗'}")
    print(f"Divergence rate < 0.01: {'✓' if divergence_rate < 0.01 else '✗'}")


def create_trace_plots(result: Dict, save_path: str = 'plots/') -> None:
    """Create trace plots for model parameters."""
    print("\n3. CREATING TRACE PLOTS...")

    # Plot traces for pi (prevalence by group)
    az.plot_trace(result['idata'], var_names=['pi'], figsize=(15, 10))
    plt.suptitle('Trace Plots for Prevalence by Group (π)')
    plt.tight_layout()
    plt.savefig(f'{save_path}trace_plots_pi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved trace_plots_pi.png")

    # Plot traces for annotator parameters
    az.plot_trace(result['idata'], var_names=['s', 'c'], figsize=(15, 8))
    plt.suptitle('Trace Plots for Annotator Parameters')
    plt.tight_layout()
    plt.savefig(f'{save_path}trace_plots_annotators.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved trace_plots_annotators.png")

    # Plot hyperparameters
    az.plot_trace(result['idata'], var_names=['mu_pi', 'sigma_pi'], figsize=(12, 4))
    plt.suptitle('Trace Plots for Hyperparameters')
    plt.tight_layout()
    plt.savefig(f'{save_path}trace_plots_hyperparams.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved trace_plots_hyperparams.png")


def create_posterior_plots(result: Dict, save_path: str = 'plots/') -> None:
    """Create posterior distribution plots."""
    print("\n4. CREATING POSTERIOR DISTRIBUTION PLOTS...")

    # Prevalence by group
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_posterior(result['idata'], var_names=['pi'], ax=ax)
    ax.set_title('Prevalence by Group (π)')
    plt.tight_layout()
    plt.savefig(f'{save_path}posterior_pi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved posterior_pi.png")

    # Annotator sensitivities
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_posterior(result['idata'], var_names=['s'], ax=ax)
    ax.set_title('Annotator Sensitivities (s)')
    plt.tight_layout()
    plt.savefig(f'{save_path}posterior_sensitivities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved posterior_sensitivities.png")

    # Annotator specificities  
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_posterior(result['idata'], var_names=['c'], ax=ax)
    ax.set_title('Annotator Specificities (c)')
    plt.tight_layout()
    plt.savefig(f'{save_path}posterior_specificities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved posterior_specificities.png")

    # Hyperparameters
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_posterior(result['idata'], var_names=['mu_pi', 'sigma_pi'], ax=ax)
    ax.set_title('Hyperparameters')
    plt.tight_layout()
    plt.savefig(f'{save_path}posterior_hyperparams.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved posterior_hyperparams.png")


def create_sampling_diagnostics(result: Dict, save_path: str = 'plots/') -> None:
    """Create sampling diagnostic plots."""
    print("\n5. CREATING SAMPLING DIAGNOSTICS...")

    # Energy plot
    fig, ax = plt.subplots(figsize=(8, 6))
    az.plot_energy(result['idata'], ax=ax)
    ax.set_title('Energy Distribution')
    plt.tight_layout()
    plt.savefig(f'{save_path}energy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved energy_plot.png")

    # Rank plot
    fig, ax = plt.subplots(figsize=(8, 6))
    az.plot_rank(result['idata'], ax=ax)
    ax.set_title('Rank Plots')
    plt.tight_layout()
    plt.savefig(f'{save_path}rank_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved rank_plot.png")


def create_forest_and_pair_plots(result: Dict, save_path: str = 'plots/') -> None:
    """Create forest and pair plots."""
    print("\n6. CREATING FOREST PLOT...")
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_forest(result['idata'], var_names=['pi'], ax=ax)
    ax.set_title('Prevalence Estimates by Group (95% CI)')
    plt.tight_layout()
    plt.savefig(f'{save_path}forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved forest_plot.png")

    # Pair plot for correlations
    print("\n7. CREATING PAIR PLOT...")
    az.plot_pair(result['idata'], var_names=['pi'], figsize=(10, 8))
    plt.suptitle('Pair Plot: Prevalence Correlations')
    plt.tight_layout()
    plt.savefig(f'{save_path}pair_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved pair_plot.png")


def compare_with_true_values(result: Dict, true_pi: list, s_true: list, c_true: list, J: int) -> None:
    """Compare model estimates with true values."""
    print("\n8. COMPARISON WITH TRUE VALUES")
    print("-" * 40)
    pi_posterior = result['idata'].posterior['pi']
    for i in range(3):
        true_val = true_pi[i]
        posterior_mean = pi_posterior[i].mean().values
        posterior_std = pi_posterior[i].std().values
        ci_low = pi_posterior[i].quantile(0.025).values
        ci_high = pi_posterior[i].quantile(0.975).values
        
        print(f"Group {i}:")
        print(f"  True value: {true_val:.3f}")
        print(f"  Posterior mean: {posterior_mean:.3f} ± {posterior_std:.3f}")
        print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  True value in CI: {'Yes' if ci_low <= true_val <= ci_high else 'No'}")
        print()

    # Annotator performance comparison
    print("9. ANNOTATOR PERFORMANCE COMPARISON")
    print("-" * 40)
    s_posterior = result['idata'].posterior['s']
    c_posterior = result['idata'].posterior['c']

    # Get the actual number of annotators from the posterior
    n_annotators = s_posterior.shape[-1]
    for j in range(min(J, n_annotators)):
        s_mean = float(s_posterior[..., j].mean().values)
        c_mean = float(c_posterior[..., j].mean().values)
        s_true_val = s_true[j]
        c_true_val = c_true[j]
        
        print(f"Annotator {j}:")
        print(f"  Sensitivity: True={s_true_val:.3f}, Estimated={s_mean:.3f}")
        print(f"  Specificity: True={c_true_val:.3f}, Estimated={c_mean:.3f}")
        print()


def save_sample_data(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    save_path: str = 'data/') -> None:
    """
    Save simulated data to files for reproducibility and analysis.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset with ground truth labels
    test_df : pd.DataFrame
        Test dataset without ground truth labels
    save_path : str
        Directory to save the data files
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save training dataset (with true labels)
    train_df.to_csv(f'{save_path}train_dataset.csv', index=False)
    
    # Save test dataset (without true labels)
    test_df.to_csv(f'{save_path}test_dataset.csv', index=False)
    
    print(f"Sample data saved to {save_path}")
    print(f"- train_dataset.csv: {len(train_df)} items with ground truth")
    print(f"- test_dataset.csv: {len(test_df)} items without ground truth")


def calculate_confusion_matrices(validation_df: pd.DataFrame, annotator_cols: list) -> None:
    """
    Calculate and display confusion matrices for each annotator using validation data.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        Validation dataset with true labels
    annotator_cols : list
        List of annotator column names
    """
    print("\n" + "="*60)
    print("CONFUSION MATRICES FOR EACH ANNOTATOR")
    print("="*60)
    
    for j, col in enumerate(annotator_cols):
        # Get true labels and predictions
        true_labels = validation_df['true_label'].values
        predictions = validation_df[col].values
        
        # Calculate confusion matrix components
        tp = np.sum((true_labels == 1) & (predictions == 1))
        tn = np.sum((true_labels == 0) & (predictions == 0))
        fp = np.sum((true_labels == 0) & (predictions == 1))
        fn = np.sum((true_labels == 1) & (predictions == 0))
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"\nAnnotator {j}:")
        print(f"  Confusion Matrix:")
        print(f"    True Positives:  {tp:3d}    False Positives: {fp:3d}")
        print(f"    False Negatives: {fn:3d}    True Negatives:   {tn:3d}")
        print(f"  Sensitivity: {sensitivity:.3f}")
        print(f"  Specificity: {specificity:.3f}")
        print(f"  Accuracy:    {accuracy:.3f}")


def learn_annotator_parameters(validation_df: pd.DataFrame, draws: int = 1000, tune: int = 2000, random_seed: int = 42) -> Dict:
    """
    Learn annotator parameters (sensitivity and specificity) from validation data with ground truth.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        DataFrame containing ground truth labels and annotator assessments
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing model results, including posterior draws for annotator parameters
    """
    # Get annotator columns and true labels
    annotator_cols = [col for col in validation_df.columns if col.startswith('annotator_')]
    J = len(annotator_cols)
    z_true = validation_df['true_label'].values
    
    # Get annotations and create mask for missing values
    y_obs = validation_df[annotator_cols].values
    annotation_mask = ~np.isnan(y_obs)
    
    with pm.Model() as model:
        # Priors for sensitivity and specificity
        s = pm.Beta("s", 8, 2, shape=J)  # sensitivity
        c = pm.Beta("c", 8, 2, shape=J)  # specificity
        
        # Compute probability of positive annotation
        p_y1 = pm.math.where(
            z_true[:, None] == 1,
            s[None, :],         # If z=1, use sensitivity
            1 - c[None, :]      # If z=0, use (1-specificity)
        )
        
        # Observe annotations
        y_obs_masked = pm.math.where(annotation_mask, y_obs, 0)
        pm.Bernoulli("y", p=p_y1, observed=y_obs_masked)
        
        # Sample posterior
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.9, random_seed=random_seed)
    
    # Return inference data
    return {"idata": idata}


def predict_test_prevalence_marginalized(
    test_df: pd.DataFrame, 
    idata_params: Dict, 
    N_group: Optional[List[int]] = None,
    draws: int = 1000, 
    tune: int = 2000, 
    random_seed: int = 42
) -> Dict:
    """
    Predict group prevalences using marginalized approach with learned annotator parameters.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset with annotations but no ground truth
    idata_params : Dict
        Dict containing InferenceData with annotator parameters from Stage 1
    N_group : list, optional
        Population sizes for each group for extrapolation
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing model results, summaries, and posterior draws
    """
    # Get annotator columns
    annotator_cols = [col for col in test_df.columns if col.startswith('annotator_')]
    group_of_item = test_df['group'].values
    n_items = len(test_df)
    
    # Infer dimensions
    J = len(annotator_cols)
    G = len(np.unique(group_of_item))
    
    # Create mask for non-missing annotations
    annotation_mask = ~test_df[annotator_cols].isna()
    y_obs = test_df[annotator_cols].values
    
    # Extract learned annotator parameters from Stage 1
    stage1_idata = idata_params['idata']
    s_learned = stage1_idata.posterior['s'].mean(['chain', 'draw']).values
    c_learned = stage1_idata.posterior['c'].mean(['chain', 'draw']).values
    s_std = stage1_idata.posterior['s'].std(['chain', 'draw']).values
    c_std = stage1_idata.posterior['c'].std(['chain', 'draw']).values
    
    # Convert to Beta distribution parameters with more balanced informativeness
    # Use a weighted combination of learned parameters and original vague priors
    def beta_params_from_mean_std(mean, std, weight=1.0):
        # Clip mean to valid range
        mean = np.clip(mean, 0.01, 0.99)
        
        # Use a more conservative approach: blend learned parameters with original priors
        # Original priors: Beta(8, 2) -> mean ≈ 0.8, std ≈ 0.12
        original_mean = 0.8
        original_std = 0.12
        
        # Weighted combination
        blended_mean = weight * mean + (1 - weight) * original_mean
        blended_std = weight * std + (1 - weight) * original_std
        
        # Convert to Beta parameters using method of moments
        var = blended_std**2
        alpha = blended_mean * (blended_mean * (1 - blended_mean) / var - 1)
        beta = (1 - blended_mean) * (blended_mean * (1 - blended_mean) / var - 1)
        
        # Ensure reasonable parameters (not too peaked, not too flat)
        alpha = np.clip(alpha, 2.0, 20.0)
        beta = np.clip(beta, 2.0, 20.0)
        
        return alpha, beta
    
    s_alpha, s_beta = beta_params_from_mean_std(s_learned, s_std)
    c_alpha, c_beta = beta_params_from_mean_std(c_learned, c_std)
    
    print(f"Using learned annotator parameters as informative priors:")
    for j in range(J):
        print(f"  Annotator {j} - Sensitivity: mean={s_learned[j]:.3f}, std={s_std[j]:.3f} -> Beta({s_alpha[j]:.1f}, {s_beta[j]:.1f})")
        print(f"  Annotator {j} - Specificity: mean={c_learned[j]:.3f}, std={c_std[j]:.3f} -> Beta({c_alpha[j]:.1f}, {c_beta[j]:.1f})")
    
    with pm.Model() as model:
        # Priors for prevalence (same as original model)
        mu_pi = pm.Normal("mu_pi", 0, 2)
        sigma_pi = pm.HalfNormal("sigma_pi", 2)
        logit_pi = pm.Normal("logit_pi", mu=mu_pi, sigma=sigma_pi, shape=G)
        pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi))
        
        # Annotator parameters using learned values as informative priors
        s = pm.Beta("s", alpha=s_alpha, beta=s_beta, shape=J)
        c = pm.Beta("c", alpha=c_alpha, beta=c_beta, shape=J)
        
        # Calculate probabilities without explicit z variables
        pi_per_item = pi[group_of_item]
        p_y1 = pi_per_item[:, None] * s[None, :] + (1 - pi_per_item[:, None]) * (1 - c[None, :])
        
        # Observe test annotations
        y_obs_masked = pm.math.where(annotation_mask, y_obs, 0)
        pm.Bernoulli("y", p=p_y1, observed=y_obs_masked)
        
        # Sample posterior (not posterior predictive)
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.9,
                          random_seed=random_seed)
    
    # Extract posterior draws for pi
    pi_draws = idata.posterior["pi"].stack(sample=("chain", "draw")).values.T
    
    # Sample counts for observed items
    G_counts = np.zeros((pi_draws.shape[0], G), dtype=int)
    for g in range(G):
        n_g = np.sum(group_of_item == g)
        G_counts[:, g] = np.random.binomial(n=n_g, p=pi_draws[:, g])
    
    # Population counts if requested
    pop_counts = None
    if N_group is not None:
        pop_counts = np.zeros((pi_draws.shape[0], G), dtype=int)
        for g in range(G):
            pop_counts[:, g] = np.random.binomial(N_group[g], p=pi_draws[:, g])
    
    # Prepare summaries
    group_summary = []
    for g in range(G):
        arr = G_counts[:, g]
        group_summary.append({
            "group": g,
            "mean_sample_count": arr.mean(),
            "ci_2.5": np.percentile(arr, 2.5),
            "ci_97.5": np.percentile(arr, 97.5)
        })
    
    pop_summary = None
    if pop_counts is not None:
        pop_summary = []
        for g in range(G):
            arr = pop_counts[:, g]
            pop_summary.append({
                "group": g,
                "mean_pop_count": arr.mean(),
                "ci_2.5": np.percentile(arr, 2.5),
                "ci_97.5": np.percentile(arr, 97.5)
            })
    
    return {
        "idata": idata,
        "group_sample_counts_draws": G_counts,
        "group_summary": pd.DataFrame(group_summary),
        "pop_summary": pd.DataFrame(pop_summary) if pop_counts is not None else None
    }


def two_stage_inference(
    validation_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    N_group: Optional[List[int]] = None,
    draws: int = 1000, 
    tune: int = 2000, 
    random_seed: int = 42
) -> Dict:
    """
    Run the complete two-stage inference process.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        Validation dataset with ground truth labels
    test_df : pd.DataFrame
        Test dataset without ground truth labels
    N_group : list, optional
        Population sizes for each group
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    Dict containing results from both stages
    """
    # Stage 1: Learn annotator parameters from validation data
    print("\nStage 1: Learning annotator parameters from validation data...")
    stage1_results = learn_annotator_parameters(
        validation_df, draws=draws, tune=tune, random_seed=random_seed
    )
    
    # Stage 2: Estimate group prevalences in test data
    print("Stage 2: Estimating group prevalences from test data...")
    stage2_results = predict_test_prevalence_marginalized(
        test_df, stage1_results, N_group=N_group,
        draws=draws, tune=tune, random_seed=random_seed
    )
    
    # Return combined results
    return {
        "stage1_results": stage1_results,
        "stage2_results": stage2_results
    }


def two_stage_posterior_predictive(
    validation_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    N_group: Optional[List[int]] = None,
    draws: int = 1000, 
    tune: int = 2000, 
    random_seed: int = 42
) -> Dict:
    """
    Run two-stage inference using posterior predictive sampling instead of informative priors.
    
    This approach learns annotator parameters from validation data, then uses 
    sample_posterior_predictive to make predictions on test data with a separate model.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        Validation dataset with ground truth labels
    test_df : pd.DataFrame
        Test dataset without ground truth labels
    N_group : list, optional
        Population sizes for each group
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    Dict containing results from both stages
    """
    # Stage 1: Learn annotator parameters from validation data
    print("\nStage 1: Learning annotator parameters from validation data...")
    stage1_results = learn_annotator_parameters(
        validation_df, draws=draws, tune=tune, random_seed=random_seed
    )
    
    # Stage 2: Use posterior predictive sampling for test data
    print("Stage 2: Making predictions using posterior predictive sampling...")
    stage2_results = predict_test_with_posterior_predictive(
        test_df, stage1_results, N_group=N_group,
        draws=draws, tune=tune, random_seed=random_seed
    )
    
    # Return combined results
    return {
        "stage1_results": stage1_results,
        "stage2_results": stage2_results
    }


def two_stage_traditional_inference(
    validation_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    N_group: Optional[List[int]] = None,
    draws: int = 1000, 
    tune: int = 2000, 
    random_seed: int = 42
) -> Dict:
    """
    Run two-stage inference using traditional Dawid-Skene with informative priors.
    
    This approach learns annotator parameters from validation data, then uses them
    as informative priors in a traditional Dawid-Skene model on test data.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        Validation dataset with ground truth labels
    test_df : pd.DataFrame
        Test dataset without ground truth labels
    N_group : list, optional
        Population sizes for each group
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    Dict containing results from both stages
    """
    # Stage 1: Learn annotator parameters from validation data
    print("\nStage 1: Learning annotator parameters from validation data...")
    stage1_results = learn_annotator_parameters(
        validation_df, draws=draws, tune=tune, random_seed=random_seed
    )
    
    # Stage 2: Use traditional model with informative priors
    print("Stage 2: Estimating group prevalences using traditional model with informative priors...")
    stage2_results = predict_test_prevalence_traditional(
        test_df, stage1_results, N_group=N_group,
        draws=draws, tune=tune, random_seed=random_seed
    )
    
    # Return combined results
    return {
        "stage1_results": stage1_results,
        "stage2_results": stage2_results
    }


def two_stage_traditional_posterior_predictive(
    validation_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    N_group: Optional[List[int]] = None,
    draws: int = 1000, 
    tune: int = 2000, 
    random_seed: int = 42
) -> Dict:
    """
    Run two-stage inference using traditional Dawid-Skene with posterior predictive sampling.
    
    This approach learns annotator parameters from validation data, then uses them
    as fixed values in a traditional Dawid-Skene model with posterior predictive sampling.
    
    Parameters:
    -----------
    validation_df : pd.DataFrame
        Validation dataset with ground truth labels
    test_df : pd.DataFrame
        Test dataset without ground truth labels
    N_group : list, optional
        Population sizes for each group
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    Dict containing results from both stages
    """
    # Stage 1: Learn annotator parameters from validation data
    print("\nStage 1: Learning annotator parameters from validation data...")
    stage1_results = learn_annotator_parameters(
        validation_df, draws=draws, tune=tune, random_seed=random_seed
    )
    
    # Stage 2: Use traditional model with posterior predictive sampling
    print("Stage 2: Making predictions using traditional model with posterior predictive sampling...")
    stage2_results = predict_test_traditional_with_posterior_predictive(
        test_df, stage1_results, N_group=N_group,
        draws=draws, tune=tune, random_seed=random_seed
    )
    
    # Return combined results
    return {
        "stage1_results": stage1_results,
        "stage2_results": stage2_results
    }


def predict_test_prevalence_traditional(
    test_df: pd.DataFrame, 
    idata_params: Dict, 
    N_group: Optional[List[int]] = None,
    draws: int = 1000, 
    tune: int = 2000, 
    random_seed: int = 42
) -> Dict:
    """
    Predict group prevalences using traditional Dawid-Skene with learned annotator parameters as informative priors.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset with annotations but no ground truth
    idata_params : Dict
        Dict containing InferenceData with annotator parameters from Stage 1
    N_group : list, optional
        Population sizes for each group for extrapolation
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing model results, summaries, and posterior draws
    """
    # Get annotator columns
    annotator_cols = [col for col in test_df.columns if col.startswith('annotator_')]
    group_of_item = test_df['group'].values
    n_items = len(test_df)
    
    # Infer dimensions
    J = len(annotator_cols)
    G = len(np.unique(group_of_item))
    
    # Create mask for non-missing annotations
    annotation_mask = ~test_df[annotator_cols].isna()
    y_obs = test_df[annotator_cols].values
    
    # Extract learned annotator parameters from Stage 1
    stage1_idata = idata_params['idata']
    s_learned = stage1_idata.posterior['s'].mean(['chain', 'draw']).values
    c_learned = stage1_idata.posterior['c'].mean(['chain', 'draw']).values
    s_std = stage1_idata.posterior['s'].std(['chain', 'draw']).values
    c_std = stage1_idata.posterior['c'].std(['chain', 'draw']).values
    
    # Convert to Beta distribution parameters with more balanced informativeness
    def beta_params_from_mean_std(mean, std, weight=1.0):
        # Clip mean to valid range
        mean = np.clip(mean, 0.01, 0.99)
        
        # Use a more conservative approach: blend learned parameters with original priors
        # Original priors: Beta(8, 2) -> mean ≈ 0.8, std ≈ 0.12
        original_mean = 0.8
        original_std = 0.12
        
        # Weighted combination
        blended_mean = weight * mean + (1 - weight) * original_mean
        blended_std = weight * std + (1 - weight) * original_std
        
        # Convert to Beta parameters using method of moments
        var = blended_std**2
        alpha = blended_mean * (blended_mean * (1 - blended_mean) / var - 1)
        beta = (1 - blended_mean) * (blended_mean * (1 - blended_mean) / var - 1)
        
        # Ensure reasonable parameters (not too peaked, not too flat)
        alpha = np.clip(alpha, 2.0, 20.0)
        beta = np.clip(beta, 2.0, 20.0)
        
        return alpha, beta
    
    s_alpha, s_beta = beta_params_from_mean_std(s_learned, s_std)
    c_alpha, c_beta = beta_params_from_mean_std(c_learned, c_std)
    
    print(f"Using learned annotator parameters as informative priors for traditional model:")
    for j in range(J):
        print(f"  Annotator {j} - Sensitivity: mean={s_learned[j]:.3f}, std={s_std[j]:.3f} -> Beta({s_alpha[j]:.1f}, {s_beta[j]:.1f})")
        print(f"  Annotator {j} - Specificity: mean={c_learned[j]:.3f}, std={c_std[j]:.3f} -> Beta({c_alpha[j]:.1f}, {c_beta[j]:.1f})")
    
    with pm.Model() as model:
        # Priors for prevalence (same as original model)
        mu_pi = pm.Normal("mu_pi", 0, 2)
        sigma_pi = pm.HalfNormal("sigma_pi", 2)
        logit_pi = pm.Normal("logit_pi", mu=mu_pi, sigma=sigma_pi, shape=G)
        pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi))
        
        # Annotator parameters using learned values as informative priors
        s = pm.Beta("s", alpha=s_alpha, beta=s_beta, shape=J)
        c = pm.Beta("c", alpha=c_alpha, beta=c_beta, shape=J)
        
        # Individual true labels (explicit latent variables)
        pi_per_item = pi[group_of_item]
        z = pm.Bernoulli("z", p=pi_per_item, shape=n_items)
        
        # Observed annotations
        p_y1 = pm.math.switch(pm.math.eq(z[:, None], 1), s[None, :], 1 - c[None, :])
        y_obs_masked = pm.math.where(annotation_mask, y_obs, 0)
        pm.Bernoulli("y", p=p_y1, observed=y_obs_masked)
        
        # Sample posterior
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.8,
                          random_seed=random_seed)
    
    # Extract posterior draws for pi
    pi_draws = idata.posterior["pi"].stack(sample=("chain", "draw")).values.T
    
    # Sample counts for observed items
    G_counts = np.zeros((pi_draws.shape[0], G), dtype=int)
    for g in range(G):
        n_g = np.sum(group_of_item == g)
        G_counts[:, g] = np.random.binomial(n=n_g, p=pi_draws[:, g])
    
    # Population counts if requested
    pop_counts = None
    if N_group is not None:
        pop_counts = np.zeros((pi_draws.shape[0], G), dtype=int)
        for g in range(G):
            pop_counts[:, g] = np.random.binomial(N_group[g], p=pi_draws[:, g])
    
    # Prepare summaries
    group_summary = []
    for g in range(G):
        arr = G_counts[:, g]
        group_summary.append({
            "group": g,
            "mean_sample_count": arr.mean(),
            "ci_2.5": np.percentile(arr, 2.5),
            "ci_97.5": np.percentile(arr, 97.5)
        })
    
    pop_summary = None
    if pop_counts is not None:
        pop_summary = []
        for g in range(G):
            arr = pop_counts[:, g]
            pop_summary.append({
                "group": g,
                "mean_pop_count": arr.mean(),
                "ci_2.5": np.percentile(arr, 2.5),
                "ci_97.5": np.percentile(arr, 97.5)
            })
    
    return {
        "idata": idata,
        "group_sample_counts_draws": G_counts,
        "group_summary": pd.DataFrame(group_summary),
        "pop_summary": pd.DataFrame(pop_summary) if pop_counts is not None else None
    }


def predict_test_traditional_with_posterior_predictive(
    test_df: pd.DataFrame, 
    idata_params: Dict, 
    N_group: Optional[List[int]] = None,
    draws: int = 1000, 
    tune: int = 2000, 
    random_seed: int = 42
) -> Dict:
    """
    Predict test data using traditional Dawid-Skene with posterior predictive sampling and learned annotator parameters.
    
    This approach uses the learned annotator parameters as fixed values in a traditional model
    and samples the group prevalences and individual labels.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset with annotations but no ground truth
    idata_params : Dict
        Dict containing InferenceData with annotator parameters from Stage 1
    N_group : list, optional
        Population sizes for each group for extrapolation
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing model results, summaries, and posterior draws
    """
    # Get annotator columns
    annotator_cols = [col for col in test_df.columns if col.startswith('annotator_')]
    group_of_item = test_df['group'].values
    n_items = len(test_df)
    
    # Infer dimensions
    J = len(annotator_cols)
    G = len(np.unique(group_of_item))
    
    # Create mask for non-missing annotations
    annotation_mask = ~test_df[annotator_cols].isna()
    y_obs = test_df[annotator_cols].values
    
    # Extract learned annotator parameters from Stage 1
    stage1_idata = idata_params['idata']
    s_learned = stage1_idata.posterior['s'].mean(['chain', 'draw']).values
    c_learned = stage1_idata.posterior['c'].mean(['chain', 'draw']).values
    
    print(f"Using learned annotator parameters as fixed values for traditional model:")
    for j in range(J):
        print(f"  Annotator {j}: s={s_learned[j]:.3f}, c={c_learned[j]:.3f}")
    
    # Create model with fixed annotator parameters
    with pm.Model() as model:
        # Group prevalence parameters
        mu_pi = pm.Normal("mu_pi", 0, 2)
        sigma_pi = pm.HalfNormal("sigma_pi", 2)
        logit_pi = pm.Normal("logit_pi", mu=mu_pi, sigma=sigma_pi, shape=G)
        pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi))
        
        # Individual true labels (latent variables)
        pi_per_item = pi[group_of_item]
        z = pm.Bernoulli("z", p=pi_per_item, shape=n_items)
        
        # Fixed annotator parameters (learned from Stage 1) - use Data
        s = pm.Data("s", s_learned)
        c = pm.Data("c", c_learned)
        
        # Observed annotations
        p_y1 = pm.math.switch(pm.math.eq(z[:, None], 1), s[None, :], 1 - c[None, :])
        y_obs_masked = pm.math.where(annotation_mask, y_obs, 0)
        pm.Bernoulli("y", p=p_y1, observed=y_obs_masked)
        
        # Sample posterior
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.8, random_seed=random_seed)
    
    # Extract results
    pi_samples = idata.posterior['pi'].stack(sample=('chain', 'draw')).values.T
    z_samples = idata.posterior['z'].stack(sample=('chain', 'draw')).values.T
    
    # Calculate group-level summaries
    group_summary = []
    for g in range(G):
        group_mask = group_of_item == g
        group_pi_samples = pi_samples[:, g]
        
        group_summary.append({
            'group': g,
            'n_items': np.sum(group_mask),
            'posterior_prevalence_mean': np.mean(group_pi_samples),
            'posterior_prevalence_std': np.std(group_pi_samples),
            'posterior_prevalence_hdi_low': np.percentile(group_pi_samples, 2.5),
            'posterior_prevalence_hdi_high': np.percentile(group_pi_samples, 97.5)
        })
    
    # Calculate individual label summaries
    individual_summary = []
    for i in range(n_items):
        z_samples_i = z_samples[:, i]
        individual_summary.append({
            'item': i,
            'group': group_of_item[i],
            'posterior_label_mean': np.mean(z_samples_i),
            'posterior_label_std': np.std(z_samples_i),
            'posterior_label_hdi_low': np.percentile(z_samples_i, 2.5),
            'posterior_label_hdi_high': np.percentile(z_samples_i, 97.5)
        })
    
    # Population extrapolation if N_group is provided
    pop_summary = None
    if N_group is not None:
        pop_counts = np.zeros((pi_samples.shape[0], G), dtype=int)
        for g in range(G):
            pop_counts[:, g] = np.random.binomial(N_group[g], p=pi_samples[:, g])
        
        pop_summary = []
        for g in range(G):
            pop_counts_g = pop_counts[:, g]
            pop_summary.append({
                'group': g,
                'population_size': N_group[g],
                'posterior_count_mean': np.mean(pop_counts_g),
                'posterior_count_std': np.std(pop_counts_g),
                'posterior_count_hdi_low': np.percentile(pop_counts_g, 2.5),
                'posterior_count_high': np.percentile(pop_counts_g, 97.5)
            })
    
    return {
        "idata": idata,
        "group_summary": pd.DataFrame(group_summary),
        "individual_summary": pd.DataFrame(individual_summary),
        "pop_summary": pd.DataFrame(pop_summary) if pop_summary is not None else None,
        "method": "traditional_posterior_predictive"
    }


def predict_test_with_posterior_predictive(
    test_df: pd.DataFrame, 
    idata_params: Dict, 
    N_group: Optional[List[int]] = None,
    draws: int = 1000, 
    tune: int = 2000, 
    random_seed: int = 42
) -> Dict:
    """
    Predict test data using posterior predictive sampling with learned annotator parameters.
    
    This approach uses the learned annotator parameters as fixed values in a new model
    and samples the group prevalences and individual labels.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataset with annotations but no ground truth
    idata_params : Dict
        Dict containing InferenceData with annotator parameters from Stage 1
    N_group : list, optional
        Population sizes for each group for extrapolation
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing model results, summaries, and posterior draws
    """
    # Get annotator columns
    annotator_cols = [col for col in test_df.columns if col.startswith('annotator_')]
    group_of_item = test_df['group'].values
    n_items = len(test_df)
    
    # Infer dimensions
    J = len(annotator_cols)
    G = len(np.unique(group_of_item))
    
    # Create mask for non-missing annotations
    annotation_mask = ~test_df[annotator_cols].isna()
    y_obs = test_df[annotator_cols].values
    
    # Extract learned annotator parameters from Stage 1
    stage1_idata = idata_params['idata']
    s_learned = stage1_idata.posterior['s'].mean(['chain', 'draw']).values
    c_learned = stage1_idata.posterior['c'].mean(['chain', 'draw']).values
    
    print(f"Using learned annotator parameters:")
    for j in range(J):
        print(f"  Annotator {j}: s={s_learned[j]:.3f}, c={c_learned[j]:.3f}")
    
    # Create model with fixed annotator parameters
    with pm.Model() as model:
        # Group prevalence parameters
        mu_pi = pm.Normal("mu_pi", 0, 2)
        sigma_pi = pm.HalfNormal("sigma_pi", 2)
        logit_pi = pm.Normal("logit_pi", mu=mu_pi, sigma=sigma_pi, shape=G)
        pi = pm.Deterministic("pi", pm.math.sigmoid(logit_pi))
        
        # Individual true labels (latent variables)
        pi_per_item = pi[group_of_item]
        z = pm.Bernoulli("z", p=pi_per_item, shape=n_items)
        
        # Fixed annotator parameters (learned from Stage 1) - use Data
        s = pm.Data("s", s_learned)
        c = pm.Data("c", c_learned)
        
        # Observed annotations
        p_y1 = pm.math.switch(pm.math.eq(z[:, None], 1), s[None, :], 1 - c[None, :])
        y_obs_masked = pm.math.where(annotation_mask, y_obs, 0)
        pm.Bernoulli("y", p=p_y1, observed=y_obs_masked)
        
        # Sample posterior
        idata = pm.sample(draws=draws, tune=tune, target_accept=0.8, random_seed=random_seed)
    
    # Extract results
    pi_samples = idata.posterior['pi'].stack(sample=('chain', 'draw')).values.T
    z_samples = idata.posterior['z'].stack(sample=('chain', 'draw')).values.T
    
    # Calculate group-level summaries
    group_summary = []
    for g in range(G):
        group_mask = group_of_item == g
        group_pi_samples = pi_samples[:, g]
        
        group_summary.append({
            'group': g,
            'n_items': np.sum(group_mask),
            'posterior_prevalence_mean': np.mean(group_pi_samples),
            'posterior_prevalence_std': np.std(group_pi_samples),
            'posterior_prevalence_hdi_low': np.percentile(group_pi_samples, 2.5),
            'posterior_prevalence_hdi_high': np.percentile(group_pi_samples, 97.5)
        })
    
    # Calculate individual label summaries
    individual_summary = []
    for i in range(n_items):
        z_samples_i = z_samples[:, i]
        individual_summary.append({
            'item': i,
            'group': group_of_item[i],
            'posterior_label_mean': np.mean(z_samples_i),
            'posterior_label_std': np.std(z_samples_i),
            'posterior_label_hdi_low': np.percentile(z_samples_i, 2.5),
            'posterior_label_hdi_high': np.percentile(z_samples_i, 97.5)
        })
    
    # Population extrapolation if N_group is provided
    pop_summary = None
    if N_group is not None:
        pop_counts = np.zeros((pi_samples.shape[0], G), dtype=int)
        for g in range(G):
            pop_counts[:, g] = np.random.binomial(N_group[g], p=pi_samples[:, g])
        
        pop_summary = []
        for g in range(G):
            pop_counts_g = pop_counts[:, g]
            pop_summary.append({
                'group': g,
                'population_size': N_group[g],
                'posterior_count_mean': np.mean(pop_counts_g),
                'posterior_count_std': np.std(pop_counts_g),
                'posterior_count_hdi_low': np.percentile(pop_counts_g, 2.5),
                'posterior_count_high': np.percentile(pop_counts_g, 97.5)
            })
    
    return {
        "idata": idata,
        "group_summary": pd.DataFrame(group_summary),
        "individual_summary": pd.DataFrame(individual_summary),
        "pop_summary": pd.DataFrame(pop_summary) if pop_summary is not None else None,
        "method": "posterior_predictive"
    }


def fit_joint_traditional_model(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                               draws: int = 1000, tune: int = 1000, random_seed: int = 42) -> Dict:
    """
    Fit joint traditional Dawid-Skene model using both train and test data.
    
    This approach uses both training data (with ground truth) and test data (without ground truth)
    in a single joint model to estimate annotator parameters and group prevalences.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset with ground truth labels
    test_df : pd.DataFrame
        Test dataset without ground truth labels
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing model results, summaries, and posterior draws
    """
    # Get annotator columns
    annotator_cols = [col for col in train_df.columns if col.startswith('annotator_')]
    J = len(annotator_cols)
    
    # Train arrays
    y_train = train_df[annotator_cols].values  # shape (n_train, J)
    z_train_obs = train_df["true_label"].values.astype("int8")
    group_train = train_df["group"].values.astype("int64")
    
    # Test arrays
    y_test = test_df[annotator_cols].values    # shape (n_test, J)
    group_test = test_df["group"].values.astype("int64")
    
    # Counts
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    G = int(max(group_train.max(), group_test.max()) + 1)
    
    # Convert annotation arrays with NaNs to masked arrays so PyMC treats missing entries as missing
    y_train_masked = np.ma.masked_invalid(y_train)
    y_test_masked = np.ma.masked_invalid(y_test)
    
    with pm.Model() as model:
        # Annotator parameters
        s = pm.Beta("s", 8, 2, shape=J)
        c = pm.Beta("c", 8, 2, shape=J)
        
        # Group prevalence hierarchy
        mu_pi = pm.Normal("mu_pi", 0, 2)
        sigma_pi = pm.HalfNormal("sigma_pi", 2)
        pi_raw = pm.Normal("pi_raw", 0, 1, shape=G)
        pi = pm.Deterministic("pi", pm.math.sigmoid(mu_pi + sigma_pi * pi_raw))  # (G,)
        
        # TRAIN: true labels observed -> inform s,c and pi
        z_train = pm.Bernoulli("z_train", p=pi[group_train], observed=z_train_obs)
        
        # TEST: latent true labels
        z_test = pm.Bernoulli("z_test", p=pi[group_test], shape=n_test)
        
        # Annotation probabilities conditioned on z for train and test
        # result shapes: (n_train, J) and (n_test, J)
        p_y1_train = pm.math.where(pm.math.eq(z_train[:, None], 1), s[None, :], 1 - c[None, :])
        p_y1_test = pm.math.where(pm.math.eq(z_test[:, None], 1), s[None, :], 1 - c[None, :])
        
        # Observe annotations (masked arrays OK)
        pm.Bernoulli("y_train", p=p_y1_train, observed=y_train_masked)
        pm.Bernoulli("y_test", p=p_y1_test, observed=y_test_masked)
        
        # Sample jointly
        idata = pm.sample(draws=draws, tune=tune, random_seed=random_seed, target_accept=0.9)
    
    # Extract posterior draws for pi
    pi_draws = idata.posterior["pi"].stack(sample=("chain", "draw")).values.T
    
    # Sample counts for test items
    G_counts = np.zeros((pi_draws.shape[0], G), dtype=int)
    for g in range(G):
        n_g = np.sum(group_test == g)
        G_counts[:, g] = np.random.binomial(n=n_g, p=pi_draws[:, g])
    
    # Prepare summaries
    group_summary = []
    for g in range(G):
        arr = G_counts[:, g]
        group_summary.append({
            "group": g,
            "mean_sample_count": arr.mean(),
            "ci_2.5": np.percentile(arr, 2.5),
            "ci_97.5": np.percentile(arr, 97.5)
        })
    
    return {
        "idata": idata,
        "group_sample_counts_draws": G_counts,
        "group_summary": pd.DataFrame(group_summary),
        "method": "joint_traditional"
    }


def fit_joint_marginalized_model(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                draws: int = 1000, tune: int = 1000, random_seed: int = 42) -> Dict:
    """
    Fit joint marginalized Dawid-Skene model using both train and test data.
    
    This approach uses both training data (with ground truth) and test data (without ground truth)
    in a single joint marginalized model to estimate annotator parameters and group prevalences.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset with ground truth labels
    test_df : pd.DataFrame
        Test dataset without ground truth labels
    draws : int
        Number of posterior draws
    tune : int
        Number of tuning steps
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict containing model results, summaries, and posterior draws
    """
    # Get annotator columns
    annotator_cols = [col for col in train_df.columns if col.startswith('annotator_')]
    J = len(annotator_cols)
    
    # Train arrays
    y_train = train_df[annotator_cols].values  # shape (n_train, J)
    z_train_obs = train_df["true_label"].values.astype("int8")
    group_train = train_df["group"].values.astype("int64")
    
    # Test arrays
    y_test = test_df[annotator_cols].values    # shape (n_test, J)
    group_test = test_df["group"].values.astype("int64")
    
    # Counts
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    G = int(max(group_train.max(), group_test.max()) + 1)
    
    # Convert annotation arrays with NaNs to masked arrays so PyMC treats missing entries as missing
    y_train_masked = np.ma.masked_invalid(y_train)
    y_test_masked = np.ma.masked_invalid(y_test)
    
    with pm.Model() as model:
        # Annotator parameters
        s = pm.Beta("s", 8, 2, shape=J)
        c = pm.Beta("c", 8, 2, shape=J)
        
        # Group prevalence hierarchy
        mu_pi = pm.Normal("mu_pi", 0, 2)
        sigma_pi = pm.HalfNormal("sigma_pi", 2)
        pi_raw = pm.Normal("pi_raw", 0, 1, shape=G)
        pi = pm.Deterministic("pi", pm.math.sigmoid(mu_pi + sigma_pi * pi_raw))  # (G,)
        
        # TRAIN: true labels observed -> inform s,c and pi
        z_train = pm.Bernoulli("z_train", p=pi[group_train], observed=z_train_obs)
        
        # TEST: marginalized approach - no explicit z variables
        # Calculate probabilities for test items using marginalized approach
        pi_per_test_item = pi[group_test]
        p_y1_test = pi_per_test_item[:, None] * s[None, :] + (1 - pi_per_test_item[:, None]) * (1 - c[None, :])
        
        # Annotation probabilities for train items (explicit z)
        p_y1_train = pm.math.where(pm.math.eq(z_train[:, None], 1), s[None, :], 1 - c[None, :])
        
        # Observe annotations (masked arrays OK)
        pm.Bernoulli("y_train", p=p_y1_train, observed=y_train_masked)
        pm.Bernoulli("y_test", p=p_y1_test, observed=y_test_masked)
        
        # Sample jointly
        idata = pm.sample(draws=draws, tune=tune, random_seed=random_seed, target_accept=0.9)
    
    # Extract posterior draws for pi
    pi_draws = idata.posterior["pi"].stack(sample=("chain", "draw")).values.T
    
    # Sample counts for test items
    G_counts = np.zeros((pi_draws.shape[0], G), dtype=int)
    for g in range(G):
        n_g = np.sum(group_test == g)
        G_counts[:, g] = np.random.binomial(n=n_g, p=pi_draws[:, g])
    
    # Prepare summaries
    group_summary = []
    for g in range(G):
        arr = G_counts[:, g]
        group_summary.append({
            "group": g,
            "mean_sample_count": arr.mean(),
            "ci_2.5": np.percentile(arr, 2.5),
            "ci_97.5": np.percentile(arr, 97.5)
        })
    
    return {
        "idata": idata,
        "group_sample_counts_draws": G_counts,
        "group_summary": pd.DataFrame(group_summary),
        "method": "joint_marginalized"
    }


def compare_all_approaches(marginalized_result: Dict, traditional_result: Dict, 
                          two_stage_marginalized_result: Dict, two_stage_marginalized_pp_result: Dict,
                          two_stage_traditional_result: Dict, two_stage_traditional_pp_result: Dict,
                          joint_traditional_result: Dict, joint_marginalized_result: Dict,
                          true_pi: list, true_params: Dict, G: int) -> None:
    """
    Comprehensive comparison between all eight approaches.
    
    Parameters:
    -----------
    marginalized_result : Dict
        Results from marginalized DS approach (1-stage)
    traditional_result : Dict
        Results from traditional DS approach (1-stage)
    two_stage_marginalized_result : Dict
        Results from two-stage marginalized approach with informative priors
    two_stage_marginalized_pp_result : Dict
        Results from two-stage marginalized approach with posterior predictive sampling
    two_stage_traditional_result : Dict
        Results from two-stage traditional approach with informative priors
    two_stage_traditional_pp_result : Dict
        Results from two-stage traditional approach with posterior predictive sampling
    joint_traditional_result : Dict
        Results from joint traditional approach
    joint_marginalized_result : Dict
        Results from joint marginalized approach
    true_pi : list
        True prevalence values
    true_params : Dict
        True parameter values
    G : int
        Number of groups
    """
    print("\n" + "="*120)
    print("COMPREHENSIVE ALL EIGHT APPROACHES COMPARISON")
    print("="*120)
    
    # Extract prevalence estimates
    marginalized_pi = marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    traditional_pi = traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_marginalized_pi = two_stage_marginalized_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_marginalized_pp_pi = two_stage_marginalized_pp_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_traditional_pi = two_stage_traditional_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_traditional_pp_pi = two_stage_traditional_pp_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    
    # Create comprehensive comparison plot
    create_all_approaches_comparison_plot(marginalized_result, traditional_result, 
                                        two_stage_marginalized_result, two_stage_marginalized_pp_result,
                                        two_stage_traditional_result, two_stage_traditional_pp_result,
                                        joint_traditional_result, joint_marginalized_result, true_pi, G)
    
    # Statistical performance comparison
    create_all_approaches_statistical_comparison(marginalized_result, traditional_result, 
                                                two_stage_marginalized_result, two_stage_marginalized_pp_result,
                                                two_stage_traditional_result, two_stage_traditional_pp_result,
                                                joint_traditional_result, joint_marginalized_result, true_pi, G)
    
    print("\nAll approaches comparison plots saved to plots/ directory")


def create_all_approaches_comparison_plot(marginalized_result: Dict, traditional_result: Dict, 
                                        two_stage_marginalized_result: Dict, two_stage_marginalized_pp_result: Dict,
                                        two_stage_traditional_result: Dict, two_stage_traditional_pp_result: Dict,
                                        joint_traditional_result: Dict, joint_marginalized_result: Dict,
                                        true_pi: list, G: int) -> None:
    """Create comprehensive comparison plot for all eight approaches."""
    marginalized_pi = marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    traditional_pi = traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_marginalized_pi = two_stage_marginalized_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_marginalized_pp_pi = two_stage_marginalized_pp_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_traditional_pi = two_stage_traditional_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_traditional_pp_pi = two_stage_traditional_pp_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    joint_traditional_pi = joint_traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    joint_marginalized_pi = joint_marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    
    # Extract posterior samples for CI calculation
    marginalized_samples = marginalized_result['idata'].posterior["pi"].stack(sample=('chain', 'draw')).values.T
    traditional_samples = traditional_result['idata'].posterior["pi"].stack(sample=('chain', 'draw')).values.T
    two_stage_marginalized_samples = two_stage_marginalized_result['stage2_results']['idata'].posterior["pi"].stack(sample=('chain', 'draw')).values.T
    two_stage_marginalized_pp_samples = two_stage_marginalized_pp_result['stage2_results']['idata'].posterior["pi"].stack(sample=('chain', 'draw')).values.T
    two_stage_traditional_samples = two_stage_traditional_result['stage2_results']['idata'].posterior["pi"].stack(sample=('chain', 'draw')).values.T
    two_stage_traditional_pp_samples = two_stage_traditional_pp_result['stage2_results']['idata'].posterior["pi"].stack(sample=('chain', 'draw')).values.T
    joint_traditional_samples = joint_traditional_result['idata'].posterior["pi"].stack(sample=('chain', 'draw')).values.T
    joint_marginalized_samples = joint_marginalized_result['idata'].posterior["pi"].stack(sample=('chain', 'draw')).values.T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot 1: Bar chart with all approaches and CI error bars
    positions = np.arange(G)
    width = 0.10  # Reduced width to fit 8 approaches
    
    # Plot bars
    ax1.bar(positions - 3.5*width, true_pi, width=width, color='green', alpha=0.8, label='True Prevalence')
    ax1.bar(positions - 2.5*width, marginalized_pi, width=width, color='blue', alpha=0.7, label='Marginalized (1-stage)')
    ax1.bar(positions - 1.5*width, traditional_pi, width=width, color='red', alpha=0.7, label='Traditional (1-stage)')
    ax1.bar(positions - 0.5*width, two_stage_marginalized_pi, width=width, color='orange', alpha=0.7, label='Marginalized 2-stage (Priors)')
    ax1.bar(positions + 0.5*width, two_stage_marginalized_pp_pi, width=width, color='purple', alpha=0.7, label='Marginalized 2-stage (PP)')
    ax1.bar(positions + 1.5*width, two_stage_traditional_pi, width=width, color='brown', alpha=0.7, label='Traditional 2-stage (Priors)')
    ax1.bar(positions + 2.5*width, two_stage_traditional_pp_pi, width=width, color='pink', alpha=0.7, label='Traditional 2-stage (PP)')
    ax1.bar(positions + 3.5*width, joint_traditional_pi, width=width, color='cyan', alpha=0.7, label='Joint Traditional')
    ax1.bar(positions + 4.5*width, joint_marginalized_pi, width=width, color='magenta', alpha=0.7, label='Joint Marginalized')
    
    # Add CI error bars for each approach
    approaches = ['Marginalized (1-stage)', 'Traditional (1-stage)', 'Marginalized 2-stage (Priors)', 
                  'Marginalized 2-stage (PP)', 'Traditional 2-stage (Priors)', 'Traditional 2-stage (PP)',
                  'Joint Traditional', 'Joint Marginalized']
    pi_estimates = [marginalized_pi, traditional_pi, two_stage_marginalized_pi, 
                   two_stage_marginalized_pp_pi, two_stage_traditional_pi, two_stage_traditional_pp_pi,
                   joint_traditional_pi, joint_marginalized_pi]
    pi_samples = [marginalized_samples, traditional_samples, two_stage_marginalized_samples,
                  two_stage_marginalized_pp_samples, two_stage_traditional_samples, two_stage_traditional_pp_samples,
                  joint_traditional_samples, joint_marginalized_samples]
    colors = ['blue', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
    bar_positions = [positions - 2.5*width, positions - 1.5*width, positions - 0.5*width, 
                    positions + 0.5*width, positions + 1.5*width, positions + 2.5*width,
                    positions + 3.5*width, positions + 4.5*width]
    
    for approach, pi_est, samples, color, bar_pos in zip(approaches, pi_estimates, pi_samples, colors, bar_positions):
        for g in range(G):
            # Calculate 95% CI
            ci_low = np.percentile(samples[:, g], 2.5)
            ci_high = np.percentile(samples[:, g], 97.5)
            
            # Add error bars
            ax1.errorbar(bar_pos[g], pi_est[g], 
                        yerr=[[pi_est[g] - ci_low], [ci_high - pi_est[g]]], 
                        fmt='none', color='black', capsize=2, capthick=1)
    
    ax1.set_title('Prevalence Estimation Comparison - All Six Approaches', fontsize=16)
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Prevalence')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'Group {g}' for g in range(G)])
    ax1.set_ylim(0, 1)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error comparison
    errors = []
    for pi_est in pi_estimates:
        error = np.abs(pi_est - true_pi)
        errors.append(error)
    
    x = np.arange(len(approaches))
    for i, (approach, error, color) in enumerate(zip(approaches, errors, colors)):
        ax2.bar(x[i], np.mean(error), color=color, alpha=0.7, label=approach)
        ax2.errorbar(x[i], np.mean(error), yerr=np.std(error), fmt='none', color='black', capsize=5)
    
    ax2.set_title('Mean Absolute Error by Approach', fontsize=16)
    ax2.set_xlabel('Approach')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Approach {i+1}' for i in range(len(approaches))], rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/all_approaches_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved all_approaches_comparison.png")


def create_all_approaches_statistical_comparison(marginalized_result: Dict, traditional_result: Dict, 
                                                two_stage_marginalized_result: Dict, two_stage_marginalized_pp_result: Dict,
                                                two_stage_traditional_result: Dict, two_stage_traditional_pp_result: Dict,
                                                joint_traditional_result: Dict, joint_marginalized_result: Dict,
                                                true_pi: list, G: int) -> None:
    """Create statistical performance comparison for all eight approaches."""
    approaches = ['Marginalized (1-stage)', 'Traditional (1-stage)', 'Marginalized 2-stage (Priors)', 
                  'Marginalized 2-stage (PP)', 'Traditional 2-stage (Priors)', 'Traditional 2-stage (PP)',
                  'Joint Traditional', 'Joint Marginalized']
    
    # Extract prevalence estimates
    marginalized_pi = marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    traditional_pi = traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_marginalized_pi = two_stage_marginalized_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_marginalized_pp_pi = two_stage_marginalized_pp_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_traditional_pi = two_stage_traditional_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_traditional_pp_pi = two_stage_traditional_pp_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    joint_traditional_pi = joint_traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    joint_marginalized_pi = joint_marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    
    pi_estimates = [marginalized_pi, traditional_pi, two_stage_marginalized_pi, 
                   two_stage_marginalized_pp_pi, two_stage_traditional_pi, two_stage_traditional_pp_pi,
                   joint_traditional_pi, joint_marginalized_pi]
    
    # Calculate metrics
    print("\nSTATISTICAL PERFORMANCE COMPARISON - ALL EIGHT APPROACHES")
    print("=" * 100)
    print(f"{'Approach':<30} {'RMSE':<10} {'MAE':<10} {'Max Error':<12} {'Coverage':<10}")
    print("-" * 100)
    
    for approach, pi_est in zip(approaches, pi_estimates):
        rmse = np.sqrt(np.mean((pi_est - true_pi) ** 2))
        mae = np.mean(np.abs(pi_est - true_pi))
        max_error = np.max(np.abs(pi_est - true_pi))
        
        # Calculate coverage (how many true values are within reasonable range)
        coverage = np.mean(np.abs(pi_est - true_pi) < 0.1)  # Within 0.1 of true value
        
        print(f"{approach:<30} {rmse:<10.4f} {mae:<10.4f} {max_error:<12.4f} {coverage:<10.2%}")
    
    # Group-level detailed comparison
    print(f"\nDETAILED GROUP-LEVEL COMPARISON")
    print(f"{'Group':<6} {'True':<8} {'Marg(1)':<10} {'Trad(1)':<10} {'Marg2(P)':<10} {'Marg2(PP)':<10} {'Trad2(P)':<10} {'Trad2(PP)':<10} {'JointT':<10} {'JointM':<10}")
    print("-" * 120)
    for g in range(G):
        print(f"{g:<6} {true_pi[g]:<8.3f} {marginalized_pi[g]:<10.3f} {traditional_pi[g]:<10.3f} "
              f"{two_stage_marginalized_pi[g]:<10.3f} {two_stage_marginalized_pp_pi[g]:<10.3f} "
              f"{two_stage_traditional_pi[g]:<10.3f} {two_stage_traditional_pp_pi[g]:<10.3f} "
              f"{joint_traditional_pi[g]:<10.3f} {joint_marginalized_pi[g]:<10.3f}")
    
    # Performance ranking
    print(f"\nPERFORMANCE RANKING (by RMSE):")
    rmse_values = []
    for pi_est in pi_estimates:
        rmse = np.sqrt(np.mean((pi_est - true_pi) ** 2))
        rmse_values.append(rmse)
    
    sorted_indices = np.argsort(rmse_values)
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. {approaches[idx]}: RMSE = {rmse_values[idx]:.4f}")


def analyze_approach_performance(marginalized_result: Dict, traditional_result: Dict, 
                                two_stage_marginalized_result: Dict, two_stage_marginalized_pp_result: Dict,
                                two_stage_traditional_result: Dict, two_stage_traditional_pp_result: Dict,
                                joint_traditional_result: Dict, joint_marginalized_result: Dict,
                                true_pi: list, true_params: Dict, G: int) -> None:
    """
    Provide detailed analysis and insights about the performance of all eight approaches.
    
    Parameters:
    -----------
    marginalized_result : Dict
        Results from marginalized DS approach (1-stage)
    traditional_result : Dict
        Results from traditional DS approach (1-stage)
    two_stage_marginalized_result : Dict
        Results from two-stage marginalized approach with informative priors
    two_stage_marginalized_pp_result : Dict
        Results from two-stage marginalized approach with posterior predictive sampling
    two_stage_traditional_result : Dict
        Results from two-stage traditional approach with informative priors
    two_stage_traditional_pp_result : Dict
        Results from two-stage traditional approach with posterior predictive sampling
    joint_traditional_result : Dict
        Results from joint traditional approach
    joint_marginalized_result : Dict
        Results from joint marginalized approach
    true_pi : list
        True prevalence values
    true_params : Dict
        True parameter values
    G : int
        Number of groups
    """
    print("\n" + "="*120)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*120)
    
    # Extract prevalence estimates
    marginalized_pi = marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    traditional_pi = traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_marginalized_pi = two_stage_marginalized_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_marginalized_pp_pi = two_stage_marginalized_pp_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_traditional_pi = two_stage_traditional_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    two_stage_traditional_pp_pi = two_stage_traditional_pp_result['stage2_results']['idata'].posterior["pi"].mean(["chain", "draw"]).values
    joint_traditional_pi = joint_traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    joint_marginalized_pi = joint_marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    
    approaches = ['Marginalized (1-stage)', 'Traditional (1-stage)', 'Marginalized 2-stage (Priors)', 
                  'Marginalized 2-stage (PP)', 'Traditional 2-stage (Priors)', 'Traditional 2-stage (PP)',
                  'Joint Traditional', 'Joint Marginalized']
    pi_estimates = [marginalized_pi, traditional_pi, two_stage_marginalized_pi, 
                   two_stage_marginalized_pp_pi, two_stage_traditional_pi, two_stage_traditional_pp_pi,
                   joint_traditional_pi, joint_marginalized_pi]
    
    # Calculate comprehensive metrics
    metrics = []
    for i, (approach, pi_est) in enumerate(zip(approaches, pi_estimates)):
        rmse = np.sqrt(np.mean((pi_est - true_pi) ** 2))
        mae = np.mean(np.abs(pi_est - true_pi))
        max_error = np.max(np.abs(pi_est - true_pi))
        bias = np.mean(pi_est - true_pi)
        
        # Calculate coverage (how many true values are within reasonable range)
        coverage = np.mean(np.abs(pi_est - true_pi) < 0.1)  # Within 0.1 of true value
        
        # Calculate consistency (how similar are estimates across groups)
        consistency = 1.0 - np.std(pi_est)  # Higher is better (closer to 1)
        
        metrics.append({
            'approach': approach,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'bias': bias,
            'coverage': coverage,
            'consistency': consistency
        })
    
    # Sort by RMSE
    metrics.sort(key=lambda x: x['rmse'])
    
    print("\n1. OVERALL PERFORMANCE RANKING (by RMSE):")
    print("-" * 80)
    for i, metric in enumerate(metrics):
        print(f"{i+1:2d}. {metric['approach']:<35} RMSE: {metric['rmse']:.4f}, MAE: {metric['mae']:.4f}, Coverage: {metric['coverage']:.1%}")
    
    # Best and worst performers
    best = metrics[0]
    worst = metrics[-1]
    
    print(f"\n2. BEST PERFORMER: {best['approach']}")
    print(f"   - RMSE: {best['rmse']:.4f}")
    print(f"   - MAE: {best['mae']:.4f}")
    print(f"   - Coverage: {best['coverage']:.1%}")
    print(f"   - Bias: {best['bias']:.4f}")
    
    print(f"\n3. WORST PERFORMER: {worst['approach']}")
    print(f"   - RMSE: {worst['rmse']:.4f}")
    print(f"   - MAE: {worst['mae']:.4f}")
    print(f"   - Coverage: {worst['coverage']:.1%}")
    print(f"   - Bias: {worst['bias']:.4f}")
    
    # Compare 1-stage vs 2-stage approaches
    print(f"\n4. 1-STAGE vs 2-STAGE COMPARISON:")
    print("-" * 50)
    
    # Marginalized approaches
    marg_1stage = next(m for m in metrics if m['approach'] == 'Marginalized (1-stage)')
    marg_2stage_priors = next(m for m in metrics if m['approach'] == 'Marginalized 2-stage (Priors)')
    marg_2stage_pp = next(m for m in metrics if m['approach'] == 'Marginalized 2-stage (PP)')
    
    print(f"Marginalized approaches:")
    print(f"  1-stage:           RMSE = {marg_1stage['rmse']:.4f}")
    print(f"  2-stage (Priors):  RMSE = {marg_2stage_priors['rmse']:.4f} (Δ = {marg_2stage_priors['rmse'] - marg_1stage['rmse']:+.4f})")
    print(f"  2-stage (PP):      RMSE = {marg_2stage_pp['rmse']:.4f} (Δ = {marg_2stage_pp['rmse'] - marg_1stage['rmse']:+.4f})")
    
    # Traditional approaches
    trad_1stage = next(m for m in metrics if m['approach'] == 'Traditional (1-stage)')
    trad_2stage_priors = next(m for m in metrics if m['approach'] == 'Traditional 2-stage (Priors)')
    trad_2stage_pp = next(m for m in metrics if m['approach'] == 'Traditional 2-stage (PP)')
    
    print(f"\nTraditional approaches:")
    print(f"  1-stage:           RMSE = {trad_1stage['rmse']:.4f}")
    print(f"  2-stage (Priors):  RMSE = {trad_2stage_priors['rmse']:.4f} (Δ = {trad_2stage_priors['rmse'] - trad_1stage['rmse']:+.4f})")
    print(f"  2-stage (PP):      RMSE = {trad_2stage_pp['rmse']:.4f} (Δ = {trad_2stage_pp['rmse'] - trad_1stage['rmse']:+.4f})")
    
    # Compare marginalized vs traditional
    print(f"\n5. MARGINALIZED vs TRADITIONAL COMPARISON:")
    print("-" * 50)
    
    # Find best marginalized and best traditional
    marginalized_approaches = [m for m in metrics if 'Marginalized' in m['approach']]
    traditional_approaches = [m for m in metrics if 'Traditional' in m['approach']]
    
    best_marginalized = min(marginalized_approaches, key=lambda x: x['rmse'])
    best_traditional = min(traditional_approaches, key=lambda x: x['rmse'])
    
    print(f"Best Marginalized: {best_marginalized['approach']} (RMSE = {best_marginalized['rmse']:.4f})")
    print(f"Best Traditional:  {best_traditional['approach']} (RMSE = {best_traditional['rmse']:.4f})")
    print(f"Winner: {'Marginalized' if best_marginalized['rmse'] < best_traditional['rmse'] else 'Traditional'}")
    
    # Group-level analysis
    print(f"\n6. GROUP-LEVEL ANALYSIS:")
    print("-" * 50)
    print(f"{'Group':<6} {'True':<8} {'Best Est':<12} {'Best Method':<25} {'Error':<8}")
    print("-" * 70)
    
    for g in range(G):
        true_val = true_pi[g]
        group_errors = []
        for approach, pi_est in zip(approaches, pi_estimates):
            error = abs(pi_est[g] - true_val)
            group_errors.append((approach, pi_est[g], error))
        
        # Find best estimate for this group
        best_approach, best_est, best_error = min(group_errors, key=lambda x: x[2])
        print(f"{g:<6} {true_val:<8.3f} {best_est:<12.3f} {best_approach:<25} {best_error:<8.3f}")
    
    # Recommendations
    print(f"\n7. RECOMMENDATIONS:")
    print("-" * 50)
    
    if best['approach'].startswith('Marginalized'):
        print("✓ Marginalized approaches generally perform better")
    else:
        print("✓ Traditional approaches generally perform better")
    
    if any('2-stage' in m['approach'] for m in metrics[:3]):  # Top 3 performers
        print("✓ Two-stage approaches show benefits when validation data is available")
    else:
        print("✓ Single-stage approaches are sufficient for this dataset")
    
    if best['coverage'] > 0.8:
        print("✓ High coverage indicates good uncertainty quantification")
    else:
        print("⚠ Low coverage suggests uncertainty estimates may be too narrow")
    
    if abs(best['bias']) < 0.05:
        print("✓ Low bias indicates good calibration")
    else:
        print("⚠ High bias suggests systematic estimation errors")
    
    print(f"\n8. KEY INSIGHTS:")
    print("-" * 50)
    print(f"• The best approach is: {best['approach']}")
    print(f"• Performance gap between best and worst: {worst['rmse'] - best['rmse']:.4f} RMSE")
    print(f"• Two-stage approaches {'improve' if any('2-stage' in m['approach'] for m in metrics[:2]) else 'do not improve'} performance")
    print(f"• Marginalized approaches {'outperform' if best['approach'].startswith('Marginalized') else 'do not outperform'} traditional approaches")
    print(f"• Overall model quality: {'Excellent' if best['rmse'] < 0.05 else 'Good' if best['rmse'] < 0.1 else 'Fair'}")


def main():
    """Main execution function."""
    # Simulate data with train/test split
    print("Simulating data with train/test split...")
    data_dict = simulate_and_split_data(
        n_total=500,
        train_fraction=0.2,
        n_annotators=5,
        n_groups=3,
        balanced_groups=True,
        seed=42
    )
    
    # Extract datasets and parameters
    train_df = data_dict['train_df']
    test_df = data_dict['test_df']
    test_df_with_labels = data_dict['test_df_with_labels']
    true_params = data_dict['true_params']
    true_counts = data_dict['true_counts']
    
    # Set model parameters
    J = len([col for col in train_df.columns if col.startswith('annotator_')])
    G = len(np.unique(train_df['group']))
    
    # Define population sizes for extrapolation (3x the test set size per group)
    N_group = []
    for g in range(G):
        group_size = np.sum(test_df['group'] == g)
        N_group.append(group_size * 3)  # 3x extrapolation
    
    # Save sample data
    print("Saving sample data...")
    save_sample_data(train_df, test_df)
    
    # Calculate confusion matrices for training data
    print("\nCalculating annotator performance on training data:")
    annotator_cols = [col for col in train_df.columns if col.startswith('annotator_')]
    calculate_confusion_matrices(train_df, annotator_cols)
    
    # Print true group counts and prevalences
    print("\nTrue group counts in training data:")
    print(true_counts['train'])
    print("\nTrue group counts in test data:")
    print(true_counts['test'])

    # Marginalized approach: Fit the marginalized model on test data
    print("\nMarginalized approach: Fitting marginalized Dawid-Skene model on test data...")
    marginalized_result = ds_marginalized_population(
        test_df,
        J=J, G=G,
        N_group=N_group,
        draws=1000, tune=2000,  # Restored to original values
    )
    
    # Traditional approach: Fit the traditional model on test data
    print("\nTraditional approach: Fitting traditional Dawid-Skene model on test data...")
    traditional_result = traditional_dawid_skene(
        test_df,
        J=J, G=G,
        N_group=N_group,
        draws=1000, tune=2000,  # Restored to original values
    )
    
    # Two-stage marginalized approach with informative priors: Learn from validation, predict on test
    print("\nTwo-stage marginalized approach (informative priors): Learning from validation data...")
    two_stage_marginalized_result = two_stage_inference(
        train_df,  # Use training data as validation
        test_df,
        N_group=N_group,
        draws=1000, tune=2000,  # Restored to original values
    )
    
    # Two-stage marginalized approach with posterior predictive: Learn from validation, predict on test
    print("\nTwo-stage marginalized approach (posterior predictive): Learning from validation data...")
    two_stage_marginalized_pp_result = two_stage_posterior_predictive(
        train_df,  # Use training data as validation
        test_df,
        N_group=N_group,
        draws=1000, tune=2000,  # Restored to original values
    )
    
    # Two-stage traditional approach with informative priors: Learn from validation, predict on test
    print("\nTwo-stage traditional approach (informative priors): Learning from validation data...")
    two_stage_traditional_result = two_stage_traditional_inference(
        train_df,  # Use training data as validation
        test_df,
        N_group=N_group,
        draws=1000, tune=2000,  # Restored to original values
    )
    
    # Two-stage traditional approach with posterior predictive: Learn from validation, predict on test
    print("\nTwo-stage traditional approach (posterior predictive): Learning from validation data...")
    two_stage_traditional_pp_result = two_stage_traditional_posterior_predictive(
        train_df,  # Use training data as validation
        test_df,
        N_group=N_group,
        draws=1000, tune=2000,  # Restored to original values
    )
    
    # Joint traditional approach: Use both train and test data in a single model
    print("\nJoint traditional approach: Fitting joint model with both train and test data...")
    joint_traditional_result = fit_joint_traditional_model(
        train_df,
        test_df,
        draws=1000, tune=2000,  # Restored to original values
    )
    
    # Joint marginalized approach: Use both train and test data in a single marginalized model
    print("\nJoint marginalized approach: Fitting joint marginalized model with both train and test data...")
    joint_marginalized_result = fit_joint_marginalized_model(
        train_df,
        test_df,
        draws=1000, tune=2000,  # Restored to original values
    )
    
    # Use marginalized result for diagnostics
    result = marginalized_result
    
    # Get true prevalence values
    true_pi = true_params['pi']
    
    # Compare approaches
    print("\nComparing all eight approaches:")
    print("\nMarginalized (1-stage) approach sample-level counts:")
    print(marginalized_result['group_summary'])
    print("\nTraditional (1-stage) approach sample-level counts:")
    print(traditional_result['group_summary'])
    print("\nMarginalized 2-stage (informative priors) approach sample-level counts:")
    print(two_stage_marginalized_result['stage2_results']['group_summary'])
    print("\nMarginalized 2-stage (posterior predictive) approach sample-level counts:")
    print(two_stage_marginalized_pp_result['stage2_results']['group_summary'])
    print("\nTraditional 2-stage (informative priors) approach sample-level counts:")
    print(two_stage_traditional_result['stage2_results']['group_summary'])
    print("\nTraditional 2-stage (posterior predictive) approach sample-level counts:")
    print(two_stage_traditional_pp_result['stage2_results']['group_summary'])
    print("\nJoint traditional approach sample-level counts:")
    print(joint_traditional_result['group_summary'])
    print("\nJoint marginalized approach sample-level counts:")
    print(joint_marginalized_result['group_summary'])
    print("\nTrue counts in test data:")
    print(true_counts['test'])
    
    # Create comprehensive comparison plots and analysis
    compare_all_approaches(marginalized_result, traditional_result, 
                          two_stage_marginalized_result, two_stage_marginalized_pp_result,
                          two_stage_traditional_result, two_stage_traditional_pp_result,
                          joint_traditional_result, joint_marginalized_result,
                          true_pi, true_params, G)
    
    # Detailed performance analysis
    analyze_approach_performance(marginalized_result, traditional_result, 
                               two_stage_marginalized_result, two_stage_marginalized_pp_result,
                               two_stage_traditional_result, two_stage_traditional_pp_result,
                               joint_traditional_result, joint_marginalized_result,
                               true_pi, true_params, G)

    # Create visualizations
    print("\nCreating visualizations...")
    create_sample_count_plots(result, G)
    create_population_count_plots(result, test_df['group'].values, N_group, G)
    create_comparison_plots(result, test_df['group'].values, N_group, G)

    # Print summary tables
    print("Sample-level counts:")
    print(result['group_summary'])
    print("\nPopulation-level counts:")
    print(result['pop_summary'])

    # Run posterior predictive checks
    print("\nRunning posterior predictive checks...")
    ppc_results = run_posterior_predictive_checks(result, test_df, test_df_with_labels, G)
    
    # Calculate evaluation metrics
    print("\nCalculating evaluation metrics...")
    eval_metrics = calculate_evaluation_metrics(result, test_df_with_labels, true_params, G)
    
    # Explain PPC
    explain_ppc()
    
    # Run diagnostics and create plots
    run_model_diagnostics(result)
    create_trace_plots(result)
    create_posterior_plots(result)
    create_sampling_diagnostics(result)
    create_forest_and_pair_plots(result)
    compare_with_true_values(result, true_params['pi'], true_params['s'], true_params['c'], J)

    print("\n" + "="*60)
    print("ARVIZ EXPLORATION COMPLETED")
    print("="*60)
    print("Generated plots:")
    print("- plots/trace_plots_pi.png")
    print("- plots/trace_plots_annotators.png")
    print("- plots/trace_plots_hyperparams.png")
    print("- plots/posterior_pi.png")
    print("- plots/posterior_sensitivities.png")
    print("- plots/posterior_specificities.png")
    print("- plots/posterior_hyperparams.png")
    print("- plots/energy_plot.png")
    print("- plots/rank_plot.png")
    print("- plots/forest_plot.png")
    print("- plots/pair_plot.png")
    print("- plots/approach_comparison.png")
    print("- plots/posterior_predictive_check.png")


def compare_marginalized_vs_traditional(marginalized_result: Dict, traditional_result: Dict, 
                                       true_pi: list, true_params: Dict, G: int) -> None:
    """
    Comprehensive comparison between marginalized and traditional Dawid-Skene approaches.
    
    Parameters:
    -----------
    marginalized_result : Dict
        Results from marginalized DS approach
    traditional_result : Dict
        Results from traditional DS approach
    true_pi : list
        True prevalence values
    true_params : Dict
        True parameter values
    G : int
        Number of groups
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MARGINALIZED vs TRADITIONAL COMPARISON")
    print("="*80)
    
    # Extract prevalence estimates
    marginalized_pi = marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    traditional_pi = traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    
    # 1. Prevalence Comparison Plot
    create_prevalence_comparison_plot(marginalized_result, traditional_result, true_pi, G)
    
    # 2. Annotator Parameter Comparison
    create_annotator_comparison_plot(marginalized_result, traditional_result, true_params, G)
    
    # 3. Statistical Performance Comparison
    create_statistical_comparison(marginalized_result, traditional_result, true_pi, true_params, G)
    
    # 4. Posterior Distribution Comparison
    create_posterior_distribution_comparison(marginalized_result, traditional_result, G)
    
    # 5. Convergence Diagnostics Comparison
    create_convergence_comparison(marginalized_result, traditional_result, G)
    
    print("\nComparison plots saved to plots/ directory")


def create_prevalence_comparison_plot(marginalized_result: Dict, traditional_result: Dict, 
                                    true_pi: list, G: int) -> None:
    """Create prevalence estimation comparison plot."""
    marginalized_pi = marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    traditional_pi = traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart with error bars
    positions = np.arange(G)
    width = 0.25
    
    ax1.bar(positions - width, true_pi, width=width, color='green', alpha=0.7, label='True Prevalence')
    ax1.bar(positions, marginalized_pi, width=width, color='blue', alpha=0.7, label='Marginalized')
    ax1.bar(positions + width, traditional_pi, width=width, color='red', alpha=0.7, label='Traditional')
    
    # Add error bars
    for i, g in enumerate(range(G)):
        # Marginalized approach
        pi_samples = marginalized_result['idata'].posterior["pi"][:, :, g].values.flatten()
        ci_low = np.percentile(pi_samples, 2.5)
        ci_high = np.percentile(pi_samples, 97.5)
        ax1.errorbar(positions[i], marginalized_pi[i], 
                    yerr=[[marginalized_pi[i]-ci_low], [ci_high-marginalized_pi[i]]], 
                   fmt='none', color='black', capsize=5)
        
        # Traditional approach
        pi_samples = traditional_result['idata'].posterior["pi"][:, :, g].values.flatten()
        ci_low = np.percentile(pi_samples, 2.5)
        ci_high = np.percentile(pi_samples, 97.5)
        ax1.errorbar(positions[i] + width, traditional_pi[i], 
                    yerr=[[traditional_pi[i]-ci_low], [ci_high-traditional_pi[i]]], 
                   fmt='none', color='black', capsize=5)
    
    ax1.set_title('Prevalence Estimation Comparison')
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Prevalence')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'Group {g}' for g in range(G)])
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot - Marginalized vs Traditional
    ax2.scatter(marginalized_pi, traditional_pi, s=100, alpha=0.7, c=['blue', 'red', 'green'])
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Agreement')
    ax2.set_xlabel('Marginalized Estimates')
    ax2.set_ylabel('Traditional Estimates')
    ax2.set_title('Marginalized vs Traditional Estimates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add group labels
    for i, g in enumerate(range(G)):
        ax2.annotate(f'Group {g}', (marginalized_pi[i], traditional_pi[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('plots/marginalized_vs_traditional_prevalence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved marginalized_vs_traditional_prevalence.png")


def create_annotator_comparison_plot(marginalized_result: Dict, traditional_result: Dict, 
                                   true_params: Dict, G: int) -> None:
    """Create annotator parameter comparison plot."""
    J = len(true_params['s'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract annotator parameters
    marg_s = marginalized_result['idata'].posterior['s'].mean(['chain', 'draw']).values
    marg_c = marginalized_result['idata'].posterior['c'].mean(['chain', 'draw']).values
    trad_s = traditional_result['idata'].posterior['s'].mean(['chain', 'draw']).values
    trad_c = traditional_result['idata'].posterior['c'].mean(['chain', 'draw']).values
    
    # Sensitivity comparison
    axes[0, 0].scatter(true_params['s'], marg_s, s=100, alpha=0.7, label='Marginalized', color='blue')
    axes[0, 0].scatter(true_params['s'], trad_s, s=100, alpha=0.7, label='Traditional', color='red')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('True Sensitivity')
    axes[0, 0].set_ylabel('Estimated Sensitivity')
    axes[0, 0].set_title('Sensitivity Estimation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Specificity comparison
    axes[0, 1].scatter(true_params['c'], marg_c, s=100, alpha=0.7, label='Marginalized', color='blue')
    axes[0, 1].scatter(true_params['c'], trad_c, s=100, alpha=0.7, label='Traditional', color='red')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('True Specificity')
    axes[0, 1].set_ylabel('Estimated Specificity')
    axes[0, 1].set_title('Specificity Estimation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Marginalized vs Traditional sensitivity
    axes[1, 0].scatter(marg_s, trad_s, s=100, alpha=0.7, c=range(J), cmap='tab10')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('Marginalized Sensitivity')
    axes[1, 0].set_ylabel('Traditional Sensitivity')
    axes[1, 0].set_title('Sensitivity: Marginalized vs Traditional')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Marginalized vs Traditional specificity
    axes[1, 1].scatter(marg_c, trad_c, s=100, alpha=0.7, c=range(J), cmap='tab10')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('Marginalized Specificity')
    axes[1, 1].set_ylabel('Traditional Specificity')
    axes[1, 1].set_title('Specificity: Marginalized vs Traditional')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add annotator labels
    for j in range(J):
        axes[1, 0].annotate(f'A{j}', (marg_s[j], trad_s[j]), xytext=(5, 5), textcoords='offset points')
        axes[1, 1].annotate(f'A{j}', (marg_c[j], trad_c[j]), xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('plots/marginalized_vs_traditional_annotators.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved marginalized_vs_traditional_annotators.png")


def create_statistical_comparison(marginalized_result: Dict, traditional_result: Dict, 
                                true_pi: list, true_params: Dict, G: int) -> None:
    """Create statistical performance comparison."""
    marginalized_pi = marginalized_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    traditional_pi = traditional_result['idata'].posterior["pi"].mean(["chain", "draw"]).values
    
    # Calculate metrics
    marg_rmse = np.sqrt(np.mean((marginalized_pi - true_pi) ** 2))
    trad_rmse = np.sqrt(np.mean((traditional_pi - true_pi) ** 2))
    
    marg_mae = np.mean(np.abs(marginalized_pi - true_pi))
    trad_mae = np.mean(np.abs(traditional_pi - true_pi))
    
    # Calculate CI widths
    marg_ci_widths = []
    trad_ci_widths = []
    
    for g in range(G):
        marg_samples = marginalized_result['idata'].posterior["pi"][:, :, g].values.flatten()
        trad_samples = traditional_result['idata'].posterior["pi"][:, :, g].values.flatten()
        
        marg_ci_width = np.percentile(marg_samples, 97.5) - np.percentile(marg_samples, 2.5)
        trad_ci_width = np.percentile(trad_samples, 97.5) - np.percentile(trad_samples, 2.5)
        
        marg_ci_widths.append(marg_ci_width)
        trad_ci_widths.append(trad_ci_width)
    
    # Print comparison table
    print("\nSTATISTICAL PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<20} {'Marginalized':<15} {'Traditional':<15} {'Difference':<15}")
    print("-" * 50)
    print(f"{'RMSE':<20} {marg_rmse:<15.4f} {trad_rmse:<15.4f} {marg_rmse-trad_rmse:<15.4f}")
    print(f"{'MAE':<20} {marg_mae:<15.4f} {trad_mae:<15.4f} {marg_mae-trad_mae:<15.4f}")
    print(f"{'Avg CI Width':<20} {np.mean(marg_ci_widths):<15.4f} {np.mean(trad_ci_widths):<15.4f} {np.mean(marg_ci_widths)-np.mean(trad_ci_widths):<15.4f}")
    
    # Group-level comparison
    print(f"\n{'Group':<8} {'True':<8} {'Marginalized':<15} {'Traditional':<15} {'Marg Error':<12} {'Trad Error':<12}")
    print("-" * 80)
    for g in range(G):
        marg_error = abs(marginalized_pi[g] - true_pi[g])
        trad_error = abs(traditional_pi[g] - true_pi[g])
        print(f"{g:<8} {true_pi[g]:<8.3f} {marginalized_pi[g]:<15.3f} {traditional_pi[g]:<15.3f} {marg_error:<12.3f} {trad_error:<12.3f}")

def create_posterior_distribution_comparison(marginalized_result: Dict, traditional_result: Dict, G: int) -> None:
    """Create posterior distribution comparison plots."""
    fig, axes = plt.subplots(1, G, figsize=(5*G, 5))
    if G == 1:
        axes = [axes]
    
    for g in range(G):
        marg_samples = marginalized_result['idata'].posterior["pi"][:, :, g].values.flatten()
        trad_samples = traditional_result['idata'].posterior["pi"][:, :, g].values.flatten()
        
        axes[g].hist(marg_samples, bins=50, alpha=0.6, label='Marginalized', color='blue', density=True)
        axes[g].hist(trad_samples, bins=50, alpha=0.6, label='Traditional', color='red', density=True)
        
        axes[g].axvline(np.mean(marg_samples), color='blue', linestyle='--', alpha=0.8, label='Marg Mean')
        axes[g].axvline(np.mean(trad_samples), color='red', linestyle='--', alpha=0.8, label='Trad Mean')
        
        axes[g].set_title(f'Group {g} Posterior Distributions')
        axes[g].set_xlabel('Prevalence')
        axes[g].set_ylabel('Density')
        axes[g].legend()
        axes[g].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/marginalized_vs_traditional_posteriors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved marginalized_vs_traditional_posteriors.png")


def create_convergence_comparison(marginalized_result: Dict, traditional_result: Dict, G: int) -> None:
    """Create convergence diagnostics comparison."""
    print("\nCONVERGENCE DIAGNOSTICS COMPARISON")
    print("-" * 50)
    
    # Marginalized diagnostics
    marg_summary = az.summary(marginalized_result['idata'])
    marg_max_rhat = marg_summary['r_hat'].max()
    marg_min_ess = marg_summary['ess_bulk'].min()
    marg_divergences = marginalized_result['idata'].sample_stats.divergences.sum().values
    marg_div_rate = marg_divergences / marginalized_result['idata'].sample_stats.divergences.size
    
    # Traditional diagnostics
    trad_summary = az.summary(traditional_result['idata'])
    trad_max_rhat = trad_summary['r_hat'].max()
    trad_min_ess = trad_summary['ess_bulk'].min()
    trad_divergences = traditional_result['idata'].sample_stats.divergences.sum().values
    trad_div_rate = trad_divergences / traditional_result['idata'].sample_stats.divergences.size
    
    print(f"{'Metric':<20} {'Marginalized':<15} {'Traditional':<15} {'Better':<15}")
    print("-" * 50)
    print(f"{'Max R-hat':<20} {marg_max_rhat:<15.3f} {trad_max_rhat:<15.3f} {'Marg' if marg_max_rhat < trad_max_rhat else 'Trad':<15}")
    print(f"{'Min ESS':<20} {marg_min_ess:<15.0f} {trad_min_ess:<15.0f} {'Marg' if marg_min_ess > trad_min_ess else 'Trad':<15}")
    print(f"{'Divergence Rate':<20} {marg_div_rate:<15.3f} {trad_div_rate:<15.3f} {'Marg' if marg_div_rate < trad_div_rate else 'Trad':<15}")
    
    # Create trace plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Marginalized traces
    az.plot_trace(marginalized_result['idata'], var_names=['pi'], figsize=(8, 4))
    plt.suptitle('Marginalized: Prevalence Traces')
    plt.tight_layout()
    plt.savefig('plots/marginalized_traces.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Traditional traces
    az.plot_trace(traditional_result['idata'], var_names=['pi'], figsize=(8, 4))
    plt.suptitle('Traditional: Prevalence Traces')
    plt.tight_layout()
    plt.savefig('plots/traditional_traces.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create new figure for diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R-hat comparison
    marg_rhat = marg_summary['r_hat'].values
    trad_rhat = trad_summary['r_hat'].values
    
    axes[0].bar(range(len(marg_rhat)), marg_rhat, alpha=0.7, label='Marginalized', color='blue')
    axes[0].bar(range(len(trad_rhat)), trad_rhat, alpha=0.7, label='Traditional', color='red')
    axes[0].axhline(y=1.01, color='black', linestyle='--', alpha=0.5, label='Target (1.01)')
    axes[0].set_title('R-hat Comparison')
    axes[0].set_xlabel('Parameter Index')
    axes[0].set_ylabel('R-hat')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ESS comparison
    marg_ess = marg_summary['ess_bulk'].values
    trad_ess = trad_summary['ess_bulk'].values
    
    axes[1].bar(range(len(marg_ess)), marg_ess, alpha=0.7, label='Marginalized', color='blue')
    axes[1].bar(range(len(trad_ess)), trad_ess, alpha=0.7, label='Traditional', color='red')
    axes[1].axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Target (100)')
    axes[1].set_title('ESS Comparison')
    axes[1].set_xlabel('Parameter Index')
    axes[1].set_ylabel('Effective Sample Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/marginalized_vs_traditional_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved marginalized_vs_traditional_convergence.png")


def run_posterior_predictive_checks(result: Dict, test_df: pd.DataFrame, test_df_with_labels: pd.DataFrame, 
                                G: int, save_path: str = 'plots/') -> Dict:
    """
    Run posterior predictive checks to assess model fit.
    
    Parameters:
    -----------
    result : Dict
        Dict containing model results
    test_df : pd.DataFrame
        Test dataset without true labels
    test_df_with_labels : pd.DataFrame
        Test dataset with true labels for evaluation
    G : int
        Number of groups
    save_path : str
        Directory to save the plots
        
    Returns:
    --------
    Dict with PPC statistics and comparison metrics
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*60)
    print("POSTERIOR PREDICTIVE CHECKS")
    print("="*60)
    
    # Get model parameters and data
    idata = result["idata"]
    group_of_item = test_df['group'].values
    
    # Extract observed counts from test data
    observed_counts = []
    for g in range(G):
        group_mask = test_df_with_labels['group'] == g
        group_count = np.sum(test_df_with_labels.loc[group_mask, 'true_label'])
        observed_counts.append(group_count)
    
    observed_counts = np.array(observed_counts)
    
    # Extract posterior samples for pi
    pi_draws = idata.posterior["pi"].stack(sample=("chain", "draw")).values.T
    
    # Generate posterior predictive counts
    n_samples = len(pi_draws)
    ppc_counts = np.zeros((n_samples, G), dtype=int)
    
    for g in range(G):
        n_g = np.sum(group_of_item == g)
        ppc_counts[:, g] = np.random.binomial(n=n_g, p=pi_draws[:, g])
    
    # Calculate PPC statistics
    ppc_stats = []
    for g in range(G):
        observed = observed_counts[g]
        predicted = ppc_counts[:, g]
        mean_pred = np.mean(predicted)
        std_pred = np.std(predicted)
        ci_low = np.percentile(predicted, 2.5)
        ci_high = np.percentile(predicted, 97.5)
        p_value = np.mean(predicted >= observed) if observed > mean_pred else np.mean(predicted <= observed)
        
        # Adjust p-value to be two-sided
        p_value = min(p_value, 1 - p_value) * 2
        
        # Check if observed value is within CI
        in_ci = ci_low <= observed <= ci_high
        
        ppc_stats.append({
            'group': g,
            'observed': observed,
            'predicted_mean': mean_pred,
            'predicted_std': std_pred,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_value,
            'in_ci': in_ci
        })
    
    # Create PPC visualization
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Count comparisons with error bars
    groups = np.arange(G)
    pred_means = [stat['predicted_mean'] for stat in ppc_stats]
    
    # Plot predicted vs observed
    axs[0].bar(groups - 0.2, observed_counts, width=0.4, color='blue', alpha=0.7, label='Observed')
    axs[0].bar(groups + 0.2, pred_means, width=0.4, color='red', alpha=0.7, label='Predicted')
    
    # Add error bars for prediction
    for i, g in enumerate(groups):
        ci_low = ppc_stats[i]['ci_low']
        ci_high = ppc_stats[i]['ci_high']
        axs[0].errorbar(g + 0.2, pred_means[i], yerr=[[pred_means[i] - ci_low], [ci_high - pred_means[i]]], 
                    fmt='none', color='black', capsize=5)
    
    axs[0].set_xlabel('Group')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Posterior Predictive Check: Group Counts')
    axs[0].set_xticks(groups)
    axs[0].set_xticklabels([f'Group {g}' for g in groups])
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Density plots of posterior predictive vs observed
    for g in range(G):
        sns.kdeplot(ppc_counts[:, g], ax=axs[1], label=f'Group {g}')
        axs[1].axvline(observed_counts[g], color=f'C{g}', linestyle='--')
    
    axs[1].set_xlabel('Count')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Posterior Predictive Distributions')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}posterior_predictive_check.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Convert stats to DataFrame for printing
    ppc_df = pd.DataFrame(ppc_stats)
    
    # Print PPC statistics
    print("\nPosterior Predictive Check Statistics:")
    print(ppc_df[['group', 'observed', 'predicted_mean', 'predicted_std', 'p_value', 'in_ci']])
    
    # Overall evaluation
    all_in_ci = all(ppc_df['in_ci'])
    min_p_value = ppc_df['p_value'].min()
    
    print("\nPPC Summary:")
    print(f"All observed values within 95% CI: {'Yes' if all_in_ci else 'No'}")
    print(f"Minimum p-value: {min_p_value:.3f}")
    
    if all_in_ci and min_p_value > 0.05:
        print("Conclusion: Good model fit - observed data consistent with posterior predictions")
    elif all_in_ci:
        print("Conclusion: Acceptable model fit - observed data within prediction intervals")
    else:
        print("Conclusion: Poor model fit - observed data inconsistent with model predictions")
    
    return {
        'ppc_stats': ppc_df,
        'ppc_counts': ppc_counts,
        'observed_counts': observed_counts,
        'all_in_ci': all_in_ci,
        'min_p_value': min_p_value
    }


def calculate_evaluation_metrics(result: Dict, test_df_with_labels: pd.DataFrame, 
                                true_params: Dict, G: int) -> Dict:
    """
    Calculate comprehensive evaluation metrics for model performance.
    
    Parameters:
    -----------
    result : Dict
        Dict containing model results
    test_df_with_labels : pd.DataFrame
        Test dataset with true labels for evaluation
    true_params : Dict
        True parameter values
    G : int
        Number of groups
        
    Returns:
    --------
    Dict containing evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Extract posterior samples
    idata = result["idata"]
    pi_draws = idata.posterior["pi"].stack(sample=("chain", "draw")).values.T
    
    # Calculate group-level metrics
    group_metrics = []
    for g in range(G):
        # True prevalence
        true_pi = true_params['pi'][g]
        
        # Predicted prevalence (posterior mean)
        pred_pi = np.mean(pi_draws[:, g])
        pred_std = np.std(pi_draws[:, g])
        pred_ci_low = np.percentile(pi_draws[:, g], 2.5)
        pred_ci_high = np.percentile(pi_draws[:, g], 97.5)
        
        # True counts in test data
        group_mask = test_df_with_labels['group'] == g
        true_count = np.sum(test_df_with_labels.loc[group_mask, 'true_label'])
        n_group = np.sum(group_mask)
        true_observed_prevalence = true_count / n_group if n_group > 0 else 0
        
        # Bias and RMSE
        bias = pred_pi - true_pi
        rmse = np.sqrt(np.mean((pi_draws[:, g] - true_pi) ** 2))
        
        # Coverage (is true value in CI?)
        coverage = pred_ci_low <= true_pi <= pred_ci_high
        
        group_metrics.append({
            'group': g,
            'true_prevalence': true_pi,
            'predicted_prevalence': pred_pi,
            'predicted_std': pred_std,
            'predicted_ci_low': pred_ci_low,
            'predicted_ci_high': pred_ci_high,
            'true_observed_prevalence': true_observed_prevalence,
            'bias': bias,
            'rmse': rmse,
            'coverage': coverage,
            'true_count': true_count,
            'n_group': n_group
        })
    
    # Overall metrics
    overall_bias = np.mean([m['bias'] for m in group_metrics])
    overall_rmse = np.mean([m['rmse'] for m in group_metrics])
    coverage_rate = np.mean([m['coverage'] for m in group_metrics])
    
    # Print results
    print("\nGroup-level Performance:")
    metrics_df = pd.DataFrame(group_metrics)
    print(metrics_df[['group', 'true_prevalence', 'predicted_prevalence', 'bias', 'rmse', 'coverage']])
    
    print(f"\nOverall Performance:")
    print(f"Mean bias: {overall_bias:.4f}")
    print(f"Mean RMSE: {overall_rmse:.4f}")
    print(f"Coverage rate: {coverage_rate:.2%}")
    
    return {
        'group_metrics': metrics_df,
        'overall_bias': overall_bias,
        'overall_rmse': overall_rmse,
        'coverage_rate': coverage_rate
    }


def explain_ppc():
    """Explain posterior predictive checks"""
    explanation = """
Posterior Predictive Checks (PPC) Explanation:

What are PPCs?
-------------
Posterior Predictive Checks assess whether a model can generate data that resembles the observed data.
They help validate if our model is capturing the important features of the data-generating process.

How they work:
-------------
1. We generate simulated data using parameters drawn from the posterior distribution
2. We compare these simulations to the actual observed data
3. If the observed data looks like it could have been generated by our model, the model is adequate

What we're checking:
------------------
- Group counts: For each group, we check if the observed count of positive cases falls within
  the 95% credible interval of counts generated from our posterior samples
- P-values: Two-sided probability of observing a value as extreme as the actual observation
  under our model (values near 0 or 1 suggest poor fit)

How to interpret:
---------------
- All observed values within 95% CI: Model is consistent with data
- P-values > 0.05: No significant discrepancy between model and data
- Visual assessment: Observed values should be near the center of posterior predictive distributions

Why this matters:
---------------
Good posterior predictive checks give us confidence that:
1. Our model can reproduce the key features of the data
2. Our group prevalence estimates are reliable
3. The model's uncertainty estimates are well-calibrated

In a Dawid-Skene model, PPCs help confirm that our estimates of annotator reliability and
group prevalences are generating realistic patterns of counts.
"""
    print(explanation)
    return explanation


if __name__ == "__main__":
    main()