"""
Hamilton plotting functions for ML Headcount Pipeline.

This module contains Hamilton-specific plotting functions that wrap the base
plotting functions for use in the Hamilton pipeline.
"""

from hamilton.function_modifiers import check_output, source, cache, save_to
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# LOG-DEBIASING PLOTS
# ============================================================================

@check_output(importance="fail")
def log_debias_plots(
    log_debias_results: Dict[str, pd.DataFrame],
    output_dir: str
) -> str:
    """
    Generate plots for log-debiasing results.
    
    Args:
        log_debias_results: Dictionary containing debiased results and filtered subsets
        output_dir: Directory to save plots
        
    Returns:
        Path to the saved plot directory
    """
    from .scripts.visualization.estimate_comparison import plot_debiased_ml_estimates_comparison
    import matplotlib.pyplot as plt
    
    logger.info("Generating log-debiasing plots...")
    
    # Create plots directory if it doesn't exist
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots for each dataset
    for dataset_name, df in log_debias_results.items():
        if len(df) == 0:
            logger.warning(f"No data available for {dataset_name}, skipping plot")
            continue
            
        logger.info(f"Generating plot for {dataset_name} with {len(df)} companies")
        
        # Filter out organizations with ml_consensus_round < 1 (matching probit plot behavior)
        before_filter = len(df)
        df = df.dropna(subset=['ml_consensus_round'])
        df_filtered = df[df['ml_consensus_round'] >= 1].copy()
        after_filter = len(df_filtered)
        logger.info(f"Filtered out {before_filter - after_filter} organizations with ml_consensus_round < 1")
        logger.info(f"Remaining organizations for plotting: {after_filter}")
        
        if after_filter == 0:
            logger.warning(f"No data available for {dataset_name} after filtering, skipping plot")
            continue
        
        # Create the comparison plot
        fig, ax, df_processed = plot_debiased_ml_estimates_comparison(
            df_filtered, 
            y_scale="log", 
            headcount_threshold=None
        )
        
        # Save the plot
        plot_filename = f"log_debias_{dataset_name}_comparison.png"
        plot_path = plots_dir / plot_filename
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved plot for {dataset_name} to {plot_path}")
    
    logger.info(f"Generated log-debiasing plots in {plots_dir}")
    return str(plots_dir)


@check_output(importance="fail")
def combined_estimates_comparison_plot(
    final_results_main_orgs: pd.DataFrame,
    log_debias_results: Dict[str, pd.DataFrame],
    dawid_skene_test_data: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    output_dir: str
) -> str:
    """
    Create a combined comparison plot showing both probit bootstrap and log-debias estimates.
    
    Uses probit estimates for filtering and sorting to ensure consistent comparison.
    Log-debias estimates are plotted below (lower z-order), probit above (higher z-order).
    
    Log-debias uses company-level aggregates from the company database (filter_broad_yes, etc.),
    NOT synthetic data. Log-debias works independently using these aggregates.
    
    Args:
        final_results_main_orgs: Final results with probit estimates and properly resolved company names
        log_debias_results: Dictionary containing debiased results (uses company-level aggregates)
        dawid_skene_test_data: Test data for individual annotator estimates
        company_database_complete: Company database with company-level aggregates
        output_dir: Directory to save plots
        
    Returns:
        Path to the saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger.info("Creating combined estimates comparison plot...")
    
    # Get the 'all_orgs' dataset from log-debias results
    log_debias_df = log_debias_results.get('all_orgs', pd.DataFrame())
    
    if len(log_debias_df) == 0:
        logger.warning("No log-debias data available, skipping plot")
        return ""
    
    if len(final_results_main_orgs) == 0:
        logger.warning("No probit bootstrap data available, skipping plot")
        return ""
    
    # Filter probit results to mean >= 1 (using real data first, synthetic as fallback)
    # Use final_results_main_orgs which has properly resolved company names
    # It has both 'mean' (real data) and 'synthetic_mean' (synthetic data)
    
    # Determine which companies should use synthetic probit estimates:
    # 1. Companies with no employee-level data
    companies_with_employee_data = set(dawid_skene_test_data['company_id'].unique())
    
    # 2. Companies where total_headcount and claude_total_accepted differ by more than 3x
    # Note: claude_total_accepted is Claude's estimate of ML employees, not total employees
    # We need to check if there's a column for Claude's total employee estimate
    # For now, we'll use claude_total_accepted as a proxy (though it's ML-specific)
    # If the column doesn't exist, we'll skip the headcount mismatch check
    company_db_cols = ['linkedin_id', 'total_headcount']
    if 'claude_total_accepted' in company_database_complete.columns:
        company_db_cols.append('claude_total_accepted')
    
    company_db_subset = company_database_complete[company_db_cols].copy()
    final_results_with_checks = final_results_main_orgs.merge(company_db_subset, on='linkedin_id', how='left')
    final_results_with_checks['has_employee_data'] = final_results_with_checks['linkedin_id'].isin(companies_with_employee_data)
    
    # Check headcount mismatch if we have the column
    if 'claude_total_accepted' in final_results_with_checks.columns and final_results_with_checks['claude_total_accepted'].notna().any():
        final_results_with_checks['headcount_ratio'] = final_results_with_checks['total_headcount'] / (final_results_with_checks['claude_total_accepted'] + 1e-6)
        final_results_with_checks['headcount_mismatch'] = (final_results_with_checks['headcount_ratio'] > 3) | (final_results_with_checks['headcount_ratio'] < 1/3)
    else:
        # No Claude total column or all NaN, so no headcount mismatch check
        final_results_with_checks['headcount_mismatch'] = False
    
    # Use synthetic probit estimates as fallback when:
    # - No employee-level data, OR
    # - total_headcount and claude_total_employees differ by more than 3x
    final_results_with_checks['use_synthetic_probit'] = (~final_results_with_checks['has_employee_data']) | final_results_with_checks['headcount_mismatch']
    
    # Create effective probit estimates: use real data if available, otherwise adjusted synthetic
    # Note: final_results_main_orgs has 'mean', 'q10', etc. (real data) and 'adjusted_synthetic_mean', etc. (adjusted synthetic)
    final_results_with_checks['probit_mean'] = final_results_with_checks['mean'].fillna(final_results_with_checks['adjusted_synthetic_mean'])
    final_results_with_checks['probit_q10'] = final_results_with_checks['q10'].fillna(final_results_with_checks['adjusted_synthetic_q10'])
    final_results_with_checks['probit_q50'] = final_results_with_checks['q50'].fillna(final_results_with_checks['adjusted_synthetic_q50'])
    final_results_with_checks['probit_q90'] = final_results_with_checks['q90'].fillna(final_results_with_checks['adjusted_synthetic_q90'])
    
    # Filter to companies with probit estimate >= 1
    probit_filtered = final_results_with_checks[
        final_results_with_checks['probit_mean'] >= 1
    ].copy()
    
    logger.info(f"Filtered probit results to {len(probit_filtered)} companies with probit_mean >= 1")
    logger.info(f"  Using real data: {(probit_filtered['mean'].notna()).sum()} companies")
    logger.info(f"  Using synthetic data: {(probit_filtered['use_synthetic_probit'] & probit_filtered['synthetic_mean'].notna()).sum()} companies")
    
    if len(probit_filtered) == 0:
        logger.warning("No probit companies with probit_mean >= 1, skipping plot")
        return ""
    
    # Merge on linkedin_id
    # Map log-debias company_id to linkedin_id for merging
    log_debias_df = log_debias_df.copy()
    if 'linkedin_id' not in log_debias_df.columns:
        # Try to get linkedin_id from company_id if they're the same
        log_debias_df['linkedin_id'] = log_debias_df.get('company_id', log_debias_df.index)
    
    # Identify available columns in dataframes
    probit_org_col = None
    if 'Organization Name' in probit_filtered.columns:
        probit_org_col = 'Organization Name'
    elif 'organization_name' in probit_filtered.columns:
        probit_org_col = 'organization_name'
    
    log_debias_org_col = None
    log_debias_cols_to_merge = ['linkedin_id', 'ml_consensus_round', 'ml_lower80_round', 'ml_upper80_round']
    if 'Organization Name' in log_debias_df.columns:
        log_debias_org_col = 'Organization Name'
        log_debias_cols_to_merge.append('Organization Name')
    elif 'organization_name' in log_debias_df.columns:
        log_debias_org_col = 'organization_name'
        log_debias_cols_to_merge.append('organization_name')
    
    # Merge the datasets
    merged_df = probit_filtered.merge(
        log_debias_df[log_debias_cols_to_merge],
        on='linkedin_id',
        how='left'
    )
    
    # Use effective probit estimates for plotting
    merged_df['mean'] = merged_df['probit_mean']
    merged_df['q10'] = merged_df['probit_q10']
    merged_df['q50'] = merged_df['probit_q50']
    merged_df['q90'] = merged_df['probit_q90']
    
    # Ensure Organization Name column exists (standardize to 'Organization Name')
    if 'Organization Name' not in merged_df.columns:
        if probit_org_col:
            merged_df['Organization Name'] = probit_filtered[probit_org_col].values
        elif log_debias_org_col:
            merged_df['Organization Name'] = merged_df[log_debias_org_col]
        elif 'organization_name' in merged_df.columns:
            merged_df['Organization Name'] = merged_df['organization_name']
        else:
            # Fallback: use linkedin_id if no name available
            merged_df['Organization Name'] = merged_df['linkedin_id']
    
    # Fill missing Organization Name values
    if merged_df['Organization Name'].isna().any():
        if probit_org_col:
            merged_df['Organization Name'] = merged_df['Organization Name'].fillna(probit_filtered[probit_org_col])
        elif log_debias_org_col and log_debias_org_col in merged_df.columns:
            merged_df['Organization Name'] = merged_df['Organization Name'].fillna(merged_df[log_debias_org_col])
        elif 'organization_name' in merged_df.columns:
            merged_df['Organization Name'] = merged_df['Organization Name'].fillna(merged_df['organization_name'])
    
    # Log-debias should use company-level aggregates from company_database_complete
    # NOT synthetic data. Log-debias works independently using filter_broad_yes, etc.
    # So we don't need any synthetic data fallback logic for log-debias
    
    # Sort by probit mean (ascending for reversed order)
    merged_df = merged_df.sort_values('mean', ascending=True).reset_index(drop=True)
    
    logger.info(f"Merged {len(merged_df)} companies for plotting")
    
    # Compute individual annotator estimates by company
    logger.info("Computing individual annotator estimates by company...")
    
    # Get annotator columns from test data
    annotator_cols = [col for col in dawid_skene_test_data.columns if col not in ['company_id', 'group']]
    logger.info(f"Found annotator columns: {annotator_cols}")
    
    # Compute annotator counts by company
    annotator_counts = {}
    for company_id in dawid_skene_test_data['company_id'].unique():
        company_mask = dawid_skene_test_data['company_id'] == company_id
        company_data = dawid_skene_test_data[company_mask]
        
        # Sum annotations for each annotator
        for col in annotator_cols:
            if col not in annotator_counts:
                annotator_counts[col] = {}
            annotator_counts[col][company_id] = company_data[col].sum()
    
    # Map annotator columns to the expected names
    annotator_mapping = {
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no',
        'llm_gemini_2_5_flash': 'gemini_total_accepted',
        'llm_sonnet_4': 'claude_total_accepted',
        'llm_gpt_5_mini': 'gpt5_total_accepted'
    }
    
    # Add individual annotator estimates to plot data
    for original_col, expected_col in annotator_mapping.items():
        if original_col in annotator_counts:
            merged_df[expected_col] = merged_df['linkedin_id'].map(annotator_counts[original_col]).fillna(0)
    
    # Create custom plot with color differentiation
    fig, ax = plt.subplots(figsize=(18, 9))
    x = np.arange(len(merged_df))
    
    # Define colors from Tol Bright palette
    probit_color = '#4477AA'  # Blue
    log_debias_color = '#EE6677'  # Red/Pink
    
    # Colors for individual annotators (keyword filters and LLMs)
    filter_colors = ['#4477AA', '#EE6677', '#228833']  # Different from main estimates
    llm_colors = ['#CCBB44', '#66CCEE', '#AA3377']
    
    # Plot individual annotator estimates first (lowest z-order)
    from .scripts.visualization.estimate_comparison import TOL_BRIGHT
    
    # Keyword filters - plot all 3 with labels
    filter_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    filter_labels = ['Filter Broad Yes', 'Filter Strict No', 'Filter Combined']
    filter_markers = ['o', 'v', 'X']
    for i, col in enumerate(filter_cols):
        if col in merged_df.columns:
            y = merged_df[col].values
            mask = (~np.isnan(y)) & (y > 0)
            x_pos = x[mask] + (i - len(filter_cols)/2) * 0.08
            ax.scatter(x_pos, y[mask], s=20, marker=filter_markers[i], c=filter_colors[i], 
                      alpha=0.4, label=filter_labels[i], zorder=1)
    
    # LLMs - plot all 3 with labels
    llm_cols = ['gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted']
    llm_labels = ['Gemini', 'Claude (Sonnet-4)', 'GPT-5-Mini']
    llm_markers = ['s', '^', 'D']
    for i, col in enumerate(llm_cols):
        if col in merged_df.columns:
            y = merged_df[col].values
            mask = (~np.isnan(y)) & (y > 0)
            x_pos = x[mask] + (i - len(llm_cols)/2) * 0.08
            ax.scatter(x_pos, y[mask], s=40, marker=llm_markers[i], c=llm_colors[i], 
                      edgecolors='black', linewidths=0.5, alpha=0.5, 
                      label=llm_labels[i], zorder=1)
    
    # Plot log-debias estimates (medium z-order, red)
    log_debias_present = merged_df['ml_consensus_round'].notna()
    if log_debias_present.sum() > 0:
        y_lower = merged_df[log_debias_present]['ml_lower80_round'].values
        y_upper = merged_df[log_debias_present]['ml_upper80_round'].values
        y_center = merged_df[log_debias_present]['ml_consensus_round'].values
        yerr_lower = y_center - y_lower
        yerr_upper = y_upper - y_center
        
        ax.errorbar(x[log_debias_present], y_center,
                   yerr=[yerr_lower, yerr_upper],
                   fmt='o', mfc='white', mec=log_debias_color, mew=1.8, ms=6,
                   ecolor=log_debias_color, elinewidth=1.2, capsize=2.5, capthick=1.2,
                   alpha=0.5,
                   label='Log-Debias Estimate (80% CI)', zorder=2)
    
    # Plot probit estimates on top (highest z-order, blue)
    y_lower_probit = merged_df['q10'].values
    y_upper_probit = merged_df['q90'].values
    y_center_probit = merged_df['mean'].values
    yerr_lower_p = y_center_probit - y_lower_probit
    yerr_upper_p = y_upper_probit - y_center_probit
    
    ax.errorbar(x, y_center_probit,
               yerr=[yerr_lower_p, yerr_upper_p],
               fmt='o', mfc='white', mec=probit_color, mew=1.8, ms=8,
               ecolor=probit_color, elinewidth=1.2, capsize=2.5, capthick=1.2,
               alpha=0.5,
               label='Probit Bootstrap Estimate (80% CI)', zorder=3)
    
    # Set y-axis to log scale with same limits as original plots
    ax.set_yscale('log')
    
    # Calculate max value including all CI upper bounds and annotator estimates
    all_values = []
    
    # Include annotator estimates
    for col in ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no', 
               'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted']:
        if col in merged_df.columns:
            all_values.extend(merged_df[col].dropna().tolist())
    
    # Include probit CI upper bounds
    if 'q90' in merged_df.columns:
        all_values.extend(merged_df['q90'].dropna().tolist())
    
    # Include log-debias CI upper bounds
    if 'ml_upper80_round' in merged_df.columns:
        all_values.extend(merged_df['ml_upper80_round'].dropna().tolist())
    
    if all_values:
        max_value = max(all_values)
        # Ensure minimum of 5e3, but use max value if larger
        y_max = max(5000, max_value * 1.1)
        ax.set_ylim(1, y_max)
        logger.info(f"Set Y-axis limits: 1 to {y_max:.1f} (max value: {max_value:.1f}, includes CI upper bounds)")
    else:
        ax.set_ylim(1, 5000)  # Default to 5e3 if no data
    
    # Labels and title
    # Get organization names, handling missing values properly
    if 'Organization Name' in merged_df.columns:
        org_names = merged_df['Organization Name'].fillna(merged_df.get('company_name', merged_df['linkedin_id'])).astype(str).tolist()
    elif 'company_name' in merged_df.columns:
        org_names = merged_df['company_name'].fillna(merged_df['linkedin_id']).astype(str).tolist()
    else:
        org_names = merged_df['linkedin_id'].astype(str).tolist()
    
    # Replace 'nan' strings with linkedin_id
    org_names = [name if name.lower() != 'nan' else str(merged_df.iloc[i]['linkedin_id']) for i, name in enumerate(org_names)]
    
    ax.set_xticks(x)
    ax.set_xticklabels(org_names, rotation=45, ha='right', fontsize=7)
    
    ax.set_xlabel('Organizations (sorted by probit bootstrap estimate)', fontsize=12)
    ax.set_ylabel('ML Headcount Estimate', fontsize=12)
    ax.set_title('Probit Bootstrap and Log-Debias Estimates Comparison\n(All organizations with probit estimate ≥ 1)', fontsize=14, pad=60)
    
    # Grid and styling
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(loc='upper left', fontsize=9, frameon=False, ncol=2)
    
    # Layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.85, right=0.98)
    
    # Save the plot
    plot_filename = "combined_estimates_comparison.png"
    plot_path = Path(output_dir) / "plots" / plot_filename
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved combined estimates plot to {plot_path}")
    return str(plot_path)


@check_output(importance="fail")
def log_debias_uncertainty_plots(
    log_debias_results: Dict[str, pd.DataFrame],
    output_dir: str
) -> str:
    """
    Generate uncertainty analysis plots for log-debiasing results.
    
    Args:
        log_debias_results: Dictionary containing debiased results and filtered subsets
        output_dir: Directory to save plots
        
    Returns:
        Path to the saved plot directory
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger.info("Generating log-debiasing uncertainty plots...")
    
    # Create plots directory if it doesn't exist
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Generate uncertainty plots for each dataset
    for dataset_name, df in log_debias_results.items():
        if len(df) == 0 or 'uncertainty_factor_x' not in df.columns:
            logger.warning(f"No uncertainty data available for {dataset_name}, skipping plot")
            continue
            
        logger.info(f"Generating uncertainty plot for {dataset_name} with {len(df)} companies")
        
        # Create uncertainty analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Uncertainty factor distribution
        uncertainty_factors = df['uncertainty_factor_x'].dropna()
        ax1.hist(uncertainty_factors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Uncertainty Factor (×/÷)')
        ax1.set_ylabel('Number of Companies')
        ax1.set_title(f'Uncertainty Factor Distribution\n{dataset_name.title()}')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_uncertainty = uncertainty_factors.mean()
        median_uncertainty = uncertainty_factors.median()
        ax1.axvline(mean_uncertainty, color='red', linestyle='--', label=f'Mean: {mean_uncertainty:.2f}')
        ax1.axvline(median_uncertainty, color='orange', linestyle='--', label=f'Median: {median_uncertainty:.2f}')
        ax1.legend()
        
        # Plot 2: ML consensus vs uncertainty
        if 'ml_consensus_round' in df.columns:
            ml_consensus = df['ml_consensus_round'].dropna()
            uncertainty = df['uncertainty_factor_x'].dropna()
            
            # Align the data
            common_idx = ml_consensus.index.intersection(uncertainty.index)
            if len(common_idx) > 0:
                scatter = ax2.scatter(ml_consensus.loc[common_idx], uncertainty.loc[common_idx], 
                                   alpha=0.6, s=50, color='green')
                ax2.set_xlabel('ML Consensus Count')
                ax2.set_ylabel('Uncertainty Factor (×/÷)')
                ax2.set_title(f'ML Consensus vs Uncertainty\n{dataset_name.title()}')
                ax2.set_xscale('log')
                ax2.grid(True, alpha=0.3)
                
                # Add correlation coefficient
                correlation = np.corrcoef(ml_consensus.loc[common_idx], uncertainty.loc[common_idx])[0, 1]
                ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"log_debias_{dataset_name}_uncertainty.png"
        plot_path = plots_dir / plot_filename
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved uncertainty plot for {dataset_name} to {plot_path}")
    
    logger.info(f"Generated log-debiasing uncertainty plots in {plots_dir}")
    return str(plots_dir)


# ============================================================================
# HELPER FUNCTIONS FOR REUSABLE PLOT LOGIC
# ============================================================================

def _create_uncertainty_plot(df: pd.DataFrame, pipeline_name: str, output_path: str) -> str:
    """Helper function to create uncertainty plots for any pipeline."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if len(df) == 0 or 'uncertainty_factor_x' not in df.columns:
        return output_path
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    uncertainty_factors = df['uncertainty_factor_x'].dropna()
    ax1.hist(uncertainty_factors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Uncertainty Factor (×/÷)')
    ax1.set_ylabel('Number of Companies')
    ax1.set_title(f'Uncertainty Factor Distribution\n{pipeline_name}')
    ax1.grid(True, alpha=0.3)
    
    mean_uncertainty = uncertainty_factors.mean()
    median_uncertainty = uncertainty_factors.median()
    ax1.axvline(mean_uncertainty, color='red', linestyle='--', label=f'Mean: {mean_uncertainty:.2f}')
    ax1.axvline(median_uncertainty, color='orange', linestyle='--', label=f'Median: {median_uncertainty:.2f}')
    ax1.legend()
    
    if 'ml_consensus_round' in df.columns:
        ml_consensus = df['ml_consensus_round'].dropna()
        uncertainty = df['uncertainty_factor_x'].dropna()
        common_idx = ml_consensus.index.intersection(uncertainty.index)
        if len(common_idx) > 0:
            ax2.scatter(ml_consensus.loc[common_idx], uncertainty.loc[common_idx], 
                       alpha=0.6, s=50, color='green')
            ax2.set_xlabel('ML Consensus Count')
            ax2.set_ylabel('Uncertainty Factor (×/÷)')
            ax2.set_title(f'ML Consensus vs Uncertainty\n{pipeline_name}')
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)
            correlation = np.corrcoef(ml_consensus.loc[common_idx], uncertainty.loc[common_idx])[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _create_combined_estimates_plot(
    probit_results: pd.DataFrame,
    log_debias_all_orgs: pd.DataFrame,
    dawid_skene_test_data: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    pipeline_name: str,
    output_path: str
) -> str:
    """Helper function to create combined estimates comparison plot for any pipeline."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plots_dir = Path(output_path).parent
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if len(log_debias_all_orgs) == 0 or len(probit_results) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data to plot', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Combined Estimates Comparison ({pipeline_name})')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    probit_filtered = probit_results[probit_results['mean'] >= 1].copy()
    
    if len(probit_filtered) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No probit companies with mean >= 1', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Combined Estimates Comparison ({pipeline_name})')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    log_debias_df = log_debias_all_orgs.copy()
    if 'linkedin_id' not in log_debias_df.columns:
        log_debias_df['linkedin_id'] = log_debias_df.get('company_id', log_debias_df.index)
    
    # Identify available columns in dataframes
    probit_org_col = None
    if 'Organization Name' in probit_filtered.columns:
        probit_org_col = 'Organization Name'
    elif 'organization_name' in probit_filtered.columns:
        probit_org_col = 'organization_name'
    
    log_debias_org_col = None
    log_debias_cols_to_merge = ['linkedin_id', 'ml_consensus_round', 'ml_lower80_round', 'ml_upper80_round']
    if 'Organization Name' in log_debias_df.columns:
        log_debias_org_col = 'Organization Name'
        log_debias_cols_to_merge.append('Organization Name')
    elif 'organization_name' in log_debias_df.columns:
        log_debias_org_col = 'organization_name'
        log_debias_cols_to_merge.append('organization_name')
    
    merged_df = probit_filtered.merge(
        log_debias_df[log_debias_cols_to_merge],
        on='linkedin_id',
        how='left'
    )
    
    # Ensure Organization Name column exists (standardize to 'Organization Name')
    if 'Organization Name' not in merged_df.columns:
        if probit_org_col:
            merged_df['Organization Name'] = probit_filtered[probit_org_col].values
        elif log_debias_org_col:
            merged_df['Organization Name'] = merged_df[log_debias_org_col]
        elif 'organization_name' in merged_df.columns:
            merged_df['Organization Name'] = merged_df['organization_name']
        else:
            # Fallback: use linkedin_id if no name available
            merged_df['Organization Name'] = merged_df['linkedin_id']
    
    # Fill missing Organization Name values
    if merged_df['Organization Name'].isna().any():
        if probit_org_col:
            merged_df['Organization Name'] = merged_df['Organization Name'].fillna(probit_filtered[probit_org_col])
        elif log_debias_org_col and log_debias_org_col in merged_df.columns:
            merged_df['Organization Name'] = merged_df['Organization Name'].fillna(merged_df[log_debias_org_col])
        elif 'organization_name' in merged_df.columns:
            merged_df['Organization Name'] = merged_df['Organization Name'].fillna(merged_df['organization_name'])
    
    # Log-debias uses company-level aggregates from company_database_complete
    # NOT synthetic data. Log-debias works independently using filter_broad_yes, etc.
    # So we don't need any synthetic data fallback logic here
    
    merged_df = merged_df.sort_values('mean', ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(range(len(merged_df)), merged_df['mean'], label='Probit Mean', alpha=0.6)
    if 'ml_consensus_round' in merged_df.columns:
        ax.scatter(range(len(merged_df)), merged_df['ml_consensus_round'], label='Log-Debias Consensus', alpha=0.6)
    ax.set_xlabel('Company Index')
    ax.set_ylabel('ML Headcount Estimate')
    ax.set_title(f'Combined Estimates Comparison ({pipeline_name})')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _create_log_debias_plot(
    log_debias_results: Dict[str, pd.DataFrame],
    output_path: str,
    pipeline_name: str
) -> str:
    """Helper function to create log-debiasing plots for any pipeline."""
    from .scripts.visualization.estimate_comparison import plot_debiased_ml_estimates_comparison
    import matplotlib.pyplot as plt
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, df in log_debias_results.items():
        if len(df) == 0:
            continue
        
        df = df.dropna(subset=['ml_consensus_round'])
        df_filtered = df[df['ml_consensus_round'] >= 1].copy()
        
        if len(df_filtered) == 0:
            continue
        
        fig, ax, df_processed = plot_debiased_ml_estimates_comparison(
            df_filtered, 
            y_scale="log", 
            headcount_threshold=None
        )
        
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        break  # Only save first dataset
    
    return output_path


# Main pipeline log-debias plot functions
@check_output(importance="fail")
def log_debias_plots_main(
    log_debias_results_main: Dict[str, pd.DataFrame],
    log_debias_plots_main_output_path: str
) -> str:
    """Generate log-debiasing plots for main pipeline."""
    logger.info("Generating log-debiasing plots for main pipeline...")
    return _create_log_debias_plot(log_debias_results_main, log_debias_plots_main_output_path, "Main Pipeline")


@check_output(importance="fail")
def log_debias_uncertainty_plots_main(
    log_debias_results_main: Dict[str, pd.DataFrame],
    log_debias_uncertainty_plots_main_output_path: str
) -> str:
    """Generate uncertainty plots for main pipeline."""
    logger.info("Generating uncertainty plots for main pipeline...")
    
    for dataset_name, df in log_debias_results_main.items():
        if len(df) > 0:
            return _create_uncertainty_plot(df, "Main Pipeline", log_debias_uncertainty_plots_main_output_path)
    
    return log_debias_uncertainty_plots_main_output_path


@check_output(importance="fail")
def combined_estimates_comparison_plot_main(
    final_results_main_orgs: pd.DataFrame,
    log_debias_results_main: Dict[str, pd.DataFrame],
    dawid_skene_test_data_main: pd.DataFrame,
    company_database_complete: pd.DataFrame,
    combined_estimates_comparison_plot_main_output_path: str
) -> str:
    """Create combined comparison plot for main pipeline."""
    logger.info("Creating combined estimates comparison plot for main pipeline...")
    
    all_orgs = log_debias_results_main.get('all_orgs', pd.DataFrame())
    return _create_combined_estimates_plot(
        final_results_main_orgs,
        all_orgs,
        dawid_skene_test_data_main,
        company_database_complete,
        "Main Pipeline",
        combined_estimates_comparison_plot_main_output_path
    )


# Comparator ML pipeline log-debias plot functions
@check_output(importance="fail")
def log_debias_plots_comparator_ml(
    log_debias_results_comparator_ml: Dict[str, pd.DataFrame],
    log_debias_plots_comparator_ml_output_path: str
) -> str:
    """Generate log-debiasing plots for comparator ML pipeline."""
    logger.info("Generating log-debiasing plots for comparator ML pipeline...")
    return _create_log_debias_plot(log_debias_results_comparator_ml, log_debias_plots_comparator_ml_output_path, "Comparator ML Pipeline")


@check_output(importance="fail")
def log_debias_uncertainty_plots_comparator_ml(
    log_debias_results_comparator_ml: Dict[str, pd.DataFrame],
    log_debias_uncertainty_plots_comparator_ml_output_path: str
) -> str:
    """Generate uncertainty plots for comparator ML pipeline."""
    logger.info("Generating uncertainty plots for comparator ML pipeline...")
    
    for dataset_name, df in log_debias_results_comparator_ml.items():
        if len(df) > 0:
            return _create_uncertainty_plot(df, "Comparator ML Pipeline", log_debias_uncertainty_plots_comparator_ml_output_path)
    
    return log_debias_uncertainty_plots_comparator_ml_output_path


@check_output(importance="fail")
def combined_estimates_comparison_plot_comparator_ml(
    correlated_probit_bootstrap_results_comparator_ml: pd.DataFrame,
    log_debias_results_comparator_ml: Dict[str, pd.DataFrame],
    dawid_skene_test_data_comparator_ml: pd.DataFrame,
    company_database_comparator_ml: pd.DataFrame,
    combined_estimates_comparison_plot_comparator_ml_output_path: str
) -> str:
    """Create combined comparison plot for comparator ML pipeline."""
    logger.info("Creating combined estimates comparison plot for comparator ML pipeline...")
    
    all_orgs = log_debias_results_comparator_ml.get('all_orgs', pd.DataFrame())
    return _create_combined_estimates_plot(
        correlated_probit_bootstrap_results_comparator_ml,
        all_orgs,
        dawid_skene_test_data_comparator_ml,
        company_database_comparator_ml,
        "Comparator ML Pipeline",
        combined_estimates_comparison_plot_comparator_ml_output_path
    )


# Comparator Non-ML pipeline log-debias plot functions
@check_output(importance="fail")
def log_debias_plots_comparator_non_ml(
    log_debias_results_comparator_non_ml: Dict[str, pd.DataFrame],
    log_debias_plots_comparator_non_ml_output_path: str
) -> str:
    """Generate log-debiasing plots for comparator Non-ML pipeline."""
    logger.info("Generating log-debiasing plots for comparator Non-ML pipeline...")
    return _create_log_debias_plot(log_debias_results_comparator_non_ml, log_debias_plots_comparator_non_ml_output_path, "Comparator Non-ML Pipeline")


@check_output(importance="fail")
def log_debias_uncertainty_plots_comparator_non_ml(
    log_debias_results_comparator_non_ml: Dict[str, pd.DataFrame],
    log_debias_uncertainty_plots_comparator_non_ml_output_path: str
) -> str:
    """Generate uncertainty plots for comparator Non-ML pipeline."""
    logger.info("Generating uncertainty plots for comparator Non-ML pipeline...")
    
    for dataset_name, df in log_debias_results_comparator_non_ml.items():
        if len(df) > 0:
            return _create_uncertainty_plot(df, "Comparator Non-ML Pipeline", log_debias_uncertainty_plots_comparator_non_ml_output_path)
    
    return log_debias_uncertainty_plots_comparator_non_ml_output_path


@check_output(importance="fail")
def combined_estimates_comparison_plot_comparator_non_ml(
    correlated_probit_bootstrap_results_comparator_non_ml: pd.DataFrame,
    log_debias_results_comparator_non_ml: Dict[str, pd.DataFrame],
    dawid_skene_test_data_comparator_non_ml: pd.DataFrame,
    company_database_comparator_non_ml: pd.DataFrame,
    combined_estimates_comparison_plot_comparator_non_ml_output_path: str
) -> str:
    """Create combined comparison plot for comparator Non-ML pipeline."""
    logger.info("Creating combined estimates comparison plot for comparator Non-ML pipeline...")
    
    all_orgs = log_debias_results_comparator_non_ml.get('all_orgs', pd.DataFrame())
    return _create_combined_estimates_plot(
        correlated_probit_bootstrap_results_comparator_non_ml,
        all_orgs,
        dawid_skene_test_data_comparator_non_ml,
        company_database_comparator_non_ml,
        "Comparator Non-ML Pipeline",
        combined_estimates_comparison_plot_comparator_non_ml_output_path
    )
