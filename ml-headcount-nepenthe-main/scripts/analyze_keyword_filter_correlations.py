#!/usr/bin/env python3
"""
Analyze correlation structure using only keyword filter annotators (3 filters).

This script:
1. Loads correlation matrices
2. Extracts only keyword filter annotators (3x3 submatrix)
3. Recomputes distances and statistics
4. Compares to full 6-annotator analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 80)
    print("KEYWORD FILTER CORRELATION ANALYSIS (3 annotators only)")
    print("=" * 80)
    print()
    
    # Load validation matrix
    val_matrix = pd.read_csv('outputs/diagnostics/validation_covariance_matrix.csv', index_col=0)
    
    # Identify keyword filter annotators
    keyword_annotators = [a for a in val_matrix.index if 'filter' in a.lower()]
    print(f"Keyword filter annotators: {keyword_annotators}")
    print()
    
    # Extract 3x3 submatrix for keyword filters
    val_matrix_filters = val_matrix.loc[keyword_annotators, keyword_annotators]
    print("Validation correlation matrix (keyword filters only):")
    print(val_matrix_filters.round(3))
    print()
    
    # Load company matrices
    company_df = pd.read_csv('outputs/diagnostics/company_covariance_matrices.csv')
    
    # Extract keyword filter correlations
    filter_corr_cols = []
    for i, annot_i in enumerate(keyword_annotators):
        for j, annot_j in enumerate(keyword_annotators):
            col_name = f"{annot_i}_{annot_j}"
            if col_name in company_df.columns:
                filter_corr_cols.append(col_name)
    
    print(f"Found {len(filter_corr_cols)} keyword filter correlation pairs")
    print()
    
    # Compute statistics
    print("=" * 80)
    print("VALIDATION STATISTICS (keyword filters only)")
    print("=" * 80)
    
    val_upper = np.triu(val_matrix_filters.values, k=1)
    val_corrs = val_upper[val_upper != 0]
    
    print(f"Mean correlation: {val_corrs.mean():.4f}")
    print(f"Std correlation: {val_corrs.std():.4f}")
    print(f"Min correlation: {val_corrs.min():.4f}")
    print(f"Max correlation: {val_corrs.max():.4f}")
    print()
    
    # Company statistics
    print("=" * 80)
    print("COMPANY STATISTICS (keyword filters only)")
    print("=" * 80)
    
    company_filter_corrs = company_df[filter_corr_cols].values.flatten()
    company_filter_corrs = company_filter_corrs[~np.isnan(company_filter_corrs)]
    
    print(f"Mean correlation: {company_filter_corrs.mean():.4f}")
    print(f"Std correlation: {company_filter_corrs.std():.4f}")
    print(f"Min correlation: {company_filter_corrs.min():.4f}")
    print(f"Max correlation: {company_filter_corrs.max():.4f}")
    print()
    
    # Compute distances
    print("=" * 80)
    print("DISTANCE ANALYSIS (keyword filters only)")
    print("=" * 80)
    
    # Flatten validation matrix (upper triangle)
    val_corrs_flat = val_corrs
    
    # Compute distances for each company
    distances = []
    company_ids = []
    
    for idx, row in company_df.iterrows():
        company_corrs = row[filter_corr_cols].values.astype(float)
        if np.isnan(company_corrs).any():
            continue
        
        # Compute distance (only use the 3 unique pairs from upper triangle)
        # filter_corr_cols has 9 elements (3x3 matrix), but we only need 3 unique pairs
        # Extract upper triangle indices: (0,1), (0,2), (1,2)
        upper_triangle_indices = [0, 1, 3]  # Indices for (0,1), (0,2), (1,2) in flattened 3x3
        company_corrs_upper = company_corrs[upper_triangle_indices]
        
        dist = np.abs(company_corrs_upper - val_corrs_flat).mean()
        distances.append(dist)
        company_ids.append(row['company_id'])
    
    distances = np.array(distances)
    
    print(f"Mean distance: {distances.mean():.4f}")
    print(f"Std distance: {distances.std():.4f}")
    print(f"Min distance: {distances.min():.4f}")
    print(f"Max distance: {distances.max():.4f}")
    print(f"Median distance: {np.median(distances):.4f}")
    print()
    
    # Compare to full analysis
    print("=" * 80)
    print("COMPARISON: KEYWORD FILTERS vs ALL ANNOTATORS")
    print("=" * 80)
    
    # Load full analysis summary
    summary = pd.read_csv('outputs/diagnostics/latent_cov_diagnostic_summary.csv')
    
    full_val_mean = summary[summary['metric'] == 'mean_correlation']['value'].iloc[0]
    full_company_mean = summary[summary['metric'] == 'mean_correlation']['value'].iloc[1]
    full_mean_dist = summary[summary['metric'] == 'mean_distance_to_validation']['value'].iloc[0]
    
    print(f"{'Metric':<40} {'All Annotators':<20} {'Filters Only':<20} {'Difference':<20}")
    print("-" * 100)
    print(f"{'Validation mean correlation':<40} {full_val_mean:<20.4f} {val_corrs.mean():<20.4f} {val_corrs.mean() - full_val_mean:<20.4f}")
    print(f"{'Company mean correlation':<40} {full_company_mean:<20.4f} {company_filter_corrs.mean():<20.4f} {company_filter_corrs.mean() - full_company_mean:<20.4f}")
    print(f"{'Mean distance to validation':<40} {full_mean_dist:<20.4f} {distances.mean():<20.4f} {distances.mean() - full_mean_dist:<20.4f}")
    print()
    
    # Save results
    results = pd.DataFrame({
        'company_id': company_ids,
        'distance_keyword_filters': distances
    })
    
    output_path = Path('outputs/diagnostics/keyword_filter_correlation_analysis.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")
    print()
    
    # Save validation matrix
    val_output_path = Path('outputs/diagnostics/validation_covariance_matrix_keyword_filters.csv')
    val_matrix_filters.to_csv(val_output_path)
    print(f"Saved validation matrix to: {val_output_path}")
    print()
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'metric': [
            'validation_mean_correlation',
            'validation_std_correlation',
            'validation_min_correlation',
            'validation_max_correlation',
            'company_mean_correlation',
            'company_std_correlation',
            'company_min_correlation',
            'company_max_correlation',
            'mean_distance',
            'std_distance',
            'min_distance',
            'max_distance',
            'median_distance'
        ],
        'value': [
            val_corrs.mean(),
            val_corrs.std(),
            val_corrs.min(),
            val_corrs.max(),
            company_filter_corrs.mean(),
            company_filter_corrs.std(),
            company_filter_corrs.min(),
            company_filter_corrs.max(),
            distances.mean(),
            distances.std(),
            distances.min(),
            distances.max(),
            np.median(distances)
        ]
    })
    
    summary_path = Path('outputs/diagnostics/keyword_filter_correlation_summary.csv')
    summary_stats.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to: {summary_path}")
    print()

if __name__ == '__main__':
    main()
