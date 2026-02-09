#!/usr/bin/env python3
"""
Simple diagnostic to compare company-level aggregates in the company database
with the actual individual profile-level data used in the probit bootstrap.

This script uses cached Hamilton outputs to avoid re-running the full pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_cached_data():
    """Load cached data from Hamilton outputs."""
    base_path = Path("outputs")
    
    # Load company database
    company_db_path = "raw_data_search/2025-08-05_systematic_search_all.xlsx"
    company_db = pd.read_excel(company_db_path, sheet_name="ML Consultancies_merged_with_co")
    
    # Extract linkedin_id from URLs
    company_db['linkedin_id'] = company_db['LinkedIn'].str.extract(r'/company/([^/]+)$')
    
    # Load test data (from cached CSV if available, otherwise need to run pipeline)
    test_data_path = base_path / "dawid_skene_test_data.csv"
    if test_data_path.exists():
        test_data = pd.read_csv(test_data_path)
    else:
        print(f"Test data not found at {test_data_path}")
        print("Please run the pipeline first to generate cached data.")
        return None, None
    
    return company_db, test_data


def compute_actual_aggregates(test_data: pd.DataFrame) -> pd.DataFrame:
    """Compute company-level aggregates from test data."""
    # Sum up all positive annotations per company (treating NaN as 0 for sum)
    company_agg = test_data.groupby('company_id').agg({
        'llm_gemini_2_5_flash': lambda x: x.fillna(0).sum(),
        'llm_sonnet_4': lambda x: x.fillna(0).sum(),
        'llm_gpt_5_mini': lambda x: x.fillna(0).sum(),
        'filter_broad_yes': lambda x: x.fillna(0).sum(),
        'filter_strict_no': lambda x: x.fillna(0).sum(),
        'filter_broad_yes_strict_no': lambda x: x.fillna(0).sum(),
    }).reset_index()
    
    company_agg = company_agg.rename(columns={
        'llm_gemini_2_5_flash': 'gemini_total_accepted_actual',
        'llm_sonnet_4': 'claude_total_accepted_actual',
        'llm_gpt_5_mini': 'gpt5_total_accepted_actual',
    })
    
    company_agg['total_profiles_actual'] = test_data.groupby('company_id').size().values
    
    return company_agg


def main():
    """Compare company-level aggregates with actual test data."""
    print("Loading data...")
    company_db, test_data = load_cached_data()
    
    if company_db is None or test_data is None:
        return
    
    print(f"Company database: {len(company_db)} companies")
    print(f"Test data: {len(test_data)} profiles across {test_data['company_id'].nunique()} companies")
    
    # Compute actual aggregates from test data
    print("\nComputing actual aggregates from test data...")
    actual_agg = compute_actual_aggregates(test_data)
    
    # Merge with company database
    print("\nMerging with company database...")
    comparison = company_db[['linkedin_id', 'Organization Name', 
                              'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted',
                              'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no',
                              'Total Headcount']].copy()
    
    comparison = comparison.merge(
        actual_agg,
        left_on='linkedin_id',
        right_on='company_id',
        how='left',
        suffixes=('_db', '_actual')
    )
    
    # Compute discrepancies
    comparison['gemini_diff'] = comparison['gemini_total_accepted'].fillna(0) - comparison['gemini_total_accepted_actual'].fillna(0)
    comparison['claude_diff'] = comparison['claude_total_accepted'].fillna(0) - comparison['claude_total_accepted_actual'].fillna(0)
    comparison['gpt5_diff'] = comparison['gpt5_total_accepted'].fillna(0) - comparison['gpt5_total_accepted_actual'].fillna(0)
    
    # Find companies with large discrepancies
    large_discrepancies = comparison[
        ((comparison['gemini_diff'] > 10) | 
         (comparison['claude_diff'] > 10) | 
         (comparison['gpt5_diff'] > 10)) &
        (comparison['gemini_total_accepted_actual'].notna())
    ].sort_values('gemini_diff', ascending=False)
    
    # Find companies in DB but not in test data
    companies_in_db_not_in_test = comparison[
        (comparison['gemini_total_accepted_actual'].isna()) &
        (comparison['gemini_total_accepted'].notna()) &
        (comparison['gemini_total_accepted'] > 0)
    ].sort_values('gemini_total_accepted', ascending=False)
    
    # Save results
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_dir / "company_aggregates_comparison.csv", index=False)
    companies_in_db_not_in_test.to_csv(output_dir / "companies_in_db_not_in_test.csv", index=False)
    large_discrepancies.to_csv(output_dir / "large_discrepancies.csv", index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total companies in database: {len(company_db)}")
    print(f"Companies in test data: {len(actual_agg)}")
    print(f"Companies in DB but not in test (with positives in DB): {len(companies_in_db_not_in_test)}")
    print(f"Companies with large discrepancies (>10): {len(large_discrepancies)}")
    
    if len(companies_in_db_not_in_test) > 0:
        print("\nTop 10 companies in DB but not in test (by gemini_total_accepted):")
        cols = ['Organization Name', 'linkedin_id', 'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted']
        print(companies_in_db_not_in_test[cols].head(10).to_string(index=False))
    
    if len(large_discrepancies) > 0:
        print("\nTop 10 companies with large discrepancies:")
        cols = ['Organization Name', 'linkedin_id', 
                'gemini_total_accepted', 'gemini_total_accepted_actual', 'gemini_diff',
                'claude_total_accepted', 'claude_total_accepted_actual', 'claude_diff',
                'gpt5_total_accepted', 'gpt5_total_accepted_actual', 'gpt5_diff']
        print(large_discrepancies[cols].head(10).to_string(index=False))
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()





