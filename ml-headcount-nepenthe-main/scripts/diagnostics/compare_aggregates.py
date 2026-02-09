#!/usr/bin/env python3
"""
Compare company-level aggregates in the company database with actual test data aggregates.

The company database has pre-computed aggregates (gpt5_total_accepted, etc.) that may
come from a different source or include more profiles than what's actually used in the
probit bootstrap pipeline.

The log_debias_company_aggregates.csv contains aggregates computed from the actual
test data used in the bootstrap (via .sum() on the test data).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml_headcount.hamilton_dataloaders import company_database_complete, INPUT_DATA_CONFIG
from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline

def main():
    """Compare company database aggregates with test data aggregates."""
    
    # Load config directly
    import yaml
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load company database
    print("Loading company database...")
    data_dir = Path(config['data_paths']['data_dir'])
    company_db_path = data_dir / INPUT_DATA_CONFIG['company_database_complete']['file_path']
    
    company_db = company_database_complete(
        df=pd.read_excel(
            company_db_path,
            sheet_name=INPUT_DATA_CONFIG['company_database_complete']['sheet_name']
        )
    )
    
    print(f"  Companies in database: {len(company_db)}")
    print(f"  Companies with gemini_total_accepted: {company_db['gemini_total_accepted'].notna().sum()}")
    
    # Load log-debias aggregates (computed from actual test data)
    print("\nLoading log-debias aggregates (from test data)...")
    log_debias_path = Path("outputs/log_debias_company_aggregates.csv")
    if not log_debias_path.exists():
        print(f"ERROR: {log_debias_path} not found. Please run the pipeline first.")
        return
    
    log_debias = pd.read_csv(log_debias_path)
    print(f"  Companies in test data: {len(log_debias)}")
    
    # Merge and compare
    print("\nMerging and comparing...")
    merged = company_db[['linkedin_id', 'organization_name', 
                         'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted',
                         'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no',
                         'total_headcount']].merge(
        log_debias[['company_id', 'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted',
                    'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']],
        left_on='linkedin_id',
        right_on='company_id',
        how='outer',
        suffixes=('_db', '_actual')
    )
    
    # Compute differences
    merged['gemini_diff'] = merged['gemini_total_accepted_db'].fillna(0) - merged['gemini_total_accepted_actual'].fillna(0)
    merged['claude_diff'] = merged['claude_total_accepted_db'].fillna(0) - merged['claude_total_accepted_actual'].fillna(0)
    merged['gpt5_diff'] = merged['gpt5_total_accepted_db'].fillna(0) - merged['gpt5_total_accepted_actual'].fillna(0)
    
    # Find large discrepancies (DB > actual by >10)
    large_diff = merged[
        ((merged['gemini_diff'] > 10) | (merged['claude_diff'] > 10) | (merged['gpt5_diff'] > 10)) &
        (merged['gemini_total_accepted_actual'].notna())
    ].sort_values('gemini_diff', ascending=False)
    
    # Companies in DB but not in test data (with positives in DB)
    missing = merged[
        (merged['gemini_total_accepted_actual'].isna()) &
        (merged['gemini_total_accepted_db'].notna()) &
        (merged['gemini_total_accepted_db'] > 0)
    ].sort_values('gemini_total_accepted_db', ascending=False)
    
    # Companies in test data but not in DB
    extra = merged[
        (merged['gemini_total_accepted_db'].isna()) &
        (merged['gemini_total_accepted_actual'].notna()) &
        (merged['gemini_total_accepted_actual'] > 0)
    ].sort_values('gemini_total_accepted_actual', ascending=False)
    
    # Save results
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged.to_csv(output_dir / "company_aggregates_comparison.csv", index=False)
    large_diff.to_csv(output_dir / "large_discrepancies.csv", index=False)
    missing.to_csv(output_dir / "companies_in_db_not_in_test.csv", index=False)
    extra.to_csv(output_dir / "companies_in_test_not_in_db.csv", index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total companies in database: {len(company_db)}")
    print(f"Companies in test data: {len(log_debias)}")
    print(f"Companies with large discrepancies (>10): {len(large_diff)}")
    print(f"Companies in DB but not in test (with positives): {len(missing)}")
    print(f"Companies in test but not in DB (with positives): {len(extra)}")
    
    if len(large_diff) > 0:
        print("\n" + "="*80)
        print("TOP 20 COMPANIES WITH LARGE DISCREPANCIES (DB > Actual by >10)")
        print("="*80)
        cols = ['organization_name', 'linkedin_id', 
                'gemini_total_accepted_db', 'gemini_total_accepted_actual', 'gemini_diff',
                'claude_total_accepted_db', 'claude_total_accepted_actual', 'claude_diff',
                'gpt5_total_accepted_db', 'gpt5_total_accepted_actual', 'gpt5_diff',
                'total_headcount']
        print(large_diff[cols].head(20).to_string(index=False))
    
    if len(missing) > 0:
        print("\n" + "="*80)
        print("TOP 20 COMPANIES IN DB BUT NOT IN TEST DATA (with positives in DB)")
        print("="*80)
        cols = ['organization_name', 'linkedin_id', 
                'gemini_total_accepted_db', 'claude_total_accepted_db', 'gpt5_total_accepted_db',
                'total_headcount']
        print(missing[cols].head(20).to_string(index=False))
    
    if len(extra) > 0:
        print("\n" + "="*80)
        print("TOP 20 COMPANIES IN TEST BUT NOT IN DB (with positives in test)")
        print("="*80)
        cols = ['company_id', 'organization_name',
                'gemini_total_accepted_actual', 'claude_total_accepted_actual', 'gpt5_total_accepted_actual']
        print(extra[cols].head(20).to_string(index=False))
    
    print(f"\n\nResults saved to {output_dir}/")
    print("\nKey findings:")
    print("1. Large discrepancies suggest the DB aggregates include profiles not in test data")
    print("2. Missing companies may be filtered out due to:")
    print("   - No size data (total_headcount missing)")
    print("   - Too few profiles (< min_profiles)")
    print("   - Not in top max_companies")
    print("   - Random sampling if max_items exceeded")
    print("3. Check filtering logic in dawid_skene_filtered_dataset()")


if __name__ == "__main__":
    main()

