#!/usr/bin/env python3
"""
Diagnostic script to compare company-level aggregates in the company database
with the actual individual profile-level data used in the probit bootstrap.

This helps identify discrepancies where company-level aggregates show many positives
but the probit bootstrap estimates are zero, suggesting missing or misused individual data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import ml_headcount modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_headcount.hamilton_dataloaders import (
    dawid_skene_annotation_data,
    dawid_skene_test_data__real,
    dawid_skene_filtered_dataset,
    company_database_complete,
    dawid_skene_linkedin_profiles,
    dawid_skene_keyword_filters,
    dawid_skene_llm_results
)
from ml_headcount.hamilton_pipeline import MLHeadcountPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_actual_aggregates_from_test_data(test_data: pd.DataFrame) -> pd.DataFrame:
    """Compute company-level aggregates from the actual test data."""
    annotator_cols = [
        'llm_gemini_2_5_flash', 'llm_sonnet_4', 'llm_gpt_5_mini',
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    
    # Group by company and compute sums (treating NaN as 0 for counting)
    company_agg = test_data.groupby('company_id').agg({
        'llm_gemini_2_5_flash': ['sum', 'count'],
        'llm_sonnet_4': ['sum', 'count'],
        'llm_gpt_5_mini': ['sum', 'count'],
        'filter_broad_yes': ['sum', 'count'],
        'filter_strict_no': ['sum', 'count'],
        'filter_broad_yes_strict_no': ['sum', 'count'],
    })
    
    # Flatten column names
    company_agg.columns = ['_'.join(col).strip() for col in company_agg.columns.values]
    
    # Rename to match company database column names
    company_agg = company_agg.rename(columns={
        'llm_gemini_2_5_flash_sum': 'gemini_total_accepted_actual',
        'llm_sonnet_4_sum': 'claude_total_accepted_actual',
        'llm_gpt_5_mini_sum': 'gpt5_total_accepted_actual',
        'filter_broad_yes_sum': 'filter_broad_yes_actual',
        'filter_strict_no_sum': 'filter_strict_no_actual',
        'filter_broad_yes_strict_no_sum': 'filter_broad_yes_strict_no_actual',
        'llm_gemini_2_5_flash_count': 'gemini_count_actual',
        'llm_sonnet_4_count': 'claude_count_actual',
        'llm_gpt_5_mini_count': 'gpt5_count_actual',
    })
    
    company_agg = company_agg.reset_index()
    company_agg['total_profiles_actual'] = test_data.groupby('company_id').size().values
    
    return company_agg


def compute_aggregates_from_annotation_data(annotation_data: pd.DataFrame) -> pd.DataFrame:
    """Compute company-level aggregates from the full annotation data (before filtering)."""
    annotator_cols = [
        'llm_gemini-2.5-flash', 'llm_sonnet-4', 'llm_gpt-5-mini',
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    
    # Only use columns that exist
    available_cols = [col for col in annotator_cols if col in annotation_data.columns]
    
    if not available_cols:
        logger.warning("No annotator columns found in annotation data")
        return pd.DataFrame()
    
    # Group by company and compute sums (treating NaN as 0 for counting)
    agg_dict = {}
    for col in available_cols:
        agg_dict[col] = ['sum', 'count']
    
    company_agg = annotation_data.groupby('company_id').agg(agg_dict)
    company_agg.columns = ['_'.join(col).strip() for col in company_agg.columns.values]
    
    # Rename to match company database column names
    rename_map = {
        'llm_gemini-2.5-flash_sum': 'gemini_total_accepted_full',
        'llm_sonnet-4_sum': 'claude_total_accepted_full',
        'llm_gpt-5-mini_sum': 'gpt5_total_accepted_full',
        'llm_gemini-2.5-flash_count': 'gemini_count_full',
        'llm_sonnet-4_count': 'claude_count_full',
        'llm_gpt-5-mini_count': 'gpt5_count_full',
    }
    
    company_agg = company_agg.rename(columns=rename_map)
    company_agg = company_agg.reset_index()
    company_agg['total_profiles_full'] = annotation_data.groupby('company_id').size().values
    
    return company_agg


def main():
    """Compare company-level aggregates with actual test data."""
    logger.info("Loading data for comparison...")
    
    # Initialize pipeline to get config
    pipeline = MLHeadcountPipeline(config_path="config/default.yaml")
    config = pipeline.config
    
    # Load company database
    logger.info("Loading company database...")
    company_db = company_database_complete(
        df=pd.read_excel(
            config['data_paths']['company_database_complete_file_path'],
            sheet_name=config['data_paths']['company_database_complete_sheet_name']
        )
    )
    
    # Load annotation data (before filtering)
    logger.info("Loading annotation data...")
    annotation_data = dawid_skene_annotation_data(
        dawid_skene_linkedin_profiles=dawid_skene_linkedin_profiles(
            dawid_skene_linkedin_profiles_raw_file_path=config['data_paths']['dawid_skene_linkedin_profiles_raw_file_path'],
            dawid_skene_linkedin_profiles_big_consulting_file_path=config['data_paths']['dawid_skene_linkedin_profiles_big_consulting_file_path'],
            dawid_skene_linkedin_profiles_comparator_file_path=config['data_paths']['dawid_skene_linkedin_profiles_comparator_file_path'],
            enable_85k_profiles=config.get('enable_85k_profiles', True),
            enable_big_consulting=config.get('enable_big_consulting', True),
            enable_comparator=config.get('enable_comparator', True)
        ),
        dawid_skene_keyword_filters=dawid_skene_keyword_filters(
            dawid_skene_keyword_filters_raw_file_path=config['data_paths']['dawid_skene_keyword_filters_raw_file_path'],
            dawid_skene_keyword_filters_big_consulting_file_path=config['data_paths']['dawid_skene_keyword_filters_big_consulting_file_path'],
            dawid_skene_keyword_filters_comparator_file_path=config['data_paths']['dawid_skene_keyword_filters_comparator_file_path'],
            enable_85k_profiles=config.get('enable_85k_profiles', True),
            enable_big_consulting=config.get('enable_big_consulting', True),
            enable_comparator=config.get('enable_comparator', True)
        ),
        dawid_skene_llm_results=dawid_skene_llm_results(
            dawid_skene_llm_results_dir_file_path=config['data_paths']['dawid_skene_llm_results_dir_file_path'],
            dawid_skene_llm_results_dir_big_consulting_file_path=config['data_paths']['dawid_skene_llm_results_dir_big_consulting_file_path'],
            dawid_skene_llm_results_dir_comparator_file_path=config['data_paths']['dawid_skene_llm_results_dir_comparator_file_path'],
            enable_85k_profiles=config.get('enable_85k_profiles', True),
            enable_big_consulting=config.get('enable_big_consulting', True),
            enable_comparator=config.get('enable_comparator', True)
        )
    )
    
    # Load filtered dataset
    logger.info("Loading filtered dataset...")
    filtered_dataset = dawid_skene_filtered_dataset(
        dawid_skene_annotation_data=annotation_data,
        company_database_complete=company_db,
        ds_max_items=config.get('dawid_skene', {}).get('max_items', 500000),
        ds_max_companies=config.get('dawid_skene', {}).get('max_companies', 500),
        ds_min_profiles=config.get('dawid_skene', {}).get('min_profiles', 0)
    )
    
    # Load test data
    logger.info("Loading test data...")
    test_data = dawid_skene_test_data__real(dawid_skene_filtered_dataset=filtered_dataset)
    
    # Compute aggregates from full annotation data
    logger.info("Computing aggregates from full annotation data...")
    full_aggregates = compute_aggregates_from_annotation_data(annotation_data)
    
    # Compute aggregates from test data
    logger.info("Computing aggregates from test data...")
    test_aggregates = compute_actual_aggregates_from_test_data(test_data)
    
    # Merge with company database
    logger.info("Merging with company database...")
    comparison = company_db[['linkedin_id', 'organization_name', 
                              'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted',
                              'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no',
                              'total_headcount']].copy()
    
    # Merge with full aggregates
    if not full_aggregates.empty:
        comparison = comparison.merge(
            full_aggregates,
            left_on='linkedin_id',
            right_on='company_id',
            how='left',
            suffixes=('_db', '_full')
        )
    
    # Merge with test aggregates
    comparison = comparison.merge(
        test_aggregates,
        left_on='linkedin_id',
        right_on='company_id',
        how='left',
        suffixes=('', '_test')
    )
    
    # Compute discrepancies
    logger.info("Computing discrepancies...")
    
    # Companies in database but not in test data
    companies_in_db_not_in_test = comparison[
        (comparison['gemini_total_accepted_actual'].isna()) &
        (comparison['gemini_total_accepted'].notna()) &
        (comparison['gemini_total_accepted'] > 0)
    ]
    
    # Companies with large discrepancies
    comparison['gemini_diff'] = comparison['gemini_total_accepted'] - comparison['gemini_total_accepted_actual'].fillna(0)
    comparison['claude_diff'] = comparison['claude_total_accepted'] - comparison['claude_total_accepted_actual'].fillna(0)
    comparison['gpt5_diff'] = comparison['gpt5_total_accepted'] - comparison['gpt5_total_accepted_actual'].fillna(0)
    
    large_discrepancies = comparison[
        ((comparison['gemini_diff'] > 10) | 
         (comparison['claude_diff'] > 10) | 
         (comparison['gpt5_diff'] > 10)) &
        (comparison['gemini_total_accepted_actual'].notna())
    ]
    
    # Save results
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_dir / "company_aggregates_comparison.csv", index=False)
    companies_in_db_not_in_test.to_csv(output_dir / "companies_in_db_not_in_test.csv", index=False)
    large_discrepancies.to_csv(output_dir / "large_discrepancies.csv", index=False)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total companies in database: {len(company_db)}")
    logger.info(f"Companies in test data: {len(test_aggregates)}")
    logger.info(f"Companies in DB but not in test (with positives in DB): {len(companies_in_db_not_in_test)}")
    logger.info(f"Companies with large discrepancies (>10): {len(large_discrepancies)}")
    
    if len(companies_in_db_not_in_test) > 0:
        logger.info("\nTop 10 companies in DB but not in test (by gemini_total_accepted):")
        top_missing = companies_in_db_not_in_test.nlargest(10, 'gemini_total_accepted')[
            ['organization_name', 'linkedin_id', 'gemini_total_accepted', 'claude_total_accepted', 'gpt5_total_accepted']
        ]
        print(top_missing.to_string())
    
    if len(large_discrepancies) > 0:
        logger.info("\nTop 10 companies with large discrepancies:")
        top_discrep = large_discrepancies.nlargest(10, 'gemini_diff')[
            ['organization_name', 'linkedin_id', 
             'gemini_total_accepted', 'gemini_total_accepted_actual', 'gemini_diff',
             'claude_total_accepted', 'claude_total_accepted_actual', 'claude_diff',
             'gpt5_total_accepted', 'gpt5_total_accepted_actual', 'gpt5_diff']
        ]
        print(top_discrep.to_string())
    
    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()





