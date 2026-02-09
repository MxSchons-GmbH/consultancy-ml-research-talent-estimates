"""
Hamilton dataloaders for ML Headcount Pipeline

This module contains all @dataloader functions for loading and validating
input data files using Pandera schemas.
"""

from hamilton.function_modifiers import dataloader, load_from, save_to, source, check_output, cache, config
from pandera.typing import DataFrame
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import json
import logging
import os
from pathlib import Path

from .schemas import (
    CVDataSchema, ValidationCVsSchema, AffiliationDataSchema, 
    CompanyDatabaseBaseSchema, CompanyDatabaseCompleteSchema, LinkedInDataSchema,
    CompanySizeDataSchema, DawidSkeneCompressedDataSchema, DawidSkeneValidationDataSchema,
    DawidSkeneTestDataSchema, TitlesDataSchema
)
import numpy as np
from scipy.stats import truncnorm

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS FOR REUSABLE LOGIC
# ============================================================================

def _process_keyword_filter_file(keyword_file: Path) -> pd.DataFrame:
    """Helper function to process a single keyword filter CSV file."""
    df = pd.read_csv(keyword_file)
    df = df.rename(columns={'id': 'public_identifier'})
    
    # Compute compound filters
    df['filter_broad_yes'] = ((df['ml_match'] == 1) & (df['broad_match'] == 1)).astype(int)
    df['filter_strict_no'] = ((df['ml_match'] == 1) & (df['strict_no_match'] == 1)).astype(int)
    df['filter_broad_yes_strict_no'] = ((df['ml_match'] == 1) & (df['broad_match'] == 1) & (df['strict_no_match'] == 1)).astype(int)
    
    return df[['public_identifier', 'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']]


# ============================================================================
# INPUT DATA CONFIGURATION
# ============================================================================

# Define input data files with their paths relative to data_dir
# These will be expanded to full paths in hamilton_pipeline.py
INPUT_DATA_CONFIG = {
    "cv_data": {
        "file_path": "raw_data_cvs/2025-08-05_CV_rated.tsv"
    },
    "validation_cvs": {
        "file_path": "raw_data_cvs/Validation Set/2025-08-06_validation_cvs_rated.xlsx",
        "sheet_name": "Validation full"
    },
    "affiliation_data": {
        "file_path": "raw_data_search/2025-08-04_arxiv_kaggle_all_affiliations_cleaned.csv"
    },
    "company_database": {
        "file_path": "raw_data_cvs/2025-08-07 included_companies.tsv"
    },
    "company_database_complete": {
        "file_path": "raw_data_search/2025-08-05_systematic_search_all.xlsx",
        "sheet_name": "ML Consultancies_merged_with_co"
    },
    "company_database_comparator_ml": {
        "file_path": "comparators/2025-08-15_compartor_analysis.xlsx",
        "sheet_name": "ML ORGS"
    },
    "company_database_comparator_non_ml": {
        "file_path": "comparators/2025-08-15_compartor_analysis.xlsx",
        "sheet_name": "Non ML ORGs"
    },
    "keyword_filter_data": {
        "file_path": "outputs_keyword_search_CVs/cv_keywords_FIXED_frequency_analysis (1).xlsx",
        "sheet_name": "cv_keywords_FIXED_frequency_ana"
    },
    "dawid_skene_linkedin_profiles_raw": {
        "file_path": "input profiles/2025-08-07 profiles.jsonl.gz"
    },
    "dawid_skene_linkedin_profiles_big_consulting": {
        "file_path": "input profiles/2025-08-14-big-consulting.jsonl.gz"
    },
    "dawid_skene_linkedin_profiles_comparator": {
        "file_path": "input profiles/2025-08-15-comparator.jsonl.gz"
    },
    "dawid_skene_keyword_filters_raw": {
        "file_path": "outputs_keyword_search_CVs/keyword_filter_output/2025-08-07 profiles_filtered.csv"
    },
    "dawid_skene_keyword_filters_big_consulting": {
        "file_path": "outputs_keyword_search_CVs/keyword_filter_output/2025-08-14-big-consulting_filtered.csv"
    },
    "dawid_skene_keyword_filters_comparator": {
        "file_path": "outputs_keyword_search_CVs/keyword_filter_output/2025-08-15-comparator_filtered.csv"
    },
    "dawid_skene_llm_results_dir": {
        "file_path": "results_85k_evaluations"
    },
    "dawid_skene_llm_results_dir_big_consulting": {
        "file_path": "results_49k_evaluation_big_companies"
    },
    "dawid_skene_llm_results_dir_comparator": {
        "file_path": "results_115k_comparator"
    }
}

# Define output data files with their paths relative to output_dir
OUTPUT_DATA_CONFIG = {
    # Text analysis output paths
    "preprocessed_text_output_path": "preprocessed_text.csv",
    "cv_keyword_frequencies_output_path": "cv_keyword_frequencies.csv",
    "keyword_extraction_results_output_path": "keyword_extraction_results.csv",
    "clustering_results_output_path": "clustering_results.json",
    "discriminative_keywords_output_path": "discriminative_keywords.csv",
    "keyword_lists_output_path": "keyword_lists.csv",
    "keyword_extraction_report_output_path": "keyword_extraction_report.json",
    "confusion_matrix_metrics_output_path": "confusion_matrix_metrics.csv",
    "filtered_keywords_list_output_path": "filtered_keywords_list.json",
    "keyword_extraction_results_filtered_output_path": "keyword_extraction_results_filtered.csv",
    
    # Data cleaning output paths
    "non_academic_affiliations_output_path": "non_academic_affiliations.csv",
    "cleaned_affiliations_output_path": "cleaned_affiliations.csv",
    
    # Additional output paths for JSON transformations
    "cv_keyword_clusters_path": "cv_keyword_clusters.json",
    "keywords_path": "keywords.json",
    "keyword_extraction_report_path": "keyword_extraction_report.json",
    
    # Dawid-Skene output paths
    "dawid_skene_linkedin_profiles_output_path": "dawid_skene_linkedin_profiles.csv",
    "dawid_skene_linkedin_profiles_main_output_path": "dawid_skene_linkedin_profiles_main.csv",
    "dawid_skene_linkedin_profiles_comparator_output_path": "dawid_skene_linkedin_profiles_comparator.csv",
    "dawid_skene_keyword_filters_output_path": "dawid_skene_keyword_filters.csv",
    "dawid_skene_keyword_filters_main_output_path": "dawid_skene_keyword_filters_main.csv",
    "dawid_skene_keyword_filters_comparator_output_path": "dawid_skene_keyword_filters_comparator.csv",
    "dawid_skene_annotation_data_output_path": "dawid_skene_annotation_data.csv",
    "dawid_skene_annotation_data_main_output_path": "dawid_skene_annotation_data_main.csv",
    "dawid_skene_annotation_data_comparator_ml_output_path": "dawid_skene_annotation_data_comparator_ml.csv",
    "dawid_skene_annotation_data_comparator_non_ml_output_path": "dawid_skene_annotation_data_comparator_non_ml.csv",
    "dawid_skene_validation_data_output_path": "dawid_skene_validation_data.csv",
    "dawid_skene_filtered_dataset_output_path": "dawid_skene_filtered_dataset.csv",
    "dawid_skene_filtered_dataset_main_output_path": "dawid_skene_filtered_dataset_main.csv",
    "dawid_skene_filtered_dataset_comparator_ml_output_path": "dawid_skene_filtered_dataset_comparator_ml.csv",
    "dawid_skene_filtered_dataset_comparator_non_ml_output_path": "dawid_skene_filtered_dataset_comparator_non_ml.csv",
    "dawid_skene_test_data_output_path": "dawid_skene_test_data.csv",
    "dawid_skene_test_data_main_output_path": "dawid_skene_test_data_main.csv",
    "dawid_skene_test_data_comparator_ml_output_path": "dawid_skene_test_data_comparator_ml.csv",
    "dawid_skene_test_data_comparator_non_ml_output_path": "dawid_skene_test_data_comparator_non_ml.csv",
    
    # Log-debiasing output paths
    "log_debias_company_aggregates_output_path": "log_debias_company_aggregates.csv",
    "log_debias_company_aggregates_main_output_path": "log_debias_company_aggregates_main.csv",
    "log_debias_company_aggregates_comparator_ml_output_path": "log_debias_company_aggregates_comparator_ml.csv",
    "log_debias_company_aggregates_comparator_non_ml_output_path": "log_debias_company_aggregates_comparator_non_ml.csv",
    "log_debias_summary_output_path": "log_debias_summary.json",
    "log_debias_plots_output_path": "log_debias_plots.png",
    "log_debias_plots_main_output_path": "plots/log_debias_plots_main.png",
    "log_debias_plots_comparator_ml_output_path": "plots/log_debias_plots_comparator_ml.png",
    "log_debias_plots_comparator_non_ml_output_path": "plots/log_debias_plots_comparator_non_ml.png",
    "log_debias_uncertainty_plots_output_path": "log_debias_uncertainty_plots.png",
    "log_debias_uncertainty_plots_main_output_path": "plots/log_debias_uncertainty_plots_main.png",
    "log_debias_uncertainty_plots_comparator_ml_output_path": "plots/log_debias_uncertainty_plots_comparator_ml.png",
    "log_debias_uncertainty_plots_comparator_non_ml_output_path": "plots/log_debias_uncertainty_plots_comparator_non_ml.png",
    "combined_estimates_comparison_plot_output_path": "combined_estimates_comparison.png",
    "combined_estimates_comparison_plot_main_output_path": "plots/combined_estimates_comparison_plot_main.png",
    "combined_estimates_comparison_plot_comparator_ml_output_path": "plots/combined_estimates_comparison_plot_comparator_ml.png",
    "combined_estimates_comparison_plot_comparator_non_ml_output_path": "plots/combined_estimates_comparison_plot_comparator_non_ml.png",
    "log_debias_all_orgs_output_path": "log_debias_all_orgs.csv",
    "log_debias_orgs_ml_output_path": "log_debias_orgs_ml.csv",
    "log_debias_orgs_talent_dense_output_path": "log_debias_orgs_talent_dense.csv",
    "log_debias_orgs_stage5_work_trial_output_path": "log_debias_orgs_stage5_work_trial.csv",
    "log_debias_orgs_enterprise_500ml_0p5pct_output_path": "log_debias_orgs_enterprise_500ml_0p5pct.csv",
    "log_debias_orgs_midscale_50ml_1pct_output_path": "log_debias_orgs_midscale_50ml_1pct.csv",
    "log_debias_orgs_boutique_10ml_5pct_output_path": "log_debias_orgs_boutique_10ml_5pct.csv",
    "log_debias_orgs_stage5_work_trial_recommended_output_path": "log_debias_orgs_stage5_work_trial_recommended.csv",
    "log_debias_results_main_orgs_output_path": "log_debias_results_main_orgs.csv",
    "log_debias_results_comparator_ml_orgs_output_path": "log_debias_results_comparator_ml_orgs.csv",
    "log_debias_results_comparator_non_ml_orgs_output_path": "log_debias_results_comparator_non_ml_orgs.csv",
    
    # Probit bootstrap output paths
    "correlated_probit_bootstrap_results_output_path": "correlated_probit_bootstrap_results.csv",
    "correlated_probit_bootstrap_results_synthetic_output_path": "correlated_probit_bootstrap_results_synthetic.csv",
    "correlated_probit_bootstrap_results_main_output_path": "correlated_probit_bootstrap_results_main.csv",
    "correlated_probit_bootstrap_results_comparator_ml_output_path": "correlated_probit_bootstrap_results_comparator_ml.csv",
    "correlated_probit_bootstrap_results_comparator_non_ml_output_path": "correlated_probit_bootstrap_results_comparator_non_ml.csv",
    "correlated_probit_bootstrap_distributions_output_path": "correlated_probit_bootstrap_distributions.json",
    "correlated_probit_bootstrap_plots_with_annotators_output_path": "correlated_probit_bootstrap_plots_with_annotators.png",
    "correlated_probit_bootstrap_plots_with_annotators_main_output_path": "plots/correlated_probit_bootstrap_plots_with_annotators_main.png",
    "correlated_probit_bootstrap_plots_with_annotators_comparator_ml_output_path": "plots/correlated_probit_bootstrap_plots_with_annotators_comparator_ml.png",
    "correlated_probit_bootstrap_plots_with_annotators_comparator_non_ml_output_path": "plots/correlated_probit_bootstrap_plots_with_annotators_comparator_non_ml.png",
    "probit_bootstrap_talent_landscape_plot_output_path": "probit_bootstrap_talent_landscape_plot.png",
    "probit_bootstrap_talent_landscape_plot_main_output_path": "plots/probit_bootstrap_talent_landscape_plot_main.png",
    "probit_bootstrap_talent_landscape_plot_comparator_ml_output_path": "plots/probit_bootstrap_talent_landscape_plot_comparator_ml.png",
    "probit_bootstrap_talent_landscape_plot_comparator_non_ml_output_path": "plots/probit_bootstrap_talent_landscape_plot_comparator_non_ml.png",
    "probit_bootstrap_per_organization_estimates_plot_output_path": "probit_bootstrap_per_organization_estimates_plot.png",
    "probit_bootstrap_per_organization_estimates_plot_main_output_path": "plots/probit_bootstrap_per_organization_estimates_plot_main.png",
    
    # Synthetic data generation output paths
    "tetrachoric_correlation_matrix_output_path": "tetrachoric_correlation_matrix.npy",
    "test_keyword_filter_correlation_matrix_output_path": "test_keyword_filter_correlation_matrix.npy",
    "companies_needing_synthetic_data_output_path": "companies_needing_synthetic_data.csv",
    "probit_bootstrap_per_organization_estimates_plot_comparator_ml_output_path": "plots/probit_bootstrap_per_organization_estimates_plot_comparator_ml.png",
    "probit_bootstrap_per_organization_estimates_plot_comparator_non_ml_output_path": "plots/probit_bootstrap_per_organization_estimates_plot_comparator_non_ml.png",
    "probit_bootstrap_prior_distribution_plot_output_path": "probit_bootstrap_prior_distribution_plot.png",
    "probit_bootstrap_prior_distribution_plot_main_output_path": "plots/probit_bootstrap_prior_distribution_plot_main.png",
    "probit_bootstrap_prior_distribution_plot_comparator_ml_output_path": "plots/probit_bootstrap_prior_distribution_plot_comparator_ml.png",
    "probit_bootstrap_prior_distribution_plot_comparator_non_ml_output_path": "plots/probit_bootstrap_prior_distribution_plot_comparator_non_ml.png",
    "keyword_annotator_prevalence_plot_output_path": "plots/keyword_annotator_prevalence_plot.png",
    "empirical_keyword_correlations_pairplot_output_path": "plots/empirical_keyword_correlations_pairplot.png",
    "real_vs_synthetic_scatter_plot_output_path": "plots/real_vs_synthetic_scatter_plot.png",
    "synthetic_real_ratio_distributions_plot_output_path": "plots/synthetic_real_ratio_distributions_plot.png",
    "probit_results_main_orgs_output_path": "probit_results_main_orgs.csv",
    "probit_results_comparator_ml_orgs_output_path": "probit_results_comparator_ml_orgs.csv",
    "probit_results_comparator_non_ml_orgs_output_path": "probit_results_comparator_non_ml_orgs.csv",
    "real_employee_level_data_all_output_path": "real_employee_level_data_all.csv",
    "synthetic_employee_level_data_all_output_path": "synthetic_employee_level_data_all.csv",
    "final_results_main_orgs_output_path": "final_results_main_orgs.csv",
    "final_results_core_orgs_output_path": "final_results_core_orgs.csv",
    "final_results_comparator_ml_output_path": "final_results_comparator_ml.csv",
    "final_results_comparator_non_ml_output_path": "final_results_comparator_non_ml.csv",
    "final_results_all_output_path": "final_results_all.csv",
    "annotator_metrics_output_path": "annotator_metrics.csv",
    "annotator_performance_sens_vs_spec_output_path": "annotator_performance_sens_vs_spec.png",
    "bias_vs_accuracy_output_path": "bias_vs_accuracy.png",
    "annotator_bias_analysis_output_path": "annotator_bias_analysis.png",
    "latent_covariance_parallel_plot_output_path": "diagnostics/latent_cov_parallel.png",
    "latent_covariance_parallel_coordinates_data_output_path": "diagnostics/latent_cov_parallel_data.csv",
    "latent_covariance_diagnostic_summary_output_path": "diagnostics/latent_cov_diagnostic_summary.csv",
    "validation_covariance_matrix_output_path": "diagnostics/validation_covariance_matrix.csv",
    "company_covariance_matrices_output_path": "diagnostics/company_covariance_matrices.csv",
    "empirical_correlations_by_class_output_path": "empirical_correlations_by_class.png",
    "real_vs_synthetic_comparison_plot_output_path": "plots/real_vs_synthetic_comparison.png",
    "real_vs_synthetic_per_company_plot_output_path": "plots/real_vs_synthetic_per_company.png"
}

# ============================================================================
# CORE DATALOADERS
# ============================================================================

@check_output(schema=CVDataSchema, importance="fail")
@load_from.csv(
    path=source('cv_data_file_path'),
    sep='\t'
)
def cv_data(df: pd.DataFrame) -> DataFrame[CVDataSchema]:
    # Clean up unnamed columns and map profile_summary to cv_text
    cleaned_data = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')])
    
    # Map profile_summary to cv_text for schema compatibility
    if 'profile_summary' in cleaned_data.columns and 'cv_text' not in cleaned_data.columns:
        cleaned_data = cleaned_data.rename(columns={'profile_summary': 'cv_text'})
    
    # Convert category to string format for schema compatibility
    if 'category' in cleaned_data.columns:
        cleaned_data['category'] = cleaned_data['category'].astype(str)
    
    return cleaned_data


@check_output(schema=ValidationCVsSchema, importance="fail")
@load_from.excel(
    path=source('validation_cvs_file_path'),
    sheet_name=source('validation_cvs_sheet_name')
)
def validation_cvs(df: pd.DataFrame) -> DataFrame[ValidationCVsSchema]:
    # Clean up unnamed columns
    cleaned_data = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')])
    return cleaned_data


@check_output(schema=AffiliationDataSchema, importance="fail")
@load_from.csv(
    path=source('affiliation_data_file_path')
)
def affiliation_data(df: pd.DataFrame) -> DataFrame[AffiliationDataSchema]:
    # Normalize column names to match schema (lowercase)
    df = df.rename(columns={'Affiliation': 'affiliation'})
    return df


@check_output(schema=CompanyDatabaseBaseSchema, importance="fail")
@load_from.csv(
    path=source('company_database_file_path'),
    sep='\t'
)
def company_database(df: pd.DataFrame) -> DataFrame[CompanyDatabaseBaseSchema]:
    # The included_companies.tsv file contains LinkedIn company IDs and employee counts
    # Map column names to match schema - these should be the actual column names in the TSV
    # We'll use the LinkedIn company ID as the id field
    if 'company_id' in df.columns:
        df['id'] = df['company_id']
    elif 'linkedin_id' in df.columns:
        df['id'] = df['linkedin_id']
    else:
        # If no clear ID column, use the first column as ID
        df['id'] = df.iloc[:, 0]
    
    # Map employee count column
    if 'employees_in_linkedin' in df.columns:
        df['employees'] = df['employees_in_linkedin']
    elif 'employees' in df.columns:
        df['employees'] = df['employees']
    else:
        # If no employee column, create a default
        df['employees'] = 100  # Default value
    
    # Use employees as total_headcount for now
    df['total_headcount'] = df['employees']
    
    # Create company_size based on total_headcount
    def categorize_size(headcount):
        if pd.isna(headcount) or headcount == 0:
            return 'Unknown'
        elif headcount < 10:
            return 'Startup'
        elif headcount < 50:
            return 'Small'
        elif headcount < 200:
            return 'Medium'
        elif headcount < 1000:
            return 'Large'
        else:
            return 'Enterprise'
    df['company_size'] = df['total_headcount'].apply(categorize_size)
    
    # Handle null values in required columns
    df['company_size'] = df['company_size'].fillna('Unknown')
    df['total_headcount'] = df['total_headcount'].fillna(0).astype('int64')
    df['employees'] = df['employees'].fillna(0).astype('int64')
    
    logger.info(f"Loaded company database: {len(df)} companies")
    return df


@check_output(schema=CompanyDatabaseCompleteSchema, importance="fail")
@load_from.excel(
    path=source('company_database_complete_file_path'),
    sheet_name=source('company_database_complete_sheet_name')
)
def company_database_complete(df: pd.DataFrame) -> DataFrame[CompanyDatabaseCompleteSchema]:
    # Map column names to match schema
    column_mapping = {
        'ID': 'id',
        'Organization Name': 'organization_name',
        'LinkedIn': 'linkedin_id',
        'Headquarters Location': 'headquarters_location',
        'country': 'country',
        'subregion': 'subregion',
        'Stage Reached': 'stage_reached',
        'category': 'category',
        'Total Headcount': 'total_headcount',
        'Founded Date': 'founded_date',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no',
        'claude_total_accepted': 'claude_total_accepted',
        'gpt5_total_accepted': 'gpt5_total_accepted',
        'gemini_total_accepted': 'gemini_total_accepted',
        'ml_share': 'ml_share',
        'max_headcount': 'max_headcount',
        'max_population': 'max_population',
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Convert id column to integer for schema validation
    if 'id' in df.columns:
        df['id'] = df['id'].astype(int)
    
    # Keep original column names for scripts compatibility
    if 'headquarters_location' in df.columns:
        df['Headquarters Location'] = df['headquarters_location']
    if 'total_headcount' in df.columns:
        df['Total Headcount'] = df['total_headcount']
    if 'organization_name' in df.columns:
        df['Organization Name'] = df['organization_name']
    if 'stage_reached' in df.columns:
        df['Stage Reached'] = df['stage_reached']
    if 'founded_date' in df.columns:
        df['Founded Date'] = df['founded_date']
    # Keep max_headcount and max_population as boolean columns
    if 'max_headcount' in df.columns:
        # Ensure boolean type
        df['max_headcount'] = df['max_headcount'].astype(bool)
    if 'max_population' in df.columns:
        # Ensure boolean type
        df['max_population'] = df['max_population'].astype(bool)
    
    # Add missing columns with default values
    if 'country' not in df.columns:
        df['country'] = 'Unknown'
    if 'subregion' not in df.columns:
        df['subregion'] = 'Unknown'
    if 'ml_share' not in df.columns:
        df['ml_share'] = 0.0
    if 'ml_share_lower80' not in df.columns:
        df['ml_share_lower80'] = 0.0
    if 'ml_share_upper80' not in df.columns:
        df['ml_share_upper80'] = 0.0
    if 'max_headcount' not in df.columns:
        df['max_headcount'] = False
    if 'max_population' not in df.columns:
        df['max_population'] = False
    
    # Convert ML estimation columns to float64 to match schema
    ml_columns = ['claude_total_accepted', 'gpt5_total_accepted', 'gemini_total_accepted']
    for col in ml_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
        else:
            # Add missing columns with float64 dtype
            df[col] = pd.Series([None] * len(df), dtype='float64')
    
    # Convert filter columns to float64 to prevent Int64 issues in downstream processing
    filter_columns = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    for col in filter_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # Handle null values in string columns - only fillna if column is string/object type
    string_columns = ['headquarters_location', 'country', 'subregion', 'stage_reached', 'category']
    for col in string_columns:
        if col in df.columns:
            # Only fillna if column is string/object type, not numeric
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].fillna('Unknown')
            else:
                # For numeric columns, convert to string first
                df[col] = df[col].astype(str).fillna('Unknown')
    
    # Drop founded_date column if it exists (not needed for pipeline)
    if 'founded_date' in df.columns:
        df = df.drop(columns=['founded_date'])
    
    # Extract company ID from LinkedIn URLs for matching with annotation data
    # LinkedIn URLs are like: https://www.linkedin.com/company/company-name
    if 'linkedin_id' in df.columns:
        df['linkedin_id'] = df['linkedin_id'].str.extract(r'/company/([^/]+)$')
        logger.info(f"Extracted LinkedIn company IDs from URLs")
    
    # Fill NaN linkedin_id with id column as fallback for consistent identifiers
    # This ensures companies without LinkedIn URLs can still be matched in joins
    if 'linkedin_id' in df.columns and 'id' in df.columns:
        nan_mask = df['linkedin_id'].isna()
        if nan_mask.any():
            df.loc[nan_mask, 'linkedin_id'] = df.loc[nan_mask, 'id'].astype(str)
            logger.info(f"Filled {nan_mask.sum()} NaN linkedin_id values with id column")
    
    return df


@check_output(schema=CompanyDatabaseCompleteSchema, importance="fail")
@load_from.excel(
    path=source('company_database_comparator_ml_file_path'),
    sheet_name=source('company_database_comparator_ml_sheet_name')
)
def company_database_comparator_ml(df: pd.DataFrame) -> DataFrame[CompanyDatabaseCompleteSchema]:
    """
    Load comparator ML organizations company database.
    
    Uses the same schema as company_database_complete but handles different column names
    and LinkedIn URL patterns (both /company/ and /showcase/).
    """
    # Map column names to match schema
    column_mapping = {
        'ID': 'id',
        'Organization Name': 'organization_name',
        'LinkedIn': 'linkedin_id',
        'Headquarters Location': 'headquarters_location',
        'Stage Reached': 'stage_reached',
        'category': 'category',
        'Total Headcount': 'total_headcount',
        'Founded Date': 'founded_date',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no',
        'claude_total_accepted': 'claude_total_accepted',
        'gpt5_total_accepted': 'gpt5_total_accepted',
        'gemini_total_accepted': 'gemini_total_accepted',
        'max_headcount_more_than_ten': 'max_headcount',  # Map different column name
        'max_population': 'max_population',
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Convert id column - handle string IDs like "comparator_1"
    if 'id' in df.columns:
        # Try to convert to int, but keep as string if it fails (for comparator IDs)
        try:
            df['id'] = df['id'].astype(int)
        except (ValueError, TypeError):
            # For comparator IDs like "comparator_1", create numeric IDs
            df['id'] = pd.factorize(df['id'])[0] + 1
    
    # Keep original column names for scripts compatibility
    if 'headquarters_location' in df.columns:
        df['Headquarters Location'] = df['headquarters_location']
    if 'total_headcount' in df.columns:
        df['Total Headcount'] = df['total_headcount']
    if 'organization_name' in df.columns:
        df['Organization Name'] = df['organization_name']
    if 'stage_reached' in df.columns:
        df['Stage Reached'] = df['stage_reached']
    if 'founded_date' in df.columns:
        df['Founded Date'] = df['founded_date']
    
    # Keep max_headcount and max_population as boolean columns
    if 'max_headcount' in df.columns:
        df['max_headcount'] = df['max_headcount'].astype(bool)
    if 'max_population' in df.columns:
        df['max_population'] = df['max_population'].astype(bool)
    
    # Add missing columns with default values
    if 'country' not in df.columns:
        df['country'] = 'Unknown'
    if 'subregion' not in df.columns:
        df['subregion'] = 'Unknown'
    if 'ml_share' not in df.columns:
        df['ml_share'] = 0.0
    if 'ml_share_lower80' not in df.columns:
        df['ml_share_lower80'] = 0.0
    if 'ml_share_upper80' not in df.columns:
        df['ml_share_upper80'] = 0.0
    if 'max_headcount' not in df.columns:
        df['max_headcount'] = False
    if 'max_population' not in df.columns:
        df['max_population'] = False
    
    # Convert ML estimation columns to float64 to match schema
    ml_columns = ['claude_total_accepted', 'gpt5_total_accepted', 'gemini_total_accepted']
    for col in ml_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
        else:
            df[col] = pd.Series([None] * len(df), dtype='float64')
    
    # Convert filter columns to float64
    filter_columns = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    for col in filter_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # Handle null values in string columns - only fillna if column is string/object type
    string_columns = ['headquarters_location', 'country', 'subregion', 'stage_reached', 'category']
    for col in string_columns:
        if col in df.columns:
            # Only fillna if column is string/object type, not numeric
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].fillna('Unknown')
            else:
                # For numeric columns, convert to string first
                df[col] = df[col].astype(str).fillna('Unknown')
    
    # Drop founded_date column if it exists (not needed for pipeline)
    if 'founded_date' in df.columns:
        df = df.drop(columns=['founded_date'])
    
    # Extract company ID from LinkedIn URLs - handle both /company/ and /showcase/ patterns
    if 'linkedin_id' in df.columns:
        # Handle both /company/ and /showcase/ patterns
        df['linkedin_id'] = df['linkedin_id'].str.extract(r'/(?:company|showcase)/([^/]+)/?$')
        logger.info(f"Extracted LinkedIn company IDs from URLs (handling both /company/ and /showcase/ patterns)")
    
    logger.info(f"Loaded comparator ML company database: {len(df)} companies")
    return df


@check_output(schema=CompanyDatabaseCompleteSchema, importance="fail")
@load_from.excel(
    path=source('company_database_comparator_non_ml_file_path'),
    sheet_name=source('company_database_comparator_non_ml_sheet_name')
)
def company_database_comparator_non_ml(df: pd.DataFrame) -> DataFrame[CompanyDatabaseCompleteSchema]:
    """
    Load comparator Non-ML organizations company database.
    
    Uses the same schema as company_database_complete but handles different column names
    and LinkedIn URL patterns (both /company/ and /showcase/).
    """
    # Map column names to match schema (same as ML comparator)
    column_mapping = {
        'ID': 'id',
        'Organization Name': 'organization_name',
        'LinkedIn': 'linkedin_id',
        'Headquarters Location': 'headquarters_location',
        'Stage Reached': 'stage_reached',
        'category': 'category',
        'Total Headcount': 'total_headcount',
        'Founded Date': 'founded_date',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no',
        'claude_total_accepted': 'claude_total_accepted',
        'gpt5_total_accepted': 'gpt5_total_accepted',
        'gemini_total_accepted': 'gemini_total_accepted',
        'max_headcount_more_than_ten': 'max_headcount',  # Map different column name
        'max_population': 'max_population',
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Convert id column - handle string IDs like "comparator_21"
    if 'id' in df.columns:
        try:
            df['id'] = df['id'].astype(int)
        except (ValueError, TypeError):
            # For comparator IDs like "comparator_21", create numeric IDs
            df['id'] = pd.factorize(df['id'])[0] + 1
    
    # Keep original column names for scripts compatibility
    if 'headquarters_location' in df.columns:
        df['Headquarters Location'] = df['headquarters_location']
    if 'total_headcount' in df.columns:
        df['Total Headcount'] = df['total_headcount']
    if 'organization_name' in df.columns:
        df['Organization Name'] = df['organization_name']
    if 'stage_reached' in df.columns:
        df['Stage Reached'] = df['stage_reached']
    if 'founded_date' in df.columns:
        df['Founded Date'] = df['founded_date']
    
    # Keep max_headcount and max_population as boolean columns
    if 'max_headcount' in df.columns:
        df['max_headcount'] = df['max_headcount'].astype(bool)
    if 'max_population' in df.columns:
        df['max_population'] = df['max_population'].astype(bool)
    
    # Add missing columns with default values
    if 'country' not in df.columns:
        df['country'] = 'Unknown'
    if 'subregion' not in df.columns:
        df['subregion'] = 'Unknown'
    if 'ml_share' not in df.columns:
        df['ml_share'] = 0.0
    if 'ml_share_lower80' not in df.columns:
        df['ml_share_lower80'] = 0.0
    if 'ml_share_upper80' not in df.columns:
        df['ml_share_upper80'] = 0.0
    if 'max_headcount' not in df.columns:
        df['max_headcount'] = False
    if 'max_population' not in df.columns:
        df['max_population'] = False
    
    # Convert ML estimation columns to float64 to match schema
    ml_columns = ['claude_total_accepted', 'gpt5_total_accepted', 'gemini_total_accepted']
    for col in ml_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
        else:
            df[col] = pd.Series([None] * len(df), dtype='float64')
    
    # Convert filter columns to float64
    filter_columns = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    for col in filter_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # Handle null values in string columns - only fillna if column is string/object type
    string_columns = ['headquarters_location', 'country', 'subregion', 'stage_reached', 'category']
    for col in string_columns:
        if col in df.columns:
            # Only fillna if column is string/object type, not numeric
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].fillna('Unknown')
            else:
                # For numeric columns, convert to string first
                df[col] = df[col].astype(str).fillna('Unknown')
    
    # Drop founded_date column if it exists (not needed for pipeline)
    if 'founded_date' in df.columns:
        df = df.drop(columns=['founded_date'])
    
    # Extract company ID from LinkedIn URLs - handle both /company/ and /showcase/ patterns
    if 'linkedin_id' in df.columns:
        # Handle both /company/ and /showcase/ patterns
        df['linkedin_id'] = df['linkedin_id'].str.extract(r'/(?:company|showcase)/([^/]+)/?$')
        logger.info(f"Extracted LinkedIn company IDs from URLs (handling both /company/ and /showcase/ patterns)")
    
    logger.info(f"Loaded comparator Non-ML company database: {len(df)} companies")
    return df


@cache()
@check_output(schema=LinkedInDataSchema, importance="fail")
def linkedin_data(
    dawid_skene_linkedin_profiles_raw_file_path: str,
    dawid_skene_linkedin_profiles_big_consulting_file_path: str,
    dawid_skene_linkedin_profiles_comparator_file_path: str,
    enable_85k_profiles: bool,
    enable_big_consulting: bool,
    enable_comparator: bool
) -> DataFrame[LinkedInDataSchema]:
    """
    Load LinkedIn data from enabled datasets with Pandera validation.
    
    Loads and concatenates profiles from enabled datasets, removing duplicates.
    
    Args:
        dawid_skene_linkedin_profiles_raw_file_path: Path to 85k profiles file
        dawid_skene_linkedin_profiles_big_consulting_file_path: Path to big consulting profiles file
        dawid_skene_linkedin_profiles_comparator_file_path: Path to comparator profiles file
        enable_85k_profiles: Whether to include 85k profiles dataset
        enable_big_consulting: Whether to include big consulting profiles dataset
        enable_comparator: Whether to include comparator profiles dataset
        
    Returns:
        Validated LinkedIn data DataFrame (deduplicated)
    """
    logger.info("Loading LinkedIn data for linkedin subgraph...")
    
    # Define dataset configurations
    datasets = [
        {
            'name': '85k profiles',
            'path': dawid_skene_linkedin_profiles_raw_file_path,
            'enabled': enable_85k_profiles
        },
        {
            'name': 'big consulting',
            'path': dawid_skene_linkedin_profiles_big_consulting_file_path,
            'enabled': enable_big_consulting
        },
        {
            'name': 'comparator',
            'path': dawid_skene_linkedin_profiles_comparator_file_path,
            'enabled': enable_comparator
        }
    ]
    
    # Load raw data from enabled datasets
    import gzip
    all_raw_data = []
    seen_ids = set()
    
    for dataset_config in datasets:
        if not dataset_config['enabled']:
            logger.info(f"Skipping {dataset_config['name']} dataset (disabled)")
            continue
        
        profile_file = Path(dataset_config['path'])
        if not profile_file.exists():
            raise FileNotFoundError(f"LinkedIn profiles file not found: {profile_file}")
        
        logger.info(f"Loading {dataset_config['name']} from {profile_file}")
        dataset_count = 0
        duplicate_count = 0
        
        with gzip.open(profile_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    profile = json.loads(line.strip())
                    linkedin_id = profile.get('linkedin_id') or profile.get('id')
                    
                    # Skip duplicates
                    if linkedin_id in seen_ids:
                        duplicate_count += 1
                        continue
                    
                    all_raw_data.append(profile)
                    seen_ids.add(linkedin_id)
                    dataset_count += 1
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid profile in {dataset_config['name']}: {e}")
                    continue
        
        logger.info(f"Loaded {dataset_count} unique profiles from {dataset_config['name']} ({duplicate_count} duplicates filtered)")
    
    if not all_raw_data:
        raise ValueError("No LinkedIn profiles loaded. Please enable at least one dataset.")
    
    # Convert using existing function
    from .scripts.data_ingestion.linkedin_conversion import convert_linkedin_json_to_csv
    df = convert_linkedin_json_to_csv(all_raw_data)
    
    logger.info(f"Total: {len(df)} LinkedIn profiles (after deduplication)")
    
    return df


# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

# Output paths are now defined inline with @save_to decorators using source('output_dir')


@check_output(importance="fail")
@load_from.excel(
    path=source('keyword_filter_data_file_path'),
    sheet_name=source('keyword_filter_data_sheet_name')
)
def keyword_filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load keyword filter data from XLSX file.
    
    Args:
        df: DataFrame loaded from XLSX file with filter information
        
    Returns:
        DataFrame with keyword filter data including 'Filter out' column
    """
    logger.info(f"Processing keyword filter data: {len(df)} rows")
    
    try:
        # The DataFrame is already loaded by the @load_from decorator
        logger.info(f"Loaded keyword filter data: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to process keyword filter data: {str(e)}")
        raise

# All utility functions removed - paths are now defined inline

# ============================================================================
# DAWID-SKENE DATALOADERS
# ============================================================================

@save_to.csv(
    path=source('dawid_skene_linkedin_profiles_output_path')
)
@cache()
@check_output(schema=LinkedInDataSchema, importance="fail")
def dawid_skene_linkedin_profiles(
    dawid_skene_linkedin_profiles_raw_file_path: str,
    dawid_skene_linkedin_profiles_big_consulting_file_path: str,
    dawid_skene_linkedin_profiles_comparator_file_path: str,
    enable_85k_profiles: bool,
    enable_big_consulting: bool,
    enable_comparator: bool
) -> DataFrame[LinkedInDataSchema]:
    """
    Load LinkedIn profiles for Dawid-Skene analysis from enabled datasets.
    
    Loads and concatenates profiles from enabled datasets, removing duplicates
    based on linkedin_id.
    
    Args:
        dawid_skene_linkedin_profiles_raw_file_path: Path to 85k profiles file
        dawid_skene_linkedin_profiles_big_consulting_file_path: Path to big consulting profiles file
        dawid_skene_linkedin_profiles_comparator_file_path: Path to comparator profiles file
        enable_85k_profiles: Whether to include 85k profiles dataset
        enable_big_consulting: Whether to include big consulting profiles dataset
        enable_comparator: Whether to include comparator profiles dataset
        
    Returns:
        DataFrame with LinkedIn profile data (deduplicated)
    """
    logger.info("Loading LinkedIn profiles for Dawid-Skene analysis...")
    
    # Define dataset configurations
    datasets = [
        {
            'name': '85k profiles',
            'path': dawid_skene_linkedin_profiles_raw_file_path,
            'enabled': enable_85k_profiles
        },
        {
            'name': 'big consulting',
            'path': dawid_skene_linkedin_profiles_big_consulting_file_path,
            'enabled': enable_big_consulting
        },
        {
            'name': 'comparator',
            'path': dawid_skene_linkedin_profiles_comparator_file_path,
            'enabled': enable_comparator
        }
    ]
    
    # Load profiles from enabled datasets
    import gzip
    all_profiles = []
    seen_ids = set()  # Track linkedin_ids to filter duplicates
    
    for dataset_config in datasets:
        if not dataset_config['enabled']:
            logger.info(f"Skipping {dataset_config['name']} dataset (disabled)")
            continue
        
        profile_file = Path(dataset_config['path'])
        if not profile_file.exists():
            raise FileNotFoundError(f"LinkedIn profiles file not found: {profile_file}")
        
        logger.info(f"Loading {dataset_config['name']} from {profile_file}")
        dataset_count = 0
        duplicate_count = 0
        
        with gzip.open(profile_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    profile = json.loads(line.strip())
                    linkedin_id = profile.get('linkedin_id') or profile.get('id')
                    
                    # Skip duplicates
                    if linkedin_id in seen_ids:
                        duplicate_count += 1
                        continue
                    
                    # Extract relevant information from the real LinkedIn profile
                    current_company = profile.get('current_company', {})
                    
                    # Build profile summary from available text fields
                    summary_parts = []
                    if profile.get('about'):
                        summary_parts.append(profile['about'])
                    if profile.get('headline'):
                        summary_parts.append(profile['headline'])
                    
                    # Add experience descriptions
                    experiences = profile.get('experience', [])
                    if isinstance(experiences, list):
                        for exp in experiences[:3]:  # Limit to first 3 experiences
                            if isinstance(exp, dict) and exp.get('description'):
                                summary_parts.append(exp['description'])
                    
                    profile_summary = ' '.join(summary_parts) if summary_parts else ''
                    
                    all_profiles.append({
                        'public_identifier': linkedin_id,  # Map linkedin_id to public_identifier
                        'profile_summary': profile_summary,
                        'company_id': current_company.get('company_id', ''),
                        'company_name': current_company.get('name', '')
                    })
                    
                    seen_ids.add(linkedin_id)
                    dataset_count += 1
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid profile in {dataset_config['name']}: {e}")
                    continue
        
        logger.info(f"Loaded {dataset_count} unique profiles from {dataset_config['name']} ({duplicate_count} duplicates filtered)")
    
    if not all_profiles:
        raise ValueError("No LinkedIn profiles loaded. Please enable at least one dataset.")
    
    df = pd.DataFrame(all_profiles)
    
    # Filter out profiles without company information
    df = df[df['company_id'].notna() & (df['company_id'] != '')]
    
    logger.info(f"Total: {len(df)} LinkedIn profiles with company information (after deduplication)")
    
    return df


@save_to.csv(
    path=source('dawid_skene_linkedin_profiles_main_output_path')
)
@cache()
@check_output(schema=LinkedInDataSchema, importance="fail")
def dawid_skene_linkedin_profiles_main(
    dawid_skene_linkedin_profiles_raw_file_path: str,
    dawid_skene_linkedin_profiles_big_consulting_file_path: str,
    enable_85k_profiles: bool,
    enable_big_consulting: bool
) -> DataFrame[LinkedInDataSchema]:
    """
    Load LinkedIn profiles for main pipeline (85k + big_consulting, excluding comparator).
    
    Returns:
        DataFrame with LinkedIn profile data (deduplicated)
    """
    logger.info("Loading LinkedIn profiles for main pipeline (85k + big_consulting)...")
    
    # Define dataset configurations (only main datasets)
    datasets = [
        {
            'name': '85k profiles',
            'path': dawid_skene_linkedin_profiles_raw_file_path,
            'enabled': enable_85k_profiles
        },
        {
            'name': 'big consulting',
            'path': dawid_skene_linkedin_profiles_big_consulting_file_path,
            'enabled': enable_big_consulting
        }
    ]
    
    # Load profiles from enabled datasets
    import gzip
    all_profiles = []
    seen_ids = set()
    
    for dataset_config in datasets:
        if not dataset_config['enabled']:
            logger.info(f"Skipping {dataset_config['name']} dataset (disabled)")
            continue
        
        profile_file = Path(dataset_config['path'])
        if not profile_file.exists():
            raise FileNotFoundError(f"LinkedIn profiles file not found: {profile_file}")
        
        logger.info(f"Loading {dataset_config['name']} from {profile_file}")
        dataset_count = 0
        duplicate_count = 0
        
        with gzip.open(profile_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    profile = json.loads(line.strip())
                    linkedin_id = profile.get('linkedin_id') or profile.get('id')
                    
                    if linkedin_id in seen_ids:
                        duplicate_count += 1
                        continue
                    
                    current_company = profile.get('current_company', {})
                    summary_parts = []
                    if profile.get('about'):
                        summary_parts.append(profile['about'])
                    if profile.get('headline'):
                        summary_parts.append(profile['headline'])
                    
                    experiences = profile.get('experience', [])
                    if isinstance(experiences, list):
                        for exp in experiences[:3]:
                            if isinstance(exp, dict) and exp.get('description'):
                                summary_parts.append(exp['description'])
                    
                    profile_summary = ' '.join(summary_parts) if summary_parts else ''
                    
                    all_profiles.append({
                        'public_identifier': linkedin_id,
                        'profile_summary': profile_summary,
                        'company_id': current_company.get('company_id', ''),
                        'company_name': current_company.get('name', '')
                    })
                    
                    seen_ids.add(linkedin_id)
                    dataset_count += 1
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid profile in {dataset_config['name']}: {e}")
                    continue
        
        logger.info(f"Loaded {dataset_count} unique profiles from {dataset_config['name']} ({duplicate_count} duplicates filtered)")
    
    if not all_profiles:
        raise ValueError("No LinkedIn profiles loaded for main pipeline. Please enable at least one main dataset.")
    
    df = pd.DataFrame(all_profiles)
    df = df[df['company_id'].notna() & (df['company_id'] != '')]
    
    logger.info(f"Main pipeline: {len(df)} LinkedIn profiles with company information (after deduplication)")
    return df


@save_to.csv(
    path=source('dawid_skene_linkedin_profiles_comparator_output_path')
)
@cache()
@check_output(schema=LinkedInDataSchema, importance="fail")
def dawid_skene_linkedin_profiles_comparator(
    dawid_skene_linkedin_profiles_comparator_file_path: str,
    enable_comparator: bool
) -> DataFrame[LinkedInDataSchema]:
    """
    Load LinkedIn profiles for comparator pipeline (comparator dataset only).
    
    Returns:
        DataFrame with LinkedIn profile data
    """
    if not enable_comparator:
        logger.info("Comparator dataset disabled, returning empty DataFrame")
        return pd.DataFrame(columns=['public_identifier', 'profile_summary', 'company_id', 'company_name'])
    
    logger.info("Loading LinkedIn profiles for comparator pipeline...")
    
    import gzip
    all_profiles = []
    seen_ids = set()
    
    profile_file = Path(dawid_skene_linkedin_profiles_comparator_file_path)
    if not profile_file.exists():
        raise FileNotFoundError(f"LinkedIn profiles file not found: {profile_file}")
    
    logger.info(f"Loading comparator from {profile_file}")
    dataset_count = 0
    duplicate_count = 0
    
    with gzip.open(profile_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                profile = json.loads(line.strip())
                linkedin_id = profile.get('linkedin_id') or profile.get('id')
                
                if linkedin_id in seen_ids:
                    duplicate_count += 1
                    continue
                
                current_company = profile.get('current_company', {})
                summary_parts = []
                if profile.get('about'):
                    summary_parts.append(profile['about'])
                if profile.get('headline'):
                    summary_parts.append(profile['headline'])
                
                experiences = profile.get('experience', [])
                if isinstance(experiences, list):
                    for exp in experiences[:3]:
                        if isinstance(exp, dict) and exp.get('description'):
                            summary_parts.append(exp['description'])
                
                profile_summary = ' '.join(summary_parts) if summary_parts else ''
                
                all_profiles.append({
                    'public_identifier': linkedin_id,
                    'profile_summary': profile_summary,
                    'company_id': current_company.get('company_id', ''),
                    'company_name': current_company.get('name', '')
                })
                
                seen_ids.add(linkedin_id)
                dataset_count += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping invalid profile in comparator: {e}")
                continue
    
    logger.info(f"Loaded {dataset_count} unique profiles from comparator ({duplicate_count} duplicates filtered)")
    
    if not all_profiles:
        logger.warning("No LinkedIn profiles loaded for comparator pipeline")
        return pd.DataFrame(columns=['public_identifier', 'profile_summary', 'company_id', 'company_name'])
    
    df = pd.DataFrame(all_profiles)
    df = df[df['company_id'].notna() & (df['company_id'] != '')]
    
    logger.info(f"Comparator pipeline: {len(df)} LinkedIn profiles with company information (after deduplication)")
    return df

@save_to.csv(
    path=source('dawid_skene_keyword_filters_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_keyword_filters(
    dawid_skene_keyword_filters_raw_file_path: str,
    dawid_skene_keyword_filters_big_consulting_file_path: str,
    dawid_skene_keyword_filters_comparator_file_path: str,
    enable_85k_profiles: bool,
    enable_big_consulting: bool,
    enable_comparator: bool
) -> pd.DataFrame:
    """
    Load real keyword filter results for Dawid-Skene analysis from enabled datasets and compute compound filters.
    
    Loads and concatenates keyword filters from enabled datasets, removing duplicates.
    
    Computes the three compound filters used as Dawid-Skene annotators per the paper:
    - filter_broad_yes: ML Selection AND Broad_yes (ml_match AND broad_match)
    - filter_strict_no: ML Selection AND Strict_no (ml_match AND strict_no_match)
    - filter_broad_yes_strict_no: ML Selection AND Broad_yes AND Strict_no (all three)
    
    Args:
        dawid_skene_keyword_filters_raw_file_path: Path to 85k keyword filter file
        dawid_skene_keyword_filters_big_consulting_file_path: Path to big consulting keyword filter file
        dawid_skene_keyword_filters_comparator_file_path: Path to comparator keyword filter file
        enable_85k_profiles: Whether to include 85k profiles dataset
        enable_big_consulting: Whether to include big consulting profiles dataset
        enable_comparator: Whether to include comparator profiles dataset
        
    Returns:
        DataFrame with compound keyword filter results (deduplicated)
    """
    logger.info("Loading keyword filter results for Dawid-Skene analysis...")
    
    from .keyword_filters import filter_broad_yes, filter_strict_no, filter_broad_yes_strict_no
    
    # Define dataset configurations
    datasets = [
        {
            'name': '85k profiles',
            'path': dawid_skene_keyword_filters_raw_file_path,
            'enabled': enable_85k_profiles
        },
        {
            'name': 'big consulting',
            'path': dawid_skene_keyword_filters_big_consulting_file_path,
            'enabled': enable_big_consulting
        },
        {
            'name': 'comparator',
            'path': dawid_skene_keyword_filters_comparator_file_path,
            'enabled': enable_comparator
        }
    ]
    
    # Load keyword filters from enabled datasets
    all_filters = []
    
    for dataset_config in datasets:
        if not dataset_config['enabled']:
            logger.info(f"Skipping {dataset_config['name']} keyword filters (disabled)")
            continue
        
        keyword_file = Path(dataset_config['path'])
        if not keyword_file.exists():
            raise FileNotFoundError(f"Keyword filter file not found: {keyword_file}")
        
        logger.info(f"Loading keyword filters from {dataset_config['name']}")
        
        df = _process_keyword_filter_file(keyword_file)
        all_filters.append(df)
        logger.info(f"Loaded {len(df)} keyword filters from {dataset_config['name']}")
    
    if not all_filters:
        raise ValueError("No keyword filters loaded. Please enable at least one dataset.")
    
    # Concatenate and deduplicate
    keyword_filters = pd.concat(all_filters, ignore_index=True)
    
    # Remove duplicates (keep first occurrence)
    initial_count = len(keyword_filters)
    keyword_filters = keyword_filters.drop_duplicates(subset=['public_identifier'], keep='first')
    duplicates_removed = initial_count - len(keyword_filters)
    
    logger.info(f"Total: {len(keyword_filters)} unique keyword filters (removed {duplicates_removed} duplicates)")
    logger.info(f"  - filter_broad_yes: {keyword_filters['filter_broad_yes'].sum()} positives ({keyword_filters['filter_broad_yes'].mean():.1%})")
    logger.info(f"  - filter_strict_no: {keyword_filters['filter_strict_no'].sum()} positives ({keyword_filters['filter_strict_no'].mean():.1%})")
    logger.info(f"  - filter_broad_yes_strict_no: {keyword_filters['filter_broad_yes_strict_no'].sum()} positives ({keyword_filters['filter_broad_yes_strict_no'].mean():.1%})")
    
    return keyword_filters


@save_to.csv(
    path=source('dawid_skene_keyword_filters_main_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_keyword_filters_main(
    dawid_skene_keyword_filters_raw_file_path: str,
    dawid_skene_keyword_filters_big_consulting_file_path: str,
    enable_85k_profiles: bool,
    enable_big_consulting: bool
) -> pd.DataFrame:
    """Load keyword filters for main pipeline (85k + big_consulting)."""
    logger.info("Loading keyword filter results for main pipeline...")
    
    datasets = [
        {
            'name': '85k profiles',
            'path': dawid_skene_keyword_filters_raw_file_path,
            'enabled': enable_85k_profiles
        },
        {
            'name': 'big consulting',
            'path': dawid_skene_keyword_filters_big_consulting_file_path,
            'enabled': enable_big_consulting
        }
    ]
    
    all_filters = []
    for dataset_config in datasets:
        if not dataset_config['enabled']:
            continue
        
        keyword_file = Path(dataset_config['path'])
        if not keyword_file.exists():
            raise FileNotFoundError(f"Keyword filter file not found: {keyword_file}")
        
        df = _process_keyword_filter_file(keyword_file)
        all_filters.append(df)
    
    if not all_filters:
        raise ValueError("No keyword filters loaded for main pipeline.")
    
    keyword_filters = pd.concat(all_filters, ignore_index=True)
    keyword_filters = keyword_filters.drop_duplicates(subset=['public_identifier'], keep='first')
    
    logger.info(f"Main pipeline: {len(keyword_filters)} unique keyword filters")
    return keyword_filters


@save_to.csv(
    path=source('dawid_skene_keyword_filters_comparator_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_keyword_filters_comparator(
    dawid_skene_keyword_filters_comparator_file_path: str,
    enable_comparator: bool
) -> pd.DataFrame:
    """Load keyword filters for comparator pipeline."""
    if not enable_comparator:
        return pd.DataFrame(columns=['public_identifier', 'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'])
    
    logger.info("Loading keyword filter results for comparator pipeline...")
    
    keyword_file = Path(dawid_skene_keyword_filters_comparator_file_path)
    if not keyword_file.exists():
        raise FileNotFoundError(f"Keyword filter file not found: {keyword_file}")
    
    df = _process_keyword_filter_file(keyword_file)
    
    logger.info(f"Comparator pipeline: {len(df)} keyword filters")
    return df

@cache()
def dawid_skene_llm_results(
    dawid_skene_llm_results_dir_file_path: str,
    dawid_skene_llm_results_dir_big_consulting_file_path: str,
    dawid_skene_llm_results_dir_comparator_file_path: str,
    enable_85k_profiles: bool,
    enable_big_consulting: bool,
    enable_comparator: bool
) -> Dict[str, pd.DataFrame]:
    """
    Load real LLM evaluation results for Dawid-Skene analysis from enabled datasets.
    
    Loads and concatenates LLM results from enabled datasets, removing duplicates.
    
    Args:
        dawid_skene_llm_results_dir_file_path: Path to 85k LLM results directory
        dawid_skene_llm_results_dir_big_consulting_file_path: Path to big consulting LLM results directory
        dawid_skene_llm_results_dir_comparator_file_path: Path to comparator LLM results directory
        enable_85k_profiles: Whether to include 85k profiles dataset
        enable_big_consulting: Whether to include big consulting profiles dataset
        enable_comparator: Whether to include comparator profiles dataset
        
    Returns:
        Dictionary mapping model names to their evaluation results (deduplicated)
    """
    logger.info("Loading LLM evaluation results for Dawid-Skene analysis...")
    
    # Define dataset configurations with their model subdirectory mappings
    datasets = [
        {
            'name': '85k profiles',
            'path': dawid_skene_llm_results_dir_file_path,
            'enabled': enable_85k_profiles,
            'model_dirs': {
                'gemini-2.5-flash': 'gemini-2.5-flash-t0.0',
                'gpt-5-mini': 'gpt-5-mini',
                'sonnet-4': 'sonnet-4-t0.0'
            }
        },
        {
            'name': 'big consulting',
            'path': dawid_skene_llm_results_dir_big_consulting_file_path,
            'enabled': enable_big_consulting,
            'model_dirs': {
                'gemini-2.5-flash': 'gemini-2.5-flash',
                'gpt-5-mini': 'gpt-5-mini',
                'sonnet-4': 'sonnet-4'
            }
        },
        {
            'name': 'comparator',
            'path': dawid_skene_llm_results_dir_comparator_file_path,
            'enabled': enable_comparator,
            'model_dirs': {
                'gemini-2.5-flash': 'gemini-2.5-flash',
                'gpt-5-mini': 'gpt-5-mini',
                'sonnet-4': 'sonnet-4'
            }
        }
    ]
    
    # Collect evaluations by model across all enabled datasets
    model_evaluations = {model: [] for model in ['gemini-2.5-flash', 'gpt-5-mini', 'sonnet-4']}
    
    for dataset_config in datasets:
        if not dataset_config['enabled']:
            logger.info(f"Skipping {dataset_config['name']} LLM results (disabled)")
            continue
        
        results_dir = Path(dataset_config['path'])
        if not results_dir.exists():
            raise FileNotFoundError(f"LLM results directory not found: {results_dir}")
        
        logger.info(f"Loading LLM results from {dataset_config['name']}")
        
        for model, model_dir in dataset_config['model_dirs'].items():
            model_path = results_dir / model_dir
            
            if not model_path.exists():
                logger.warning(f"Model directory not found: {model_path}")
                continue
                
            # Load all JSONL files for this model
            dataset_evals = []
            for jsonl_file in model_path.glob("batch_results_*.jsonl"):
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            batch_data = json.loads(line.strip())
                            for result in batch_data.get('results', []):
                                dataset_evals.append({
                                    'linkedin_id': result.get('linkedin_id'),
                                    'evaluation': result.get('evaluation'),
                                    'model': model
                                })
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Skipping invalid batch result in {jsonl_file}: {e}")
                            continue
            
            if dataset_evals:
                model_evaluations[model].extend(dataset_evals)
                logger.info(f"  Loaded {len(dataset_evals)} evaluations for {model} from {dataset_config['name']}")
            else:
                logger.warning(f"  No evaluations found for {model} in {dataset_config['name']}")
    
    # Convert to DataFrames and deduplicate
    llm_results = {}
    for model, evaluations in model_evaluations.items():
        if not evaluations:
            logger.warning(f"No evaluations found for model {model} in any enabled dataset")
            continue
        
        # Convert to DataFrame and create binary classification
        df = pd.DataFrame(evaluations)
        df['llm_accept'] = (df['evaluation'] == 'ACCEPT').astype(int)
        
        # Group by linkedin_id and take the first evaluation (in case of duplicates)
        initial_count = len(df)
        df_grouped = df.groupby('linkedin_id').first().reset_index()
        duplicates_removed = initial_count - len(df_grouped)
        
        llm_results[model] = df_grouped[['linkedin_id', 'llm_accept']].rename(
            columns={'linkedin_id': 'public_identifier', 'llm_accept': f'llm_{model}'}
        )
        
        logger.info(f"Total for {model}: {len(df_grouped)} unique evaluations (removed {duplicates_removed} duplicates)")
    
    logger.info(f"Loaded LLM results for {len(llm_results)} models")
    return llm_results


@cache()
def dawid_skene_llm_results_main(
    dawid_skene_llm_results_dir_file_path: str,
    dawid_skene_llm_results_dir_big_consulting_file_path: str,
    enable_85k_profiles: bool,
    enable_big_consulting: bool
) -> Dict[str, pd.DataFrame]:
    """Load LLM results for main pipeline (85k + big_consulting)."""
    logger.info("Loading LLM evaluation results for main pipeline...")
    
    datasets = [
        {
            'name': '85k profiles',
            'path': dawid_skene_llm_results_dir_file_path,
            'enabled': enable_85k_profiles,
            'model_dirs': {
                'gemini-2.5-flash': 'gemini-2.5-flash-t0.0',
                'gpt-5-mini': 'gpt-5-mini',
                'sonnet-4': 'sonnet-4-t0.0'
            }
        },
        {
            'name': 'big consulting',
            'path': dawid_skene_llm_results_dir_big_consulting_file_path,
            'enabled': enable_big_consulting,
            'model_dirs': {
                'gemini-2.5-flash': 'gemini-2.5-flash',
                'gpt-5-mini': 'gpt-5-mini',
                'sonnet-4': 'sonnet-4'
            }
        }
    ]
    
    model_evaluations = {model: [] for model in ['gemini-2.5-flash', 'gpt-5-mini', 'sonnet-4']}
    
    for dataset_config in datasets:
        if not dataset_config['enabled']:
            continue
        
        results_dir = Path(dataset_config['path'])
        if not results_dir.exists():
            raise FileNotFoundError(f"LLM results directory not found: {results_dir}")
        
        for model, model_dir in dataset_config['model_dirs'].items():
            model_path = results_dir / model_dir
            if not model_path.exists():
                continue
            
            dataset_evals = []
            for jsonl_file in model_path.glob("batch_results_*.jsonl"):
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            batch_data = json.loads(line.strip())
                            for result in batch_data.get('results', []):
                                dataset_evals.append({
                                    'linkedin_id': result.get('linkedin_id'),
                                    'evaluation': result.get('evaluation'),
                                    'model': model
                                })
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            if dataset_evals:
                model_evaluations[model].extend(dataset_evals)
    
    llm_results = {}
    for model, evaluations in model_evaluations.items():
        if not evaluations:
            continue
        
        df = pd.DataFrame(evaluations)
        df['llm_accept'] = (df['evaluation'] == 'ACCEPT').astype(int)
        df_grouped = df.groupby('linkedin_id').first().reset_index()
        
        llm_results[model] = df_grouped[['linkedin_id', 'llm_accept']].rename(
            columns={'linkedin_id': 'public_identifier', 'llm_accept': f'llm_{model}'}
        )
    
    logger.info(f"Main pipeline: Loaded LLM results for {len(llm_results)} models")
    return llm_results


@cache()
def dawid_skene_llm_results_comparator(
    dawid_skene_llm_results_dir_comparator_file_path: str,
    enable_comparator: bool
) -> Dict[str, pd.DataFrame]:
    """Load LLM results for comparator pipeline."""
    if not enable_comparator:
        return {}
    
    logger.info("Loading LLM evaluation results for comparator pipeline...")
    
    results_dir = Path(dawid_skene_llm_results_dir_comparator_file_path)
    if not results_dir.exists():
        raise FileNotFoundError(f"LLM results directory not found: {results_dir}")
    
    model_dirs = {
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gpt-5-mini': 'gpt-5-mini',
        'sonnet-4': 'sonnet-4'
    }
    
    model_evaluations = {model: [] for model in ['gemini-2.5-flash', 'gpt-5-mini', 'sonnet-4']}
    
    for model, model_dir in model_dirs.items():
        model_path = results_dir / model_dir
        if not model_path.exists():
            continue
        
        dataset_evals = []
        for jsonl_file in model_path.glob("batch_results_*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        batch_data = json.loads(line.strip())
                        for result in batch_data.get('results', []):
                            dataset_evals.append({
                                'linkedin_id': result.get('linkedin_id'),
                                'evaluation': result.get('evaluation'),
                                'model': model
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        if dataset_evals:
            model_evaluations[model].extend(dataset_evals)
    
    llm_results = {}
    for model, evaluations in model_evaluations.items():
        if not evaluations:
            continue
        
        df = pd.DataFrame(evaluations)
        df['llm_accept'] = (df['evaluation'] == 'ACCEPT').astype(int)
        df_grouped = df.groupby('linkedin_id').first().reset_index()
        
        llm_results[model] = df_grouped[['linkedin_id', 'llm_accept']].rename(
            columns={'linkedin_id': 'public_identifier', 'llm_accept': f'llm_{model}'}
        )
    
    logger.info(f"Comparator pipeline: Loaded LLM results for {len(llm_results)} models")
    return llm_results


@save_to.csv(
    path=source('dawid_skene_annotation_data_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_annotation_data(
    dawid_skene_linkedin_profiles: DataFrame[LinkedInDataSchema],
    dawid_skene_keyword_filters: pd.DataFrame,
    dawid_skene_llm_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine all annotation data for Dawid-Skene analysis.
    
    Args:
        dawid_skene_linkedin_profiles: LinkedIn profiles data (already includes company names)
        dawid_skene_keyword_filters: Keyword filter results
        dawid_skene_llm_results: LLM evaluation results
        
    Returns:
        Combined DataFrame with all annotation data including company names
    """
    logger.info("Combining annotation data for Dawid-Skene analysis...")
    
    # Start with profiles (company names are already included from real LinkedIn data)
    annotation_df = dawid_skene_linkedin_profiles[['public_identifier', 'company_id', 'company_name']].copy()
    
    # Add keyword filters
    annotation_df = annotation_df.merge(
        dawid_skene_keyword_filters, 
        on='public_identifier', 
        how='left'
    )
    
    # Add LLM results
    for model, results in dawid_skene_llm_results.items():
        annotation_df = annotation_df.merge(
            results, 
            on='public_identifier', 
            how='left'
        )
    
    # Keep rows with missing annotations - they will be handled by the probit bootstrap
    # pipeline using marginalization over missing annotators. This preserves all data
    # while correctly handling missing values statistically.
    # Note: Missing values are represented as NaN and will be converted to None in patterns
    annotator_cols = [col for col in annotation_df.columns 
                     if col not in ['public_identifier', 'company_id', 'company_name']]
    # No longer dropping rows with missing annotations - they're handled via marginalization
    
    missing_count = annotation_df[annotator_cols].isna().any(axis=1).sum()
    if missing_count > 0:
        logger.info(f"Combined annotation data: {annotation_df.shape} ({missing_count} rows with missing annotations - will be handled via marginalization)")
    else:
        logger.info(f"Combined annotation data: {annotation_df.shape} (all annotations complete)")
    return annotation_df


@save_to.csv(
    path=source('dawid_skene_annotation_data_main_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_annotation_data_main(
    dawid_skene_linkedin_profiles_main: DataFrame[LinkedInDataSchema],
    dawid_skene_keyword_filters_main: pd.DataFrame,
    dawid_skene_llm_results_main: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Combine annotation data for main pipeline."""
    logger.info("Combining annotation data for main pipeline...")
    
    annotation_df = dawid_skene_linkedin_profiles_main[['public_identifier', 'company_id', 'company_name']].copy()
    annotation_df = annotation_df.merge(dawid_skene_keyword_filters_main, on='public_identifier', how='left')
    
    for model, results in dawid_skene_llm_results_main.items():
        annotation_df = annotation_df.merge(results, on='public_identifier', how='left')
    
    annotator_cols = [col for col in annotation_df.columns 
                     if col not in ['public_identifier', 'company_id', 'company_name']]
    # Keep rows with missing annotations - handled via marginalization in probit bootstrap
    
    missing_count = annotation_df[annotator_cols].isna().any(axis=1).sum()
    if missing_count > 0:
        logger.info(f"Main pipeline annotation data: {annotation_df.shape} ({missing_count} rows with missing annotations)")
    else:
        logger.info(f"Main pipeline annotation data: {annotation_df.shape}")
    return annotation_df


@save_to.csv(
    path=source('dawid_skene_annotation_data_comparator_ml_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_annotation_data_comparator_ml(
    dawid_skene_linkedin_profiles_comparator: DataFrame[LinkedInDataSchema],
    dawid_skene_keyword_filters_comparator: pd.DataFrame,
    dawid_skene_llm_results_comparator: Dict[str, pd.DataFrame],
    company_database_comparator_ml: DataFrame[CompanyDatabaseCompleteSchema]
) -> pd.DataFrame:
    """Combine annotation data for comparator ML pipeline, filtered by ML companies."""
    logger.info("Combining annotation data for comparator ML pipeline...")
    
    # Get ML company linkedin_ids
    ml_linkedin_ids = set(company_database_comparator_ml['linkedin_id'].dropna().unique())
    
    # Filter profiles to only ML companies
    annotation_df = dawid_skene_linkedin_profiles_comparator[
        dawid_skene_linkedin_profiles_comparator['company_id'].isin(ml_linkedin_ids)
    ][['public_identifier', 'company_id', 'company_name']].copy()
    
    if len(annotation_df) == 0:
        logger.warning("No profiles found for comparator ML companies")
        return pd.DataFrame()
    
    annotation_df = annotation_df.merge(dawid_skene_keyword_filters_comparator, on='public_identifier', how='left')
    
    for model, results in dawid_skene_llm_results_comparator.items():
        annotation_df = annotation_df.merge(results, on='public_identifier', how='left')
    
    annotator_cols = [col for col in annotation_df.columns 
                     if col not in ['public_identifier', 'company_id', 'company_name']]
    # Keep rows with missing annotations - handled via marginalization in probit bootstrap
    
    missing_count = annotation_df[annotator_cols].isna().any(axis=1).sum()
    if missing_count > 0:
        logger.info(f"Comparator ML annotation data: {annotation_df.shape} ({missing_count} rows with missing annotations)")
    else:
        logger.info(f"Comparator ML annotation data: {annotation_df.shape}")
    return annotation_df


@save_to.csv(
    path=source('dawid_skene_annotation_data_comparator_non_ml_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_annotation_data_comparator_non_ml(
    dawid_skene_linkedin_profiles_comparator: DataFrame[LinkedInDataSchema],
    dawid_skene_keyword_filters_comparator: pd.DataFrame,
    dawid_skene_llm_results_comparator: Dict[str, pd.DataFrame],
    company_database_comparator_non_ml: DataFrame[CompanyDatabaseCompleteSchema]
) -> pd.DataFrame:
    """Combine annotation data for comparator Non-ML pipeline, filtered by Non-ML companies."""
    logger.info("Combining annotation data for comparator Non-ML pipeline...")
    
    # Get Non-ML company linkedin_ids
    non_ml_linkedin_ids = set(company_database_comparator_non_ml['linkedin_id'].dropna().unique())
    
    # Filter profiles to only Non-ML companies
    annotation_df = dawid_skene_linkedin_profiles_comparator[
        dawid_skene_linkedin_profiles_comparator['company_id'].isin(non_ml_linkedin_ids)
    ][['public_identifier', 'company_id', 'company_name']].copy()
    
    if len(annotation_df) == 0:
        logger.warning("No profiles found for comparator Non-ML companies")
        return pd.DataFrame()
    
    annotation_df = annotation_df.merge(dawid_skene_keyword_filters_comparator, on='public_identifier', how='left')
    
    for model, results in dawid_skene_llm_results_comparator.items():
        annotation_df = annotation_df.merge(results, on='public_identifier', how='left')
    
    annotator_cols = [col for col in annotation_df.columns 
                     if col not in ['public_identifier', 'company_id', 'company_name']]
    # Keep rows with missing annotations - handled via marginalization in probit bootstrap
    
    missing_count = annotation_df[annotator_cols].isna().any(axis=1).sum()
    if missing_count > 0:
        logger.info(f"Comparator Non-ML annotation data: {annotation_df.shape} ({missing_count} rows with missing annotations)")
    else:
        logger.info(f"Comparator Non-ML annotation data: {annotation_df.shape}")
    return annotation_df


@save_to.csv(
    path=source('dawid_skene_validation_data_output_path')
)
@config.when(fake_data_enable=False)
@cache()
@check_output(schema=DawidSkeneValidationDataSchema, importance="fail")
def dawid_skene_validation_data__real(
    validation_cvs_with_paper_filters: pd.DataFrame
) -> DataFrame[DawidSkeneValidationDataSchema]:
    """
    Prepare validation data for Dawid-Skene analysis.
    
    Uses 6 annotators as per the paper:
    - 3 LLM annotators (all with prompt 1): gemini-2.5-flash, sonnet-4, gpt-5-mini
    - 3 keyword filters: ml_match, broad_match, strict_no_match
    
    Args:
        validation_cvs_with_paper_filters: Validation CVs with 6 annotator columns
        
    Returns:
        DataFrame with validation data formatted for Dawid-Skene analysis
    """
    logger.info("Preparing validation data for Dawid-Skene analysis...")
    
    # Define annotator columns: 3 LLMs + 3 compound keyword filters
    # Note: Using dashes in LLM names to match test data convention
    llm_annotators = [
        'llm_gemini-2.5-flash',  # prompt-1-gemini-2.5-flash
        'llm_sonnet-4',          # prompt-1-claude-sonnet-4-20250514
        'llm_gpt-5-mini'         # prompt-1-gpt-5-mini-2025-08-07
    ]
    # Use compound filters as per the paper's Dawid-Skene annotator specification
    keyword_filter_annotators = [
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    
    all_annotator_cols = llm_annotators + keyword_filter_annotators
    
    # Filter validation data to only include rows with all annotator scores
    validation_df = validation_cvs_with_paper_filters[['cv_text', 'category'] + all_annotator_cols].copy()
    validation_df = validation_df.dropna(subset=all_annotator_cols, how='any')
    
    # category is already a binary label (0 or 1) - use it directly as the true label
    validation_df['category'] = validation_df['category'].astype(int)
    
    # No group column needed - validation data is used only to estimate annotator confusion matrices
    
    # Rename annotator columns to human-readable Python-safe names (underscores instead of hyphens/dots)
    # CRITICAL: The order of annotators must match the test data order
    # Order is: llm_gemini-2.5-flash, llm_sonnet-4, llm_gpt-5-mini, 
    #           filter_broad_yes, filter_strict_no, filter_broad_yes_strict_no
    annotator_rename = {
        'llm_gemini-2.5-flash': 'llm_gemini_2_5_flash',
        'llm_sonnet-4': 'llm_sonnet_4',
        'llm_gpt-5-mini': 'llm_gpt_5_mini',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no'
    }
    validation_df = validation_df.rename(columns=annotator_rename)
    
    # Ensure annotator columns are int (they might be float from LLM results)
    for col in annotator_rename.values():
        validation_df[col] = validation_df[col].astype(int)
    
    logger.info(f"Prepared validation data: {len(validation_df)} rows with {len(all_annotator_cols)} annotators")
    logger.info(f"  - {len(llm_annotators)} LLM annotators (prompt 1): {', '.join(llm_annotators)}")
    logger.info(f"  - {len(keyword_filter_annotators)} compound keyword filters: {', '.join(keyword_filter_annotators)}")
    logger.info(f"  - Annotator column order: {list(annotator_rename.values())}")
    return validation_df

@save_to.csv(
    path=source('dawid_skene_filtered_dataset_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_filtered_dataset(
    dawid_skene_annotation_data: pd.DataFrame,
    company_database_complete: DataFrame[CompanyDatabaseCompleteSchema],
    ds_max_items: int,
    ds_max_companies: int,
    ds_min_profiles: int
) -> pd.DataFrame:
    """
    Filter dataset for Dawid-Skene analysis, only including companies with size data.
    
    Args:
        dawid_skene_annotation_data: Combined annotation data
        company_database: Company database with size information
        ds_max_items: Maximum number of items to include
        ds_max_companies: Maximum number of companies to include
        ds_min_profiles: Minimum profiles per company
        
    Returns:
        Filtered DataFrame ready for Dawid-Skene analysis
    """
    logger.info(f"Filtering dataset: max_items={ds_max_items}, max_companies={ds_max_companies}, min_profiles={ds_min_profiles}")
    
    # First, filter to only companies that have size data in the database
    companies_with_size_data = company_database_complete[
        (company_database_complete['total_headcount'].notna()) & 
        (company_database_complete['total_headcount'] > 0) &
        (company_database_complete['linkedin_id'].notna())
    ]['linkedin_id'].unique()
    
    logger.info(f"Found {len(companies_with_size_data)} companies with size data in database")
    
    # Debug: Check the data types and sample values
    logger.info(f"Company database linkedin_id type: {company_database_complete['linkedin_id'].dtype}")
    logger.info(f"Company database linkedin_id sample values: {company_database_complete['linkedin_id'].head().tolist()}")
    logger.info(f"Companies with size data sample: {list(companies_with_size_data[:5])}")
    logger.info(f"Annotation data company_id type: {dawid_skene_annotation_data['company_id'].dtype}")
    logger.info(f"Annotation data company_id sample values: {dawid_skene_annotation_data['company_id'].head().tolist()}")
    
    # Filter annotation data to only include companies with size data
    annotation_with_size_data = dawid_skene_annotation_data[
        dawid_skene_annotation_data['company_id'].isin(companies_with_size_data)
    ].copy()
    
    logger.info(f"Annotation data after filtering for size data: {len(annotation_with_size_data)} profiles")
    
    # Filter companies with minimum profiles
    company_counts = annotation_with_size_data['company_id'].value_counts()
    valid_companies = company_counts[company_counts >= ds_min_profiles].index
    
    if len(valid_companies) == 0:
        raise ValueError(f"No companies found with at least {ds_min_profiles} profiles and size data")
    
    # Select the largest companies by profile count (up to max_companies)
    if len(valid_companies) > ds_max_companies:
        # Sort by profile count (descending) and take the top max_companies
        valid_companies = company_counts[valid_companies].nlargest(ds_max_companies).index
    
    # Filter data
    filtered_df = annotation_with_size_data[
        annotation_with_size_data['company_id'].isin(valid_companies)
    ].copy()
    
    # Limit to max_items
    if len(filtered_df) > ds_max_items:
        filtered_df = filtered_df.sample(n=ds_max_items, random_state=42)
    
    logger.info(f"Filtered dataset: {len(filtered_df)} profiles, {len(valid_companies)} companies (all with size data)")
    return filtered_df


@save_to.csv(
    path=source('dawid_skene_test_data_output_path')
)
@config.when(fake_data_enable=False)
@cache()
@check_output(schema=DawidSkeneTestDataSchema, importance="fail")
def dawid_skene_test_data__real(
    dawid_skene_filtered_dataset: pd.DataFrame
) -> DataFrame[DawidSkeneTestDataSchema]:
    """
    Prepare test data for Dawid-Skene analysis.
    
    Args:
        dawid_skene_filtered_dataset: Filtered main dataset
        
    Returns:
        DataFrame with test data formatted for Dawid-Skene analysis
    """
    logger.info("Preparing test data for Dawid-Skene analysis...")
    
    # Get annotator columns (exclude metadata columns)
    # These come from dawid_skene_annotation_data which has columns from:
    # 1. LLM results (llm_gemini-2.5-flash, llm_sonnet-4, llm_gpt-5-mini)
    # 2. Keyword filters (filter_broad_yes, filter_strict_no, filter_broad_yes_strict_no)
    annotator_cols = [col for col in dawid_skene_filtered_dataset.columns 
                     if col not in ['public_identifier', 'company_id', 'company_name']]
    
    # Create test dataframe with required columns
    test_df = dawid_skene_filtered_dataset[['company_id'] + annotator_cols].copy()
    
    # Add group column mapping company_id to sequential group indices
    # Note: Validation data uses group=0, test data uses groups 0, 1, 2, ...
    # This means group 0 will have both validation items AND test items from company 0
    # The hierarchical model will estimate a separate prevalence pi[g] for each group
    unique_companies = sorted(test_df['company_id'].unique())
    company_to_group = {company_id: idx for idx, company_id in enumerate(unique_companies)}
    test_df['group'] = test_df['company_id'].map(company_to_group)
    
    # Rename annotator columns to human-readable Python-safe names (must match validation data)
    # CRITICAL: The order must match validation data order
    # Expected order: llm_gemini-2.5-flash, llm_sonnet-4, llm_gpt-5-mini,
    #                 filter_broad_yes, filter_strict_no, filter_broad_yes_strict_no
    expected_order = [
        'llm_gemini-2.5-flash', 'llm_sonnet-4', 'llm_gpt-5-mini',
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    
    # Verify all expected columns exist
    missing_cols = [col for col in expected_order if col not in annotator_cols]
    if missing_cols:
        logger.warning(f"Missing expected annotator columns: {missing_cols}")
        # Use whatever columns exist, but in a consistent order
        annotator_cols_sorted = [col for col in expected_order if col in annotator_cols]
    else:
        annotator_cols_sorted = expected_order
    
    # Rename to human-readable Python-safe names (underscores instead of hyphens/dots)
    annotator_rename = {
        'llm_gemini-2.5-flash': 'llm_gemini_2_5_flash',
        'llm_sonnet-4': 'llm_sonnet_4',
        'llm_gpt-5-mini': 'llm_gpt_5_mini',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no'
    }
    test_df = test_df.rename(columns=annotator_rename)
    
    # Convert all annotator columns to float64, preserving NaN values for missing annotations
    # This ensures consistent dtype (float64) for all annotators, allowing NaN for missing values
    for col in annotator_rename.values():
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').astype('float64')
        # Keep as float64 to preserve NaN values - they'll be handled in the bootstrap
    
    logger.info(f"Prepared test data: {len(test_df)} rows with {len(annotator_cols_sorted)} annotators across {len(unique_companies)} companies")
    logger.info(f"  - Test data groups: {min(test_df['group'])} to {max(test_df['group'])}")
    logger.info(f"  - Annotator column order: {list(annotator_rename.values())}")
    return test_df


@save_to.csv(
    path=source('dawid_skene_filtered_dataset_main_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_filtered_dataset_main(
    dawid_skene_annotation_data_main: pd.DataFrame,
    company_database_complete: DataFrame[CompanyDatabaseCompleteSchema],
    ds_max_items: int,
    ds_max_companies: int,
    ds_min_profiles: int
) -> pd.DataFrame:
    """Filter dataset for main pipeline Dawid-Skene analysis."""
    logger.info(f"Filtering main dataset: max_items={ds_max_items}, max_companies={ds_max_companies}, min_profiles={ds_min_profiles}")
    
    companies_with_size_data = company_database_complete[
        (company_database_complete['total_headcount'].notna()) & 
        (company_database_complete['total_headcount'] > 0) &
        (company_database_complete['linkedin_id'].notna())
    ]['linkedin_id'].unique()
    
    annotation_with_size_data = dawid_skene_annotation_data_main[
        dawid_skene_annotation_data_main['company_id'].isin(companies_with_size_data)
    ].copy()
    
    company_counts = annotation_with_size_data['company_id'].value_counts()
    valid_companies = company_counts[company_counts >= ds_min_profiles].index
    
    if len(valid_companies) > ds_max_companies:
        valid_companies = company_counts[valid_companies].nlargest(ds_max_companies).index
    
    filtered_df = annotation_with_size_data[
        annotation_with_size_data['company_id'].isin(valid_companies)
    ].copy()
    
    if len(filtered_df) > ds_max_items:
        filtered_df = filtered_df.sample(n=ds_max_items, random_state=42)
    
    logger.info(f"Main filtered dataset: {len(filtered_df)} profiles, {len(valid_companies)} companies")
    return filtered_df


@save_to.csv(
    path=source('dawid_skene_filtered_dataset_comparator_ml_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_filtered_dataset_comparator_ml(
    dawid_skene_annotation_data_comparator_ml: pd.DataFrame,
    company_database_comparator_ml: DataFrame[CompanyDatabaseCompleteSchema],
    ds_max_items: int,
    ds_max_companies: int,
    ds_min_profiles: int
) -> pd.DataFrame:
    """Filter dataset for comparator ML pipeline Dawid-Skene analysis."""
    if len(dawid_skene_annotation_data_comparator_ml) == 0:
        return pd.DataFrame()
    
    logger.info(f"Filtering comparator ML dataset: max_items={ds_max_items}, max_companies={ds_max_companies}, min_profiles={ds_min_profiles}")
    
    companies_with_size_data = company_database_comparator_ml[
        (company_database_comparator_ml['total_headcount'].notna()) & 
        (company_database_comparator_ml['total_headcount'] > 0) &
        (company_database_comparator_ml['linkedin_id'].notna())
    ]['linkedin_id'].unique()
    
    annotation_with_size_data = dawid_skene_annotation_data_comparator_ml[
        dawid_skene_annotation_data_comparator_ml['company_id'].isin(companies_with_size_data)
    ].copy()
    
    company_counts = annotation_with_size_data['company_id'].value_counts()
    valid_companies = company_counts[company_counts >= ds_min_profiles].index
    
    if len(valid_companies) > ds_max_companies:
        valid_companies = company_counts[valid_companies].nlargest(ds_max_companies).index
    
    filtered_df = annotation_with_size_data[
        annotation_with_size_data['company_id'].isin(valid_companies)
    ].copy()
    
    if len(filtered_df) > ds_max_items:
        filtered_df = filtered_df.sample(n=ds_max_items, random_state=42)
    
    logger.info(f"Comparator ML filtered dataset: {len(filtered_df)} profiles, {len(valid_companies)} companies")
    return filtered_df


@save_to.csv(
    path=source('dawid_skene_filtered_dataset_comparator_non_ml_output_path')
)
@cache()
@check_output(importance="fail")
def dawid_skene_filtered_dataset_comparator_non_ml(
    dawid_skene_annotation_data_comparator_non_ml: pd.DataFrame,
    company_database_comparator_non_ml: DataFrame[CompanyDatabaseCompleteSchema],
    ds_max_items: int,
    ds_max_companies: int,
    ds_min_profiles: int
) -> pd.DataFrame:
    """Filter dataset for comparator Non-ML pipeline Dawid-Skene analysis."""
    if len(dawid_skene_annotation_data_comparator_non_ml) == 0:
        return pd.DataFrame()
    
    logger.info(f"Filtering comparator Non-ML dataset: max_items={ds_max_items}, max_companies={ds_max_companies}, min_profiles={ds_min_profiles}")
    
    companies_with_size_data = company_database_comparator_non_ml[
        (company_database_comparator_non_ml['total_headcount'].notna()) & 
        (company_database_comparator_non_ml['total_headcount'] > 0) &
        (company_database_comparator_non_ml['linkedin_id'].notna())
    ]['linkedin_id'].unique()
    
    annotation_with_size_data = dawid_skene_annotation_data_comparator_non_ml[
        dawid_skene_annotation_data_comparator_non_ml['company_id'].isin(companies_with_size_data)
    ].copy()
    
    company_counts = annotation_with_size_data['company_id'].value_counts()
    valid_companies = company_counts[company_counts >= ds_min_profiles].index
    
    if len(valid_companies) > ds_max_companies:
        valid_companies = company_counts[valid_companies].nlargest(ds_max_companies).index
    
    filtered_df = annotation_with_size_data[
        annotation_with_size_data['company_id'].isin(valid_companies)
    ].copy()
    
    if len(filtered_df) > ds_max_items:
        filtered_df = filtered_df.sample(n=ds_max_items, random_state=42)
    
    logger.info(f"Comparator Non-ML filtered dataset: {len(filtered_df)} profiles, {len(valid_companies)} companies")
    return filtered_df


@save_to.csv(
    path=source('dawid_skene_test_data_main_output_path')
)
@config.when(fake_data_enable=False)
@cache()
@check_output(schema=DawidSkeneTestDataSchema, importance="fail")
def dawid_skene_test_data_main(
    dawid_skene_filtered_dataset_main: pd.DataFrame
) -> DataFrame[DawidSkeneTestDataSchema]:
    """Prepare test data for main pipeline Dawid-Skene analysis."""
    logger.info("Preparing test data for main pipeline...")
    
    annotator_cols = [col for col in dawid_skene_filtered_dataset_main.columns 
                     if col not in ['public_identifier', 'company_id', 'company_name']]
    
    test_df = dawid_skene_filtered_dataset_main[['company_id'] + annotator_cols].copy()
    
    unique_companies = sorted(test_df['company_id'].unique())
    company_to_group = {company_id: idx for idx, company_id in enumerate(unique_companies)}
    test_df['group'] = test_df['company_id'].map(company_to_group)
    
    expected_order = [
        'llm_gemini-2.5-flash', 'llm_sonnet-4', 'llm_gpt-5-mini',
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    
    annotator_rename = {
        'llm_gemini-2.5-flash': 'llm_gemini_2_5_flash',
        'llm_sonnet-4': 'llm_sonnet_4',
        'llm_gpt-5-mini': 'llm_gpt_5_mini',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no'
    }
    
    test_df = test_df.rename(columns=annotator_rename)
    
    # Convert all annotator columns to float64, preserving NaN values for missing annotations
    # This ensures consistent dtype (float64) for all annotators, allowing NaN for missing values
    for col in annotator_rename.values():
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').astype('float64')
        # Keep as float64 to preserve NaN values - they'll be handled in the bootstrap
    
    logger.info(f"Main test data: {len(test_df)} rows across {len(unique_companies)} companies")
    return test_df


@save_to.csv(
    path=source('dawid_skene_test_data_comparator_ml_output_path')
)
@config.when(fake_data_enable=False)
@cache()
@check_output(schema=DawidSkeneTestDataSchema, importance="fail")
def dawid_skene_test_data_comparator_ml(
    dawid_skene_filtered_dataset_comparator_ml: pd.DataFrame
) -> DataFrame[DawidSkeneTestDataSchema]:
    """Prepare test data for comparator ML pipeline Dawid-Skene analysis."""
    if len(dawid_skene_filtered_dataset_comparator_ml) == 0:
        # Create empty DataFrame with correct dtypes for schema validation
        empty_df = pd.DataFrame({
            'company_id': pd.Series(dtype='object'),
            'group': pd.Series(dtype='int64'),
            'llm_gemini_2_5_flash': pd.Series(dtype='int64'),
            'llm_sonnet_4': pd.Series(dtype='int64'),
            'llm_gpt_5_mini': pd.Series(dtype='int64'),
            'filter_broad_yes': pd.Series(dtype='int64'),
            'filter_strict_no': pd.Series(dtype='int64'),
            'filter_broad_yes_strict_no': pd.Series(dtype='int64')
        })
        return empty_df
    
    logger.info("Preparing test data for comparator ML pipeline...")
    
    annotator_cols = [col for col in dawid_skene_filtered_dataset_comparator_ml.columns 
                     if col not in ['public_identifier', 'company_id', 'company_name']]
    
    test_df = dawid_skene_filtered_dataset_comparator_ml[['company_id'] + annotator_cols].copy()
    
    unique_companies = sorted(test_df['company_id'].unique())
    company_to_group = {company_id: idx for idx, company_id in enumerate(unique_companies)}
    test_df['group'] = test_df['company_id'].map(company_to_group)
    
    annotator_rename = {
        'llm_gemini-2.5-flash': 'llm_gemini_2_5_flash',
        'llm_sonnet-4': 'llm_sonnet_4',
        'llm_gpt-5-mini': 'llm_gpt_5_mini',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no'
    }
    
    test_df = test_df.rename(columns=annotator_rename)
    
    # Convert all annotator columns to float64, preserving NaN values for missing annotations
    # This ensures consistent dtype (float64) for all annotators, allowing NaN for missing values
    for col in annotator_rename.values():
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').astype('float64')
        # Keep as float64 to preserve NaN values - they'll be handled in the bootstrap
    
    logger.info(f"Comparator ML test data: {len(test_df)} rows across {len(unique_companies)} companies")
    return test_df


@save_to.csv(
    path=source('dawid_skene_test_data_comparator_non_ml_output_path')
)
@config.when(fake_data_enable=False)
@cache()
@check_output(schema=DawidSkeneTestDataSchema, importance="fail")
def dawid_skene_test_data_comparator_non_ml(
    dawid_skene_filtered_dataset_comparator_non_ml: pd.DataFrame
) -> DataFrame[DawidSkeneTestDataSchema]:
    """Prepare test data for comparator Non-ML pipeline Dawid-Skene analysis."""
    if len(dawid_skene_filtered_dataset_comparator_non_ml) == 0:
        # Create empty DataFrame with correct dtypes for schema validation
        empty_df = pd.DataFrame({
            'company_id': pd.Series(dtype='object'),
            'group': pd.Series(dtype='int64'),
            'llm_gemini_2_5_flash': pd.Series(dtype='int64'),
            'llm_sonnet_4': pd.Series(dtype='int64'),
            'llm_gpt_5_mini': pd.Series(dtype='int64'),
            'filter_broad_yes': pd.Series(dtype='int64'),
            'filter_strict_no': pd.Series(dtype='int64'),
            'filter_broad_yes_strict_no': pd.Series(dtype='int64')
        })
        return empty_df
    
    logger.info("Preparing test data for comparator Non-ML pipeline...")
    
    annotator_cols = [col for col in dawid_skene_filtered_dataset_comparator_non_ml.columns 
                     if col not in ['public_identifier', 'company_id', 'company_name']]
    
    test_df = dawid_skene_filtered_dataset_comparator_non_ml[['company_id'] + annotator_cols].copy()
    
    unique_companies = sorted(test_df['company_id'].unique())
    company_to_group = {company_id: idx for idx, company_id in enumerate(unique_companies)}
    test_df['group'] = test_df['company_id'].map(company_to_group)
    
    annotator_rename = {
        'llm_gemini-2.5-flash': 'llm_gemini_2_5_flash',
        'llm_sonnet-4': 'llm_sonnet_4',
        'llm_gpt-5-mini': 'llm_gpt_5_mini',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no'
    }
    
    test_df = test_df.rename(columns=annotator_rename)
    
    # Convert all annotator columns to float64, preserving NaN values for missing annotations
    # This ensures consistent dtype (float64) for all annotators, allowing NaN for missing values
    for col in annotator_rename.values():
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').astype('float64')
        # Keep as float64 to preserve NaN values - they'll be handled in the bootstrap
    
    logger.info(f"Comparator Non-ML test data: {len(test_df)} rows across {len(unique_companies)} companies")
    return test_df


@save_to.csv(path=source("real_employee_level_data_all_output_path"))
@cache()
def real_employee_level_data_all(
    dawid_skene_test_data_main: pd.DataFrame,
    dawid_skene_test_data_comparator_ml: pd.DataFrame,
    dawid_skene_test_data_comparator_non_ml: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all real employee-level datasets with dataset_source column.
    
    Combines main, comparator_ml, and comparator_non_ml datasets into a single
    DataFrame with a dataset_source column indicating origin. Recomputes group
    numbers sequentially across all companies to ensure uniqueness.
    
    Args:
        dawid_skene_test_data_main: Main dataset employee data
        dawid_skene_test_data_comparator_ml: Comparator ML dataset employee data
        dawid_skene_test_data_comparator_non_ml: Comparator Non-ML dataset employee data
        
    Returns:
        Combined DataFrame with dataset_source column and recomputed group numbers
    """
    logger.info("Combining all real employee-level datasets...")
    
    # Add dataset_source column to each dataset
    main_df = dawid_skene_test_data_main.copy()
    main_df['dataset_source'] = 'main'
    
    comp_ml_df = dawid_skene_test_data_comparator_ml.copy()
    comp_ml_df['dataset_source'] = 'comparator_ml'
    
    comp_non_ml_df = dawid_skene_test_data_comparator_non_ml.copy()
    comp_non_ml_df['dataset_source'] = 'comparator_non_ml'
    
    # Concatenate all datasets
    combined_df = pd.concat([main_df, comp_ml_df, comp_non_ml_df], ignore_index=True)
    
    # Recompute group sequentially across all companies
    # This ensures unique groups (0 to N-1) matching the number of unique companies
    unique_companies = sorted(combined_df['company_id'].unique())
    company_to_group = {company_id: idx for idx, company_id in enumerate(unique_companies)}
    combined_df['group'] = combined_df['company_id'].map(company_to_group)
    
    # Reorder columns: company_id, group, dataset_source, then annotators
    annotator_cols = [
        'llm_gemini_2_5_flash', 'llm_sonnet_4', 'llm_gpt_5_mini',
        'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no'
    ]
    column_order = ['company_id', 'group', 'dataset_source'] + annotator_cols
    combined_df = combined_df[column_order]
    
    logger.info(f"Combined real employee data: {len(combined_df):,} rows across {len(unique_companies)} companies")
    logger.info(f"  - Main: {(combined_df['dataset_source'] == 'main').sum():,} rows")
    logger.info(f"  - Comparator ML: {(combined_df['dataset_source'] == 'comparator_ml').sum():,} rows")
    logger.info(f"  - Comparator Non-ML: {(combined_df['dataset_source'] == 'comparator_non_ml').sum():,} rows")
    
    return combined_df


@config.when(fake_data_enable=True)
def fake_data_ground_truth(
    fake_data_test_prevalence_log_mean: float,
    fake_data_test_prevalence_log_std: float,
    fake_data_test_prevalence_log_min: float,
    fake_data_test_prevalence_log_max: float,
    fake_data_validation_observed_positive_rate: float,
    fake_data_annotator_filter_broad_yes_sensitivity: float,
    fake_data_annotator_filter_broad_yes_specificity: float,
    fake_data_annotator_filter_strict_no_sensitivity: float,
    fake_data_annotator_filter_strict_no_specificity: float,
    fake_data_annotator_filter_broad_yes_strict_no_sensitivity: float,
    fake_data_annotator_filter_broad_yes_strict_no_specificity: float,
    fake_data_annotator_llm_gemini_2_5_flash_sensitivity: float,
    fake_data_annotator_llm_gemini_2_5_flash_specificity: float,
    fake_data_annotator_llm_sonnet_4_sensitivity: float,
    fake_data_annotator_llm_sonnet_4_specificity: float,
    fake_data_annotator_llm_gpt_5_mini_sensitivity: float,
    fake_data_annotator_llm_gpt_5_mini_specificity: float,
    ds_random_seed: int,
    fake_data_test_num_companies: int
) -> Dict[str, Any]:
    """
    Generate ground truth parameters for fake data (test and validation).
    
    Only runs when fake_data_enable=True (via @config.when decorator).
    
    This creates the true parameters for both test and validation data:
    - Test data: Company prevalences using double-truncated lognormal + annotator confusion matrices
    - Validation data: True labels to match target positive rate + same confusion matrices
    
    Args:
        fake_data_test_prevalence_log_mean: Mean prevalence in log10 space for test data
        fake_data_test_prevalence_log_std: Standard deviation in log10 space for test data
        fake_data_test_prevalence_log_min: Lower truncation in log10 space for test data
        fake_data_test_prevalence_log_max: Upper truncation in log10 space for test data
        fake_data_validation_observed_positive_rate: Target positive rate for validation data
        fake_data_annotator_*_sensitivity: True positive rate for each annotator
        fake_data_annotator_*_specificity: True negative rate for each annotator
        ds_random_seed: Random seed for reproducibility
        fake_data_test_num_companies: Number of companies to generate for test data
        
    Returns:
        Dictionary with ground truth parameters for both test and validation data
    """
    logger.info(f"Generating ground truth for fake data (seed={ds_random_seed})...")
    logger.info(f"  Test data: {fake_data_test_num_companies} companies, lognormal(μ={fake_data_test_prevalence_log_mean}, σ={fake_data_test_prevalence_log_std}) truncated to [10^{fake_data_test_prevalence_log_min}, 10^{fake_data_test_prevalence_log_max}]")
    logger.info(f"  Validation data: target positive rate = {fake_data_validation_observed_positive_rate:.1%}")
    
    # Annotator names in order (must match everywhere)
    annotator_names = [
        'llm_gemini_2_5_flash',
        'llm_sonnet_4',
        'llm_gpt_5_mini',
        'filter_broad_yes',
        'filter_strict_no',
        'filter_broad_yes_strict_no'
    ]
    
    sensitivities_dict = {
        'llm_gemini_2_5_flash': fake_data_annotator_llm_gemini_2_5_flash_sensitivity,
        'llm_sonnet_4': fake_data_annotator_llm_sonnet_4_sensitivity,
        'llm_gpt_5_mini': fake_data_annotator_llm_gpt_5_mini_sensitivity,
        'filter_broad_yes': fake_data_annotator_filter_broad_yes_sensitivity,
        'filter_strict_no': fake_data_annotator_filter_strict_no_sensitivity,
        'filter_broad_yes_strict_no': fake_data_annotator_filter_broad_yes_strict_no_sensitivity,
    }
    
    specificities_dict = {
        'llm_gemini_2_5_flash': fake_data_annotator_llm_gemini_2_5_flash_specificity,
        'llm_sonnet_4': fake_data_annotator_llm_sonnet_4_specificity,
        'llm_gpt_5_mini': fake_data_annotator_llm_gpt_5_mini_specificity,
        'filter_broad_yes': fake_data_annotator_filter_broad_yes_specificity,
        'filter_strict_no': fake_data_annotator_filter_strict_no_specificity,
        'filter_broad_yes_strict_no': fake_data_annotator_filter_broad_yes_strict_no_specificity,
    }
    
    # Generate test data company prevalences using double-truncated lognormal distribution
    rng = np.random.default_rng(ds_random_seed + 1)  # Different seed from validation
    
    # Convert truncation bounds to standard normal space
    a = (fake_data_test_prevalence_log_min - fake_data_test_prevalence_log_mean) / fake_data_test_prevalence_log_std
    b = (fake_data_test_prevalence_log_max - fake_data_test_prevalence_log_mean) / fake_data_test_prevalence_log_std
    
    # Sample from truncated normal distribution in log10 space
    log_prevalences = truncnorm.rvs(
        a=a, b=b, 
        loc=fake_data_test_prevalence_log_mean, 
        scale=fake_data_test_prevalence_log_std,
        size=fake_data_test_num_companies,
        random_state=rng
    )
    test_company_prevalences = 10 ** log_prevalences
    
    ground_truth = {
        'test_data': {
            'company_prevalences': test_company_prevalences,
            'sensitivities': sensitivities_dict,
            'specificities': specificities_dict
        },
        'validation_data': {
            'observed_positive_rate': fake_data_validation_observed_positive_rate,
            'sensitivities': sensitivities_dict,  # Same as test
            'specificities': specificities_dict   # Same as test
        }
    }
    
    logger.info(f"Generated ground truth:")
    logger.info(f"  Test data - Company prevalences (linear): min={test_company_prevalences.min():.3e}, median={np.median(test_company_prevalences):.3e}, max={test_company_prevalences.max():.3f}")
    logger.info(f"  Test data - Company prevalences (log10): min={np.log10(test_company_prevalences.min()):.2f}, median={np.log10(np.median(test_company_prevalences)):.2f}, max={np.log10(test_company_prevalences.max()):.2f}")
    logger.info(f"  Test data - Lognormal parameters: μ={fake_data_test_prevalence_log_mean:.2f}, σ={fake_data_test_prevalence_log_std:.2f}")
    logger.info(f"  Validation data - Target positive rate: {fake_data_validation_observed_positive_rate:.1%}")
    logger.info(f"  Annotator sensitivities: {sensitivities_dict}")
    logger.info(f"  Annotator specificities: {specificities_dict}")
    
    return ground_truth


@save_to.csv(
    path=source('dawid_skene_test_data_output_path')
)
@config.when(fake_data_enable=True)
@cache()
@check_output(schema=DawidSkeneTestDataSchema, importance="fail")
def dawid_skene_test_data__fake(
    fake_data_ground_truth: Dict[str, Any],
    fake_data_test_num_profiles: int,
    ds_random_seed: int,
    fake_data_test_num_companies: int
) -> DataFrame[DawidSkeneTestDataSchema]:
    """
    Generate fake test data for Dawid-Skene model evaluation.
    
    This generates synthetic test data using the ground truth parameters
    for company prevalences and annotator confusion matrices.
    
    Args:
        fake_data_ground_truth: Ground truth parameters for both test and validation data
        fake_data_test_num_profiles: Number of fake test profiles to generate
        ds_random_seed: Random seed for reproducibility
        fake_data_test_num_companies: Number of companies to generate
        
    Returns:
        DataFrame with fake test data matching the schema
    """
    logger.info(f"Generating fake test data with {fake_data_test_num_profiles} profiles across {fake_data_test_num_companies} companies (seed={ds_random_seed})...")
    
    # Set random seed
    rng = np.random.default_rng(ds_random_seed + 1)  # Different seed from validation
    
    # Extract test data ground truth
    test_ground_truth = fake_data_ground_truth['test_data']
    company_prevalences = test_ground_truth['company_prevalences']
    sensitivities_dict = test_ground_truth['sensitivities']
    specificities_dict = test_ground_truth['specificities']
    
    # Annotator names in order
    annotator_names = list(sensitivities_dict.keys())
    sensitivities = [sensitivities_dict[name] for name in annotator_names]
    specificities = [specificities_dict[name] for name in annotator_names]
    
    # Distribute profiles across companies (roughly equal, but with some variation)
    base_profiles_per_company = fake_data_test_num_profiles // fake_data_test_num_companies
    extra_profiles = fake_data_test_num_profiles % fake_data_test_num_companies
    profiles_per_company = np.full(fake_data_test_num_companies, base_profiles_per_company)
    profiles_per_company[:extra_profiles] += 1
    
    # Generate test data for each company
    all_data = []
    
    for company_idx in range(fake_data_test_num_companies):
        n_profiles = profiles_per_company[company_idx]
        prevalence = company_prevalences[company_idx]
        company_id = f'company_{company_idx:03d}'
        
        # Generate true labels for this company
        true_labels = rng.binomial(1, prevalence, size=n_profiles)
        
        # Generate annotations for each profile
        company_annotations = {annotator: [] for annotator in annotator_names}
        
        for true_label in true_labels:
            for annotator, sens, spec in zip(annotator_names, sensitivities, specificities):
                # Generate annotation based on confusion matrix
                if true_label == 1:
                    # P(Y=1 | Z=1) = sensitivity
                    annotation = rng.binomial(1, sens)
                else:
                    # P(Y=1 | Z=0) = 1 - specificity
                    annotation = rng.binomial(1, 1 - spec)
                
                company_annotations[annotator].append(annotation)
        
        # Create DataFrame for this company
        company_df = pd.DataFrame({
            'company_id': company_id,
            'group': company_idx,
            **company_annotations
        })
        
        all_data.append(company_df)
    
    # Combine all companies
    test_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Generated fake test data: {len(test_df)} rows with {len(annotator_names)} annotators across {fake_data_test_num_companies} companies")
    logger.info(f"  - Company prevalences: min={company_prevalences.min():.3f}, mean={company_prevalences.mean():.3f}, max={company_prevalences.max():.3f}")
    logger.info(f"  - Profiles per company: min={profiles_per_company.min()}, mean={profiles_per_company.mean():.1f}, max={profiles_per_company.max()}")
    
    return test_df


@save_to.csv(
    path=source('dawid_skene_validation_data_output_path')
)
@config.when(fake_data_enable=True)
@cache()
@check_output(schema=DawidSkeneValidationDataSchema, importance="fail")
def dawid_skene_validation_data__fake(
    fake_data_ground_truth: Dict[str, Any],
    fake_data_validation_num_profiles: int,
    ds_random_seed: int
) -> DataFrame[DawidSkeneValidationDataSchema]:
    """
    Generate fake validation data for Dawid-Skene model evaluation.
    
    This generates synthetic validation data using two-stage generation:
    1. Generate true labels to match target positive rate
    2. Generate annotations using confusion matrices
    
    Args:
        fake_data_ground_truth: Ground truth parameters for both test and validation data
        fake_data_validation_num_profiles: Number of fake validation profiles to generate
        ds_random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with fake validation data matching the schema
    """
    logger.info(f"Generating fake validation data with {fake_data_validation_num_profiles} profiles (seed={ds_random_seed})...")
    
    # Set random seed
    rng = np.random.default_rng(ds_random_seed + 2)  # Different seed from test data
    
    # Extract validation data ground truth
    validation_ground_truth = fake_data_ground_truth['validation_data']
    target_positive_rate = validation_ground_truth['observed_positive_rate']
    sensitivities_dict = validation_ground_truth['sensitivities']
    specificities_dict = validation_ground_truth['specificities']
    
    # Annotator names in order
    annotator_names = list(sensitivities_dict.keys())
    sensitivities = [sensitivities_dict[name] for name in annotator_names]
    specificities = [specificities_dict[name] for name in annotator_names]
    
    # Stage 1: Generate true labels to match target positive rate
    num_positive = int(fake_data_validation_num_profiles * target_positive_rate)
    true_labels = np.zeros(fake_data_validation_num_profiles, dtype=int)
    true_labels[:num_positive] = 1
    rng.shuffle(true_labels)  # Randomize order
    
    # Stage 2: Generate annotations using confusion matrices
    annotations = {}
    for annotator, sens, spec in zip(annotator_names, sensitivities, specificities):
        annotator_labels = []
        for true_label in true_labels:
            if true_label == 1:
                # P(Y=1 | Z=1) = sensitivity
                annotation = rng.binomial(1, sens)
            else:
                # P(Y=1 | Z=0) = 1 - specificity
                annotation = rng.binomial(1, 1 - spec)
            annotator_labels.append(annotation)
        annotations[annotator] = annotator_labels
    
    # Create DataFrame
    # Note: category is the true label column (0 or 1), cv_text is required by schema
    validation_df = pd.DataFrame({
        'cv_text': [f'Fake CV text for validation profile {i:06d}' for i in range(fake_data_validation_num_profiles)],
        'category': true_labels,  # Use true_labels as category (the ground truth)
        **annotations
    })
    
    # Convert annotator columns to int
    for annotator in annotator_names:
        validation_df[annotator] = validation_df[annotator].astype(int)
    
    actual_positive_rate = true_labels.mean()
    logger.info(f"Generated fake validation data: {len(validation_df)} rows with {len(annotator_names)} annotators")
    logger.info(f"  - Target positive rate: {target_positive_rate:.1%}")
    logger.info(f"  - Actual positive rate: {actual_positive_rate:.1%}")
    logger.info(f"  - True labels: {true_labels.sum()} positive, {len(true_labels) - true_labels.sum()} negative")
    
    return validation_df


@load_from.excel(
    path=source('company_database_complete_file_path'),
    sheet_name=source('company_database_complete_sheet_name')
)
def systematic_search_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load systematic search data for ground truth comparison.
    
    This loads the same file as company_database_complete but returns it
    in raw form (with original column names) for use in plotting.
    
    Returns:
        DataFrame with systematic search data including estimator columns
    """
    logger.info(f"Loaded systematic search data: {len(df)} companies")
    return df


# ============================================================================
# LOG-DEBIASING OUTPUT PATHS
# ============================================================================

def log_debias_output_paths(data_dir: str) -> dict:
    """
    Define output paths for log-debiasing results.
    
    Args:
        data_dir: Base data directory
    
    Returns:
        Dictionary of output paths
    """
    log_debias_dir = os.path.join(data_dir, "log_debias")
    plots_dir = os.path.join(log_debias_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    return {
        "log_debias_company_aggregates_output_path": os.path.join(log_debias_dir, "log_debias_company_aggregates.csv"),
        "log_debias_summary_output_path": os.path.join(log_debias_dir, "log_debias_summary.json"),
        "log_debias_all_orgs_output_path": os.path.join(log_debias_dir, "log_debias_all_orgs.csv"),
        "log_debias_orgs_ml_output_path": os.path.join(log_debias_dir, "log_debias_orgs_ml.csv"),
        "log_debias_orgs_talent_dense_output_path": os.path.join(log_debias_dir, "log_debias_orgs_talent_dense.csv"),
        "log_debias_orgs_stage5_work_trial_output_path": os.path.join(log_debias_dir, "log_debias_orgs_stage5_work_trial.csv"),
        "log_debias_orgs_enterprise_500ml_0p5pct_output_path": os.path.join(log_debias_dir, "log_debias_orgs_enterprise_500ml_0p5pct.csv"),
        "log_debias_orgs_midscale_50ml_1pct_output_path": os.path.join(log_debias_dir, "log_debias_orgs_midscale_50ml_1pct.csv"),
        "log_debias_orgs_boutique_10ml_5pct_output_path": os.path.join(log_debias_dir, "log_debias_orgs_boutique_10ml_5pct.csv"),
        "log_debias_orgs_stage5_work_trial_recommended_output_path": os.path.join(log_debias_dir, "log_debias_orgs_stage5_work_trial_recommended.csv"),
        "log_debias_plots_output_path": plots_dir,
        "log_debias_uncertainty_plots_output_path": plots_dir
    }

