"""
Pandera schemas for ML Headcount Pipeline

This module defines comprehensive data validation schemas using Pandera
for all input data, intermediate data, and output data in the pipeline.
"""

import pandera.pandas as pa
from pandera.typing import DataFrame, Series
from typing import Optional

# ============================================================================
# INPUT DATA SCHEMAS
# ============================================================================

class CVDataSchema(pa.DataFrameModel):
    """Schema for CV data input"""
    cv_text: Series[str] = pa.Field(description="CV text content")
    category: Series[str] = pa.Field(description="Category classification")

class ValidationCVsSchema(pa.DataFrameModel):
    """Schema for validation CVs input with LLM annotations"""
    cv_text: Series[str] = pa.Field(description="CV text content")
    category: Series[int] = pa.Field(description="Category classification")
    # LLM annotators (all using prompt 1)
    class Config:
        strict = False  # Allow extra columns for LLM results with varying names

class AffiliationDataSchema(pa.DataFrameModel):
    """Schema for affiliation data input"""
    affiliation: Series[str] = pa.Field( description="Organization affiliation")
    count: Series[int] = pa.Field(ge=0, description="Affiliation count")

class CompanyDatabaseBaseSchema(pa.DataFrameModel):
    """Base schema for company database input (without ML estimates)"""
    id: Series[str] = pa.Field( description="Company identifier")
    employees: Series[int] = pa.Field(ge=0, description="Employee count")
    company_size: Series[str] = pa.Field( description="Company size category")
    employees_in_linkedin: Series[int] = pa.Field(ge=0, description="LinkedIn employee count")

class CompanyDatabaseCompleteSchema(pa.DataFrameModel):
    """Schema for complete company database with all required columns"""
    id: Series[int] = pa.Field(description="Company identifier")
    organization_name: Series[str] = pa.Field(description="Organization name")
    linkedin_id: Series[str] = pa.Field(nullable=True, description="LinkedIn company identifier")
    headquarters_location: Series[str] = pa.Field(nullable=True, description="Headquarters location")
    country: Series[str] = pa.Field(nullable=True, description="Country")
    subregion: Series[str] = pa.Field(nullable=True, description="Subregion")
    stage_reached: Series[str] = pa.Field(nullable=True, description="Stage reached")
    category: Series[str] = pa.Field(nullable=True, description="Category classification")
    total_headcount: Series[int] = pa.Field(ge=0, description="Total headcount")
    max_headcount: Series[bool] = pa.Field(nullable=True, description="Maximum headcount flag")
    max_population: Series[bool] = pa.Field(nullable=True, description="Maximum population flag")
    # ML estimation columns
    filter_broad_yes: Series[float] = pa.Field(nullable=True, ge=0)
    filter_strict_no: Series[float] = pa.Field(nullable=True, ge=0)
    filter_broad_yes_strict_no: Series[float] = pa.Field(nullable=True, ge=0)
    claude_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    gpt5_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    gemini_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    # Additional columns for reporting
    ml_share: Series[float] = pa.Field(nullable=True, ge=0)
    ml_share_lower80: Series[float] = pa.Field(nullable=True, ge=0)
    ml_share_upper80: Series[float] = pa.Field(nullable=True, ge=0)

class LinkedInDataSchema(pa.DataFrameModel):
    """Schema for processed LinkedIn data"""
    public_identifier: Series[str] = pa.Field(description="LinkedIn profile identifier")
    profile_summary: Series[str] = pa.Field(description="Profile summary text")
    company_id: Series[str] = pa.Field(description="Current company identifier")
    company_name: Series[str] = pa.Field(description="Current company name")

class CompanySizeDataSchema(pa.DataFrameModel):
    """Schema for company size data used in Dawid-Skene analysis"""
    company_id: Series[str] = pa.Field( description="Company identifier")
    company_name: Series[str] = pa.Field( description="Company name")
    company_size: Series[str] = pa.Field( description="Company size category")
    linkedin_employees: Series[int] = pa.Field(ge=0, description="LinkedIn employee count")
    population_size: Series[int] = pa.Field(ge=1, description="Population size for Dawid-Skene model")
    company_index: Series[int] = pa.Field(ge=0, description="Company index for model")
    ds_group_index: Series[int] = pa.Field(ge=0, description="Dawid-Skene group index (same as company_index for consistency)")

class DawidSkeneCompressedDataSchema(pa.DataFrameModel):
    """Schema for compressed Dawid-Skene annotation data with variable number of annotators"""
    company_idx: Series[int] = pa.Field(ge=0, description="Company index for model")
    count: Series[int] = pa.Field(ge=1, description="Count of this pattern for this company")
    
    @pa.check("company_idx")
    def check_company_idx(cls, series):
        return series >= 0
    
    @pa.check("count")
    def check_count(cls, series):
        return series >= 1

# ============================================================================
# INTERMEDIATE DATA SCHEMAS
# ============================================================================

class KeywordFrequenciesSchema(pa.DataFrameModel):
    """Schema for CV keyword frequencies"""
    phrase: Series[str] = pa.Field(description="Extracted phrase")
    count: Series[int] = pa.Field(ge=1, description="Phrase count")

class DiscriminativeKeywordsSchema(pa.DataFrameModel):
    """Schema for discriminative keywords"""
    category: Series[int] = pa.Field(description="Category (0 or 1)")
    keyword: Series[str] = pa.Field( description="Discriminative keyword")
    score: Series[float] = pa.Field(ge=0, le=1, description="Discriminative score")
    # Additional columns required by keyword_filtering script
    total_raw_frequency: Series[int] = pa.Field(ge=0, description="Total raw frequency")
    discriminative_score: Series[float] = pa.Field(ge=0, le=1, description="Discriminative score")
    raw_category_specificity: Series[float] = pa.Field(ge=0, le=1, description="Raw category specificity")

class KeywordsListsSchema(pa.DataFrameModel):
    """Schema for keyword lists"""
    strict_yes: Series[str] = pa.Field(nullable=True, description="Strict yes keywords")
    strict_no: Series[str] = pa.Field(nullable=True, description="Strict no keywords")
    broad_yes: Series[str] = pa.Field(nullable=True, description="Broad yes keywords")
    broad_no: Series[str] = pa.Field(nullable=True, description="Broad no keywords")

class ValidationCVsScoredSchema(pa.DataFrameModel):
    """Schema for scored validation CVs"""
    cv_text: Series[str] = pa.Field(description="CV text content")
    category: Series[int] = pa.Field(description="Category (0 or 1)")
    # Scoring method columns (will be added dynamically) - using int64 for 0/1 values
    strict_yes_strict_no: Series[int] = pa.Field(nullable=True, ge=0, le=1)
    strict_yes_broad_no: Series[int] = pa.Field(nullable=True, ge=0, le=1)
    broad_yes_broad_no: Series[int] = pa.Field(nullable=True, ge=0, le=1)
    broad_yes_strict_no: Series[int] = pa.Field(nullable=True, ge=0, le=1)
    strict_yes_only: Series[int] = pa.Field(nullable=True, ge=0, le=1)
    broad_yes_only: Series[int] = pa.Field(nullable=True, ge=0, le=1)
    strict_no_only: Series[int] = pa.Field(nullable=True, ge=0, le=1)
    broad_no_only: Series[int] = pa.Field(nullable=True, ge=0, le=1)

class ConfusionMatrixMetricsSchema(pa.DataFrameModel):
    """Schema for confusion matrix metrics"""
    method: Series[str] = pa.Field(description="Scoring method name")
    TP: Series[int] = pa.Field(ge=0, description="True Positives")
    FP: Series[int] = pa.Field(ge=0, description="False Positives")
    FN: Series[int] = pa.Field(ge=0, description="False Negatives")
    TN: Series[int] = pa.Field(ge=0, description="True Negatives")
    sensitivity: Series[float] = pa.Field(ge=0, le=1, description="Sensitivity (True Positive Rate)")
    specificity: Series[float] = pa.Field(ge=0, le=1, description="Specificity (True Negative Rate)")
    precision: Series[float] = pa.Field(ge=0, le=1, description="Precision")
    recall: Series[float] = pa.Field(ge=0, le=1, description="Recall")
    F1: Series[float] = pa.Field(ge=0, le=1, description="F1 Score")
    accuracy: Series[float] = pa.Field(ge=0, le=1, description="Accuracy")

class CompanyDatabaseWithSubregionsSchema(pa.DataFrameModel):
    """Schema for company database with geographic subregions"""
    id: Series[int] = pa.Field(description="Company identifier")
    organization_name: Series[str] = pa.Field(description="Organization name")
    headquarters_location: Series[str] = pa.Field(nullable=True, description="Headquarters location")
    country: Series[str] = pa.Field(nullable=True, description="Country")
    subregion: Series[str] = pa.Field(nullable=True, description="Subregion")
    stage_reached: Series[str] = pa.Field(nullable=True, description="Stage reached")
    category: Series[str] = pa.Field(nullable=True, description="Category classification")
    total_headcount: Series[int] = pa.Field(ge=0, description="Total headcount")
    # ML estimation columns
    filter_broad_yes: Series[float] = pa.Field(nullable=True, ge=0)
    filter_strict_no: Series[float] = pa.Field(nullable=True, ge=0)
    filter_broad_yes_strict_no: Series[float] = pa.Field(nullable=True, ge=0)
    claude_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    gpt5_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    gemini_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    # Additional columns for reporting (may be in percentage format > 1)
    ml_share: Series[float] = pa.Field(nullable=True, ge=0)
    ml_share_lower80: Series[float] = pa.Field(nullable=True, ge=0)
    ml_share_upper80: Series[float] = pa.Field(nullable=True, ge=0)

class DebiasedOrganizationsSchema(pa.DataFrameModel):
    """Schema for debiased organizations"""
    id: Series[int] = pa.Field(description="Company identifier")
    organization_name: Series[str] = pa.Field(description="Organization name")
    headquarters_location: Series[str] = pa.Field(nullable=True, description="Headquarters location")
    country: Series[str] = pa.Field(nullable=True, description="Country")
    subregion: Series[str] = pa.Field(nullable=True, description="Subregion")
    stage_reached: Series[str] = pa.Field(nullable=True, description="Stage reached")
    category: Series[str] = pa.Field(nullable=True, description="Category classification")
    total_headcount: Series[int] = pa.Field(ge=0, description="Total headcount")
    # ML estimation columns
    filter_broad_yes: Series[float] = pa.Field(nullable=True, ge=0)
    filter_strict_no: Series[float] = pa.Field(nullable=True, ge=0)
    filter_broad_yes_strict_no: Series[float] = pa.Field(nullable=True, ge=0)
    claude_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    gpt5_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    gemini_total_accepted: Series[float] = pa.Field(nullable=True, ge=0)
    # Additional columns for reporting
    ml_share: Series[float] = pa.Field(nullable=True, ge=0)
    ml_share_lower80: Series[float] = pa.Field(nullable=True, ge=0)
    ml_share_upper80: Series[float] = pa.Field(nullable=True, ge=0)

# ============================================================================
# OUTPUT DATA SCHEMAS
# ============================================================================

class SummaryStatisticsSchema(pa.DataFrameModel):
    """Schema for summary statistics output"""
    metric: Series[str] = pa.Field( description="Statistical metric name")
    value: Series[float] = pa.Field(ge=0, description="Metric value")
    category: Series[str] = pa.Field(nullable=True, description="Category if applicable")

class OrganizationsListSchema(pa.DataFrameModel):
    """Schema for organizations list output"""
    company_name: Series[str] = pa.Field( description="Company name")
    employees: Series[float] = pa.Field(ge=0, description="Employee count")
    headquarters_location: Series[str] = pa.Field( description="Headquarters location")
    country: Series[str] = pa.Field( description="Country")
    subregion: Series[str] = pa.Field( description="Subregion")
    consensus_ml_estimate: Series[float] = pa.Field(ge=0, description="Consensus ML estimate")
    confidence_interval_lower: Series[float] = pa.Field(ge=0, description="Lower confidence bound")
    confidence_interval_upper: Series[float] = pa.Field(ge=0, description="Upper confidence bound")

class KeywordVisualizationSchema(pa.DataFrameModel):
    """Schema for keyword visualization data"""
    visualization_type: Series[str] = pa.Field(description="Type of visualization")
    file_path: Series[str] = pa.Field(description="Path to visualization file")
    category: Series[str] = pa.Field(nullable=True, description="Category for category-specific visualizations")

class KeywordExtractionResultsSchema(pa.DataFrameModel):
    """Schema for keyword extraction results"""
    keyword: Series[str] = pa.Field(description="Extracted keyword")
    category: Series[int] = pa.Field(description="Category classification")
    frequency: Series[int] = pa.Field(ge=1, description="Keyword frequency")
    score: Series[float] = pa.Field(ge=-1, le=1, description="Keyword score (can be negative)")

class DawidSkeneValidationDataSchema(pa.DataFrameModel):
    """Schema for Dawid-Skene validation data with 6 annotators"""
    cv_text: Series[str] = pa.Field(description="CV text content")
    category: Series[int] = pa.Field(ge=0, le=1, description="True binary label (0 or 1)")
    # 6 annotators in order: 3 LLMs + 3 keyword filters
    llm_gemini_2_5_flash: Series[int] = pa.Field(ge=0, le=1, description="Gemini 2.5 Flash LLM annotator")
    llm_sonnet_4: Series[int] = pa.Field(ge=0, le=1, description="Claude Sonnet 4 LLM annotator")
    llm_gpt_5_mini: Series[int] = pa.Field(ge=0, le=1, description="GPT-5 Mini LLM annotator")
    filter_broad_yes: Series[int] = pa.Field(ge=0, le=1, description="Broad keyword filter")
    filter_strict_no: Series[int] = pa.Field(ge=0, le=1, description="Strict keyword filter")
    filter_broad_yes_strict_no: Series[int] = pa.Field(ge=0, le=1, description="Combined keyword filter")

class DawidSkeneTestDataSchema(pa.DataFrameModel):
    """Schema for Dawid-Skene test data with 6 annotators
    
    Supports missing values (NaN) for annotators, which are handled via marginalization
    in the probit bootstrap pipeline.
    """
    company_id: Series[str] = pa.Field(description="Company identifier")
    group: Series[int] = pa.Field(ge=0, description="Group identifier")
    # 6 annotators in order: 3 LLMs + 3 keyword filters (same order as validation)
    # Allow nullable float to preserve NaN values for missing annotations
    llm_gemini_2_5_flash: Series[float] = pa.Field(nullable=True, ge=0, le=1, description="Gemini 2.5 Flash LLM annotator (0/1/NaN)")
    llm_sonnet_4: Series[float] = pa.Field(nullable=True, ge=0, le=1, description="Claude Sonnet 4 LLM annotator (0/1/NaN)")
    llm_gpt_5_mini: Series[float] = pa.Field(nullable=True, ge=0, le=1, description="GPT-5 Mini LLM annotator (0/1/NaN)")
    filter_broad_yes: Series[float] = pa.Field(nullable=True, ge=0, le=1, description="Broad keyword filter (0/1/NaN)")
    filter_strict_no: Series[float] = pa.Field(nullable=True, ge=0, le=1, description="Strict keyword filter (0/1/NaN)")
    filter_broad_yes_strict_no: Series[float] = pa.Field(nullable=True, ge=0, le=1, description="Combined keyword filter (0/1/NaN)")

class TitlesDataSchema(pa.DataFrameModel):
    """Schema for job titles data"""
    title: Series[str] = pa.Field(description="Job title")
    count: Series[int] = pa.Field(ge=1, description="Title occurrence count")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_dataframe(df: DataFrame, schema: pa.DataFrameModel) -> DataFrame:
    """
    Validate a DataFrame against a Pandera schema.
    
    Args:
        df: DataFrame to validate
        schema: Pandera schema to validate against
        
    Returns:
        Validated DataFrame
        
    Raises:
        pa.errors.SchemaError: If validation fails
    """
    return schema.validate(df)

def get_schema_for_data_type(data_type: str) -> pa.DataFrameModel:
    """
    Get the appropriate schema for a given data type.
    
    Args:
        data_type: Type of data (e.g., 'cv_data', 'validation_cvs', etc.)
        
    Returns:
        Corresponding Pandera schema
    """
    schema_mapping = {
        'cv_data': CVDataSchema,
        'validation_cvs': ValidationCVsSchema,
        'affiliation_data': AffiliationDataSchema,
        'company_database': CompanyDatabaseCompleteSchema,
        'linkedin_data': LinkedInDataSchema,
        'keyword_frequencies': KeywordFrequenciesSchema,
        'discriminative_keywords': DiscriminativeKeywordsSchema,
        'keywords_lists': KeywordsListsSchema,
        'validation_cvs_scored': ValidationCVsScoredSchema,
        'company_database_with_subregions': CompanyDatabaseWithSubregionsSchema,
        'debiased_organizations': DebiasedOrganizationsSchema,
        'summary_statistics': SummaryStatisticsSchema,
        'organizations_list': OrganizationsListSchema,
        'keyword_visualization': KeywordVisualizationSchema,
    }
    
    if data_type not in schema_mapping:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return schema_mapping[data_type]
