"""
Hamilton processors for ML Headcount Pipeline

This module contains all processing functions that transform data
through the pipeline using Hamilton's function-based approach.
Functions can dynamically execute locally or remotely based on configuration.
"""

from hamilton.function_modifiers import check_output, save_to, source, cache
from pandera.typing import DataFrame
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import re
from pathlib import Path

from .schemas import (
    CVDataSchema, ValidationCVsSchema, AffiliationDataSchema,
    CompanyDatabaseCompleteSchema, KeywordFrequenciesSchema, DiscriminativeKeywordsSchema, KeywordsListsSchema,
    ValidationCVsScoredSchema, LinkedInDataSchema, KeywordVisualizationSchema,
    CompanySizeDataSchema, DawidSkeneCompressedDataSchema, ConfusionMatrixMetricsSchema,
    KeywordExtractionResultsSchema, DawidSkeneValidationDataSchema, DawidSkeneTestDataSchema, TitlesDataSchema
)
from .scripts.statistical.debiasing import debias_ml_headcount_estimates
from .synthetic_data_generation import (
    estimate_test_keyword_filter_correlation_matrix,
    identify_companies_needing_synthetic_data
)
import numpy as np

logger = logging.getLogger(__name__)

# Text preprocessing functions
def preprocess_text(text: str) -> str:
    """
    Preprocess text for keyword extraction: lowercase, trim whitespace, remove short standalone numbers.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    text = text.lower()  # Lowercase
    text = " ".join(text.split())  # Collapse whitespace
    text = re.sub(r"\b\d{1,2}\b", "", text)  # Remove 1-2 digit numbers
    return text

# Import Modal functions for dynamic execution
from .modal_functions import (
    run_keybert_extraction,
    run_keyword_clustering
)
from .execution_config import execution_mode_is_remote

# ============================================================================
# DYNAMIC EXECUTION CONFIGURATION
# ============================================================================

def _execute_modal_function(modal_func, *args, **kwargs):
    """Execute a Modal function either locally or remotely based on configuration.
    
    Args:
        modal_func: The Modal function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function execution
    """
    if execution_mode_is_remote():
        logger.info(f"Executing {getattr(modal_func, '__name__', 'modal_function')} remotely on Modal")
        return modal_func.remote(*args, **kwargs)
    else:
        logger.info(f"Executing {getattr(modal_func, '__name__', 'modal_function')} locally")
        return modal_func.local(*args, **kwargs)

# ============================================================================
# TEXT ANALYSIS FUNCTIONS
# ============================================================================

@save_to.csv(path=source("preprocessed_text_output_path"))
@cache()  # Cache preprocessed text
@check_output(importance="fail")
def preprocessed_text(validation_cvs: DataFrame[ValidationCVsSchema]) -> DataFrame[ValidationCVsSchema]:
    """
    Preprocess CV text for keyword extraction and analysis.
    
    Args:
        validation_cvs: Validation CV data with cv_text column
        
    Returns:
        DataFrame with additional preprocessed_text column
    """
    logger.info("Preprocessing CV text")
    
    try:
        # Create a copy to avoid modifying the original
        result = validation_cvs.copy()
        
        # Apply text preprocessing to create processed_text column
        result['processed_text'] = result['cv_text'].apply(preprocess_text)
        
        logger.info(f"Preprocessed text for {len(result)} CVs")
        return result
    except Exception as e:
        logger.error(f"Failed to preprocess text: {str(e)}")
        raise

@save_to.csv(path=source("cv_keyword_frequencies_output_path"))
@cache()  # Cache CV keyword frequencies
@check_output(importance="fail")
def cv_keyword_frequencies(validation_cvs: DataFrame[ValidationCVsSchema]) -> DataFrame[KeywordFrequenciesSchema]:
    """
    Extract CV keyword frequencies.
    
    Args:
        validation_cvs: Validation CV data with text and categories
        
    Returns:
        DataFrame with keyword frequencies by category
    """
    logger.info("Extracting CV keyword frequencies")
    
    try:
        from .scripts.data_ingestion.cv_keywords import extract_cv_keyword_frequencies
        result = extract_cv_keyword_frequencies(validation_cvs)
        logger.info(f"Extracted keyword frequencies: {len(result)} rows")
        return result
    except Exception as e:
        logger.error(f"Failed to extract CV keyword frequencies: {str(e)}")
        raise

@save_to.csv(path=source("keyword_extraction_results_output_path"))
@cache()  # Cache keyword extraction results
@check_output(schema=KeywordExtractionResultsSchema, importance="fail")
def keyword_extraction_results(
    preprocessed_text: DataFrame[ValidationCVsSchema],
    ke_model_name: str,
    ke_batch_size: int,
    ke_max_seq_length: int,
    ke_top_n: int,
    ke_ngram_max: int
) -> DataFrame[KeywordExtractionResultsSchema]:
    """
    Run KeyBERT keyword extraction using unified local/remote execution.
    
    Args:
        preprocessed_text: Preprocessed CV data with processed_text column
        ke_model_name: Sentence transformer model name
        ke_batch_size: Batch size for embeddings
        ke_max_seq_length: Maximum sequence length
        ke_top_n: Number of keywords to extract per document
        ke_ngram_max: Maximum n-gram size (e.g., 3 for 1-3 grams)
        modal_resources: Modal resource configuration dict
        
    Returns:
        DataFrame containing keyword extraction results with scores
    """
    logger.info("Running KeyBERT keyword extraction")
    
    try:
        # Use the preprocessed data directly
        cv_df_processed = preprocessed_text.copy()
        
        # Build parameters from config
        ngram_range = (1, ke_ngram_max)
        
        logger.info(f"KeyBERT parameters: model={ke_model_name}, batch_size={ke_batch_size}, "
                   f"max_seq_length={ke_max_seq_length}, top_n={ke_top_n}, ngram_range={ngram_range}")
        
        # Use unified execution - _execute_modal_function handles local vs remote
        result = _execute_modal_function(
            run_keybert_extraction,
            cv_data=cv_df_processed,
            model_name=ke_model_name,
            batch_size=ke_batch_size,
            max_seq_length=ke_max_seq_length,
            top_n=ke_top_n,
            ngram_range=ngram_range
        )
        
        logger.info("KeyBERT keyword extraction completed")
        return result
        
    except Exception as e:
        logger.error(f"Failed to run KeyBERT keyword extraction: {str(e)}")
        raise

@save_to.json(path=source("clustering_results_output_path"))
@cache()  # Cache clustering results
@check_output(importance="fail")
def clustering_results(
    keyword_extraction_results: DataFrame[KeywordExtractionResultsSchema],
    ke_model_name: str
) -> Dict[str, Any]:
    """
    Run keyword clustering (local or remote based on configuration).
    
    Args:
        keyword_extraction_results: Results from keyword extraction (DataFrame)
        
    Returns:
        Dictionary containing clustering results
    """
    if keyword_extraction_results.empty:
        logger.info("No keywords available, returning empty clustering results")
        return {
            "discriminative_keywords": {},
            "category_clusters": {},
            "keyword_cluster_data": {}
        }
    
    logger.info("Running keyword clustering")
    
    try:
        if execution_mode_is_remote():
            # Use remote Modal implementation
            if isinstance(keyword_extraction_results, pd.DataFrame):
                # Convert DataFrame to dict format for remote function
                discriminative_keywords = {}
                for _, row in keyword_extraction_results.iterrows():
                    category = row['category']
                    keyword = row['keyword']
                    score = row.get('discriminative_score', row.get('score', 0.0))
                    
                    if category not in discriminative_keywords:
                        discriminative_keywords[category] = {}
                    discriminative_keywords[category][keyword] = score
                
                keyword_data = {
                    'discriminative_keywords': discriminative_keywords,
                    'sentence_model': None
                }
            else:
                keyword_data = keyword_extraction_results
            
            # Import the function before using it
            from .modal_functions import run_keyword_clustering
            result = _execute_modal_function(
                run_keyword_clustering,
                discriminative_keywords=keyword_data.get('discriminative_keywords', {}),
                n_clusters=5,
                min_cluster_size=2,
                model_name=ke_model_name
            )
            logger.info("Keyword clustering completed remotely")
            return result
        else:
            # Use local implementation
            from .scripts.text_analysis.keybert_extraction import run_keyword_clustering
            
            if isinstance(keyword_extraction_results, pd.DataFrame):
                # Convert DataFrame to dict format expected by local function
                discriminative_keywords = {}
                for _, row in keyword_extraction_results.iterrows():
                    category = row['category']
                    keyword = row['keyword']
                    score = row.get('discriminative_score', row.get('score', 0.0))
                    
                    if category not in discriminative_keywords:
                        discriminative_keywords[category] = {}
                    discriminative_keywords[category][keyword] = score
                
                keyword_dict = {
                    "discriminative_keywords": discriminative_keywords,
                    "sentence_model": None  # Will be loaded by local function
                }
            else:
                keyword_dict = keyword_extraction_results
            
            discriminative_keywords = keyword_dict["discriminative_keywords"]
            
            # Initialize the sentence model locally
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sentence_model = SentenceTransformer(ke_model_name, device=device)
            
            result = run_keyword_clustering(discriminative_keywords, sentence_model)
            result["discriminative_keywords"] = discriminative_keywords
            logger.info("Keyword clustering completed locally")
            return result
    except Exception as e:
        logger.error(f"Failed to run keyword clustering: {str(e)}")
        raise

@save_to.csv(path=source("discriminative_keywords_output_path"))
@cache()  # Cache discriminative keywords processing
@check_output(schema=DiscriminativeKeywordsSchema, importance="fail")
def discriminative_keywords(
    keyword_extraction_results_filtered: DataFrame[KeywordExtractionResultsSchema], 
    preprocessed_text: DataFrame[ValidationCVsSchema], 
    tfidf_ngram_range: tuple,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_max_df: float,
    dk_min_score: float,
    dk_max_keywords: int
) -> DataFrame[DiscriminativeKeywordsSchema]:
    """
    Extract discriminative keywords from filtered keyword extraction results.
    
    Args:
        keyword_extraction_results_filtered: Filtered DataFrame from keyword extraction
        preprocessed_text: Preprocessed CV data for TF-IDF calculation
        tfidf_ngram_range: N-gram range for TF-IDF
        tfidf_max_features: Maximum TF-IDF features
        tfidf_min_df: Minimum document frequency for TF-IDF
        tfidf_max_df: Maximum document frequency for TF-IDF
        dk_min_score: Minimum discriminative score threshold
        dk_max_keywords: Maximum keywords per category
        
    Returns:
        DataFrame with discriminative keywords and scores
    """
    logger.info("Extracting discriminative keywords")
    
    try:
        if keyword_extraction_results_filtered.empty:
            logger.warning("No keyword extraction results found")
            return pd.DataFrame({
                    "category": pd.Series([], dtype=str),
                    "keyword": pd.Series([], dtype=str),
                    "score": pd.Series([], dtype=float),
                    "total_raw_frequency": pd.Series([], dtype=int),
                    "discriminative_score": pd.Series([], dtype=float),
                    "raw_category_specificity": pd.Series([], dtype=float)
                })
            
        # Group keywords by category
        category_keywords = {}
        for _, row in keyword_extraction_results_filtered.iterrows():
            cat = row['category']
            keyword = row['keyword']
            frequency = row['frequency']
            
            if cat not in category_keywords:
                category_keywords[cat] = {}
            category_keywords[cat][keyword] = frequency
        
        logger.info(f"Grouped keywords by category: {list(category_keywords.keys())}")
        
        # Calculate TF-IDF scores using the existing implementation
        logger.info("Calculating TF-IDF scores...")
        from .scripts.text_analysis.keybert_extraction import calculate_tfidf_scores
        
        tfidf_scores = calculate_tfidf_scores(
            preprocessed_text, 
            ngram_range=tfidf_ngram_range,
            max_features=tfidf_max_features,
            min_df=tfidf_min_df,
            max_df=tfidf_max_df
        )
        
        # Calculate discriminative scores using existing implementation
        logger.info("Calculating discriminative scores...")
        from .scripts.text_analysis.keybert_extraction import calculate_discriminative_keywords
        from collections import Counter
        
        # Convert category_keywords to Counter format expected by the function
        category_keywords_counters = {cat: Counter(keywords) for cat, keywords in category_keywords.items()}
        
        discriminative_keywords = calculate_discriminative_keywords(
            category_keywords_counters,
            tfidf_scores,
            min_score=dk_min_score,
            max_keywords=dk_max_keywords
        )
        
        logger.info(f"Calculated discriminative keywords for {len(discriminative_keywords)} categories")
        
        # Convert to DataFrame
        results_data = []
        for category, keywords in discriminative_keywords.items():
            for keyword, score in keywords.items():
                raw_freq = category_keywords_counters[category][keyword]
                results_data.append({
                    "category": category,
                    "keyword": keyword,
                    "score": score,
                    "total_raw_frequency": raw_freq,
                    "discriminative_score": score,
                    "raw_category_specificity": score
                })
            
        result = pd.DataFrame(results_data)
        logger.info(f"Extracted discriminative keywords: {len(result)} rows")
        return result
    except Exception as e:
        logger.error(f"Failed to extract discriminative keywords: {str(e)}")
        raise

@save_to.csv(path=source("keyword_lists_output_path"))
@cache()  # Cache keyword lists generation
@check_output(schema=KeywordsListsSchema, importance="fail")
def keyword_lists(discriminative_keywords: DataFrame[DiscriminativeKeywordsSchema], kf_strict_threshold: float, kf_broad_threshold: float) -> DataFrame[KeywordsListsSchema]:
    """
    Split keywords into strict/broad lists.
    
    Args:
        discriminative_keywords: DataFrame with discriminative keywords
        
    Returns:
        DataFrame with keyword lists
    """
    logger.info("Splitting keywords into strict/broad lists")
    
    try:
        # Ensure required columns exist
        if discriminative_keywords.empty:
            logger.warning("No discriminative keywords found, returning empty lists")
            return pd.DataFrame([{
                'strict_yes': '',
                'strict_no': '',
                'broad_yes': '',
                'broad_no': ''
            }])
        
        # Ensure total_raw_frequency column exists
        if 'total_raw_frequency' not in discriminative_keywords.columns:
            discriminative_keywords = discriminative_keywords.copy()
            discriminative_keywords['total_raw_frequency'] = discriminative_keywords.get('frequency', 1)
        
        # Use the exact parameters from the original script
        # First, convert numeric categories to string categories for compatibility
        df = discriminative_keywords.copy()
        df['category'] = df['category'].map({0: 'no', 1: 'yes'})
        
        # Apply the exact filtering logic from the original script
        # 1 – baseline filter on frequency
        df = df[df["total_raw_frequency"] >= 5]

        # 2 – strict vs. broad masks (exact thresholds from original script)
        # Note: Using hardcoded values to match original script exactly
        broad_mask  = (df["discriminative_score"] >= 0.8) & (df["raw_category_specificity"] >= 0.7)
        strict_mask = (df["discriminative_score"] >= 0.9) & (df["raw_category_specificity"] >= 0.8)

        broad  = df[broad_mask]
        strict = df[strict_mask]

        def to_list(series):
            """Return Python-list literal with quoted keywords."""
            return [*series.astype(str)]

        strict_yes = to_list(strict[strict["category"].str.lower() == "yes"]["keyword"])
        strict_no  = to_list(strict[strict["category"].str.lower() == "no"]["keyword"])
        broad_yes  = to_list(broad[broad["category"].str.lower() == "yes"]["keyword"])
        broad_no   = to_list(broad[broad["category"].str.lower() == "no"]["keyword"])

        print("strict_yes =", strict_yes)
        print("strict_no  =", strict_no)
        print("broad_yes  =", broad_yes)
        print("broad_no   =", broad_no)

        print("(" + " OR ".join(f'"{kw}"' for kw in strict_yes) + ")")
        print("(" + " OR ".join(f'"{kw}"' for kw in strict_no) + ")")
        print("(" + " OR ".join(f'"{kw}"' for kw in broad_yes) + ")")
        print("(" + " OR ".join(f'"{kw}"' for kw in broad_no) + ")")
        
        # Convert to DataFrame format expected by schema
        result = pd.DataFrame([{
            'strict_yes': ', '.join(strict_yes),
            'strict_no': ', '.join(strict_no), 
            'broad_yes': ', '.join(broad_yes),
            'broad_no': ', '.join(broad_no)
        }])
        
        logger.info("Keyword lists created successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to create keyword lists: {str(e)}")
        raise

@save_to.json(path=source("keyword_extraction_report_output_path"))
@cache()  # Cache keyword extraction report generation
@check_output(importance="fail")
def keyword_extraction_report(keyword_extraction_results_filtered: DataFrame[KeywordExtractionResultsSchema]) -> Dict[str, Any]:
    """
    Generate comprehensive report for filtered keyword extraction results.
    
    Args:
        keyword_extraction_results_filtered: Filtered results from keyword extraction
        
    Returns:
        Dictionary containing formatted report data
    """
    logger.info("Generating keyword extraction report...")
    
    try:
        # The function receives a DataFrame directly
        results_df = keyword_extraction_results_filtered.copy()
        
        if results_df.empty:
            logger.warning("No keywords found in results")
            return {"error": "No keywords found"}
        
        # Calculate comprehensive statistics
        stats = {
            "total_keywords": len(results_df),
            "categories": results_df['category'].unique().tolist(),
            "keywords_per_category": results_df.groupby('category').size().to_dict(),
            "avg_frequency_per_category": results_df.groupby('category')['frequency'].mean().to_dict(),
            "top_keywords_per_category": {}
        }
        
        # Calculate top keywords per category
        for category in stats["categories"]:
            cat_data = results_df[results_df['category'] == category]
            top_keywords = cat_data.nlargest(5, 'frequency')[['keyword', 'frequency']]
            stats["top_keywords_per_category"][category] = top_keywords.to_dict('records')
        
        # Generate formatted report (convert DataFrame to dict for JSON serialization)
        report = {
            "statistics": stats,
            "sample_results": results_df.head(10).to_dict('records'),
            "report_info": {
                "total_categories": len(stats["categories"]),
                "total_keywords": stats["total_keywords"],
                "report_generated": True
            }
        }
        
        # Display formatted output
        print("\n" + "="*60)
        print("🎉 KEYWORD EXTRACTION RESULTS")
        print("="*60)
        
        print(f"📊 Total Keywords: {stats['total_keywords']}")
        print(f"📊 Categories: {stats['categories']}")
        print(f"📊 Keywords per Category: {stats['keywords_per_category']}")
        print(f"📊 Average Frequency per Category: {stats['avg_frequency_per_category']}")
        
        print(f"\n🔍 TOP KEYWORDS PER CATEGORY:")
        for category, keywords in stats["top_keywords_per_category"].items():
            print(f"\n   Category {category}:")
            for kw_data in keywords[:3]:  # Show top 3
                print(f"     • {kw_data['keyword']} (frequency: {kw_data['frequency']})")
        
        print(f"\n📋 SAMPLE RESULTS (first 10 rows):")
        if not results_df.empty:
            print(results_df.head(10).to_string(index=False))
        else:
            print("No results to display")
        
        print("\n" + "="*60)
        print("✅ KEYWORD EXTRACTION REPORT COMPLETED!")
        print("="*60)
        
        logger.info(f"Generated keyword extraction report: {stats['total_keywords']} keywords across {len(stats['categories'])} categories")
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate keyword extraction report: {str(e)}")
        raise

@cache()  # Cache keyword visualization generation
@check_output(schema=KeywordVisualizationSchema, importance="fail")
def keyword_visualization(
    keyword_extraction_results: DataFrame[KeywordExtractionResultsSchema],
    clustering_results: Dict[str, Any],
    output_dir: str
) -> DataFrame[KeywordVisualizationSchema]:
    """
    Create visualizations for keyword analysis results.
    
    Args:
        keyword_extraction_results: DataFrame from keyword extraction
        clustering_results: Results from keyword clustering
        output_dir: Directory to save visualization files
        
    Returns:
        DataFrame with visualization metadata
    """
    logger.info("Creating keyword visualizations")
    
    try:
        from .scripts.text_analysis.keybert_extraction import run_keyword_visualization
        
        # Convert DataFrame to dict format for visualization
        if keyword_extraction_results.empty:
            logger.warning("No keyword extraction results available for visualization")
            return pd.DataFrame({
                "visualization_type": pd.Series([], dtype=str),
                "file_path": pd.Series([], dtype=str),
                "category": pd.Series([], dtype=str)
            })
        
        # Convert DataFrame to discriminative keywords dict
        discriminative_keywords = {}
        for _, row in keyword_extraction_results.iterrows():
            category = row['category']
            keyword = row['keyword']
            score = row.get('discriminative_score', row.get('score', 0.0))
            
            if category not in discriminative_keywords:
                discriminative_keywords[category] = {}
            discriminative_keywords[category][keyword] = score
        
        category_clusters = clustering_results.get("category_clusters", {})
        
        if not discriminative_keywords:
            logger.warning("No discriminative keywords available for visualization")
            return pd.DataFrame({
                "visualization_type": pd.Series([], dtype=str),
                "file_path": pd.Series([], dtype=str),
                "category": pd.Series([], dtype=str)
            })
        
        # Run visualization
        viz_results = run_keyword_visualization(
            discriminative_keywords, category_clusters, output_dir
        )
        
        # Convert to DataFrame format
        viz_data = []
        for viz_type, file_path in viz_results["visualization_data"].items():
            viz_data.append({
                "visualization_type": viz_type,
                "file_path": file_path,
                "category": None  # Global visualizations don't have specific categories
            })
        
        result = pd.DataFrame(viz_data)
        logger.info(f"Created {len(result)} visualizations")
        return result
    except Exception as e:
        logger.error(f"Failed to create keyword visualizations: {str(e)}")
        raise

@cache()  # Cache validation CV scoring
@check_output(schema=ValidationCVsScoredSchema, importance="fail")
def validation_cvs_scored(validation_cvs: DataFrame[ValidationCVsSchema], 
                         keyword_lists: DataFrame[KeywordsListsSchema]) -> DataFrame[ValidationCVsScoredSchema]:
    """
    Score validation CVs with filters.
    
    Args:
        validation_cvs: Validation CVs data
        keyword_lists: Keyword lists for scoring
        
    Returns:
        DataFrame with scored validation CVs
    """
    logger.info("Scoring validation CVs with filters")
    
    try:
        # Convert keyword_lists DataFrame to dictionary format expected by scoring function
        keyword_dict = keyword_lists.to_dict('list')
        
        from .scripts.text_analysis.validation_scoring import score_validation_cvs_with_filters
        result = score_validation_cvs_with_filters(validation_cvs, keyword_dict)
        logger.info(f"Scored validation CVs: {len(result)} rows")
        return result
    except Exception as e:
        logger.error(f"Failed to score validation CVs: {str(e)}")
        raise


@cache()  # Cache paper filter application
def validation_cvs_with_paper_filters(validation_cvs: DataFrame[ValidationCVsSchema]) -> pd.DataFrame:
    """
    Prepare validation CVs with 6 annotators: 3 LLMs + 3 compound keyword filters.
    
    Extracts from validation CVs:
    - cv_text and category (true label)
    - 3 LLM results (all using prompt 1 as per the paper):
        * prompt-1-gemini-2.5-flash
        * prompt-1-claude-sonnet-4-20250514 (referred to as "sonnet-4" in paper)
        * prompt-1-gpt-5-mini-2025-08-07 (referred to as "gpt-5-mini-thinking" in paper)
    - 3 compound keyword filters (computed here as per the paper):
        * filter_broad_yes: ML Selection AND Broad_yes (ml_match AND broad_match)
        * filter_strict_no: ML Selection AND Strict_no (ml_match AND strict_no_match)
        * filter_broad_yes_strict_no: ML Selection AND Broad_yes AND Strict_no (all three)
    
    Args:
        validation_cvs: Raw validation CVs data with LLM results
        
    Returns:
        DataFrame with cv_text, category, and 6 annotator columns (0/1 binary values)
    """
    logger.info("Preparing validation CVs with 6 annotators (3 LLMs + 3 compound keyword filters)")
    
    from .keyword_filters import filter_broad_yes, filter_strict_no, filter_broad_yes_strict_no
    
    # Start with cv_text and category
    result = validation_cvs[['cv_text', 'category']].copy()
    
    # Extract the 3 LLM results (all prompt 1)
    # Note: Using dashes to match the test data naming convention
    llm_columns = {
        'prompt-1-gemini-2.5-flash': 'llm_gemini-2.5-flash',
        'prompt-1-claude-sonnet-4-20250514': 'llm_sonnet-4',
        'prompt-1-gpt-5-mini-2025-08-07': 'llm_gpt-5-mini'
    }
    
    for orig_col, new_col in llm_columns.items():
        if orig_col in validation_cvs.columns:
            result[new_col] = validation_cvs[orig_col].astype(int)
        else:
            logger.warning(f"LLM column {orig_col} not found in validation data")
            result[new_col] = 0
    
    # Apply the 3 compound keyword filter functions
    result['filter_broad_yes'] = result['cv_text'].apply(lambda x: 1 if filter_broad_yes(x) else 0)
    result['filter_strict_no'] = result['cv_text'].apply(lambda x: 1 if filter_strict_no(x) else 0)
    result['filter_broad_yes_strict_no'] = result['cv_text'].apply(lambda x: 1 if filter_broad_yes_strict_no(x) else 0)
    
    logger.info(f"Prepared validation CVs with 6 annotators for {len(result)} CVs")
    logger.info("LLM annotators (prompt 1):")
    for new_col in llm_columns.values():
        if new_col in result.columns:
            logger.info(f"  - {new_col}: {result[new_col].sum()} positives ({result[new_col].mean():.1%})")
    logger.info("Compound keyword filter annotators:")
    logger.info(f"  - filter_broad_yes: {result['filter_broad_yes'].sum()} positives ({result['filter_broad_yes'].mean():.1%})")
    logger.info(f"  - filter_strict_no: {result['filter_strict_no'].sum()} positives ({result['filter_strict_no'].mean():.1%})")
    logger.info(f"  - filter_broad_yes_strict_no: {result['filter_broad_yes_strict_no'].sum()} positives ({result['filter_broad_yes_strict_no'].mean():.1%})")
    
    return result


@save_to.csv(path=source("confusion_matrix_metrics_output_path"))
@cache()  # Cache confusion matrix analysis
@check_output(schema=ConfusionMatrixMetricsSchema, importance="fail")
def confusion_matrix_metrics(validation_cvs_scored: DataFrame[ValidationCVsScoredSchema]) -> DataFrame[ConfusionMatrixMetricsSchema]:
    """
    Calculate confusion matrix metrics for validation CV scoring methods.
    
    Args:
        validation_cvs_scored: DataFrame with scored validation CVs
        
    Returns:
        DataFrame with confusion matrix metrics for each scoring method
    """
    logger.info("Calculating confusion matrix metrics for validation scoring")
    
    try:
        from .scripts.text_analysis.confusion_matrix_analysis import analyze_validation_scoring_performance
        result = analyze_validation_scoring_performance(validation_cvs_scored)
        logger.info(f"Calculated confusion matrix metrics for {len(result)} methods")
        return result
    except Exception as e:
        logger.error(f"Failed to calculate confusion matrix metrics: {str(e)}")
        raise

# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

@save_to.csv(path=source("non_academic_affiliations_output_path"))
@cache()  # Cache academic affiliation filtering
@check_output(schema=AffiliationDataSchema, importance="fail")
def non_academic_affiliations(affiliation_data: DataFrame[AffiliationDataSchema]) -> DataFrame[AffiliationDataSchema]:
    """
    Filter out academic affiliations.
    
    Args:
        affiliation_data: Raw affiliation data
        
    Returns:
        DataFrame with non-academic affiliations
    """
    logger.info("Filtering out academic affiliations")
    
    try:
        from .scripts.data_cleaning.academic_filtering import filter_academic_affiliations
        result = filter_academic_affiliations(affiliation_data)
        logger.info(f"Filtered affiliations: {len(result)} rows")
        return result
    except Exception as e:
        logger.error(f"Failed to filter academic affiliations: {str(e)}")
        raise

@save_to.csv(path=source("cleaned_affiliations_output_path"))
@cache()  # Cache affiliation cleaning
@check_output(schema=AffiliationDataSchema, importance="fail")
def cleaned_affiliations(non_academic_affiliations: DataFrame[AffiliationDataSchema]) -> DataFrame[AffiliationDataSchema]:
    """
    Clean and split affiliations.
    
    Args:
        non_academic_affiliations: Non-academic affiliations data
        
    Returns:
        DataFrame with cleaned and split affiliations
    """
    logger.info("Cleaning and splitting affiliations")
    
    try:
        from .scripts.data_cleaning.affiliation_cleaning import clean_and_split_csv
        result = clean_and_split_csv(non_academic_affiliations)
        logger.info(f"Cleaned affiliations: {len(result)} rows")
        return result
    except Exception as e:
        logger.error(f"Failed to clean affiliations: {str(e)}")
        raise

# ============================================================================
# INTERMEDIATE FILE SAVING FUNCTIONS (with transformations)
# ============================================================================

@save_to.json(path=source("cv_keyword_clusters_path"), output_name_="cv_keyword_clusters_json")
def save_cv_keyword_clusters_json(clustering_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save CV keyword clusters to JSON.
    
    Args:
        clustering_results: Clustering results containing keyword cluster data
        
    Returns:
        Keyword cluster data
    """
    logger.info("Saving CV keyword clusters to JSON")
    return clustering_results.get("keyword_cluster_data", {})

@save_to.json(path=source("keywords_path"))
def save_keywords_json(keyword_lists: DataFrame[KeywordsListsSchema]) -> Dict[str, List[str]]:
    """
    Save keyword lists to JSON format.
    
    Args:
        keyword_lists: Keyword lists DataFrame
        
    Returns:
        Dictionary with keyword lists as arrays
    """
    logger.info(f"Saving keyword lists to JSON: {len(keyword_lists)} rows")
    
    if keyword_lists.empty:
        return {
            "strict_yes": [],
            "strict_no": [],
            "broad_yes": [],
            "broad_no": []
        }
    
    # Convert comma-separated strings to lists
    result = {}
    for col in ['strict_yes', 'strict_no', 'broad_yes', 'broad_no']:
        if col in keyword_lists.columns:
            # Split comma-separated string and strip whitespace
            keywords_str = keyword_lists[col].iloc[0] if keyword_lists[col].iloc[0] else ""
            result[col] = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
        else:
            result[col] = []
    
    return result

@save_to.json(path=source("keyword_extraction_report_path"))
def save_keyword_extraction_report_json(keyword_extraction_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save keyword extraction report to JSON.
    
    Args:
        keyword_extraction_report: Keyword extraction report dictionary
        
    Returns:
        Keyword extraction report dictionary with DataFrame converted to dict
    """
    logger.info("Saving keyword extraction report to JSON")
    
    # Convert DataFrame to dict for JSON serialization
    if 'results_df' in keyword_extraction_report:
        keyword_extraction_report['results_df'] = keyword_extraction_report['results_df'].to_dict('records')
    
    return keyword_extraction_report

@save_to.json(path=source("filtered_keywords_list_output_path"))
@cache()  # Cache filtered keywords list
@check_output(importance="fail")
def filtered_keywords_list(keyword_filter_data: pd.DataFrame) -> List[str]:
    """
    Extract list of keywords to filter out from the keyword filter data.
    
    Args:
        keyword_filter_data: DataFrame with keyword filter data including 'Filter out' column
        
    Returns:
        List of keywords that should be filtered out
    """
    logger.info("Extracting filtered keywords list")
    
    try:
        # Get keywords marked with 'x' in the 'Filter out' column
        filtered_keywords = keyword_filter_data[
            keyword_filter_data['Filter out'] == 'x'
        ]['keyword'].tolist()
        
        logger.info(f"Found {len(filtered_keywords)} keywords to filter out")
        logger.debug(f"Filtered keywords: {filtered_keywords[:10]}...")  # Show first 10
        
        return filtered_keywords
    except Exception as e:
        logger.error(f"Failed to extract filtered keywords: {str(e)}")
        raise

@save_to.csv(path=source("keyword_extraction_results_filtered_output_path"))
@cache()  # Cache filtered keyword extraction results
@check_output(schema=KeywordExtractionResultsSchema, importance="fail")
def keyword_extraction_results_filtered(
    keyword_extraction_results: DataFrame[KeywordExtractionResultsSchema], 
    filtered_keywords_list: List[str]
) -> DataFrame[KeywordExtractionResultsSchema]:
    """
    Filter out unwanted keywords from keyword extraction results.
    
    Args:
        keyword_extraction_results: Original keyword extraction results
        filtered_keywords_list: List of keywords to filter out
        
    Returns:
        Filtered keyword extraction results DataFrame
    """
    logger.info("Filtering out unwanted keywords from extraction results")
    
    try:
        # Create a copy to avoid modifying the original
        filtered_results = keyword_extraction_results.copy()
        
        # Filter out keywords that are in the filtered list
        initial_count = len(filtered_results)
        filtered_results = filtered_results[
            ~filtered_results['keyword'].isin(filtered_keywords_list)
        ]
        final_count = len(filtered_results)
        
        removed_count = initial_count - final_count
        logger.info(f"Filtered keywords: {removed_count} removed, {final_count} remaining")
        
        if removed_count > 0:
            removed_keywords = keyword_extraction_results[
                keyword_extraction_results['keyword'].isin(filtered_keywords_list)
            ]['keyword'].tolist()
            logger.info(f"Removed keywords: {removed_keywords}")
        
        return filtered_results
    except Exception as e:
        logger.error(f"Failed to filter keyword extraction results: {str(e)}")
        raise

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_intermediate_result(data: Any, output_dir: str, filename: str, file_type: str = "csv") -> str:
    """
    Save intermediate result to file.
    
    Args:
        data: Data to save
        output_dir: Output directory
        filename: Filename (without extension)
        file_type: File type (csv, tsv, json, etc.)
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir) / f"{filename}.{file_type}"
    
    if file_type == "csv":
        data.to_csv(output_path, index=False)
    elif file_type == "tsv":
        data.to_csv(output_path, sep='\t', index=False)
    elif file_type == "json":
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    logger.info(f"Saved intermediate result: {output_path}")
    return str(output_path)

# ============================================================================
# ============================================================================
# LOG-DEBIASING PROCESSORS
# ============================================================================

@save_to.csv(
    path=source("log_debias_company_aggregates_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_company_aggregates(
    dawid_skene_test_data: DataFrame[DawidSkeneTestDataSchema],
    company_database_complete: DataFrame[CompanyDatabaseCompleteSchema]
) -> pd.DataFrame:
    """
    Calculate company-level aggregates for log-debiasing from company database.
    
    Uses company-level aggregates directly from company_database_complete (filter_broad_yes, etc.)
    instead of aggregating from test data. This ensures ALL companies in the database are included,
    not just those with employee-level test data.
    
    Args:
        dawid_skene_test_data: Test data (used for reference, but not for aggregation)
        company_database_complete: Company database with pre-computed aggregates and metadata
        
    Returns:
        DataFrame with company-level aggregates for log-debiasing
    """
    logger.info("Calculating company-level aggregates for log-debiasing from company database...")
    
    # Start with company database - use pre-computed aggregates
    # Include all companies, but ensure missing annotators are NaN (not zero)
    required_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no',
                     'claude_total_accepted', 'gpt5_total_accepted', 'gemini_total_accepted']
    
    company_aggregates = company_database_complete.copy()
    
    # Convert to numeric - missing or invalid values become NaN
    llm_cols = ['claude_total_accepted', 'gpt5_total_accepted', 'gemini_total_accepted']
    filter_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    
    for col in required_cols:
        if col in company_aggregates.columns:
            company_aggregates[col] = pd.to_numeric(company_aggregates[col], errors='coerce')
            # For LLM columns: treat 0 as missing (NaN) since 0 likely means "no data" not "found 0 matches"
            # For filter columns: keep 0 as legitimate (filter could find 0 matches)
            if col in llm_cols:
                # Convert 0 to NaN for LLM columns (missing data, not zero count)
                company_aggregates[col] = company_aggregates[col].replace(0, np.nan)
    
    # Filter to companies that have at least some aggregate data (at least one non-null value)
    has_some_data = company_aggregates[required_cols].notna().any(axis=1)
    company_aggregates = company_aggregates[has_some_data].copy()
    
    logger.info(f"Found {len(company_aggregates)} companies with at least some aggregate data (out of {len(company_database_complete)} total)")
    
    # Rename columns to match debiasing function expectations
    # The company database already has the correct column names, but we need to ensure consistency
    column_mapping = {
        'llm_gemini_2_5_flash': 'gemini_total_accepted',
        'llm_sonnet_4': 'claude_total_accepted', 
        'llm_gpt_5_mini': 'gpt5_total_accepted',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no'
    }
    
    # Only rename columns that exist and need renaming
    for old_name, new_name in column_mapping.items():
        if old_name in company_aggregates.columns and new_name not in company_aggregates.columns:
            company_aggregates = company_aggregates.rename(columns={old_name: new_name})
    
    # Ensure we have the required columns (use existing if already present)
    for col in required_cols:
        if col not in company_aggregates.columns:
            logger.warning(f"Missing column {col} in company database")
    
    # Use linkedin_id as company_id for consistency
    company_aggregates['company_id'] = company_aggregates['linkedin_id']
    company_aggregates['company_name'] = company_aggregates['organization_name']
    
    # Add Organization Name column for plotting (matches probit plot format)
    company_aggregates['Organization Name'] = company_aggregates['organization_name']
    
    # Select and order columns for consistency with previous format
    output_cols = ['company_id', 'filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no',
                   'gemini_total_accepted', 'gpt5_total_accepted', 'claude_total_accepted',
                   'company_name', 'Total Headcount', 'max_headcount', 'max_population',
                   'Stage Reached', 'category', 'Organization Name']
    
    # Add any missing columns with default values
    for col in output_cols:
        if col not in company_aggregates.columns:
            if col == 'Total Headcount':
                company_aggregates[col] = company_aggregates.get('total_headcount', 1000)
            elif col == 'Stage Reached':
                company_aggregates[col] = company_aggregates.get('stage_reached', '')
            else:
                company_aggregates[col] = None
    
    # Select only the columns we need
    available_cols = [col for col in output_cols if col in company_aggregates.columns]
    company_aggregates = company_aggregates[available_cols].copy()

    logger.info(f"Created company aggregates for {len(company_aggregates)} companies")
    logger.info(f"Sample company names: {company_aggregates['company_name'].head().tolist()}")
    return company_aggregates


def _create_log_debias_company_aggregates(
    dawid_skene_test_data: DataFrame[DawidSkeneTestDataSchema],
    company_database: DataFrame[CompanyDatabaseCompleteSchema],
    pipeline_name: str
) -> pd.DataFrame:
    """
    Helper function to create company aggregates for log-debiasing.
    
    Args:
        dawid_skene_test_data: Test data with binary annotations
        company_database: Company database with metadata
        pipeline_name: Name of the pipeline for logging
        
    Returns:
        DataFrame with company-level aggregates
    """
    logger.info(f"Calculating company-level aggregates for {pipeline_name} log-debiasing from company database...")
    
    # Start with company database - use pre-computed aggregates
    # Include all companies, but ensure missing annotators are NaN (not zero)
    required_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no',
                     'claude_total_accepted', 'gpt5_total_accepted', 'gemini_total_accepted']
    
    company_aggregates = company_database.copy()
    
    # Convert to numeric - missing or invalid values become NaN
    llm_cols = ['claude_total_accepted', 'gpt5_total_accepted', 'gemini_total_accepted']
    filter_cols = ['filter_broad_yes', 'filter_strict_no', 'filter_broad_yes_strict_no']
    
    for col in required_cols:
        if col in company_aggregates.columns:
            company_aggregates[col] = pd.to_numeric(company_aggregates[col], errors='coerce')
            # For LLM columns: treat 0 as missing (NaN) since 0 likely means "no data" not "found 0 matches"
            # For filter columns: keep 0 as legitimate (filter could find 0 matches)
            if col in llm_cols:
                # Convert 0 to NaN for LLM columns (missing data, not zero count)
                company_aggregates[col] = company_aggregates[col].replace(0, np.nan)
    
    # Filter to companies that have at least some aggregate data (at least one non-null value)
    has_some_data = company_aggregates[required_cols].notna().any(axis=1)
    company_aggregates = company_aggregates[has_some_data].copy()
    
    logger.info(f"{pipeline_name}: Found {len(company_aggregates)} companies with at least some aggregate data (out of {len(company_database)} total)")
    
    # Rename columns to match debiasing function expectations
    column_mapping = {
        'llm_gemini_2_5_flash': 'gemini_total_accepted',
        'llm_sonnet_4': 'claude_total_accepted', 
        'llm_gpt_5_mini': 'gpt5_total_accepted',
        'filter_broad_yes': 'filter_broad_yes',
        'filter_strict_no': 'filter_strict_no',
        'filter_broad_yes_strict_no': 'filter_broad_yes_strict_no'
    }
    
    # Only rename columns that exist and need renaming
    for old_name, new_name in column_mapping.items():
        if old_name in company_aggregates.columns and new_name not in company_aggregates.columns:
            company_aggregates = company_aggregates.rename(columns={old_name: new_name})
    
    # Use linkedin_id as company_id for consistency
    company_aggregates['company_id'] = company_aggregates['linkedin_id']
    company_aggregates['company_name'] = company_aggregates['organization_name']
    
    # Add Organization Name column for plotting
    company_aggregates['Organization Name'] = company_aggregates['organization_name']
    
    # Ensure Total Headcount column exists
    if 'Total Headcount' not in company_aggregates.columns:
        company_aggregates['Total Headcount'] = company_aggregates.get('total_headcount', 1000)
    
    # Ensure Stage Reached column exists
    if 'Stage Reached' not in company_aggregates.columns:
        company_aggregates['Stage Reached'] = company_aggregates.get('stage_reached', '')

    logger.info(f"Created company aggregates for {pipeline_name}: {len(company_aggregates)} companies")
    return company_aggregates


@cache()
@check_output(importance="fail")
def log_debias_results(
    log_debias_company_aggregates: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Apply log-debiasing approach to generate 80% CI for ML headcount estimates.

    Args:
        log_debias_company_aggregates: Company-level aggregates with ML headcount estimates

    Returns:
        Dictionary containing debiased results and filtered subsets
    """
    logger.info("Applying log-debiasing approach to ML headcount estimates...")

    # Apply the debiasing function
    debiased_results = debias_ml_headcount_estimates(log_debias_company_aggregates)

    logger.info(f"Generated {len(debiased_results)} filtered datasets from log-debiasing")
    return debiased_results


def _create_log_debias_results(
    log_debias_company_aggregates: pd.DataFrame,
    pipeline_name: str
) -> Dict[str, pd.DataFrame]:
    """
    Helper function to apply log-debiasing approach.
    
    Args:
        log_debias_company_aggregates: Company-level aggregates with ML headcount estimates
        pipeline_name: Name of the pipeline for logging
        
    Returns:
        Dictionary containing debiased results and filtered subsets
    """
    logger.info(f"Applying log-debiasing approach for {pipeline_name}...")
    
    if len(log_debias_company_aggregates) == 0:
        return {}
    
    debiased_results = debias_ml_headcount_estimates(log_debias_company_aggregates)
    logger.info(f"{pipeline_name}: Generated {len(debiased_results)} filtered datasets from log-debiasing")
    return debiased_results


def _create_log_debias_results_orgs(
    log_debias_results: Dict[str, pd.DataFrame],
    pipeline_name: str
) -> pd.DataFrame:
    """
    Helper function to create final output table for log-debias results.
    
    Args:
        log_debias_results: Dictionary containing debiased results
        pipeline_name: Name of the pipeline for logging
        
    Returns:
        DataFrame with final output table
    """
    logger.info(f"Creating {pipeline_name} log-debias results output table...")
    
    all_orgs = log_debias_results.get('all_orgs', pd.DataFrame())
    if len(all_orgs) == 0:
        return pd.DataFrame(columns=['linkedin_id', 'ml_consensus_round', 'ml_lower80_round', 'ml_upper80_round'])
    
    # Extract linkedin_id from company_id if needed
    if 'linkedin_id' not in all_orgs.columns and 'company_id' in all_orgs.columns:
        all_orgs = all_orgs.copy()
        all_orgs['linkedin_id'] = all_orgs['company_id']
    
    result_df = all_orgs[['linkedin_id', 'ml_consensus_round', 'ml_lower80_round', 'ml_upper80_round']].copy()
    
    logger.info(f"Created {pipeline_name} log-debias results table with {len(result_df)} companies")
    return result_df


@save_to.csv(
    path=source("log_debias_company_aggregates_main_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_company_aggregates_main(
    dawid_skene_test_data_main: DataFrame[DawidSkeneTestDataSchema],
    company_database_complete: DataFrame[CompanyDatabaseCompleteSchema]
) -> pd.DataFrame:
    """Calculate company aggregates for main pipeline log-debiasing."""
    return _create_log_debias_company_aggregates(
        dawid_skene_test_data_main,
        company_database_complete,
        "Main pipeline"
    )


@save_to.csv(
    path=source("log_debias_company_aggregates_comparator_ml_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_company_aggregates_comparator_ml(
    dawid_skene_test_data_comparator_ml: DataFrame[DawidSkeneTestDataSchema],
    company_database_comparator_ml: DataFrame[CompanyDatabaseCompleteSchema]
) -> pd.DataFrame:
    """Calculate company aggregates for comparator ML pipeline log-debiasing."""
    if len(dawid_skene_test_data_comparator_ml) == 0:
        return pd.DataFrame()
    
    logger.info("Calculating company-level aggregates for comparator ML pipeline log-debiasing...")
    return _create_log_debias_company_aggregates(
        dawid_skene_test_data_comparator_ml,
        company_database_comparator_ml,
        "Comparator ML"
    )


@save_to.csv(
    path=source("log_debias_company_aggregates_comparator_non_ml_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_company_aggregates_comparator_non_ml(
    dawid_skene_test_data_comparator_non_ml: DataFrame[DawidSkeneTestDataSchema],
    company_database_comparator_non_ml: DataFrame[CompanyDatabaseCompleteSchema]
) -> pd.DataFrame:
    """Calculate company aggregates for comparator Non-ML pipeline log-debiasing."""
    if len(dawid_skene_test_data_comparator_non_ml) == 0:
        return pd.DataFrame()
    
    logger.info("Calculating company-level aggregates for comparator Non-ML pipeline log-debiasing...")
    return _create_log_debias_company_aggregates(
        dawid_skene_test_data_comparator_non_ml,
        company_database_comparator_non_ml,
        "Comparator Non-ML"
    )


@cache()
@check_output(importance="fail")
def log_debias_results_main(
    log_debias_company_aggregates_main: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Apply log-debiasing for main pipeline."""
    return _create_log_debias_results(log_debias_company_aggregates_main, "Main pipeline")


@cache()
@check_output(importance="fail")
def log_debias_results_comparator_ml(
    log_debias_company_aggregates_comparator_ml: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Apply log-debiasing for comparator ML pipeline."""
    return _create_log_debias_results(log_debias_company_aggregates_comparator_ml, "Comparator ML")


@cache()
@check_output(importance="fail")
def log_debias_results_comparator_non_ml(
    log_debias_company_aggregates_comparator_non_ml: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Apply log-debiasing for comparator Non-ML pipeline."""
    return _create_log_debias_results(log_debias_company_aggregates_comparator_non_ml, "Comparator Non-ML")


@save_to.csv(
    path=source("log_debias_results_main_orgs_output_path")
)
@check_output(importance="fail")
def log_debias_results_main_orgs(
    log_debias_results_main: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Create final output table for main pipeline log-debias results."""
    return _create_log_debias_results_orgs(log_debias_results_main, "Main pipeline")


@save_to.csv(
    path=source("log_debias_results_comparator_ml_orgs_output_path")
)
@check_output(importance="fail")
def log_debias_results_comparator_ml_orgs(
    log_debias_results_comparator_ml: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Create final output table for comparator ML pipeline log-debias results."""
    return _create_log_debias_results_orgs(log_debias_results_comparator_ml, "Comparator ML")


@save_to.csv(
    path=source("log_debias_results_comparator_non_ml_orgs_output_path")
)
@check_output(importance="fail")
def log_debias_results_comparator_non_ml_orgs(
    log_debias_results_comparator_non_ml: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Create final output table for comparator Non-ML pipeline log-debias results."""
    return _create_log_debias_results_orgs(log_debias_results_comparator_non_ml, "Comparator Non-ML")


@save_to.csv(
    path=source("log_debias_all_orgs_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_all_orgs(
    log_debias_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Save the 'all_orgs' dataset from log-debiasing results."""
    return log_debias_results.get('all_orgs', pd.DataFrame())


@save_to.csv(
    path=source("log_debias_orgs_ml_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_orgs_ml(
    log_debias_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Save the 'orgs_ML' dataset from log-debiasing results."""
    return log_debias_results.get('orgs_ML', pd.DataFrame())


@save_to.csv(
    path=source("log_debias_orgs_talent_dense_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_orgs_talent_dense(
    log_debias_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Save the 'orgs_talent_dense' dataset from log-debiasing results."""
    return log_debias_results.get('orgs_talent_dense', pd.DataFrame())


@save_to.csv(
    path=source("log_debias_orgs_stage5_work_trial_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_orgs_stage5_work_trial(
    log_debias_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Save the 'orgs_stage5_work_trial' dataset from log-debiasing results."""
    return log_debias_results.get('orgs_stage5_work_trial', pd.DataFrame())


@save_to.csv(
    path=source("log_debias_orgs_enterprise_500ml_0p5pct_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_orgs_enterprise_500ml_0p5pct(
    log_debias_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Save the 'orgs_enterprise_500ml_0p5pct' dataset from log-debiasing results."""
    return log_debias_results.get('orgs_enterprise_500ml_0p5pct', pd.DataFrame())


@save_to.csv(
    path=source("log_debias_orgs_midscale_50ml_1pct_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_orgs_midscale_50ml_1pct(
    log_debias_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Save the 'orgs_midscale_50ml_1pct' dataset from log-debiasing results."""
    return log_debias_results.get('orgs_midscale_50ml_1pct', pd.DataFrame())


@save_to.csv(
    path=source("log_debias_orgs_boutique_10ml_5pct_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_orgs_boutique_10ml_5pct(
    log_debias_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Save the 'orgs_boutique_10ml_5pct' dataset from log-debiasing results."""
    return log_debias_results.get('orgs_boutique_10ml_5pct', pd.DataFrame())


@save_to.csv(
    path=source("log_debias_orgs_stage5_work_trial_recommended_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_orgs_stage5_work_trial_recommended(
    log_debias_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Save the 'orgs_stage5_work_trial_recommended' dataset from log-debiasing results."""
    return log_debias_results.get('orgs_stage5_work_trial_recommended', pd.DataFrame())


@save_to.json(
    path=source("log_debias_summary_output_path")
)
@cache()
@check_output(importance="fail")
def log_debias_summary(
    log_debias_results: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Generate summary statistics for log-debiasing results.

    Args:
        log_debias_results: Dictionary containing debiased results and filtered subsets

    Returns:
        Dictionary with summary statistics
    """
    logger.info("Generating summary statistics for log-debiasing results...")

    summary = {}

    for dataset_name, df in log_debias_results.items():
        if len(df) == 0:
            summary[dataset_name] = {"count": 0, "message": "No data available"}
            continue

        # Calculate summary statistics
        stats = {
            "count": len(df),
            "total_ml_consensus": int(df['ml_consensus_round'].sum()) if 'ml_consensus_round' in df.columns else 0,
            "mean_ml_consensus": float(df['ml_consensus_round'].mean()) if 'ml_consensus_round' in df.columns else 0,
            "median_ml_consensus": float(df['ml_consensus_round'].median()) if 'ml_consensus_round' in df.columns else 0,
            "mean_uncertainty_factor": float(df['uncertainty_factor_x'].mean()) if 'uncertainty_factor_x' in df.columns else 0,
            "outlier_count": int(df['outlier_flag_gt3x'].sum()) if 'outlier_flag_gt3x' in df.columns else 0
        }

        # Add share statistics if available
        if 'ml_share' in df.columns:
            stats.update({
                "mean_ml_share": float(df['ml_share'].mean()),
                "median_ml_share": float(df['ml_share'].median())
            })

        summary[dataset_name] = stats

    logger.info(f"Generated summary for {len(summary)} datasets")
    return summary


# ============================================================================
# SYNTHETIC DATA GENERATION HELPERS
# ============================================================================

@cache()
@check_output(importance="fail")
def test_keyword_filter_correlation_matrix(
    dawid_skene_test_data: DataFrame[DawidSkeneTestDataSchema]
) -> np.ndarray:
    """
    Estimate 3×3 tetrachoric correlation matrix from all test data (keyword filters only).
    
    This represents the unconditional correlation structure of the latent Gaussian variables
    for the 3 keyword filter annotators, estimated from all employee-level test data.
    
    Args:
        dawid_skene_test_data: Test data with employee-level annotations
        
    Returns:
        3×3 correlation matrix (values in [-1, 1], diagonal = 1)
    """
    return estimate_test_keyword_filter_correlation_matrix(dawid_skene_test_data)


@cache()
@check_output(importance="fail")
def companies_needing_synthetic_data(
    company_database_complete: DataFrame[CompanyDatabaseCompleteSchema],
    dawid_skene_test_data: DataFrame[DawidSkeneTestDataSchema]
) -> pd.DataFrame:
    """
    Identify companies that need synthetic data generation.
    
    Companies need synthetic data if they have keyword filter aggregates in the company database
    but are missing from the employee-level test data (or have very few records).
    
    Args:
        company_database_complete: Company database with aggregates
        dawid_skene_test_data: Employee-level test data
        
    Returns:
        DataFrame with companies needing synthetic data
    """
    return identify_companies_needing_synthetic_data(company_database_complete, dawid_skene_test_data)

