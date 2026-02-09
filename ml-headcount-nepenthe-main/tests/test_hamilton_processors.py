"""
Tests for Hamilton processors module.

This module tests all processor functions in hamilton_processors.py
using real data files and real scripts, validating results with Pandera schemas.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from ml_headcount.hamilton_processors import (
    # Text analysis functions
    cv_keyword_frequencies, clustering_results, discriminative_keywords, 
    keyword_lists, validation_cvs_scored,
    
    # Data cleaning functions
    non_academic_affiliations, cleaned_affiliations,
    
    # Utility functions
    save_intermediate_result
)

# Set execution mode to remote for all tests (will be mocked)
from ml_headcount.execution_config import set_execution_mode
set_execution_mode(use_remote=True)

# Mock Modal functions to avoid actual remote calls during tests
@pytest.fixture(autouse=True)
def mock_modal_functions():
    """Mock Modal functions for all tests to avoid remote calls."""
    with patch('ml_headcount.hamilton_processors.run_keybert_extraction') as mock_keybert, \
         patch('ml_headcount.hamilton_processors.run_keyword_clustering') as mock_clustering:
        
        # Configure mocks to return appropriate data
        mock_keybert.remote.return_value = pd.DataFrame({
            'category': ['ML', 'ML', 'DS'],
            'keyword': ['python', 'tensorflow', 'statistics'],
            'score': [0.9, 0.8, 0.7],
            'frequency': [10, 8, 5],
            'discriminative_score': [0.9, 0.8, 0.7],
            'raw_category_specificity': [0.9, 0.8, 0.7],
            'total_raw_frequency': [10, 8, 5]
        })
        
        mock_clustering.remote.return_value = {
            'discriminative_keywords': {
                'ML': {'python': 0.9, 'tensorflow': 0.8},
                'DS': {'statistics': 0.7}
            },
            'category_clusters': {},
            'keyword_cluster_data': {}
        }
        
        yield {
            'keybert': mock_keybert,
            'clustering': mock_clustering
        }

from ml_headcount.schemas import (
    CVDataSchema, ValidationCVsSchema, AffiliationDataSchema, CompanyDatabaseBaseSchema,
    KeywordFrequenciesSchema, DiscriminativeKeywordsSchema, KeywordsListsSchema,
    ValidationCVsScoredSchema, CompanyDatabaseWithSubregionsSchema, DebiasedOrganizationsSchema
)
from ml_headcount.config import load_config


# Load test configuration
@pytest.fixture(scope="session")
def test_config():
    """Load test configuration from test.yaml"""
    return load_config(Path("config/test.yaml"))


class TestTextAnalysisFunctions:
    """Test text analysis processor functions."""
    
    @pytest.fixture
    def real_cv_data(self):
        """Load real CV data for testing."""
        return pd.read_csv('data/cv_data.tsv', sep='\t')
    
    @pytest.fixture
    def real_validation_cvs(self):
        """Load real validation CVs data for testing."""
        df = pd.read_excel('data/validation_cvs.xlsx', sheet_name='Validation full')
        # Clean up unnamed columns
        df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')])
        return df
    
    def test_cv_keyword_frequencies(self, real_cv_data):
        """Test CV keyword frequencies extraction with real data."""
        result = cv_keyword_frequencies(real_cv_data)
        
        # Basic checks
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'phrase' in result.columns
        assert 'count' in result.columns
        
        # Validate with schema
        validated_result = KeywordFrequenciesSchema.validate(result)
        assert len(validated_result) == len(result)
    
    
    
    def test_discriminative_keywords_with_data(self, test_config):
        """Test discriminative keywords extraction with data."""
        # Create mock keyword extraction results
        keyword_extraction_results_filtered = pd.DataFrame({
            'category': [0, 0, 1, 1],  # Use integer categories
            'keyword': ['python', 'tensorflow', 'statistics', 'pandas'],
            'frequency': [10, 8, 12, 6]
        })

        # Create mock preprocessed text data
        preprocessed_text = pd.DataFrame({
            'processed_text': [
                'python machine learning tensorflow deep learning',
                'statistics data science pandas numpy',
                'python programming software development',
                'data analysis statistics research'
            ],
            'category': [0, 1, 0, 1]  # Use integer categories
        })

        result = discriminative_keywords(
            keyword_extraction_results_filtered, 
            preprocessed_text, 
            tfidf_ngram_range=test_config.tfidf.ngram_range,
            tfidf_max_features=test_config.tfidf.max_features,
            tfidf_min_df=test_config.tfidf.min_df,
            tfidf_max_df=test_config.tfidf.max_df,
            dk_min_score=test_config.discriminative_keywords.min_score,
            dk_max_keywords=test_config.discriminative_keywords.max_keywords
        )
        
        # Basic checks
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'category' in result.columns
        assert 'keyword' in result.columns
        assert 'score' in result.columns
        
        # Validate with schema
        validated_result = DiscriminativeKeywordsSchema.validate(result)
        assert len(validated_result) == len(result)
    
    def test_discriminative_keywords_empty(self, test_config):
        """Test discriminative keywords extraction with empty data."""
        # Create empty keyword extraction results
        keyword_extraction_results_filtered = pd.DataFrame({
            'category': pd.Series([], dtype=str),
            'keyword': pd.Series([], dtype=str),
            'frequency': pd.Series([], dtype=int)
        })
        
        # Create mock preprocessed text data
        preprocessed_text = pd.DataFrame({
            'processed_text': [
                'python machine learning tensorflow deep learning',
                'statistics data science pandas numpy'
            ],
            'category': ['ML', 'DS']
        })
        
        result = discriminative_keywords(
            keyword_extraction_results_filtered, 
            preprocessed_text, 
            tfidf_ngram_range=test_config.tfidf.ngram_range,
            tfidf_max_features=test_config.tfidf.max_features,
            tfidf_min_df=test_config.tfidf.min_df,
            tfidf_max_df=test_config.tfidf.max_df,
            dk_min_score=test_config.discriminative_keywords.min_score,
            dk_max_keywords=test_config.discriminative_keywords.max_keywords
        )
        
        # Should return empty DataFrame with proper structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert 'category' in result.columns
        assert 'keyword' in result.columns
        assert 'score' in result.columns
    
    def test_keyword_lists(self):
        """Test keyword lists creation."""
        discriminative_keywords_df = pd.DataFrame({
            'category': ['yes', 'yes', 'no', 'yes', 'no'],
            'keyword': ['python', 'tensorflow', 'excel', 'pytorch', 'word'],
            'score': [0.8, 0.7, 0.9, 0.85, 0.75],
            'total_raw_frequency': [10, 8, 12, 6, 9],
            'discriminative_score': [0.8, 0.7, 0.9, 0.85, 0.75],
            'raw_category_specificity': [0.8, 0.7, 0.9, 0.85, 0.75]
        })
    
        result = keyword_lists(discriminative_keywords_df, kf_strict_threshold=0.8, kf_broad_threshold=0.5)
        
        # Basic checks - Hamilton processor returns DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'strict_yes' in result.columns
        assert 'strict_no' in result.columns
        assert 'broad_yes' in result.columns
        assert 'broad_no' in result.columns
        
        # Validate with schema
        validated_result = KeywordsListsSchema.validate(result)
        assert len(validated_result) == len(result)
    
    def test_validation_cvs_scored(self, real_validation_cvs):
        """Test validation CVs scoring."""
        # Create keyword lists DataFrame in the format expected by the function
        keyword_lists = pd.DataFrame([{
            'strict_yes': 'python, tensorflow, pytorch',
            'strict_no': 'java, excel, word',
            'broad_yes': 'machine learning, deep learning, neural network',
            'broad_no': 'cobol, fortran, pascal'
        }])
        
        result = validation_cvs_scored(real_validation_cvs, keyword_lists)
        
        # Basic checks
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'cv_text' in result.columns
        assert 'category' in result.columns


class TestDataCleaningFunctions:
    """Test data cleaning processor functions."""
    
    @pytest.fixture
    def real_affiliation_data(self):
        """Load real affiliation data for testing."""
        return pd.read_csv('data/affiliation_counts.csv')
    
    def test_non_academic_affiliations(self, real_affiliation_data):
        """Test filtering out academic affiliations."""
        # Create test data with some academic affiliations
        test_data = pd.DataFrame({
            'affiliation': [
                'University of California',  # Academic
                'Google Inc.',  # Non-academic
                'MIT',  # Academic
                'Microsoft Corporation',  # Non-academic
                'Stanford University',  # Academic
                'Apple Inc.'  # Non-academic
            ],
            'count': [100, 200, 150, 300, 120, 250]
        })
        
        result = non_academic_affiliations(test_data)
        
        # Basic checks
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'affiliation' in result.columns
        assert 'count' in result.columns
        
        # Should have fewer rows than input (academic affiliations filtered out)
        assert len(result) < len(test_data)
        # Should have only non-academic affiliations
        affiliations = result['affiliation'].str.lower()
        # Check that academic institutions are filtered out
        academic_terms = ['university', 'stanford']
        for term in academic_terms:
            assert not affiliations.str.contains(term).any(), f"Found academic term '{term}' in filtered results"
    
    def test_cleaned_affiliations(self, real_affiliation_data):
        """Test cleaning and splitting affiliations."""
        result = cleaned_affiliations(real_affiliation_data)
        
        # Basic checks
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'affiliation' in result.columns
        assert 'count' in result.columns




class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_save_intermediate_result_csv(self):
        """Test saving intermediate result as CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            data = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['a', 'b', 'c']
            })
            
            result_path = save_intermediate_result(data, temp_dir, "test_data", "csv")
            
            # Check that file was created
            assert os.path.exists(result_path)
            assert result_path.endswith("test_data.csv")
            
            # Check that data was saved correctly
            loaded_data = pd.read_csv(result_path)
            pd.testing.assert_frame_equal(data, loaded_data)
    
    def test_save_intermediate_result_tsv(self):
        """Test saving intermediate result as TSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            data = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': ['a', 'b', 'c']
            })
            
            result_path = save_intermediate_result(data, temp_dir, "test_data", "tsv")
            
            # Check that file was created
            assert os.path.exists(result_path)
            assert result_path.endswith("test_data.tsv")
            
            # Check that data was saved correctly
            loaded_data = pd.read_csv(result_path, sep='\t')
            pd.testing.assert_frame_equal(data, loaded_data)
    
    def test_save_intermediate_result_json(self):
        """Test saving intermediate result as JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            data = {"key1": "value1", "key2": [1, 2, 3]}
            
            result_path = save_intermediate_result(data, temp_dir, "test_data", "json")
            
            # Check that file was created
            assert os.path.exists(result_path)
            assert result_path.endswith("test_data.json")
            
            # Check that data was saved correctly
            with open(result_path, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == data
    
    def test_save_intermediate_result_unsupported_type(self):
        """Test saving with unsupported file type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = pd.DataFrame({'col1': [1, 2, 3]})
            
            with pytest.raises(ValueError, match="Unsupported file type"):
                save_intermediate_result(data, temp_dir, "test_data", "unsupported")


class TestErrorHandling:
    """Test error handling in processor functions."""
    
    def test_discriminative_keywords_empty_data(self, test_config):
        """Test discriminative keywords with empty data."""
        # Create empty keyword extraction results
        keyword_extraction_results_filtered = pd.DataFrame({
            'category': pd.Series([], dtype=str),
            'keyword': pd.Series([], dtype=str),
            'frequency': pd.Series([], dtype=int)
        })
        
        # Create mock preprocessed text data
        preprocessed_text = pd.DataFrame({
            'processed_text': [
                'python machine learning tensorflow deep learning',
                'statistics data science pandas numpy'
            ],
            'category': ['ML', 'DS']
        })
        
        # This should not raise an error even with empty data
        result = discriminative_keywords(
            keyword_extraction_results_filtered, 
            preprocessed_text, 
            tfidf_ngram_range=test_config.tfidf.ngram_range,
            tfidf_max_features=test_config.tfidf.max_features,
            tfidf_min_df=test_config.tfidf.min_df,
            tfidf_max_df=test_config.tfidf.max_df,
            dk_min_score=test_config.discriminative_keywords.min_score,
            dk_max_keywords=test_config.discriminative_keywords.max_keywords
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__])
