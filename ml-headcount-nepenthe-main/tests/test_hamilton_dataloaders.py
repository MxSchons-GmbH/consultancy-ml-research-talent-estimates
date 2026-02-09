"""
Tests for Hamilton dataloaders module using real data files.

This module tests all dataloader functions in hamilton_dataloaders.py
using actual data files and validates results with Pandera schemas.
"""

import pytest
import pandas as pd
import os
from pathlib import Path

from ml_headcount.hamilton_dataloaders import (
    cv_data, validation_cvs, affiliation_data, company_database, 
    linkedin_data
)
from ml_headcount.schemas import (
    CVDataSchema, ValidationCVsSchema, AffiliationDataSchema, 
    CompanyDatabaseBaseSchema, LinkedInDataSchema, CompanyDatabaseCompleteSchema
)
from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline
from ml_headcount.config import load_config


# Shared test configuration fixture
@pytest.fixture(scope="session")
def test_config():
    """Load test configuration from test.yaml"""
    return load_config(Path("config/test.yaml"))


def _test_dataloader_with_hamilton(pipeline, dataloader_name: str, data_path: str, expected_schema):
    """Helper function to test dataloaders using Hamilton execution."""
    import pandas as pd
    
    # Execute the specific dataloader function through Hamilton
    result = pipeline.dr.execute([dataloader_name], inputs={f"{dataloader_name}_path": data_path})
    
    # Hamilton returns a dict with function names as keys, extract the DataFrame
    df = result[dataloader_name]
    
    # Basic checks
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    # Validate with schema if provided
    if expected_schema:
        validated_df = expected_schema.validate(df)
        assert len(validated_df) == len(df)
    
    return df


class TestCVDataLoader:
    """Test cv_data dataloader function with real data."""
    
    @pytest.fixture
    def pipeline(self, test_config):
        """Create Hamilton pipeline for testing."""
        config_dict = test_config.to_hamilton_inputs()
        return HamiltonMLHeadcountPipeline(config_dict=config_dict)
    
    def test_cv_data_loads_real_data(self, pipeline):
        """Test that CV data loads successfully from real file."""
        data_path = "raw_data/raw_data_cvs/2025-08-05_CV_rated.tsv"
        
        df = _test_dataloader_with_hamilton(pipeline, "cv_data", data_path, CVDataSchema)
        
        # Basic checks
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert list(df.columns) == ['cv_text', 'category']
    
    def test_cv_data_schema_validation(self, pipeline):
        """Test that CV data validates against Pandera schema."""
        data_path = "raw_data/raw_data_cvs/2025-08-05_CV_rated.tsv"
        
        df = _test_dataloader_with_hamilton(pipeline, "cv_data", data_path, CVDataSchema)
        
        # Schema validation is already done in the helper function
        assert list(df.columns) == ['cv_text', 'category']


class TestValidationCVsLoader:
    """Test validation_cvs dataloader function with real data using Hamilton."""
    
    @pytest.fixture
    def pipeline(self, test_config):
        """Create Hamilton pipeline for testing."""
        config_dict = test_config.to_hamilton_inputs()
        return HamiltonMLHeadcountPipeline(config_dict=config_dict)
    
    def test_validation_cvs_loads_real_data(self, pipeline):
        """Test that validation CVs load successfully from real file."""
        data_path = "raw_data/raw_data_cvs/Validation Set/2025-08-06_validation_cvs_rated.xlsx"
        
        df = _test_dataloader_with_hamilton(pipeline, "validation_cvs", data_path, ValidationCVsSchema)
        
        # Basic checks
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_validation_cvs_schema_mismatch(self, pipeline):
        """Test that validation CVs schema doesn't match actual data."""
        data_path = "raw_data/raw_data_cvs/Validation Set/2025-08-06_validation_cvs_rated.xlsx"
        
        # Load raw data to see what we actually have
        df = pd.read_excel(data_path, sheet_name='Validation full')
        df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')])
        
        # Check what columns we actually have
        actual_columns = list(df.columns)
        expected_columns = ['public_identifier', 'profile_summary', 'category']
        
        # Document the mismatch
        assert 'cv_text' in actual_columns, f"Expected 'cv_text' in columns: {actual_columns}"
        assert 'category' in actual_columns, f"Expected 'category' in columns: {actual_columns}"
        assert 'public_identifier' not in actual_columns, f"Unexpected 'public_identifier' in columns: {actual_columns}"
    
    def test_validation_cvs_correct_data_exists(self, pipeline):
        """Test that correct validation data exists in raw_data directory."""
        # Check if the correct LinkedIn-style validation data exists
        correct_data_path = "raw_data/raw_data_cvs/Validation Set/2025-08-06_validation_set.csv"
        
        if os.path.exists(correct_data_path):
            df = pd.read_csv(correct_data_path)
            
            # Check that it has the correct structure
            assert 'public_identifier' in df.columns
            assert 'profile_summary' in df.columns
            assert len(df) > 0
            
            print(f"Found correct validation data: {len(df)} rows with LinkedIn-style structure")
        else:
            pytest.skip("Correct validation data file not found")
    
    def test_validation_cvs_cv_text_data_exists(self, pipeline):
        """Test that CV text validation data exists for scoring functions."""
        # Check if the CV text validation data exists (for scoring functions)
        cv_text_path = "data/validation_cvs.xlsx"
        
        if os.path.exists(cv_text_path):
            df = pd.read_excel(cv_text_path, sheet_name='Validation full')
            df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')])
            
            # Check that it has the structure expected by scoring functions
            assert 'cv_text' in df.columns
            assert 'category' in df.columns
            assert len(df) > 0
            
            print(f"Found CV text validation data: {len(df)} rows with cv_text structure")
            print("This data is suitable for scoring functions but not for LinkedIn-style schemas")
        else:
            pytest.skip("CV text validation data file not found")


class TestAffiliationDataLoader:
    """Test affiliation_data dataloader function with real data."""
    
    @pytest.fixture
    def pipeline(self, test_config):
        """Create Hamilton pipeline for testing."""
        config_dict = test_config.to_hamilton_inputs()
        return HamiltonMLHeadcountPipeline(config_dict=config_dict)
    
    def test_affiliation_data_loads_real_data(self, pipeline):
        """Test that affiliation data loads successfully from real file."""
        data_path = "raw_data/raw_data_search/2025-08-04_arxiv_kaggle_all_affiliations_cleaned.csv"
        
        df = _test_dataloader_with_hamilton(pipeline, "affiliation_data", data_path, AffiliationDataSchema)
        
        # Basic checks
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert list(df.columns) == ['affiliation', 'count']
    
    def test_affiliation_data_schema_validation(self, pipeline):
        """Test that affiliation data validates against Pandera schema."""
        data_path = "raw_data/raw_data_search/2025-08-04_arxiv_kaggle_all_affiliations_cleaned.csv"
        
        df = _test_dataloader_with_hamilton(pipeline, "affiliation_data", data_path, AffiliationDataSchema)
        
        # Schema validation is already done in the helper function
        assert list(df.columns) == ['affiliation', 'count']


class TestCompanyDatabaseLoader:
    """Test company_database dataloader function with real data."""
    
    @pytest.fixture
    def pipeline(self, test_config):
        """Create Hamilton pipeline for testing."""
        config_dict = test_config.to_hamilton_inputs()
        return HamiltonMLHeadcountPipeline(config_dict=config_dict)
    
    def test_company_database_better_data_exists(self, pipeline):
        """Test that better company data exists in raw_data directory."""
        # Check if the raw Crunchbase data exists and has the required fields
        crunchbase_path = "raw_data/raw_data_search/2025-08-04_crunchbase_export_ml_consultancies.csv"
        
        if os.path.exists(crunchbase_path):
            df = pd.read_csv(crunchbase_path)
            
            # Check that it has the required fields
            has_headquarters = 'Headquarters Location' in df.columns
            has_employees = 'Number of Employees' in df.columns
            has_org_name = 'Organization Name' in df.columns
            
            assert has_headquarters, "Raw Crunchbase data missing 'Headquarters Location'"
            assert has_employees, "Raw Crunchbase data missing 'Number of Employees'"
            assert has_org_name, "Raw Crunchbase data missing 'Organization Name'"
            
            print(f"Found better company data: {len(df)} rows with required fields")
            print(f"Sample headquarters: {df['Headquarters Location'].head(3).tolist()}")
        else:
            pytest.skip("Raw Crunchbase data file not found")
    
    def test_company_database_geographic_processing(self, pipeline):
        """Test that company database can be used for geographic processing."""
        # Use the complete database that has Headquarters Location
        data_path = "raw_data/raw_data_search/2025-08-05_systematic_search_all.xlsx"
        
        df = _test_dataloader_with_hamilton(pipeline, "company_database_complete", data_path, CompanyDatabaseCompleteSchema)
        
        # Check that required geographic columns are present
        assert 'headquarters_location' in df.columns
        assert 'country' in df.columns
        assert 'subregion' in df.columns


class TestLinkedInDataLoader:
    """Test linkedin_data dataloader function with real data."""
    
    @pytest.fixture
    def pipeline(self, test_config):
        """Create Hamilton pipeline for testing."""
        config_dict = test_config.to_hamilton_inputs()
        return HamiltonMLHeadcountPipeline(config_dict=config_dict)
    
    def test_linkedin_data_file_exists(self, pipeline):
        """Test that LinkedIn data file exists and is accessible."""
        data_path = "raw_data/raw_data_cvs/2025-08-07 85k profiles.jsonl"
        
        # Check if file exists
        if not os.path.exists(data_path):
            pytest.skip("LinkedIn data file not found")
        
        # Check file size
        file_size = os.path.getsize(data_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            pytest.skip("LinkedIn data file too large for testing")
        
        # Try to load the data
        try:
            df = _test_dataloader_with_hamilton(pipeline, "linkedin_data", data_path, LinkedInDataSchema)
            
            # Basic checks
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            
        except Exception as e:
            # Document any issues
            pytest.fail(f"LinkedIn data loading failed: {e}")
    
    def test_linkedin_data_schema_validation(self, pipeline):
        """Test that LinkedIn data validates against Pandera schema."""
        data_path = "raw_data/raw_data_cvs/2025-08-07 85k profiles.jsonl"
        
        if not os.path.exists(data_path):
            pytest.skip("LinkedIn data file not found")
        
        file_size = os.path.getsize(data_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            pytest.skip("LinkedIn data file too large for testing")
        
        try:
            df = _test_dataloader_with_hamilton(pipeline, "linkedin_data", data_path, LinkedInDataSchema)
            
            # Validate with Pandera schema
            validated_df = LinkedInDataSchema.validate(df)
            assert len(validated_df) == len(df)
            assert list(validated_df.columns) == ['public_identifier', 'profile_summary']
            
        except Exception as e:
            pytest.fail(f"LinkedIn data schema validation failed: {e}")


class TestDataQualityIssues:
    """Test and document data quality issues found in real data."""
    
    def test_validation_cvs_schema_mismatch(self):
        """Document the validation CVs schema mismatch."""
        data_path = "raw_data/raw_data_cvs/Validation Set/2025-08-06_validation_cvs_rated.xlsx"
        
        # Load the actual data
        df = pd.read_excel(data_path, sheet_name='Validation full')
        df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')])
        
        print(f"\nValidation CVs actual columns: {list(df.columns)}")
        print(f"Expected schema columns: ['public_identifier', 'profile_summary', 'category']")
        
        # The actual data has cv_text and category, not public_identifier and profile_summary
        assert 'cv_text' in df.columns
        assert 'category' in df.columns
        assert 'public_identifier' not in df.columns
        assert 'profile_summary' not in df.columns
    
    def test_company_database_missing_geographic_data(self):
        """Document the company database missing geographic data."""
        data_path = "raw_data/raw_data_cvs/2025-08-07 included_companies.tsv"
        
        df = pd.read_csv(data_path, sep='\t')
        
        print(f"\nCompany database columns: {list(df.columns)}")
        print("Missing 'Headquarters Location' field required by geographic processing")
        
        # Check what we have vs what's needed
        has_geographic = 'Headquarters Location' in df.columns or 'headquarters_location' in df.columns
        assert not has_geographic, "Company database should be missing geographic data"
    
    def test_data_file_sizes(self):
        """Document the sizes of data files."""
        files_to_check = [
            "data/cv_data.tsv",
            "data/affiliation_counts.csv", 
            "data/company_database.tsv",
            "data/validation_cvs.xlsx",
            "raw_data/raw_data_cvs/2025-08-07 85k profiles.jsonl"
        ]
        
        print("\nData file sizes:")
        for file_path in files_to_check:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                size_mb = size / (1024 * 1024)
                print(f"  {file_path}: {size_mb:.1f} MB")
            else:
                print(f"  {file_path}: Not found")


if __name__ == "__main__":
    pytest.main([__file__])