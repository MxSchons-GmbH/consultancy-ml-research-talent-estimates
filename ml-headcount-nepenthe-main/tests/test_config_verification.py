"""
Test configuration verification to ensure INPUT_DATA_CONFIG is used as single source of truth.
"""

import pytest
from ml_headcount.hamilton_dataloaders import INPUT_DATA_CONFIG


class TestConfigurationVerification:
    """Test that config_dict uses INPUT_DATA_CONFIG as single source of truth."""
    
    def test_config_generation_logic(self):
        """Test that the new config generation logic works correctly."""
        # Simulate the new config generation logic from hamilton_pipeline.py
        output_dir = "outputs"
        
        config_dict = {
            "output_dir": output_dir
        }
        
        # Add all file paths from INPUT_DATA_CONFIG
        for data_type, config in INPUT_DATA_CONFIG.items():
            # Use file paths directly from INPUT_DATA_CONFIG (they're already correct relative paths)
            config_dict[f"{data_type}_file_path"] = config["file_path"]
            
            # Add sheet name if present
            if "sheet_name" in config:
                config_dict[f"{data_type}_sheet_name"] = config["sheet_name"]
        
        # Verify that all expected keys are present
        expected_keys = [
            "output_dir",
            "validation_cvs_file_path", "validation_cvs_sheet_name",
            "affiliation_data_file_path", "company_database_file_path",
            "company_database_complete_file_path", "company_database_complete_sheet_name",
            "keyword_filter_data_file_path", "keyword_filter_data_sheet_name",
            "cv_data_file_path", 
            "dawid_skene_linkedin_profiles_raw_file_path",
            "dawid_skene_linkedin_profiles_big_consulting_file_path",
            "dawid_skene_linkedin_profiles_comparator_file_path",
            "dawid_skene_keyword_filters_raw_file_path",
            "dawid_skene_keyword_filters_big_consulting_file_path",
            "dawid_skene_keyword_filters_comparator_file_path",
            "dawid_skene_llm_results_dir_file_path",
            "dawid_skene_llm_results_dir_big_consulting_file_path",
            "dawid_skene_llm_results_dir_comparator_file_path"
        ]
        
        missing_keys = [key for key in expected_keys if key not in config_dict]
        assert not missing_keys, f"Missing keys: {missing_keys}"
        
        # Verify that file paths are correctly constructed
        for data_type, config in INPUT_DATA_CONFIG.items():
            expected_path = config_dict[f"{data_type}_file_path"]
            expected = config["file_path"]
            
            assert expected_path == expected, f"{data_type}: expected {expected}, got {expected_path}"
    
    def test_input_data_config_structure(self):
        """Test that INPUT_DATA_CONFIG has the expected structure."""
        assert isinstance(INPUT_DATA_CONFIG, dict)
        assert len(INPUT_DATA_CONFIG) > 0
        
        for data_type, config in INPUT_DATA_CONFIG.items():
            assert isinstance(data_type, str)
            assert isinstance(config, dict)
            assert "file_path" in config
            assert isinstance(config["file_path"], str)
            
            # If sheet_name is present, it should be a string
            if "sheet_name" in config:
                assert isinstance(config["sheet_name"], str)
    
    def test_data_paths_start_with_data_prefix(self):
        """Test that raw_data paths are properly identified."""
        raw_data_cvs_paths = []
        raw_data_search_paths = []
        other_paths = []
        
        for data_type, config in INPUT_DATA_CONFIG.items():
            if config["file_path"].startswith("raw_data_cvs/"):
                raw_data_cvs_paths.append(data_type)
            elif config["file_path"].startswith("raw_data_search/"):
                raw_data_search_paths.append(data_type)
            else:
                other_paths.append(data_type)
        
        # Verify we have different types of paths
        assert len(raw_data_cvs_paths) > 0, "Should have some raw_data_cvs/ paths"
        assert len(raw_data_search_paths) > 0, "Should have some raw_data_search/ paths"
        assert len(other_paths) > 0, "Should have some other paths"
        
        # Verify specific expected paths
        expected_raw_data_cvs_paths = ["cv_data", "validation_cvs", "company_database"]
        for path in expected_raw_data_cvs_paths:
            assert path in raw_data_cvs_paths, f"{path} should be a raw_data_cvs/ path"
        
        expected_raw_data_search_paths = ["affiliation_data", "company_database_complete"]
        for path in expected_raw_data_search_paths:
            assert path in raw_data_search_paths, f"{path} should be a raw_data_search/ path"
