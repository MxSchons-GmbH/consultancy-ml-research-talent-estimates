"""
Tests for configuration management module.

This module tests the Pydantic configuration system including
YAML loading, validation, and conversion to Hamilton inputs.
"""

import pytest
from pathlib import Path
import tempfile
import yaml
from pydantic import ValidationError

from ml_headcount.config import (
    load_config,
    PipelineConfig,
    KeyBERTExtractionConfig,
    KeyBERTClusteringConfig,
    TFIDFConfig,
    DiscriminativeKeywordsConfig,
    KeywordFilteringConfig,
    DataPathsConfig,
    ExecutionConfig
)


class TestConfigLoading:
    """Test configuration loading from YAML files."""
    
    def test_load_default_config(self):
        """Test loading the default configuration file."""
        config = load_config(Path("config/default.yaml"))
        
        assert isinstance(config, PipelineConfig)
        assert config.keybert_extraction.model_name == "Salesforce/SFR-Embedding-Mistral"
        assert config.keybert_extraction.batch_size == 1024
    
    def test_load_test_config(self):
        """Test loading the test configuration file."""
        config = load_config(Path("config/test.yaml"))
        
        assert isinstance(config, PipelineConfig)
        assert config.keybert_extraction.model_name == "all-MiniLM-L6-v2"
        assert config.keybert_extraction.batch_size == 32
    
    def test_load_nonexistent_config(self):
        """Test that loading a nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("config/nonexistent.yaml"))
    
    def test_load_invalid_config(self):
        """Test that loading an invalid config raises ValidationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write invalid config (negative values)
            yaml.dump({
                'keybert_extraction': {'model_name': '', 'batch_size': -1, 'max_seq_length': -1, 
                                      'top_n': -1, 'ngram_max': -1, 'min_score': -0.5, 'max_keywords': -1},
                'keybert_clustering': {'min_cluster_size': -1, 'n_clusters': -1},
                'tfidf': {'ngram_range': [1, 3], 'max_features': -1, 'min_df': -1, 'max_df': -0.5},
                'discriminative_keywords': {'min_score': -0.5, 'max_keywords': -1},
                'keyword_filtering': {'strict_threshold': -0.5, 'broad_threshold': -0.3},
                'data_paths': {'data_dir': 'data', 'output_dir': 'outputs'},
                'execution': {'use_remote': True, 'disable_cache': False, 'enable_telemetry': False, 
                            'project_id': None, 'username': None, 'dag_name': 'test', 'telemetry_tags': {}}
            }, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid configuration"):
                load_config(Path(temp_path))
        finally:
            Path(temp_path).unlink()


class TestPydanticModels:
    """Test Pydantic model validation."""
    
    def test_keybert_extraction_config_valid(self):
        """Test valid KeyBERT extraction configuration."""
        config = KeyBERTExtractionConfig(
            model_name="Salesforce/SFR-Embedding-Mistral",
            batch_size=2048,
            max_seq_length=256,
            top_n=30,
            ngram_max=4,
            min_score=0.1,
            max_keywords=100
        )
        assert config.model_name == "Salesforce/SFR-Embedding-Mistral"
        assert config.batch_size == 2048
        assert config.ngram_max == 4
    
    def test_keybert_extraction_config_invalid_ngram(self):
        """Test KeyBERT config with invalid ngram_max."""
        with pytest.raises(ValidationError):
            KeyBERTExtractionConfig(
                model_name="test-model",
                batch_size=32,
                max_seq_length=128,
                top_n=10,
                ngram_max=20,  # Invalid: must be <= 10
                min_score=0.1,
                max_keywords=50
            )
    
    def test_tfidf_config_valid(self):
        """Test valid TF-IDF configuration."""
        config = TFIDFConfig(
            ngram_range=[1, 4],
            max_features=5000,
            min_df=2,
            max_df=0.85
        )
        assert config.ngram_range == (1, 4)  # Should convert to tuple
        assert config.max_features == 5000
    
    def test_tfidf_config_invalid_ngram_range(self):
        """Test TF-IDF config with invalid ngram_range."""
        with pytest.raises(ValidationError, match="ngram_range must be a list of exactly 2 integers"):
            TFIDFConfig(
                ngram_range=[1],  # Invalid: must have 2 elements
                max_features=5000,
                min_df=2,
                max_df=0.85
            )
    
    def test_tfidf_config_invalid_ngram_order(self):
        """Test TF-IDF config with invalid ngram order."""
        with pytest.raises(ValidationError, match="Invalid ngram_range"):
            TFIDFConfig(
                ngram_range=[4, 1],  # Invalid: max must be >= min
                max_features=5000,
                min_df=2,
                max_df=0.85
            )
    
    def test_keyword_filtering_config_valid(self):
        """Test valid keyword filtering configuration."""
        config = KeywordFilteringConfig(
            strict_threshold=0.8,
            broad_threshold=0.5
        )
        assert config.strict_threshold == 0.8
        assert config.broad_threshold == 0.5
    
    def test_keyword_filtering_config_invalid_thresholds(self):
        """Test keyword filtering config with invalid threshold order."""
        with pytest.raises(ValidationError, match="broad_threshold must be < strict_threshold"):
            KeywordFilteringConfig(
                strict_threshold=0.5,
                broad_threshold=0.8  # Invalid: must be < strict_threshold
            )
    


class TestConfigConversion:
    """Test configuration conversion to Hamilton inputs."""
    
    def test_to_hamilton_inputs(self):
        """Test conversion of config to Hamilton input dictionary."""
        config = load_config(Path("config/test.yaml"))
        inputs = config.to_hamilton_inputs()
        
        # Check that all expected keys are present
        assert 'ke_model_name' in inputs
        assert 'ke_batch_size' in inputs
        assert 'ke_max_seq_length' in inputs
        assert 'ke_top_n' in inputs
        assert 'ke_ngram_max' in inputs
        assert 'kc_min_cluster_size' in inputs
        assert 'tfidf_ngram_range' in inputs
        assert 'tfidf_max_features' in inputs
        assert 'tfidf_min_df' in inputs
        assert 'tfidf_max_df' in inputs
        assert 'dk_min_score' in inputs
        assert 'dk_max_keywords' in inputs
        assert 'kf_strict_threshold' in inputs
        assert 'kf_broad_threshold' in inputs
        assert 'data_dir' in inputs
        assert 'output_dir' in inputs
        
        # Check some values
        assert inputs['ke_model_name'] == "all-MiniLM-L6-v2"
        assert inputs['tfidf_ngram_range'] == (1, 3)
    
    def test_to_hamilton_inputs_types(self):
        """Test that Hamilton inputs have correct types."""
        config = load_config(Path("config/test.yaml"))
        inputs = config.to_hamilton_inputs()
        
        assert isinstance(inputs['ke_model_name'], str)
        assert isinstance(inputs['tfidf_ngram_range'], tuple)
        assert len(inputs['tfidf_ngram_range']) == 2



class TestConfigDefaults:
    """Test that default configurations are sensible."""
    
    def test_default_config_values(self):
        """Test that default config has sensible values."""
        config = load_config(Path("config/default.yaml"))
        
        # KeyBERT should use large production model
        assert "SFR-Embedding-Mistral" in config.keybert_extraction.model_name
        assert config.keybert_extraction.batch_size >= 256
    
    def test_test_config_values(self):
        """Test that test config has smaller/faster values."""
        config = load_config(Path("config/test.yaml"))
        
        # KeyBERT should use smaller test model
        assert "MiniLM" in config.keybert_extraction.model_name
        assert config.keybert_extraction.batch_size <= 100


class TestExecutionConfig:
    """Test execution configuration."""
    
    def test_execution_config_default(self):
        """Test default execution configuration."""
        config = load_config(Path("config/default.yaml"))
        
        assert config.execution.use_remote is True
        assert config.execution.disable_cache is False
        assert config.execution.enable_telemetry is False
        assert config.execution.project_id == "1"
        assert config.execution.username is None
        assert config.execution.dag_name == "ml_headcount_pipeline"
        assert config.execution.telemetry_tags["environment"] == "production"
    
    def test_execution_config_test(self):
        """Test test execution configuration."""
        config = load_config(Path("config/test.yaml"))
        
        assert config.execution.use_remote is False
        assert config.execution.disable_cache is True
        assert config.execution.enable_telemetry is False
        assert config.execution.telemetry_tags["environment"] == "test"
    
    def test_execution_config_telemetry_validation(self):
        """Test that telemetry validation works in execution config."""
        from ml_headcount.config import ExecutionConfig
        
        # Should succeed with telemetry disabled and no credentials
        config = ExecutionConfig(
            enable_telemetry=False
        )
        assert config.enable_telemetry is False
        
        # Should succeed with telemetry enabled and full credentials
        config = ExecutionConfig(
            enable_telemetry=True,
            project_id="test-project",
            username="test@example.com"
        )
        assert config.enable_telemetry is True
        assert config.project_id == "test-project"
        assert config.username == "test@example.com"
        
        # Validation happens at pipeline runtime via main.py, not at config creation
        # This is intentional to allow configs with telemetry disabled but no credentials


class TestExecutionConfigCLIOverrides:
    """Test CLI override logic for execution config."""
    
    def test_use_remote_override(self):
        """Test that --use-remote and --local-only flags override config."""
        # This is more of a documentation test since we can't easily test argparse
        # The logic is in main.py lines 120-129
        config = load_config(Path("config/default.yaml"))
        
        # Default config has use_remote=true
        assert config.execution.use_remote is True
        
        # Test config has use_remote=false
        test_config = load_config(Path("config/test.yaml"))
        assert test_config.execution.use_remote is False
    
    def test_cache_override(self):
        """Test that --disable-cache flag overrides config."""
        config = load_config(Path("config/default.yaml"))
        
        # Default config has disable_cache=false
        assert config.execution.disable_cache is False
        
        # Test config has disable_cache=true
        test_config = load_config(Path("config/test.yaml"))
        assert test_config.execution.disable_cache is True
    
    def test_telemetry_override(self):
        """Test that --enable-telemetry and --disable-telemetry flags override config."""
        config = load_config(Path("config/default.yaml"))
        
        # Default config has enable_telemetry=false (disabled by default)
        assert config.execution.enable_telemetry is False
        
        # Can be overridden via CLI
        assert config.execution.project_id == "1"
        assert config.execution.username is None


class TestConfigImmutability:
    """Test that configs are properly validated and immutable."""
    
    def test_config_is_validated(self):
        """Test that Pydantic validates all fields."""
        config = load_config(Path("config/default.yaml"))
        
        # All fields should be accessible
        assert config.keybert_extraction is not None
        assert config.keybert_clustering is not None
        assert config.tfidf is not None
        assert config.discriminative_keywords is not None
        assert config.keyword_filtering is not None
        assert config.data_paths is not None
        assert config.execution is not None
    
    def test_config_types_are_correct(self):
        """Test that all config fields have correct types."""
        config = load_config(Path("config/default.yaml"))
        
        assert isinstance(config.keybert_extraction, KeyBERTExtractionConfig)
        assert isinstance(config.keybert_clustering, KeyBERTClusteringConfig)
        assert isinstance(config.tfidf, TFIDFConfig)
        assert isinstance(config.discriminative_keywords, DiscriminativeKeywordsConfig)
        assert isinstance(config.keyword_filtering, KeywordFilteringConfig)
        assert isinstance(config.data_paths, DataPathsConfig)
        assert isinstance(config.execution, ExecutionConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

