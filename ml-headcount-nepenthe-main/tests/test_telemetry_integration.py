"""
Tests for Hamilton telemetry integration.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline
from ml_headcount.config import load_config


@pytest.fixture(scope="session")
def test_config():
    """Load test configuration from test.yaml"""
    return load_config(Path("config/test.yaml"))


class TestTelemetryIntegration:
    """Test cases for telemetry integration."""
    
    def test_telemetry_disabled_by_default(self, test_config):
        """Test that telemetry is disabled by default."""
        config_dict = test_config.to_hamilton_inputs()
        pipeline = HamiltonMLHeadcountPipeline(config_dict=config_dict)
        
        assert pipeline.enable_telemetry is False
        assert pipeline.project_id is None
        assert pipeline.username is None
        assert pipeline.dag_name == "ml_headcount_pipeline"
        assert pipeline.telemetry_tags == {"environment": "DEV", "team": "ML_HEADCOUNT"}
    
    @patch('ml_headcount.hamilton_pipeline.adapters.HamiltonTracker')
    def test_telemetry_enabled_with_valid_config(self, mock_tracker_class, test_config):
        """Test that telemetry can be enabled with valid configuration."""
        # Mock the HamiltonTracker to avoid API calls
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Mock the driver building to avoid module import issues
        with patch('ml_headcount.hamilton_pipeline.driver.Builder') as mock_builder:
            mock_driver = Mock()
            mock_builder.return_value.with_modules.return_value.with_config.return_value.with_adapters.return_value.with_cache.return_value.build.return_value = mock_driver
            
            config_dict = test_config.to_hamilton_inputs()
            pipeline = HamiltonMLHeadcountPipeline(
                config_dict=config_dict,
                enable_telemetry=True,
                project_id="test-project",
                username="test@example.com",
                dag_name="test_dag",
                telemetry_tags={"environment": "TEST", "team": "TEST_TEAM"}
            )
            
            assert pipeline.enable_telemetry is True
            assert pipeline.project_id == "test-project"
            assert pipeline.username == "test@example.com"
            assert pipeline.dag_name == "test_dag"
            assert pipeline.telemetry_tags == {"environment": "TEST", "team": "TEST_TEAM"}
            
            # Verify HamiltonTracker was called with correct parameters
            mock_tracker_class.assert_called_once_with(
                project_id="test-project",
                username="test@example.com",
                dag_name="test_dag",
                tags={"environment": "TEST", "team": "TEST_TEAM"}
            )
    
    def test_telemetry_validation_missing_project_id(self, test_config):
        """Test that validation fails when project_id is missing."""
        config_dict = test_config.to_hamilton_inputs()
        with pytest.raises(ValueError, match="project_id is required when enable_telemetry=True"):
            HamiltonMLHeadcountPipeline(
                config_dict=config_dict,
                enable_telemetry=True,
                username="test@example.com"
            )
    
    @patch('ml_headcount.hamilton_pipeline.adapters.HamiltonTracker')
    def test_telemetry_validation_missing_username(self, mock_tracker_class, test_config):
        """Test that OS username is used as fallback when username is missing."""
        import getpass
        config_dict = test_config.to_hamilton_inputs()
        
        pipeline = HamiltonMLHeadcountPipeline(
            config_dict=config_dict,
            enable_telemetry=True,
            project_id="test-project"
        )
        
        # Should use OS username as fallback
        assert pipeline.username == getpass.getuser()
    
    @patch('ml_headcount.hamilton_pipeline.adapters.HamiltonTracker')
    def test_telemetry_default_values(self, mock_tracker_class, test_config):
        """Test that default values are set correctly."""
        # Mock the HamiltonTracker to avoid API calls
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Mock the driver building to avoid module import issues
        with patch('ml_headcount.hamilton_pipeline.driver.Builder') as mock_builder:
            mock_driver = Mock()
            mock_builder.return_value.with_modules.return_value.with_config.return_value.with_adapters.return_value.with_cache.return_value.build.return_value = mock_driver
            
            config_dict = test_config.to_hamilton_inputs()
            pipeline = HamiltonMLHeadcountPipeline(
                config_dict=config_dict,
                enable_telemetry=True,
                project_id="test-project",
                username="test@example.com"
            )
            
            assert pipeline.dag_name == "ml_headcount_pipeline"
            assert pipeline.telemetry_tags == {"environment": "DEV", "team": "ML_HEADCOUNT"}
    
    @patch('ml_headcount.hamilton_pipeline.adapters.HamiltonTracker')
    def test_tracker_creation_when_telemetry_enabled(self, mock_tracker_class, test_config):
        """Test that HamiltonTracker is created when telemetry is enabled."""
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Mock the driver building to avoid module import issues
        with patch('ml_headcount.hamilton_pipeline.driver.Builder') as mock_builder:
            mock_driver = Mock()
            mock_builder.return_value.with_modules.return_value.with_config.return_value.with_adapters.return_value.with_cache.return_value.build.return_value = mock_driver
            
            config_dict = test_config.to_hamilton_inputs()
            HamiltonMLHeadcountPipeline(
                config_dict=config_dict,
                enable_telemetry=True,
                project_id="test-project",
                username="test@example.com",
                dag_name="test_dag",
                telemetry_tags={"environment": "TEST"}
            )
            
            # Verify that HamiltonTracker was called with correct parameters
            mock_tracker_class.assert_called_once_with(
                project_id="test-project",
                username="test@example.com",
                dag_name="test_dag",
                tags={"environment": "TEST"}
            )
    
    @patch('ml_headcount.hamilton_pipeline.adapters.HamiltonTracker')
    def test_convenience_function_telemetry_parameters(self, mock_tracker_class):
        """Test that convenience function accepts telemetry parameters."""
        from ml_headcount.hamilton_pipeline import run_hamilton_pipeline
        
        # Mock the HamiltonTracker to avoid API calls
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        # Mock the driver building and other dependencies
        with patch('ml_headcount.hamilton_pipeline.driver.Builder') as mock_builder, \
             patch('ml_headcount.hamilton_pipeline.HamiltonMLHeadcountPipeline.run') as mock_run:
            
            mock_driver = Mock()
            mock_builder.return_value.with_modules.return_value.with_config.return_value.with_adapters.return_value.with_cache.return_value.build.return_value = mock_driver
            mock_run.return_value = {"test_output": "test_value"}
            
            # This should not raise an error for parameter validation
            result = run_hamilton_pipeline(
                config_path="config/test.yaml",
                use_remote=False,
                disable_cache=True,
                enable_telemetry=True,
                project_id="test-project",
                username="test@example.com",
                dag_name="test_dag",
                telemetry_tags={"environment": "TEST"}
            )
            
            # Verify the function completed successfully
            assert result == {"test_output": "test_value"}
            
            # Verify that HamiltonTracker was called with correct parameters
            mock_tracker_class.assert_called_once_with(
                project_id="test-project",
                username="test@example.com",
                dag_name="test_dag",
                tags={"environment": "TEST"}
            )
