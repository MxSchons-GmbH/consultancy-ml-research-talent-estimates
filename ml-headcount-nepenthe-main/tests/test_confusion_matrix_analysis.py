"""
Tests for confusion matrix analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np
from ml_headcount.scripts.text_analysis.confusion_matrix_analysis import (
    calculate_confusion_metrics,
    analyze_validation_scoring_performance,
    create_confusion_matrix_display,
    generate_confusion_matrix_analysis
)
from ml_headcount.hamilton_processors import confusion_matrix_metrics
from ml_headcount.schemas import ConfusionMatrixMetricsSchema, ValidationCVsScoredSchema


class TestConfusionMatrixAnalysis:
    """Test confusion matrix analysis functions."""
    
    def test_calculate_confusion_metrics_perfect_classifier(self):
        """Test confusion matrix metrics for perfect classifier."""
        y_true = pd.Series([1, 1, 0, 0, 1, 0])
        y_pred = pd.Series([1, 1, 0, 0, 1, 0])
        
        metrics = calculate_confusion_metrics(y_true, y_pred)
        
        assert metrics["TP"] == 3
        assert metrics["TN"] == 3
        assert metrics["FP"] == 0
        assert metrics["FN"] == 0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["F1"] == 1.0
        assert metrics["accuracy"] == 1.0
    
    def test_calculate_confusion_metrics_random_classifier(self):
        """Test confusion matrix metrics for random classifier."""
        y_true = pd.Series([1, 1, 0, 0, 1, 0])
        y_pred = pd.Series([1, 0, 1, 0, 1, 1])
        
        metrics = calculate_confusion_metrics(y_true, y_pred)
        
        assert metrics["TP"] == 2
        assert metrics["TN"] == 1
        assert metrics["FP"] == 2  # Fixed: 0->1, 1->1 = 2 FP
        assert metrics["FN"] == 1
        assert metrics["sensitivity"] == 0.667  # 2/3
        assert metrics["specificity"] == 0.333  # 1/3
        assert metrics["precision"] == 0.5  # 2/4
        assert metrics["recall"] == 0.667
        assert metrics["accuracy"] == 0.5
    
    def test_calculate_confusion_metrics_all_positive(self):
        """Test confusion matrix metrics when all predictions are positive."""
        y_true = pd.Series([1, 1, 0, 0, 1, 0])
        y_pred = pd.Series([1, 1, 1, 1, 1, 1])
        
        metrics = calculate_confusion_metrics(y_true, y_pred)
        
        assert metrics["TP"] == 3
        assert metrics["TN"] == 0
        assert metrics["FP"] == 3
        assert metrics["FN"] == 0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 0.0
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 1.0
        assert metrics["accuracy"] == 0.5
    
    def test_calculate_confusion_metrics_with_nan_values(self):
        """Test confusion matrix metrics with NaN values."""
        y_true = pd.Series([1, 1, np.nan, 0, 1, 0])
        y_pred = pd.Series([1, 0, 1, 0, 1, 1])
        
        metrics = calculate_confusion_metrics(y_true, y_pred)
        
        # NaN values should be converted to 0
        assert metrics["TP"] == 2
        assert metrics["TN"] == 1
        assert metrics["FP"] == 2  # Fixed: 0->1, 1->1 = 2 FP
        assert metrics["FN"] == 1
    
    def test_analyze_validation_scoring_performance(self):
        """Test validation scoring performance analysis."""
        # Create test data
        test_data = pd.DataFrame({
            'cv_text': ['ML engineer with Python', 'Data scientist with R', 'Software engineer', 'ML researcher'],
            'category': ['yes', 'yes', 'no', 'yes'],
            'strict_yes_strict_no': [1, 1, 0, 1],
            'strict_yes_broad_no': [1, 0, 0, 1],
            'broad_yes_broad_no': [1, 1, 0, 1],
            'broad_yes_strict_no': [1, 1, 0, 1],
            'strict_yes_only': [1, 1, 0, 1],
            'broad_yes_only': [1, 1, 0, 1],
            'strict_no_only': [0, 0, 1, 0],
            'broad_no_only': [0, 0, 1, 0]
        })
        
        result = analyze_validation_scoring_performance(test_data)
        
        # Check that we get results for all methods
        assert len(result) == 8
        assert 'method' in result.columns
        assert 'TP' in result.columns
        assert 'FP' in result.columns
        assert 'FN' in result.columns
        assert 'TN' in result.columns
        assert 'sensitivity' in result.columns
        assert 'specificity' in result.columns
        assert 'precision' in result.columns
        assert 'recall' in result.columns
        assert 'F1' in result.columns
        assert 'accuracy' in result.columns
        
        # Check that methods are sorted by F1 score
        f1_scores = result['F1'].dropna()
        if len(f1_scores) > 1:
            assert f1_scores.iloc[0] >= f1_scores.iloc[1]
    
    def test_analyze_validation_scoring_performance_missing_methods(self):
        """Test validation scoring performance with missing methods."""
        # Create test data with only some methods
        test_data = pd.DataFrame({
            'cv_text': ['ML engineer', 'Data scientist'],
            'category': ['yes', 'no'],
            'strict_yes_strict_no': [1, 0],
            'broad_yes_broad_no': [1, 0]
        })
        
        result = analyze_validation_scoring_performance(test_data)
        
        # Should only analyze available methods
        assert len(result) == 2
        assert set(result['method']) == {'strict_yes_strict_no', 'broad_yes_broad_no'}
    
    def test_analyze_validation_scoring_performance_no_methods(self):
        """Test validation scoring performance with no scoring methods."""
        test_data = pd.DataFrame({
            'cv_text': ['ML engineer', 'Data scientist'],
            'category': ['yes', 'no']
        })
        
        result = analyze_validation_scoring_performance(test_data)
        
        # Should return empty DataFrame
        assert result.empty
    
    def test_analyze_validation_scoring_performance_missing_ground_truth(self):
        """Test validation scoring performance with missing ground truth column."""
        test_data = pd.DataFrame({
            'cv_text': ['ML engineer', 'Data scientist'],
            'strict_yes_strict_no': [1, 0]
        })
        
        result = analyze_validation_scoring_performance(test_data, ground_truth_column='missing_column')
        
        # Should return empty DataFrame
        assert result.empty
    
    def test_create_confusion_matrix_display(self):
        """Test confusion matrix display creation."""
        y_true = pd.Series([1, 1, 0, 0, 1, 0])
        y_pred = pd.Series([1, 0, 1, 0, 1, 1])
        
        cm_df = create_confusion_matrix_display(y_true, y_pred, "test_method")
        
        # Check structure
        assert cm_df.shape == (2, 2)
        assert list(cm_df.index) == ["Actual: yes (1)", "Actual: no (0)"]
        assert list(cm_df.columns) == ["Predicted: yes (1)", "Predicted: no (0)"]
        
        # Check values
        assert cm_df.loc["Actual: yes (1)", "Predicted: yes (1)"] == 2  # TP
        assert cm_df.loc["Actual: yes (1)", "Predicted: no (0)"] == 1   # FN
        assert cm_df.loc["Actual: no (0)", "Predicted: yes (1)"] == 2   # FP (fixed)
        assert cm_df.loc["Actual: no (0)", "Predicted: no (0)"] == 1    # TN
    
    def test_generate_confusion_matrix_analysis(self):
        """Test comprehensive confusion matrix analysis."""
        # Create test data
        test_data = pd.DataFrame({
            'cv_text': ['ML engineer with Python', 'Data scientist with R', 'Software engineer', 'ML researcher'],
            'category': ['yes', 'yes', 'no', 'yes'],
            'strict_yes_strict_no': [1, 1, 0, 1],
            'strict_yes_broad_no': [1, 0, 0, 1],
            'broad_yes_broad_no': [1, 1, 0, 1],
            'broad_yes_strict_no': [1, 1, 0, 1],
            'strict_yes_only': [1, 1, 0, 1],
            'broad_yes_only': [1, 1, 0, 1],
            'strict_no_only': [0, 0, 1, 0],
            'broad_no_only': [0, 0, 1, 0]
        })
        
        result = generate_confusion_matrix_analysis(test_data)
        
        # Check structure
        assert 'performance_metrics' in result
        assert 'confusion_matrices' in result
        
        # Check performance metrics
        perf_df = result['performance_metrics']
        assert len(perf_df) == 8
        assert 'method' in perf_df.columns
        
        # Check confusion matrices
        cm_dict = result['confusion_matrices']
        assert len(cm_dict) == 8
        for method, cm_df in cm_dict.items():
            assert cm_df.shape == (2, 2)
            assert list(cm_df.index) == ["Actual: yes (1)", "Actual: no (0)"]
            assert list(cm_df.columns) == ["Predicted: yes (1)", "Predicted: no (0)"]


class TestHamiltonConfusionMatrixFunction:
    """Test Hamilton confusion matrix function."""
    
    def test_confusion_matrix_metrics_hamilton_function(self):
        """Test the Hamilton confusion matrix metrics function."""
        # Create test validation CVs scored data
        test_data = pd.DataFrame({
            'cv_text': ['ML engineer with Python', 'Data scientist with R', 'Software engineer', 'ML researcher'],
            'category': ['yes', 'yes', 'no', 'yes'],
            'strict_yes_strict_no': [1, 1, 0, 1],
            'strict_yes_broad_no': [1, 0, 0, 1],
            'broad_yes_broad_no': [1, 1, 0, 1],
            'broad_yes_strict_no': [1, 1, 0, 1],
            'strict_yes_only': [1, 1, 0, 1],
            'broad_yes_only': [1, 1, 0, 1],
            'strict_no_only': [0, 0, 1, 0],
            'broad_no_only': [0, 0, 1, 0]
        })
        
        # Test the Hamilton function
        result = confusion_matrix_metrics(test_data)
        
        # Check that we get a DataFrame with the expected structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 8
        assert 'method' in result.columns
        assert 'TP' in result.columns
        assert 'FP' in result.columns
        assert 'FN' in result.columns
        assert 'TN' in result.columns
        assert 'sensitivity' in result.columns
        assert 'specificity' in result.columns
        assert 'precision' in result.columns
        assert 'recall' in result.columns
        assert 'F1' in result.columns
        assert 'accuracy' in result.columns
        
        # Check that all methods are present
        expected_methods = [
            'strict_yes_strict_no', 'strict_yes_broad_no', 'broad_yes_broad_no',
            'broad_yes_strict_no', 'strict_yes_only', 'broad_yes_only',
            'strict_no_only', 'broad_no_only'
        ]
        assert set(result['method']) == set(expected_methods)
    
    def test_confusion_matrix_metrics_empty_data(self):
        """Test Hamilton confusion matrix function with empty data."""
        empty_data = pd.DataFrame()
        
        result = confusion_matrix_metrics(empty_data)
        
        # Should return empty DataFrame
        assert result.empty
    
    def test_confusion_matrix_metrics_schema_validation(self):
        """Test that the result conforms to the schema."""
        # Create test data
        test_data = pd.DataFrame({
            'cv_text': ['ML engineer', 'Data scientist'],
            'category': ['yes', 'no'],
            'strict_yes_strict_no': [1, 0],
            'broad_yes_broad_no': [1, 0]
        })
        
        result = confusion_matrix_metrics(test_data)
        
        # Validate against schema
        try:
            ConfusionMatrixMetricsSchema.validate(result)
        except Exception as e:
            pytest.fail(f"Result does not conform to schema: {e}")


class TestConfusionMatrixEdgeCases:
    """Test edge cases for confusion matrix analysis."""
    
    def test_all_same_class_predicted(self):
        """Test when all predictions are the same."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = pd.Series([1, 1, 1, 1])
        
        metrics = calculate_confusion_metrics(y_true, y_pred)
        
        assert metrics["TP"] == 2
        assert metrics["TN"] == 0
        assert metrics["FP"] == 2
        assert metrics["FN"] == 0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 0.0
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 1.0
        assert metrics["F1"] == 0.667
        assert metrics["accuracy"] == 0.5
