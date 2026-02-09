"""
Confusion matrix analysis for validation CV scoring.

This module provides functions to calculate confusion matrix metrics
for validation CV scoring against ground truth labels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def calculate_confusion_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculate confusion matrix metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        
    Returns:
        Dictionary containing TP, FP, FN, TN, sensitivity, specificity, precision, recall, F1
    """
    # Ensure both series are numeric and handle NaN values
    y_true = pd.to_numeric(y_true, errors='coerce').fillna(0).astype(int)
    y_pred = pd.to_numeric(y_pred, errors='coerce').fillna(0).astype(int)
    
    # Calculate confusion matrix components
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    
    # Calculate metrics (use 0.0 instead of NaN for undefined cases)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = sensitivity  # recall is the same as sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall)) else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(f1, 3),
        "accuracy": round(accuracy, 3)
    }

def analyze_validation_scoring_performance(
    validation_cvs_scored: pd.DataFrame,
    ground_truth_column: str = "category"
) -> pd.DataFrame:
    """
    Analyze performance of validation CV scoring methods against ground truth.
    
    Args:
        validation_cvs_scored: DataFrame with scored validation CVs
        ground_truth_column: Column name containing ground truth labels
        
    Returns:
        DataFrame with confusion matrix metrics for each scoring method
    """
    logger.info("Analyzing validation scoring performance")
    
    # Define the scoring method columns to analyze
    scoring_methods = [
        "strict_yes_strict_no",
        "strict_yes_broad_no", 
        "broad_yes_broad_no",
        "broad_yes_strict_no",
        "strict_yes_only",
        "broad_yes_only",
        "strict_no_only",
        "broad_no_only"
    ]
    
    # Filter to only include methods that exist in the DataFrame
    available_methods = [col for col in scoring_methods if col in validation_cvs_scored.columns]
    
    if not available_methods:
        logger.warning("No scoring methods found in validation data")
        return pd.DataFrame()
    
    # Get ground truth labels
    if ground_truth_column not in validation_cvs_scored.columns:
        logger.error(f"Ground truth column '{ground_truth_column}' not found")
        return pd.DataFrame()
    
    y_true = validation_cvs_scored[ground_truth_column]
    
    # Convert ground truth to binary (assuming 'yes'/'no' or 1/0)
    if y_true.dtype == 'object':
        # Convert string labels to binary
        y_true_binary = (y_true.str.lower().str.strip() == 'yes').astype(int)
    else:
        y_true_binary = y_true.astype(int)
    
    # Calculate metrics for each scoring method
    results = []
    for method in available_methods:
        logger.info(f"Analyzing method: {method}")
        
        y_pred = validation_cvs_scored[method]
        metrics = calculate_confusion_metrics(y_true_binary, y_pred)
        metrics["method"] = method
        
        results.append(metrics)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by F1 score (descending)
    results_df = results_df.sort_values("F1", ascending=False, na_position='last')
    
    logger.info(f"Analyzed {len(results)} scoring methods")
    return results_df

def create_confusion_matrix_display(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    method_name: str = "Unknown"
) -> pd.DataFrame:
    """
    Create a confusion matrix display DataFrame.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels  
        method_name: Name of the scoring method
        
    Returns:
        DataFrame formatted as confusion matrix
    """
    # Calculate confusion matrix components
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    
    # Create confusion matrix DataFrame
    cm_df = pd.DataFrame(
        [[TP, FN],
         [FP, TN]],
        index=pd.Index(["Actual: yes (1)", "Actual: no (0)"], name=""),
        columns=pd.Index(["Predicted: yes (1)", "Predicted: no (0)"], name="")
    )
    
    return cm_df

def generate_confusion_matrix_analysis(
    validation_cvs_scored: pd.DataFrame,
    ground_truth_column: str = "category",
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Generate comprehensive confusion matrix analysis for validation scoring.
    
    Args:
        validation_cvs_scored: DataFrame with scored validation CVs
        ground_truth_column: Column name containing ground truth labels
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing analysis results and file paths
    """
    logger.info("Generating confusion matrix analysis")
    
    # Analyze performance metrics
    performance_df = analyze_validation_scoring_performance(
        validation_cvs_scored, 
        ground_truth_column
    )
    
    if performance_df.empty:
        logger.warning("No performance analysis generated")
        return {"performance_metrics": pd.DataFrame(), "confusion_matrices": {}}
    
    # Generate confusion matrices for each method
    confusion_matrices = {}
    y_true = validation_cvs_scored[ground_truth_column]
    
    # Convert ground truth to binary
    if y_true.dtype == 'object':
        y_true_binary = (y_true.str.lower().str.strip() == 'yes').astype(int)
    else:
        y_true_binary = y_true.astype(int)
    
    scoring_methods = [
        "strict_yes_strict_no",
        "strict_yes_broad_no", 
        "broad_yes_broad_no",
        "broad_yes_strict_no",
        "strict_yes_only",
        "broad_yes_only",
        "strict_no_only",
        "broad_no_only"
    ]
    
    for method in scoring_methods:
        if method in validation_cvs_scored.columns:
            y_pred = validation_cvs_scored[method]
            cm_df = create_confusion_matrix_display(y_true_binary, y_pred, method)
            confusion_matrices[method] = cm_df
    
    logger.info(f"Generated confusion matrices for {len(confusion_matrices)} methods")
    
    return {
        "performance_metrics": performance_df,
        "confusion_matrices": confusion_matrices
    }
