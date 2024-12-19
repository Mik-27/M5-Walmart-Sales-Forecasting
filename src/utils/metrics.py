"""
Evaluation metrics for time series forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

from .logger import setup_logger

logger = setup_logger(__name__)


def calculate_rmse(actual: Union[np.ndarray, pd.Series], 
                   predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Squared Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def calculate_mae(actual: Union[np.ndarray, pd.Series], 
                  predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        float: MAE value
    """
    return mean_absolute_error(actual, predicted)


def calculate_mape(actual: Union[np.ndarray, pd.Series], 
                   predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        float: MAPE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return np.inf
    
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_smape(actual: Union[np.ndarray, pd.Series], 
                    predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        float: SMAPE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def calculate_wmape(actual: Union[np.ndarray, pd.Series], 
                    predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        float: WMAPE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100


def calculate_metrics(actual: Union[np.ndarray, pd.Series], 
                      predicted: Union[np.ndarray, pd.Series],
                      metrics: list = None) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics
    
    Args:
        actual: Actual values
        predicted: Predicted values
        metrics: List of metrics to calculate. If None, calculates all.
        
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'mape', 'smape', 'wmape']
    
    results = {}
    
    try:
        if 'rmse' in metrics:
            results['rmse'] = calculate_rmse(actual, predicted)
        
        if 'mae' in metrics:
            results['mae'] = calculate_mae(actual, predicted)
        
        if 'mape' in metrics:
            results['mape'] = calculate_mape(actual, predicted)
        
        if 'smape' in metrics:
            results['smape'] = calculate_smape(actual, predicted)
        
        if 'wmape' in metrics:
            results['wmape'] = calculate_wmape(actual, predicted)
        
        logger.info(f"Calculated metrics: {results}")
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise
    
    return results


class ModelEvaluator:
    """
    Model evaluation utilities
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        self.results = {}
        logger.info("Initialized ModelEvaluator")
    
    def evaluate_model(self, 
                       model_name: str,
                       actual: Union[np.ndarray, pd.Series], 
                       predicted: Union[np.ndarray, pd.Series],
                       metrics: list = None) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model_name (str): Name of the model
            actual: Actual values
            predicted: Predicted values
            metrics: List of metrics to calculate
            
        Returns:
            Dict[str, float]: Evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        results = calculate_metrics(actual, predicted, metrics)
        self.results[model_name] = results
        
        return results
    
    def compare_models(self, 
                       actual: Union[np.ndarray, pd.Series],
                       predictions: Dict[str, Union[np.ndarray, pd.Series]],
                       metrics: list = None) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            actual: Actual values
            predictions: Dictionary of model names and their predictions
            metrics: List of metrics to calculate
            
        Returns:
            pd.DataFrame: Comparison results
        """
        logger.info(f"Comparing {len(predictions)} models")
        
        comparison_results = {}
        
        for model_name, predicted in predictions.items():
            results = self.evaluate_model(model_name, actual, predicted, metrics)
            comparison_results[model_name] = results
        
        # Convert to DataFrame for easier comparison
        comparison_df = pd.DataFrame(comparison_results).T
        
        # Sort by RMSE (ascending)
        if 'rmse' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('rmse')
        
        logger.info("Model comparison completed")
        return comparison_df
    
    def get_best_model(self, metric: str = 'rmse') -> str:
        """
        Get the best performing model based on a metric
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            str: Name of the best model
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        if metric not in ['rmse', 'mae']:
            # For percentage-based metrics, lower is better
            best_model = min(self.results.keys(), 
                           key=lambda x: self.results[x].get(metric, float('inf')))
        else:
            # For absolute metrics, lower is better
            best_model = min(self.results.keys(), 
                           key=lambda x: self.results[x].get(metric, float('inf')))
        
        logger.info(f"Best model based on {metric}: {best_model}")
        return best_model
