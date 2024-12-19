"""
Base model class for time series forecasting
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Union
import joblib
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for time series forecasting models
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base model
        
        Args:
            model_name (str): Name of the model
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.parameters = kwargs
        self.metrics = {}
        
        logger.info(f"Initialized {self.model_name} model")
    
    @abstractmethod
    def fit(self, 
            train_data: Union[pd.Series, pd.DataFrame], 
            **kwargs) -> 'BaseModel':
        """
        Fit the model to training data
        
        Args:
            train_data: Training data
            **kwargs: Additional fitting parameters
            
        Returns:
            BaseModel: Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, 
                steps: int, 
                **kwargs) -> Union[np.ndarray, pd.Series]:
        """
        Make predictions
        
        Args:
            steps (int): Number of steps to forecast
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    def evaluate(self, 
                 actual: Union[np.ndarray, pd.Series], 
                 predicted: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        from ..utils.metrics import calculate_metrics
        
        metrics = calculate_metrics(actual, predicted)
        self.metrics.update(metrics)
        
        logger.info(f"{self.model_name} evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """
        Load a fitted model
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            BaseModel: Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.parameters = model_data['parameters']
        self.metrics = model_data['metrics']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'model_name': self.model_name,
            'parameters': self.parameters,
            'is_fitted': self.is_fitted,
            'metrics': self.metrics
        }
    
    def __repr__(self) -> str:
        """String representation of the model"""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_name}({status})"
