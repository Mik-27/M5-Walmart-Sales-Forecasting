"""
SARIMA model implementation for time series forecasting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
import warnings
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from .base_model import BaseModel
from ..utils.logger import setup_logger

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class SarimaModel(BaseModel):
    """
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) model implementation
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (2, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (2, 1, 1, 42),
                 enforce_stationarity: bool = False,
                 enforce_invertibility: bool = False,
                 **kwargs):
        """
        Initialize SARIMA model
        
        Args:
            order: (p, d, q) order of the ARIMA model
            seasonal_order: (P, D, Q, s) seasonal order
            enforce_stationarity: Whether to enforce stationarity
            enforce_invertibility: Whether to enforce invertibility
            **kwargs: Additional parameters
        """
        super().__init__("SARIMA", **kwargs)
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.exog_features = None
        
        logger.info(f"Initialized SARIMA model with order={order}, seasonal_order={seasonal_order}")
    
    def analyze_time_series(self, data: pd.Series, plot: bool = False) -> Dict[str, Any]:
        """
        Analyze time series for stationarity and seasonality
        
        Args:
            data (pd.Series): Time series data
            plot (bool): Whether to create plots
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info("Analyzing time series properties")
        
        # Augmented Dickey-Fuller test for stationarity
        adf_result = adfuller(data.dropna())
        
        analysis = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        if plot:
            self._plot_time_series_analysis(data)
        
        logger.info(f"ADF test p-value: {adf_result[1]:.6f}")
        logger.info(f"Series is stationary: {analysis['is_stationary']}")
        
        return analysis
    
    def _plot_time_series_analysis(self, data: pd.Series) -> None:
        """
        Create diagnostic plots for time series analysis
        
        Args:
            data (pd.Series): Time series data
        """
        try:
            # Decomposition
            decomposition = seasonal_decompose(data, model='additive', period=42)
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 16))
            
            decomposition.observed.plot(ax=axes[0], title='Observed')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            plt.show()
            
            # ACF and PACF plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            plot_acf(data.dropna(), ax=ax1, lags=50)
            plot_pacf(data.dropna(), ax=ax2, lags=50)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not create diagnostic plots: {e}")
    
    def fit(self, 
            train_data: pd.Series, 
            exog: Optional[pd.DataFrame] = None,
            **kwargs) -> 'SarimaModel':
        """
        Fit SARIMA model to training data
        
        Args:
            train_data (pd.Series): Training time series data
            exog (pd.DataFrame, optional): Exogenous variables
            **kwargs: Additional fitting parameters
            
        Returns:
            SarimaModel: Fitted model instance
        """
        logger.info("Fitting SARIMA model")
        
        try:
            # Store exogenous features for prediction
            self.exog_features = exog
            
            # Create and fit the model
            self.model = SARIMAX(
                train_data,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            
            self.fitted_model = self.model.fit(disp=False)
            self.is_fitted = True
            
            # Store training data for future reference
            self.train_data = train_data
            
            logger.info("SARIMA model fitted successfully")
            logger.info(f"Model AIC: {self.fitted_model.aic:.2f}")
            logger.info(f"Model BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {e}")
            raise
        
        return self
    
    def predict(self, 
                steps: int, 
                exog: Optional[pd.DataFrame] = None,
                return_conf_int: bool = False,
                alpha: float = 0.05) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Make predictions using fitted SARIMA model
        
        Args:
            steps (int): Number of steps to forecast
            exog (pd.DataFrame, optional): Exogenous variables for prediction period
            return_conf_int (bool): Whether to return confidence intervals
            alpha (float): Significance level for confidence intervals
            
        Returns:
            Predictions and optionally confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making SARIMA predictions for {steps} steps")
        
        try:
            # Get forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps, exog=exog)
            predictions = forecast_result.predicted_mean
            
            if return_conf_int:
                conf_int = forecast_result.conf_int(alpha=alpha)
                return predictions, conf_int
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making SARIMA predictions: {e}")
            raise
    
    def get_in_sample_predictions(self) -> pd.Series:
        """
        Get in-sample (fitted) predictions
        
        Returns:
            pd.Series: In-sample predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting in-sample predictions")
        
        return self.fitted_model.fittedvalues
    
    def get_residuals(self) -> pd.Series:
        """
        Get model residuals
        
        Returns:
            pd.Series: Model residuals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting residuals")
        
        return self.fitted_model.resid
    
    def plot_diagnostics(self) -> None:
        """
        Plot model diagnostics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        logger.info("Creating SARIMA diagnostic plots")
        
        try:
            fig = self.fitted_model.plot_diagnostics(figsize=(15, 12))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"Could not create diagnostic plots: {e}")
    
    def get_model_summary(self) -> str:
        """
        Get model summary
        
        Returns:
            str: Model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        return str(self.fitted_model.summary())
    
    def get_information_criteria(self) -> Dict[str, float]:
        """
        Get model information criteria
        
        Returns:
            Dict[str, float]: AIC, BIC, and HQIC values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting information criteria")
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic
        }
