"""
Prophet model implementation for time series forecasting
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any
import warnings
import matplotlib.pyplot as plt

try:
    import prophet
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from .base_model import BaseModel
from ..utils.logger import setup_logger

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class ProphetModel(BaseModel):
    """
    Prophet model implementation for time series forecasting
    """
    
    def __init__(self,
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 **kwargs):
        """
        Initialize Prophet model
        
        Args:
            weekly_seasonality (bool): Include weekly seasonality
            yearly_seasonality (bool): Include yearly seasonality
            daily_seasonality (bool): Include daily seasonality
            changepoint_prior_scale (float): Changepoint prior scale
            seasonality_prior_scale (float): Seasonality prior scale
            holidays_prior_scale (float): Holidays prior scale
            **kwargs: Additional parameters
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet package is not available. Please install it with: pip install prophet")
        
        super().__init__("Prophet", **kwargs)
        
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        
        self.holidays = None
        
        logger.info(f"Initialized Prophet model with weekly_seasonality={weekly_seasonality}, "
                   f"yearly_seasonality={yearly_seasonality}")
    
    def _prepare_prophet_data(self, data: pd.Series) -> pd.DataFrame:
        """
        Prepare data for Prophet format
        
        Args:
            data (pd.Series): Time series data
            
        Returns:
            pd.DataFrame: Data in Prophet format (ds, y columns)
        """
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        # Ensure datetime index
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        return prophet_data
    
    def _prepare_holidays(self, calendar: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Prepare holidays data for Prophet
        
        Args:
            calendar (pd.DataFrame, optional): Calendar data with events
            
        Returns:
            pd.DataFrame: Holidays in Prophet format
        """
        if calendar is None:
            return None
        
        # Extract events from calendar
        events = calendar[calendar['event_name_1'].notna()][['date', 'event_name_1']].copy()
        events.columns = ['ds', 'holiday']
        events['ds'] = pd.to_datetime(events['ds'])
        
        # Add second events if available
        events2 = calendar[calendar['event_name_2'].notna()][['date', 'event_name_2']].copy()
        if not events2.empty:
            events2.columns = ['ds', 'holiday']
            events2['ds'] = pd.to_datetime(events2['ds'])
            events = pd.concat([events, events2], ignore_index=True)
        
        # Remove duplicates
        events = events.drop_duplicates()
        
        logger.info(f"Prepared {len(events)} holiday events")
        return events
    
    def fit(self, 
            train_data: pd.Series,
            calendar: Optional[pd.DataFrame] = None,
            **kwargs) -> 'ProphetModel':
        """
        Fit Prophet model to training data
        
        Args:
            train_data (pd.Series): Training time series data
            calendar (pd.DataFrame, optional): Calendar data with holidays/events
            **kwargs: Additional fitting parameters
            
        Returns:
            ProphetModel: Fitted model instance
        """
        logger.info("Fitting Prophet model")
        
        try:
            # Prepare data for Prophet
            prophet_data = self._prepare_prophet_data(train_data)
            
            # Prepare holidays if calendar is provided
            if calendar is not None:
                self.holidays = self._prepare_holidays(calendar)
            
            # Create Prophet model
            self.model = Prophet(
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                holidays=self.holidays
            )
            
            # Fit the model
            self.model.fit(prophet_data)
            self.is_fitted = True
            self.train_data = train_data
            
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
        
        return self
    
    def predict(self, 
                steps: int,
                freq: str = 'D',
                include_history: bool = False) -> pd.Series:
        """
        Make predictions using fitted Prophet model
        
        Args:
            steps (int): Number of steps to forecast
            freq (str): Frequency of predictions
            include_history (bool): Whether to include historical data
            
        Returns:
            pd.Series: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making Prophet predictions for {steps} steps")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=steps, 
                freq=freq,
                include_history=include_history
            )
            
            # Make forecast
            forecast = self.model.predict(future)
            
            if include_history:
                predictions = forecast['yhat']
            else:
                predictions = forecast['yhat'].iloc[-steps:]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making Prophet predictions: {e}")
            raise
    
    def predict_with_uncertainty(self, 
                                steps: int,
                                freq: str = 'D') -> Dict[str, pd.Series]:
        """
        Make predictions with uncertainty intervals
        
        Args:
            steps (int): Number of steps to forecast
            freq (str): Frequency of predictions
            
        Returns:
            Dict[str, pd.Series]: Predictions with upper and lower bounds
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making Prophet predictions with uncertainty for {steps} steps")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps, freq=freq)
            
            # Make forecast
            forecast = self.model.predict(future)
            
            # Extract predictions and bounds
            result = {
                'yhat': forecast['yhat'].iloc[-steps:],
                'yhat_lower': forecast['yhat_lower'].iloc[-steps:],
                'yhat_upper': forecast['yhat_upper'].iloc[-steps:]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making Prophet predictions with uncertainty: {e}")
            raise
    
    def plot_forecast(self, 
                     forecast: Optional[pd.DataFrame] = None,
                     steps: int = 30) -> None:
        """
        Plot forecast using Prophet's built-in plotting
        
        Args:
            forecast (pd.DataFrame, optional): Forecast dataframe
            steps (int): Number of steps to forecast if forecast not provided
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting forecast")
        
        logger.info("Plotting Prophet forecast")
        
        try:
            if forecast is None:
                future = self.model.make_future_dataframe(periods=steps)
                forecast = self.model.predict(future)
            
            # Plot forecast
            fig = self.model.plot(forecast, figsize=(12, 6))
            plt.title('Prophet Forecast')
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not create forecast plot: {e}")
    
    def plot_components(self, 
                       forecast: Optional[pd.DataFrame] = None,
                       steps: int = 30) -> None:
        """
        Plot forecast components (trend, seasonality, etc.)
        
        Args:
            forecast (pd.DataFrame, optional): Forecast dataframe
            steps (int): Number of steps to forecast if forecast not provided
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting components")
        
        logger.info("Plotting Prophet forecast components")
        
        try:
            if forecast is None:
                future = self.model.make_future_dataframe(periods=steps)
                forecast = self.model.predict(future)
            
            # Plot components
            fig = self.model.plot_components(forecast, figsize=(12, 8))
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not create components plot: {e}")
    
    def get_changepoints(self) -> pd.DataFrame:
        """
        Get detected changepoints
        
        Returns:
            pd.DataFrame: Changepoints with dates and deltas
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting changepoints")
        
        changepoints = pd.DataFrame({
            'date': self.model.changepoints,
            'delta': self.model.params['delta']
        })
        
        return changepoints
    
    def add_custom_seasonality(self, 
                              name: str,
                              period: float,
                              fourier_order: int,
                              condition_name: Optional[str] = None) -> None:
        """
        Add custom seasonality to the model
        
        Args:
            name (str): Name of the seasonality
            period (float): Period of the seasonality
            fourier_order (int): Fourier order
            condition_name (str, optional): Name of condition column
        """
        if self.is_fitted:
            logger.warning("Model is already fitted. Custom seasonality should be added before fitting.")
            return
        
        if self.model is None:
            # Create model if not exists
            self.model = Prophet(
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                holidays=self.holidays
            )
        
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            condition_name=condition_name
        )
        
        logger.info(f"Added custom seasonality: {name} with period {period}")
    
    def add_regressor(self, name: str, prior_scale: Optional[float] = None) -> None:
        """
        Add external regressor to the model
        
        Args:
            name (str): Name of the regressor
            prior_scale (float, optional): Prior scale for the regressor
        """
        if self.is_fitted:
            logger.warning("Model is already fitted. Regressors should be added before fitting.")
            return
        
        if self.model is None:
            # Create model if not exists
            self.model = Prophet(
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                holidays=self.holidays
            )
        
        self.model.add_regressor(name, prior_scale=prior_scale)
        
        logger.info(f"Added regressor: {name}")
