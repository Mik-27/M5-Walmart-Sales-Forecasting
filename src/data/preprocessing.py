"""
Data preprocessing utilities for M5 Walmart Sales Forecasting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class M5DataPreprocessor:
    """
    Data preprocessing utilities for M5 dataset
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scalers = {}
        logger.info("Initialized M5DataPreprocessor")
    
    def reshape_sales_data(self, sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape sales data from wide to long format and merge with calendar
        
        Args:
            sales (pd.DataFrame): Sales data in wide format
            calendar (pd.DataFrame): Calendar data
            
        Returns:
            pd.DataFrame: Reshaped data with dates
        """
        logger.info("Reshaping sales data from wide to long format")
        
        # Get day columns
        day_columns = [col for col in sales.columns if col.startswith('d_')]
        
        # Melt to long format
        sales_long = sales.melt(
            id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            value_vars=day_columns,
            var_name='d',
            value_name='sales'
        )
        
        # Merge with calendar
        sales_with_dates = sales_long.merge(calendar, on='d', how='left')
        
        # Sort by id and date
        sales_with_dates = sales_with_dates.sort_values(['id', 'date'])
        
        logger.info(f"Reshaped data shape: {sales_with_dates.shape}")
        return sales_with_dates
    
    def create_lag_features(self, 
                           data: pd.DataFrame, 
                           target_col: str = 'sales',
                           lags: List[int] = [1, 2, 3, 7, 14, 28]) -> pd.DataFrame:
        """
        Create lag features for time series data
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column name
            lags (List[int]): List of lag periods
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        logger.info(f"Creating lag features for lags: {lags}")
        
        data_with_lags = data.copy()
        
        for lag in lags:
            data_with_lags[f'lag_{lag}'] = data_with_lags.groupby('id')[target_col].shift(lag)
        
        logger.info("Lag features created successfully")
        return data_with_lags
    
    def create_rolling_features(self, 
                               data: pd.DataFrame, 
                               target_col: str = 'sales',
                               windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column name
            windows (List[int]): List of window sizes
            
        Returns:
            pd.DataFrame: Data with rolling features
        """
        logger.info(f"Creating rolling features for windows: {windows}")
        
        data_with_rolling = data.copy()
        
        for window in windows:
            # Rolling mean
            data_with_rolling[f'rolling_mean_{window}'] = (
                data_with_rolling.groupby('id')[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Rolling std
            data_with_rolling[f'rolling_std_{window}'] = (
                data_with_rolling.groupby('id')[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
        
        logger.info("Rolling features created successfully")
        return data_with_rolling
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from date column
        
        Args:
            data (pd.DataFrame): Input data with date column
            
        Returns:
            pd.DataFrame: Data with temporal features
        """
        logger.info("Creating temporal features")
        
        data_with_temporal = data.copy()
        
        # Basic temporal features
        data_with_temporal['day_of_week'] = data_with_temporal['date'].dt.dayofweek
        data_with_temporal['day_of_month'] = data_with_temporal['date'].dt.day
        data_with_temporal['day_of_year'] = data_with_temporal['date'].dt.dayofyear
        data_with_temporal['week_of_year'] = data_with_temporal['date'].dt.isocalendar().week
        data_with_temporal['month'] = data_with_temporal['date'].dt.month
        data_with_temporal['quarter'] = data_with_temporal['date'].dt.quarter
        data_with_temporal['year'] = data_with_temporal['date'].dt.year
        
        # Weekend indicator
        data_with_temporal['is_weekend'] = data_with_temporal['day_of_week'].isin([5, 6]).astype(int)
        
        # Month start/end indicators
        data_with_temporal['is_month_start'] = data_with_temporal['date'].dt.is_month_start.astype(int)
        data_with_temporal['is_month_end'] = data_with_temporal['date'].dt.is_month_end.astype(int)
        
        logger.info("Temporal features created successfully")
        return data_with_temporal
    
    def create_price_features(self, 
                             data: pd.DataFrame, 
                             prices: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-related features
        
        Args:
            data (pd.DataFrame): Sales data
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with price features
        """
        logger.info("Creating price features")
        
        # Merge price data
        data_with_prices = data.merge(
            prices, 
            on=['store_id', 'item_id', 'wm_yr_wk'], 
            how='left'
        )
        
        # Fill missing prices with forward fill and backward fill
        data_with_prices['sell_price'] = (
            data_with_prices.groupby(['store_id', 'item_id'])['sell_price']
            .fillna(method='ffill')
            .fillna(method='bfill')
        )
        
        # Price change features
        data_with_prices['price_change'] = (
            data_with_prices.groupby(['store_id', 'item_id'])['sell_price']
            .diff()
        )
        
        data_with_prices['price_change_pct'] = (
            data_with_prices.groupby(['store_id', 'item_id'])['sell_price']
            .pct_change()
        )
        
        # Price momentum (rolling price change)
        data_with_prices['price_momentum_7'] = (
            data_with_prices.groupby(['store_id', 'item_id'])['sell_price']
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        
        logger.info("Price features created successfully")
        return data_with_prices
    
    def create_event_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create event-related features
        
        Args:
            data (pd.DataFrame): Input data with event columns
            
        Returns:
            pd.DataFrame: Data with event features
        """
        logger.info("Creating event features")
        
        data_with_events = data.copy()
        
        # Event indicators
        data_with_events['has_event_1'] = (~data_with_events['event_name_1'].isna()).astype(int)
        data_with_events['has_event_2'] = (~data_with_events['event_name_2'].isna()).astype(int)
        data_with_events['has_any_event'] = (
            data_with_events['has_event_1'] | data_with_events['has_event_2']
        ).astype(int)
        
        # Event type indicators
        for event_type in ['Cultural', 'National', 'Religious', 'Sporting']:
            data_with_events[f'event_type_{event_type.lower()}'] = (
                (data_with_events['event_type_1'] == event_type) |
                (data_with_events['event_type_2'] == event_type)
            ).astype(int)
        
        # SNAP indicators
        data_with_events['snap_any'] = (
            data_with_events[['snap_CA', 'snap_TX', 'snap_WI']].sum(axis=1) > 0
        ).astype(int)
        
        logger.info("Event features created successfully")
        return data_with_events
    
    def scale_features(self, 
                      data: pd.DataFrame, 
                      features: List[str],
                      scaler_type: str = 'minmax') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            data (pd.DataFrame): Input data
            features (List[str]): List of features to scale
            scaler_type (str): Type of scaler ('minmax' or 'standard')
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        logger.info(f"Scaling features with {scaler_type} scaler")
        
        data_scaled = data.copy()
        
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        
        # Fit and transform features
        data_scaled[features] = scaler.fit_transform(data[features])
        
        # Store scaler for later use
        self.scalers[f'{scaler_type}_scaler'] = scaler
        
        logger.info("Feature scaling completed")
        return data_scaled
    
    def get_highest_selling_product(self, sales: pd.DataFrame) -> str:
        """
        Get the product ID with highest total sales
        
        Args:
            sales (pd.DataFrame): Sales data
            
        Returns:
            str: Product ID with highest sales
        """
        day_columns = [col for col in sales.columns if col.startswith('d_')]
        total_sales = sales[day_columns].sum(axis=1)
        max_idx = total_sales.idxmax()
        product_id = sales.loc[max_idx, 'id']
        
        logger.info(f"Highest selling product: {product_id}")
        return product_id
    
    def prepare_time_series_data(self, 
                                data: pd.DataFrame, 
                                product_id: str) -> pd.Series:
        """
        Prepare time series data for a specific product
        
        Args:
            data (pd.DataFrame): Preprocessed data
            product_id (str): Product ID to extract
            
        Returns:
            pd.Series: Time series data for the product
        """
        logger.info(f"Preparing time series data for product: {product_id}")
        
        product_data = data[data['id'] == product_id].copy()
        product_data = product_data.sort_values('date')
        product_data = product_data.set_index('date')
        
        time_series = product_data['sales']
        
        logger.info(f"Time series data shape: {time_series.shape}")
        return time_series
