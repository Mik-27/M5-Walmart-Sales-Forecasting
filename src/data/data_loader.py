"""
Data loading utilities for M5 Walmart Sales Forecasting
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class M5DataLoader:
    """
    Data loader for M5 Walmart Sales Forecasting dataset
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader
        
        Args:
            data_path (str): Path to the directory containing the data files
        """
        self.data_path = Path(data_path)
        self.calendar = None
        self.sales = None
        self.prices = None
        self.submission = None
        
        logger.info(f"Initialized M5DataLoader with data path: {self.data_path}")
    
    def load_calendar(self, filename: str = "calendar.csv") -> pd.DataFrame:
        """
        Load calendar data
        
        Args:
            filename (str): Name of the calendar file
            
        Returns:
            pd.DataFrame: Calendar data
        """
        try:
            file_path = self.data_path / filename
            self.calendar = pd.read_csv(file_path)
            self.calendar['date'] = pd.to_datetime(self.calendar['date'])
            
            logger.info(f"Loaded calendar data: {self.calendar.shape}")
            return self.calendar
            
        except Exception as e:
            logger.error(f"Error loading calendar data: {e}")
            raise
    
    def load_sales(self, filename: str = "sales_train_validation.csv") -> pd.DataFrame:
        """
        Load sales data
        
        Args:
            filename (str): Name of the sales file
            
        Returns:
            pd.DataFrame: Sales data
        """
        try:
            file_path = self.data_path / filename
            self.sales = pd.read_csv(file_path)
            
            logger.info(f"Loaded sales data: {self.sales.shape}")
            return self.sales
            
        except Exception as e:
            logger.error(f"Error loading sales data: {e}")
            raise
    
    def load_prices(self, filename: str = "sell_prices.csv") -> pd.DataFrame:
        """
        Load price data
        
        Args:
            filename (str): Name of the prices file
            
        Returns:
            pd.DataFrame: Price data
        """
        try:
            file_path = self.data_path / filename
            self.prices = pd.read_csv(file_path)
            
            logger.info(f"Loaded price data: {self.prices.shape}")
            return self.prices
            
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            raise
    
    def load_submission(self, filename: str = "sample_submission.csv") -> pd.DataFrame:
        """
        Load sample submission data
        
        Args:
            filename (str): Name of the submission file
            
        Returns:
            pd.DataFrame: Submission data
        """
        try:
            file_path = self.data_path / filename
            self.submission = pd.read_csv(file_path)
            
            logger.info(f"Loaded submission data: {self.submission.shape}")
            return self.submission
            
        except Exception as e:
            logger.error(f"Error loading submission data: {e}")
            raise
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required data files
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Calendar, sales, and prices data
        """
        logger.info("Loading all data files...")
        
        calendar = self.load_calendar()
        sales = self.load_sales()
        prices = self.load_prices()
        
        logger.info("Successfully loaded all data files")
        return calendar, sales, prices
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about loaded datasets
        
        Returns:
            Dict[str, Any]: Information about the datasets
        """
        info = {}
        
        if self.calendar is not None:
            info['calendar'] = {
                'shape': self.calendar.shape,
                'date_range': (self.calendar['date'].min(), self.calendar['date'].max()),
                'columns': list(self.calendar.columns)
            }
        
        if self.sales is not None:
            info['sales'] = {
                'shape': self.sales.shape,
                'num_items': self.sales['item_id'].nunique(),
                'num_stores': self.sales['store_id'].nunique(),
                'num_categories': self.sales['cat_id'].nunique(),
                'columns': list(self.sales.columns)
            }
        
        if self.prices is not None:
            info['prices'] = {
                'shape': self.prices.shape,
                'price_range': (self.prices['sell_price'].min(), self.prices['sell_price'].max()),
                'columns': list(self.prices.columns)
            }
        
        return info


class DataValidator:
    """
    Data validation utilities
    """
    
    @staticmethod
    def validate_sales_data(sales: pd.DataFrame) -> bool:
        """
        Validate sales data format and content
        
        Args:
            sales (pd.DataFrame): Sales data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        # Check required columns
        if not all(col in sales.columns for col in required_columns):
            logger.error("Missing required columns in sales data")
            return False
        
        # Check for day columns
        day_columns = [col for col in sales.columns if col.startswith('d_')]
        if len(day_columns) == 0:
            logger.error("No day columns found in sales data")
            return False
        
        # Check for negative sales
        day_data = sales[day_columns]
        if (day_data < 0).any().any():
            logger.warning("Negative sales values found")
        
        logger.info("Sales data validation passed")
        return True
    
    @staticmethod
    def validate_calendar_data(calendar: pd.DataFrame) -> bool:
        """
        Validate calendar data format and content
        
        Args:
            calendar (pd.DataFrame): Calendar data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'd']
        
        # Check required columns
        if not all(col in calendar.columns for col in required_columns):
            logger.error("Missing required columns in calendar data")
            return False
        
        # Check date range
        if calendar['date'].isna().any():
            logger.error("Missing dates in calendar data")
            return False
        
        logger.info("Calendar data validation passed")
        return True
    
    @staticmethod
    def validate_prices_data(prices: pd.DataFrame) -> bool:
        """
        Validate prices data format and content
        
        Args:
            prices (pd.DataFrame): Prices data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
        
        # Check required columns
        if not all(col in prices.columns for col in required_columns):
            logger.error("Missing required columns in prices data")
            return False
        
        # Check for negative prices
        if (prices['sell_price'] < 0).any():
            logger.error("Negative prices found in prices data")
            return False
        
        logger.info("Prices data validation passed")
        return True
