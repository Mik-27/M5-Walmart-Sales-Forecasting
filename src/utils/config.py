"""
Configuration utilities for M5 Walmart Sales Forecasting
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .logger import setup_logger

logger = setup_logger(__name__)


class ConfigManager:
    """
    Configuration manager for loading and managing project settings
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        if config_path is None:
            config_path = "config/config.yaml"
        
        self.config_path = Path(config_path)
        self.config = {}
        
        if self.config_path.exists():
            self.load_config()
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            self._create_default_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_config()
            return self.config
    
    def _create_default_config(self) -> None:
        """
        Create default configuration
        """
        self.config = {
            'data': {
                'raw_data_path': 'data/raw/',
                'processed_data_path': 'data/processed/',
                'calendar_file': 'calendar.csv',
                'sales_file': 'sales_train_validation.csv',
                'prices_file': 'sell_prices.csv'
            },
            'models': {
                'sarima': {
                    'order': [2, 1, 1],
                    'seasonal_order': [2, 1, 1, 42]
                },
                'lstm': {
                    'sequence_length': 12,
                    'hidden_units': 50,
                    'epochs': 100,
                    'batch_size': 32
                },
                'prophet': {
                    'weekly_seasonality': True,
                    'yearly_seasonality': True,
                    'daily_seasonality': False
                }
            },
            'training': {
                'test_size': 100,
                'validation_size': 100,
                'random_state': 42
            },
            'output': {
                'models_path': 'output/models/',
                'predictions_path': 'output/predictions/',
                'figures_path': 'output/figures/'
            }
        }
        
        logger.info("Using default configuration")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key (str): Configuration key (supports dot notation)
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key
        
        Args:
            key (str): Configuration key (supports dot notation)
            value (Any): Value to set
        """
        keys = key.split('.')
        config_dict = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        # Set the value
        config_dict[keys[-1]] = value
        logger.info(f"Configuration key '{key}' set to: {value}")
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            output_path (str, optional): Output path for configuration file
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration
        
        Returns:
            Dict[str, Any]: Data configuration
        """
        return self.get('data', {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model-specific configuration
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model configuration
        """
        return self.get(f'models.{model_name}', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration
        
        Returns:
            Dict[str, Any]: Training configuration
        """
        return self.get('training', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output configuration
        
        Returns:
            Dict[str, Any]: Output configuration
        """
        return self.get('output', {})
    
    def validate_config(self) -> bool:
        """
        Validate configuration completeness
        
        Returns:
            bool: True if configuration is valid
        """
        required_keys = [
            'data.raw_data_path',
            'data.calendar_file',
            'data.sales_file',
            'models.sarima.order',
            'models.lstm.sequence_length',
            'training.test_size',
            'output.models_path'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def create_directories(self) -> None:
        """
        Create necessary directories based on configuration
        """
        directories = [
            self.get('data.raw_data_path'),
            self.get('data.processed_data_path'),
            self.get('output.models_path'),
            self.get('output.predictions_path'),
            self.get('output.figures_path'),
            'output/logs'
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
        
        logger.info("All necessary directories created")


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """
    Get global configuration manager instance
    
    Returns:
        ConfigManager: Global configuration manager
    """
    return config_manager


def load_config(config_path: str) -> ConfigManager:
    """
    Load configuration from specific path
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        ConfigManager: Configuration manager instance
    """
    return ConfigManager(config_path)
