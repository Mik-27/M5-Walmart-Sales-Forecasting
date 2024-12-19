"""
Logging utilities for M5 Walmart Sales Forecasting
"""

import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional
import os


def setup_logger(name: str, config_path: Optional[str] = None) -> logging.Logger:
    """
    Set up logger with configuration
    
    Args:
        name (str): Logger name
        config_path (str, optional): Path to logging configuration file
        
    Returns:
        logging.Logger: Configured logger
    """
    if config_path is None:
        # Default logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(name)
    
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path("output/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.config.dictConfig(config)
    except Exception as e:
        print(f"Error loading logging config: {e}")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
