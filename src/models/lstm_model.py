"""
LSTM model implementation for time series forecasting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, List
import warnings
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from .base_model import BaseModel
from ..utils.logger import setup_logger

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class LSTMModel(BaseModel):
    """
    LSTM (Long Short-Term Memory) model implementation for time series forecasting
    """
    
    def __init__(self,
                 sequence_length: int = 12,
                 hidden_units: int = 50,
                 dropout_rate: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.1,
                 learning_rate: float = 0.001,
                 **kwargs):
        """
        Initialize LSTM model
        
        Args:
            sequence_length (int): Length of input sequences
            hidden_units (int): Number of LSTM hidden units
            dropout_rate (float): Dropout rate for regularization
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            validation_split (float): Fraction of data for validation
            learning_rate (float): Learning rate for optimizer
            **kwargs: Additional parameters
        """
        super().__init__("LSTM", **kwargs)
        
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        
        self.scaler = MinMaxScaler()
        self.training_history = None
        
        logger.info(f"Initialized LSTM model with sequence_length={sequence_length}, "
                   f"hidden_units={hidden_units}")
    
    def _create_sequences(self, 
                         data: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data (np.ndarray): Input data
            sequence_length (int): Length of sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (sequences) and y (targets)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data
            
        Returns:
            Sequential: Compiled LSTM model
        """
        model = Sequential([
            LSTM(self.hidden_units, 
                 activation='relu', 
                 input_shape=input_shape,
                 return_sequences=True),
            Dropout(self.dropout_rate),
            
            LSTM(self.hidden_units, 
                 activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, 
            train_data: pd.Series,
            validation_data: Optional[pd.Series] = None,
            verbose: int = 0,
            **kwargs) -> 'LSTMModel':
        """
        Fit LSTM model to training data
        
        Args:
            train_data (pd.Series): Training time series data
            validation_data (pd.Series, optional): Validation data
            verbose (int): Verbosity level
            **kwargs: Additional fitting parameters
            
        Returns:
            LSTMModel: Fitted model instance
        """
        logger.info("Fitting LSTM model")
        
        try:
            # Scale the data
            train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
            
            # Create sequences
            X_train, y_train = self._create_sequences(train_scaled.flatten(), self.sequence_length)
            
            # Reshape for LSTM input
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            # Build model
            self.model = self._build_model((self.sequence_length, 1))
            
            # Setup callbacks
            callbacks = []
            if self.validation_split > 0:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Train the model
            self.training_history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=verbose,
                shuffle=False  # Important for time series
            )
            
            self.is_fitted = True
            self.train_data = train_data
            
            logger.info("LSTM model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}")
            raise
        
        return self
    
    def predict(self, 
                steps: int, 
                last_sequence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using fitted LSTM model
        
        Args:
            steps (int): Number of steps to forecast
            last_sequence (np.ndarray, optional): Last sequence for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making LSTM predictions for {steps} steps")
        
        try:
            # Use last sequence from training data if not provided
            if last_sequence is None:
                train_scaled = self.scaler.transform(self.train_data.values.reshape(-1, 1))
                last_sequence = train_scaled[-self.sequence_length:].flatten()
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Reshape for prediction
                X_pred = current_sequence.reshape(1, self.sequence_length, 1)
                
                # Make prediction
                next_pred = self.model.predict(X_pred, verbose=0)[0, 0]
                predictions.append(next_pred)
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_pred
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {e}")
            raise
    
    def predict_with_test_data(self, test_data: pd.Series) -> np.ndarray:
        """
        Make predictions using test data sequences
        
        Args:
            test_data (pd.Series): Test data
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info("Making LSTM predictions with test data")
        
        try:
            # Scale test data
            test_scaled = self.scaler.transform(test_data.values.reshape(-1, 1))
            
            # Create sequences from test data
            X_test, _ = self._create_sequences(test_scaled.flatten(), self.sequence_length)
            
            # Reshape for LSTM input
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Make predictions
            predictions_scaled = self.model.predict(X_test, verbose=0)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions with test data: {e}")
            raise
    
    def plot_training_history(self) -> None:
        """
        Plot training history
        """
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        logger.info("Plotting LSTM training history")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.training_history.history['loss'], label='Training Loss')
        if 'val_loss' in self.training_history.history:
            ax1.plot(self.training_history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot MAE
        ax2.plot(self.training_history.history['mae'], label='Training MAE')
        if 'val_mae' in self.training_history.history:
            ax2.plot(self.training_history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary
        
        Returns:
            str: Model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary
