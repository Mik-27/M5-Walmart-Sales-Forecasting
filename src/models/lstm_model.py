"""
LSTM model implementation for time series forecasting using PyTorch
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, List
import warnings
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from .base_model import BaseModel
from ..utils.logger import setup_logger

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """
    Dataset class for time series data
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMNet(nn.Module):
    """
    LSTM Neural Network Architecture
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int, 
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        super(LSTMNet, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out


class LSTMModel(BaseModel):
    """
    LSTM (Long Short-Term Memory) model implementation for time series forecasting using PyTorch
    """
    
    def __init__(self,
                 sequence_length: int = 12,
                 hidden_size: int = 50,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 validation_split: float = 0.1,
                 **kwargs):
        """
        Initialize LSTM model
        
        Args:
            sequence_length (int): Length of input sequences
            hidden_size (int): Number of LSTM hidden units
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for optimizer
            validation_split (float): Fraction of data for validation
            **kwargs: Additional parameters
        """
        super().__init__("LSTM", **kwargs)
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.scaler = MinMaxScaler()
        self.model = None
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.train_data = None
        
        logger.info(f"Initialized PyTorch LSTM model with sequence_length={sequence_length}, "
                   f"hidden_size={hidden_size}, device={self.device}")

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

    def _build_model(self, input_size: int) -> LSTMNet:
        """
        Build LSTM model architecture
        
        Args:
            input_size (int): Size of input features
            
        Returns:
            LSTMNet: PyTorch LSTM model
        """
        model = LSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            output_size=1
        )
        
        return model.to(self.device)

    def fit(self, 
            train_data: Union[pd.Series, np.ndarray],
            validation_data: Optional[Union[pd.Series, np.ndarray]] = None,
            verbose: int = 0,
            **kwargs) -> 'LSTMModel':
        """
        Fit LSTM model to training data
        
        Args:
            train_data: Training time series data
            validation_data: Validation data (optional)
            verbose (int): Verbosity level
            **kwargs: Additional fitting parameters
            
        Returns:
            LSTMModel: Fitted model instance
        """
        logger.info("Fitting PyTorch LSTM model")
        
        try:
            # Convert to numpy if pandas Series
            if isinstance(train_data, pd.Series):
                train_data = train_data.values
            
            # Scale the data
            train_scaled = self.scaler.fit_transform(train_data.reshape(-1, 1))
            
            # Create sequences
            X_train, y_train = self._create_sequences(train_scaled.flatten(), self.sequence_length)
            
            # Reshape for LSTM input (batch_size, sequence_length, input_size)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            # Split validation data if validation_split is specified
            if self.validation_split > 0:
                split_idx = int(len(X_train) * (1 - self.validation_split))
                X_val, y_val = X_train[split_idx:], y_train[split_idx:]
                X_train, y_train = X_train[:split_idx], y_train[:split_idx]
            else:
                X_val, y_val = None, None
            
            # Create datasets and dataloaders
            train_dataset = TimeSeriesDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
            
            if X_val is not None:
                val_dataset = TimeSeriesDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            else:
                val_loader = None
            
            # Build model
            self.model = self._build_model(input_size=1)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                self.training_history['train_loss'].append(train_loss)
                
                # Validation phase
                val_loss = 0.0
                if val_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                            outputs = self.model(batch_X)
                            loss = criterion(outputs.squeeze(), batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    self.training_history['val_loss'].append(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose > 0:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                
                if verbose > 0 and (epoch + 1) % 10 == 0:
                    if val_loader is not None:
                        print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    else:
                        print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.6f}")
            
            # Load best model if available
            if hasattr(self, 'best_model_state'):
                self.model.load_state_dict(self.best_model_state)
            
            self.is_fitted = True
            self.train_data = train_data
            
            logger.info("PyTorch LSTM model fitted successfully")
            
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
            self.model.eval()
            
            # Use last sequence from training data if not provided
            if last_sequence is None:
                train_scaled = self.scaler.transform(self.train_data.reshape(-1, 1))
                last_sequence = train_scaled[-self.sequence_length:].flatten()
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            with torch.no_grad():
                for _ in range(steps):
                    # Reshape for prediction
                    X_pred = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1).to(self.device)
                    
                    # Make prediction
                    next_pred = self.model(X_pred).cpu().numpy()[0, 0]
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

    def predict_with_test_data(self, test_data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Make predictions using test data sequences
        
        Args:
            test_data: Test data
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info("Making LSTM predictions with test data")
        
        try:
            # Convert to numpy if pandas Series
            if isinstance(test_data, pd.Series):
                test_data = test_data.values
            
            # Scale test data
            test_scaled = self.scaler.transform(test_data.reshape(-1, 1))
            
            # Create sequences from test data
            X_test, _ = self._create_sequences(test_scaled.flatten(), self.sequence_length)
            
            # Reshape for LSTM input
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Make predictions
            self.model.eval()
            predictions_scaled = []
            
            with torch.no_grad():
                for i in range(0, len(X_test), self.batch_size):
                    batch_X = torch.FloatTensor(X_test[i:i+self.batch_size]).to(self.device)
                    batch_pred = self.model(batch_X).cpu().numpy()
                    predictions_scaled.extend(batch_pred.flatten())
            
            # Inverse transform predictions
            predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions with test data: {e}")
            raise

    def plot_training_history(self) -> None:
        """
        Plot training history
        """
        if not self.training_history['train_loss']:
            logger.warning("No training history available")
            return
        
        logger.info("Plotting LSTM training history")
        
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Training Loss')
        if self.training_history['val_loss']:
            plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot loss in log scale
        plt.subplot(1, 2, 2)
        plt.semilogy(self.training_history['train_loss'], label='Training Loss')
        if self.training_history['val_loss']:
            plt.semilogy(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Model Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str) -> None:
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'model_params': {
                    'sequence_length': self.sequence_length,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout_rate': self.dropout_rate
                },
                'scaler': self.scaler,
                'training_history': self.training_history
            }
            
            torch.save(save_dict, filepath)
            logger.info(f"LSTM model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        """
        Load trained model
        
        Args:
            filepath (str): Path to load model from
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load model parameters
            model_params = checkpoint['model_params']
            self.sequence_length = model_params['sequence_length']
            self.hidden_size = model_params['hidden_size']
            self.num_layers = model_params['num_layers']
            self.dropout_rate = model_params['dropout_rate']
            
            # Build and load model
            self.model = self._build_model(input_size=1)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load scaler and history
            self.scaler = checkpoint['scaler']
            self.training_history = checkpoint['training_history']
            
            self.is_fitted = True
            logger.info(f"LSTM model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            raise

    def get_model_summary(self) -> str:
        """
        Get model architecture summary
        
        Returns:
            str: Model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        summary = f"""
PyTorch LSTM Model Summary:
==========================
Sequence Length: {self.sequence_length}
Hidden Size: {self.hidden_size}
Number of Layers: {self.num_layers}
Dropout Rate: {self.dropout_rate}
Device: {self.device}

Model Architecture:
{self.model}

Total Parameters: {sum(p.numel() for p in self.model.parameters())}
Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}
"""
        return summary