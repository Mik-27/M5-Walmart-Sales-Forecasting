"""
Plotting utilities for M5 Walmart Sales Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Union, Optional, Any
import warnings

from ..utils.logger import setup_logger

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class M5Visualizer:
    """
    Visualization utilities for M5 dataset and forecasting results
    """
    
    def __init__(self, style: str = 'seaborn', figsize: tuple = (12, 8)):
        """
        Initialize the visualizer
        
        Args:
            style (str): Matplotlib style
            figsize (tuple): Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Set plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
        logger.info(f"Initialized M5Visualizer with style={style}")
    
    def plot_sales_trends(self, 
                         sales_data: pd.DataFrame,
                         sample_items: List[str] = None,
                         num_samples: int = 3) -> None:
        """
        Plot sales trends for sample items
        
        Args:
            sales_data (pd.DataFrame): Sales data
            sample_items (List[str], optional): Specific items to plot
            num_samples (int): Number of random samples if sample_items not provided
        """
        logger.info("Creating sales trends plot")
        
        # Get day columns
        day_columns = [col for col in sales_data.columns if col.startswith('d_')]
        
        if sample_items is None:
            # Random sample
            sample_items = sales_data['id'].sample(num_samples).tolist()
        
        fig = make_subplots(
            rows=len(sample_items), 
            cols=1,
            subplot_titles=[f"Sales Trend for {item}" for item in sample_items]
        )
        
        colors = ['mediumseagreen', 'violet', 'dodgerblue', 'orange', 'red']
        
        for i, item_id in enumerate(sample_items):
            item_data = sales_data[sales_data['id'] == item_id][day_columns].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(item_data))),
                    y=item_data.values,
                    mode='lines',
                    name=item_id,
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(sample_items),
            title_text="Sales Trends for Sample Items"
        )
        fig.show()
    
    def plot_rolling_average(self, 
                           sales_data: pd.DataFrame,
                           calendar: pd.DataFrame,
                           selling_prices: pd.DataFrame,
                           window: int = 90) -> None:
        """
        Plot rolling average sales per store
        
        Args:
            sales_data (pd.DataFrame): Sales data
            calendar (pd.DataFrame): Calendar data
            selling_prices (pd.DataFrame): Price data
            window (int): Rolling window size
        """
        logger.info(f"Creating rolling average plot with window={window}")
        
        # Get day columns
        day_columns = [col for col in sales_data.columns if col.startswith('d_')]
        
        # Transform to long format and merge with calendar
        past_sales = sales_data.set_index('id')[day_columns].T
        past_sales = past_sales.merge(
            calendar.set_index('d')['date'],
            left_index=True,
            right_index=True,
            validate='1:1'
        ).set_index('date')
        
        store_list = selling_prices['store_id'].unique()
        
        fig = go.Figure()
        
        for store in store_list:
            store_items = [col for col in past_sales.columns if store in col]
            if store_items:
                data = past_sales[store_items].sum(axis=1).rolling(window).mean()
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name=store
                ))
        
        fig.update_layout(
            title=f"Rolling Average Sales ({window}-day window) per Store",
            xaxis_title="Date",
            yaxis_title="Sales",
            height=600
        )
        fig.show()
    
    def plot_seasonal_patterns(self, 
                              data: pd.Series,
                              decomposition_type: str = 'additive') -> None:
        """
        Plot seasonal decomposition
        
        Args:
            data (pd.Series): Time series data
            decomposition_type (str): Type of decomposition ('additive' or 'multiplicative')
        """
        logger.info(f"Creating seasonal decomposition plot ({decomposition_type})")
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                data.dropna(), 
                model=decomposition_type,
                period=42  # Weekly seasonality for daily data
            )
            
            # Create subplots
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Observed')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("statsmodels not available for seasonal decomposition")
    
    def plot_category_performance(self, 
                                sales_data: pd.DataFrame,
                                top_n: int = 10) -> None:
        """
        Plot performance by category
        
        Args:
            sales_data (pd.DataFrame): Sales data
            top_n (int): Number of top categories to show
        """
        logger.info(f"Creating category performance plot for top {top_n} categories")
        
        # Get day columns
        day_columns = [col for col in sales_data.columns if col.startswith('d_')]
        
        # Calculate total sales by category
        category_sales = sales_data.groupby('cat_id')[day_columns].sum().sum(axis=1)
        category_sales = category_sales.sort_values(ascending=False).head(top_n)
        
        # Create bar plot
        fig = px.bar(
            x=category_sales.index,
            y=category_sales.values,
            title=f"Top {top_n} Categories by Total Sales",
            labels={'x': 'Category', 'y': 'Total Sales'}
        )
        
        fig.update_layout(height=500)
        fig.show()
    
    def plot_store_performance(self, 
                             sales_data: pd.DataFrame) -> None:
        """
        Plot performance by store
        
        Args:
            sales_data (pd.DataFrame): Sales data
        """
        logger.info("Creating store performance plot")
        
        # Get day columns
        day_columns = [col for col in sales_data.columns if col.startswith('d_')]
        
        # Calculate total sales by store
        store_sales = sales_data.groupby('store_id')[day_columns].sum().sum(axis=1)
        store_sales = store_sales.sort_values(ascending=False)
        
        # Create bar plot
        fig = px.bar(
            x=store_sales.index,
            y=store_sales.values,
            title="Total Sales by Store",
            labels={'x': 'Store ID', 'y': 'Total Sales'}
        )
        
        fig.update_layout(height=500)
        fig.show()
    
    def plot_price_analysis(self, 
                          selling_prices: pd.DataFrame) -> None:
        """
        Plot price distribution and trends
        
        Args:
            selling_prices (pd.DataFrame): Price data
        """
        logger.info("Creating price analysis plots")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Price Distribution',
                'Price by Store',
                'Price Trends Over Time',
                'Price vs Sales Correlation'
            ]
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(x=selling_prices['sell_price'], nbinsx=50, name='Price Distribution'),
            row=1, col=1
        )
        
        # Price by store
        store_avg_price = selling_prices.groupby('store_id')['sell_price'].mean()
        fig.add_trace(
            go.Bar(x=store_avg_price.index, y=store_avg_price.values, name='Avg Price by Store'),
            row=1, col=2
        )
        
        # Price trends over time
        price_trends = selling_prices.groupby('wm_yr_wk')['sell_price'].mean()
        fig.add_trace(
            go.Scatter(x=price_trends.index, y=price_trends.values, mode='lines', name='Price Trend'),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=False)
        fig.show()
    
    def plot_model_comparison(self, 
                            actual: pd.Series,
                            predictions: Dict[str, pd.Series],
                            metrics: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """
        Plot model comparison
        
        Args:
            actual (pd.Series): Actual values
            predictions (Dict[str, pd.Series]): Model predictions
            metrics (Dict[str, Dict[str, float]], optional): Model metrics
        """
        logger.info(f"Creating model comparison plot for {len(predictions)} models")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Model Predictions vs Actual', 'Model Performance Metrics'],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Plot actual values
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        # Plot predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(
                go.Scatter(
                    x=pred.index if hasattr(pred, 'index') else actual.index,
                    y=pred.values if hasattr(pred, 'values') else pred,
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)])
                ),
                row=1, col=1
            )
        
        # Plot metrics if provided
        if metrics:
            metric_names = list(next(iter(metrics.values())).keys())
            model_names = list(metrics.keys())
            
            for i, metric in enumerate(metric_names):
                metric_values = [metrics[model][metric] for model in model_names]
                
                fig.add_trace(
                    go.Bar(
                        x=model_names,
                        y=metric_values,
                        name=metric.upper(),
                        offsetgroup=i
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            height=800,
            title="Model Comparison Results"
        )
        fig.show()
    
    def plot_forecast_with_confidence(self, 
                                    actual: pd.Series,
                                    forecast: pd.Series,
                                    confidence_intervals: Optional[pd.DataFrame] = None,
                                    model_name: str = "Model") -> None:
        """
        Plot forecast with confidence intervals
        
        Args:
            actual (pd.Series): Historical actual values
            forecast (pd.Series): Forecast values
            confidence_intervals (pd.DataFrame, optional): Confidence intervals
            model_name (str): Name of the model
        """
        logger.info(f"Creating forecast plot with confidence intervals for {model_name}")
        
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            mode='lines',
            name='Actual',
            line=dict(color='black')
        ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Plot confidence intervals if provided
        if confidence_intervals is not None:
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=confidence_intervals.iloc[:, 1],  # Upper bound
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=confidence_intervals.iloc[:, 0],  # Lower bound
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='Confidence Interval',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"{model_name} Forecast with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Sales",
            height=600
        )
        fig.show()
    
    def plot_residuals_analysis(self, 
                              residuals: pd.Series,
                              model_name: str = "Model") -> None:
        """
        Plot residuals analysis
        
        Args:
            residuals (pd.Series): Model residuals
            model_name (str): Name of the model
        """
        logger.info(f"Creating residuals analysis for {model_name}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals plot
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        
        # Histogram of residuals
        axes[0, 1].hist(residuals.dropna(), bins=30, alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # ACF of residuals
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals.dropna(), ax=axes[1, 1], lags=40)
            axes[1, 1].set_title('ACF of Residuals')
        except ImportError:
            axes[1, 1].text(0.5, 0.5, 'ACF plot not available\n(statsmodels required)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.suptitle(f'{model_name} Residuals Analysis')
        plt.tight_layout()
        plt.show()
    
    def save_plot(self, 
                  fig, 
                  filename: str, 
                  output_dir: str = "output/figures/",
                  format: str = "png",
                  dpi: int = 300) -> None:
        """
        Save plot to file
        
        Args:
            fig: Figure object
            filename (str): Output filename
            output_dir (str): Output directory
            format (str): File format
            dpi (int): DPI for raster formats
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path
        filepath = os.path.join(output_dir, f"{filename}.{format}")
        
        # Save plot
        if hasattr(fig, 'write_image'):  # Plotly figure
            fig.write_image(filepath)
        else:  # Matplotlib figure
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        logger.info(f"Plot saved to {filepath}")
