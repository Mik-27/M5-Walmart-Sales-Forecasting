# M5 Walmart Sales Forecasting

A comprehensive production-ready data science project for forecasting Walmart sales using multiple time series models including SARIMA, LSTM, and Prophet. This project demonstrates best practices for converting experimental notebooks into production-grade code with proper architecture, evaluation, and business impact analysis.

## Project Structure

```
M5 Walmart Sales Forecasting/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── config/
│   ├── config.yaml              # Configuration settings
│   └── logging_config.yaml      # Logging configuration
├── data/                        # Data directory
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data files
│   └── external/                # External data sources
├── src/                         # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Data loading utilities
│   │   └── preprocessing.py     # Data preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py        # Base model class
│   │   ├── sarima_model.py      # SARIMA implementation
│   │   ├── lstm_model.py        # LSTM implementation
│   │   └── prophet_model.py     # Prophet implementation
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py             # Visualization utilities
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration utilities
│       ├── logger.py            # Logging utilities
│       └── metrics.py           # Evaluation metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA notebook
│   ├── 02_feature_engineering.ipynb # Feature engineering
│   ├── 03_model_training.ipynb      # Model training
│   └── 04_model_evaluation.ipynb   # Model evaluation
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   └── test_utils/
└── output/                      # Output files
    ├── models/                  # Trained models
    ├── predictions/             # Predictions
    └── figures/                 # Generated plots
```

## Features

-   **Production-Ready Architecture**: Modular design with proper separation of concerns
-   **Multiple Forecasting Models**: SARIMA, LSTM, and Prophet with comprehensive comparison
-   **Advanced Feature Engineering**: Lag features, rolling statistics, temporal features, price features, and event indicators
-   **Comprehensive Evaluation**: Statistical validation, residual analysis, and business impact metrics
-   **Business Impact Analysis**: Revenue forecasting, inventory optimization, and operational cost analysis
-   **Robust Data Pipeline**: Data validation, preprocessing, and automated feature generation
-   **Professional Documentation**: Complete notebooks with detailed explanations and visualizations
-   **Configuration Management**: YAML-based configuration with logging and error handling
-   **Model Persistence**: Save/load functionality for all trained models
-   **Automated Workflows**: End-to-end pipeline from raw data to production predictions

## Dataset

The project uses the M5 Forecasting dataset from Walmart, which includes:

-   **sales_train_validation.csv**: Historical daily unit sales data for 3,049 products across 10 stores
-   **calendar.csv**: Date information with events, holidays, SNAP benefits, and special events
-   **sell_prices.csv**: Selling price information per store, item, and date with promotional data
-   **sample_submission.csv**: Sample submission format for competition

### Data Characteristics

-   **Time Period**: 1,913 days (approximately 5.4 years) from 2011-01-29 to 2016-05-22
-   **Products**: 3,049 unique products across 3 categories (Hobbies, Foods, Household)
-   **Stores**: 10 stores across 3 states (California, Texas, Wisconsin)
-   **Hierarchical Structure**: State → Store → Category → Department → Item level aggregations

## Quick Start

### Prerequisites

-   Python 3.8+
-   Required packages listed in `requirements.txt`

### Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd "M5 Walmart Sales Forecasting"
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the package:**

    ```bash
    pip install -e .
    ```

4. **Download the M5 dataset:**
    - Download from Kaggle: https://www.kaggle.com/c/m5-forecasting-accuracy/data
    - Place files in `data/raw/` directory

### Running the Analysis

**Option 1: Run Complete Notebooks (Recommended for first-time users)**

```bash
# 1. Exploratory Data Analysis
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Feature Engineering
jupyter notebook notebooks/02_feature_engineering.ipynb

# 3. Model Training
jupyter notebook notebooks/03_model_training.ipynb

# 4. Model Evaluation
jupyter notebook notebooks/04_model_evaluation.ipynb
```

**Option 2: Use Modular Components**

```python
# Quick example - see notebooks for detailed usage
from src.data.data_loader import M5DataLoader
from src.models.sarima_model import SarimaModel

# Load data
loader = M5DataLoader('data/raw/')
calendar, sales, prices = loader.load_all_data()

# Train model
model = SarimaModel()
model.fit(train_data)
predictions = model.predict(28)
```

## Usage

### Data Loading

```python
from src.data.data_loader import M5DataLoader

loader = M5DataLoader('data/raw/')
calendar, sales, prices = loader.load_all_data()
```

### Model Training

```python
from src.models.sarima_model import SarimaModel
from src.models.lstm_model import LSTMModel
from src.models.prophet_model import ProphetModel

# Train SARIMA model
sarima = SarimaModel()
sarima.fit(train_data)
sarima_predictions = sarima.predict(steps=28)

# Train LSTM model
lstm = LSTMModel(sequence_length=12)
lstm.fit(train_data)
lstm_predictions = lstm.predict(test_data)

# Train Prophet model
prophet_model = ProphetModel()
prophet_model.fit(train_data)
prophet_predictions = prophet_model.predict(periods=28)
```

### Model Evaluation

```python
from src.utils.metrics import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.compare_models(
    actual=test_data,
    predictions={
        'SARIMA': sarima_predictions,
        'LSTM': lstm_predictions,
        'Prophet': prophet_predictions
    }
)
```

## Models

### SARIMA (Seasonal AutoRegressive Integrated Moving Average)

-   **Best for**: Time series with clear seasonal patterns and stationarity
-   **Features**: Automatic parameter selection, seasonal decomposition, statistical significance testing
-   **Pros**: Fast training, interpretable, works well with limited data
-   **Cons**: Assumes stationarity, limited multivariate capability

### LSTM (Long Short-Term Memory)

-   **Best for**: Complex patterns and multivariate time series with rich features
-   **Features**: Sequence-to-sequence prediction, handles multiple input features, non-linear pattern recognition
-   **Pros**: Captures complex relationships, handles multiple features, good for long sequences
-   **Cons**: Requires more data, longer training time, less interpretable

### Prophet

-   **Best for**: Business forecasting with holidays and trend changes
-   **Features**: Automatic seasonality detection, holiday effects, trend change detection, uncertainty intervals
-   **Pros**: Robust to missing data, handles holidays automatically, highly interpretable components
-   **Cons**: May overfit with limited data, less flexible than deep learning approaches

## Evaluation Metrics

The project uses comprehensive evaluation metrics:

### Statistical Metrics

-   **RMSE**: Root Mean Square Error for overall accuracy
-   **MAE**: Mean Absolute Error for average deviation
-   **MAPE**: Mean Absolute Percentage Error for relative accuracy
-   **R²**: Coefficient of determination for variance explained
-   **Direction Accuracy**: Percentage of correct trend predictions

### Business Metrics

-   **Revenue Impact**: Predicted vs actual revenue differences
-   **Inventory Costs**: Carrying costs for overstocking
-   **Lost Sales**: Revenue lost from understocking
-   **Total Operational Cost**: Combined business impact

### Statistical Validation

-   **Residual Analysis**: Normality, autocorrelation, heteroscedasticity tests
-   **Ljung-Box Test**: Independence of residuals
-   **Jarque-Bera Test**: Normality of residuals
-   **Durbin-Watson Test**: Autocorrelation detection

## Configuration

The project uses YAML configuration files in the `config/` directory:

-   `config.yaml`: Main configuration settings
-   `logging_config.yaml`: Logging configuration

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## Project Outputs

### Generated Files

-   **`data/processed/sales_processed.parquet`**: Fully engineered dataset with all features
-   **`data/processed/highest_selling_product_ts.csv`**: Time series data for focused modeling
-   **`data/processed/feature_metadata.json`**: Feature engineering configuration and metadata
-   **`models/`**: Trained model files (SARIMA, LSTM, Prophet)
-   **`models/model_comparison_results.csv`**: Performance comparison across all models
-   **`models/training_metadata.json`**: Training configuration and best model information
-   **`output/evaluation/`**: Comprehensive evaluation results and business impact analysis

### Key Insights

-   **Best Performing Model**: Varies by metric - see evaluation notebooks for detailed comparison
-   **Feature Importance**: Lag features (1, 7, 28 days) are most predictive
-   **Seasonal Patterns**: Strong weekly seasonality with event-driven variations
-   **Business Impact**: Quantified inventory optimization opportunities
-   **Operational Recommendations**: Model-specific deployment strategies

## Architecture Highlights

### Modular Design

-   **Base Classes**: Abstract model interface for consistent API
-   **Data Layer**: Separate loading, validation, and preprocessing components
-   **Model Layer**: Independent model implementations with common interface
-   **Utility Layer**: Shared configuration, logging, and metrics functionality

### Production Features

-   **Error Handling**: Comprehensive exception handling with informative messages
-   **Logging**: Structured logging throughout the pipeline
-   **Configuration**: YAML-based configuration management
-   **Validation**: Data quality checks and model validation
-   **Persistence**: Save/load functionality for all components

### Best Practices Implemented

-   **Clean Code**: PEP 8 compliance, meaningful variable names, docstrings
-   **Documentation**: Comprehensive README, inline comments, notebook explanations
-   **Testing Structure**: Unit test framework setup (ready for implementation)
-   **Version Control**: Git-friendly structure with proper .gitignore
-   **Reproducibility**: Fixed random seeds and deterministic processes

## Next Steps for Production

1. **Scale to Multiple Products**: Apply pipeline to entire product catalog
2. **Automated Retraining**: Implement scheduled model retraining
3. **Real-time Prediction API**: Deploy models as REST API endpoints
4. **Monitoring & Alerting**: Implement model drift detection and performance monitoring
5. **A/B Testing Framework**: Test different models in production
6. **Integration**: Connect with inventory management and planning systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   Walmart for providing the M5 forecasting dataset
-   Kaggle for hosting the competition
-   The open-source community for the excellent libraries used in this project
