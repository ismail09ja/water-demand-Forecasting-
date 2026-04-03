# 💧 Water Demand Probability Forecasting for Urban Areas

A complete, production-ready machine learning project for forecasting daily urban water demand using **probabilistic models**. The system provides both expected values and uncertainty intervals (10%, 50%, 90% quantiles) to help urban planners and water management authorities make informed decisions.

## 🎯 Features

- **Synthetic Data Generation**: Creates realistic synthetic water demand data (2020-2024) with seasonal patterns, trends, and noise
- **Two Forecasting Models**:
  - **SARIMAX**: Time series model with seasonal components
  - **Quantile Regression**: Gradient Boosting model that predicts multiple quantiles (10%, 50%, 90%)
- **Comprehensive Evaluation**: MAE, RMSE, R², Pinball Loss, and coverage metrics
- **Rich Visualizations**: Time series plots, uncertainty intervals, residual analysis
- **Interactive Dashboard**: Streamlit web app for easy forecasting and exploration

## 📁 Project Structure

```
water-demand-forecast/
├── data/                  # Data directory
│   ├── water_demand.csv  # Generated dataset
│   └── plots/            # Generated visualization plots
├── models/               # Saved models
│   ├── sarimax_model.pkl
│   ├── quantile_models.pkl
│   └── scaler.pkl
├── src/
│   ├── generate_data.py  # Synthetic data generation
│   ├── preprocess.py     # Data preprocessing and feature engineering
│   ├── model_train.py    # Model training (SARIMAX & Quantile Regression)
│   ├── evaluate.py       # Model evaluation and comparison
│   ├── visualize.py      # Visualization functions
│   └── app.py           # Streamlit dashboard
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate Data

Generate synthetic water demand data:

```bash
python src/generate_data.py
```

This will create `data/water_demand.csv` with daily data from 2020-2024.

### 3. Train Models

Train both SARIMAX and Quantile Regression models:

```bash
python src/model_train.py
```

This will:
- Preprocess the data
- Train SARIMAX model
- Train Quantile Regression models (10%, 50%, 90% quantiles)
- Save models to `models/` directory

### 4. Evaluate Models

Evaluate and compare model performance:

```bash
python src/evaluate.py
```

This will display:
- MAE, RMSE, R², MAPE for point forecasts
- Pinball Loss for each quantile
- Coverage metrics for prediction intervals
- Model comparison summary

### 5. Generate Visualizations

Create all visualization plots:

```bash
python src/visualize.py
```

Plots will be saved to `data/plots/` directory:
- Data overview
- Actual vs Predicted comparisons
- Forecast uncertainty intervals
- Residual analysis

### 6. Run Interactive Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run src/app.py
```

The dashboard allows you to:
- Select model (SARIMAX or Quantile Regression)
- Choose forecast horizon (7-90 days)
- View interactive forecasts with uncertainty intervals
- Download forecasts as CSV
- Explore data overview and statistics

## 📊 Data Description

The synthetic dataset includes the following columns:

- **date**: Daily date (2020-01-01 to 2024-12-31)
- **temperature**: Daily temperature in Celsius (seasonal pattern)
- **rainfall**: Daily rainfall in mm (seasonal pattern with random events)
- **population_index**: Population growth index (gradual upward trend)
- **water_demand**: Daily water demand in liters (target variable)

### Data Relationships

- **Temperature**: Higher temperature → higher water demand
- **Rainfall**: More rainfall → lower demand
- **Population**: Gradual upward trend from population growth
- **Seasonality**: Weekly patterns (higher on weekends) and seasonal patterns (higher in summer)
- **Noise**: Random variations for realism

## 🤖 Models

### 1. SARIMAX (Seasonal ARIMA with Exogenous Variables)

- **Type**: Time series model
- **Parameters**: ARIMA order (1,1,1) with seasonal order (1,1,1,7)
- **Output**: Point forecasts
- **Use Case**: Baseline time series forecasting

### 2. Quantile Regression (Gradient Boosting)

- **Type**: Ensemble learning model
- **Quantiles**: 10%, 50%, 90%
- **Output**: Probabilistic forecasts with uncertainty intervals
- **Use Case**: Risk-aware forecasting with uncertainty quantification

## 📈 Evaluation Metrics

### Point Forecast Metrics
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actuals
- **RMSE** (Root Mean Squared Error): Square root of average squared differences
- **R²** (Coefficient of Determination): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Average percentage error

### Probabilistic Forecast Metrics
- **Pinball Loss**: Quantile-specific loss function
- **Coverage**: Percentage of actual values within prediction intervals

## 🎨 Visualizations

The project includes several visualization types:

1. **Data Overview**: Time series of all variables and correlation matrix
2. **Actual vs Predicted**: Comparison of forecasts with actual values
3. **Uncertainty Intervals**: Forecasts with shaded 10-90% quantile bands
4. **Residual Analysis**: Residuals over time and distribution

## 🛠️ Usage Examples

### Generate Forecast Programmatically

```python
from src.model_train import load_quantile_models, predict_quantiles
from src.preprocess import preprocess_pipeline
import joblib

# Load data and models
X_train, X_test, y_train, y_test, scaler, feature_names, df = preprocess_pipeline()
quantile_models = load_quantile_models()

# Generate predictions
predictions = predict_quantiles(quantile_models, X_test)

# Access predictions
forecast_10 = predictions[0.1]  # 10% quantile
forecast_50 = predictions[0.5]  # 50% quantile (median)
forecast_90 = predictions[0.9]  # 90% quantile
```

### Customize Data Generation

```python
from src.generate_data import generate_water_demand_data

# Generate custom date range
df = generate_water_demand_data(
    start_date='2023-01-01',
    end_date='2023-12-31',
    seed=123
)
```

## 🔧 Configuration

### Adjust Forecast Horizon

In the Streamlit app, use the sidebar slider to select forecast horizon (7-90 days).

### Modify Model Parameters

Edit `src/model_train.py` to adjust:
- SARIMAX order and seasonal order
- Quantile Regression hyperparameters (n_estimators, max_depth, learning_rate)

### Change Train/Test Split

Modify `test_size` parameter in `src/preprocess.py` (default: 0.2, i.e., 20% for testing).

## 📝 Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- streamlit
- joblib

## 🐛 Troubleshooting

### Models not found
If you get "Model file not found" error, run `python src/model_train.py` first.

### Data file not found
If you get "Data file not found" error, run `python src/generate_data.py` first.

### Memory issues
If you encounter memory issues with large datasets, reduce the date range in `generate_data.py` or adjust the train/test split.

## 📚 References

- **SARIMAX**: Statsmodels documentation on Seasonal ARIMA
- **Quantile Regression**: Scikit-learn Gradient Boosting Regressor with quantile loss
- **Streamlit**: Streamlit documentation for building dashboards

## 📄 License

This project is provided as-is for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

This project demonstrates probabilistic forecasting techniques for urban water demand management, combining traditional time series methods with modern machine learning approaches.

---

**Built with ❤️ for urban water management**

