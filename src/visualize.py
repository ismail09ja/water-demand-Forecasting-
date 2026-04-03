"""
Visualization functions for water demand forecasting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_actual_vs_predicted(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot actual vs predicted water demand.
    
    Parameters:
    -----------
    y_true : np.array
        True values
    y_pred : np.array
        Predicted values
    model_name : str
        Name of the model
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7, linewidth=2)
    plt.plot(y_pred, label='Predicted', alpha=0.7, linewidth=2)
    plt.xlabel('Time (Days)')
    plt.ylabel('Water Demand (Liters)')
    plt.title(f'Actual vs Predicted Water Demand - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return plt.gcf()


def plot_forecast_with_uncertainty(y_true, quantile_predictions, dates=None, save_path=None):
    """
    Plot forecast with uncertainty intervals (quantile bands).
    
    Parameters:
    -----------
    y_true : np.array
        True values
    quantile_predictions : dict
        Dictionary of predictions for each quantile
    dates : pd.DatetimeIndex
        Dates for x-axis
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(14, 7))
    
    # Plot true values
    if dates is not None:
        plt.plot(dates, y_true, label='Actual', color='black', linewidth=2, alpha=0.8)
    else:
        plt.plot(y_true, label='Actual', color='black', linewidth=2, alpha=0.8)
    
    # Plot quantile predictions
    if 0.5 in quantile_predictions:
        median = quantile_predictions[0.5]
        if dates is not None:
            plt.plot(dates, median, label='Median Forecast (50%)', color='blue', linewidth=2, linestyle='--')
        else:
            plt.plot(median, label='Median Forecast (50%)', color='blue', linewidth=2, linestyle='--')
    
    # Shade uncertainty intervals
    if 0.1 in quantile_predictions and 0.9 in quantile_predictions:
        lower = quantile_predictions[0.1]
        upper = quantile_predictions[0.9]
        if dates is not None:
            plt.fill_between(dates, lower, upper, alpha=0.3, color='blue', label='80% Prediction Interval')
        else:
            plt.fill_between(range(len(lower)), lower, upper, alpha=0.3, color='blue', label='80% Prediction Interval')
    
    plt.xlabel('Date')
    plt.ylabel('Water Demand (Liters)')
    plt.title('Water Demand Forecast with Uncertainty Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return plt.gcf()


def plot_residuals(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot residuals distribution and time series.
    
    Parameters:
    -----------
    y_true : np.array
        True values
    y_pred : np.array
        Predicted values
    model_name : str
        Name of the model
    save_path : str
        Path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals time series
    axes[0].plot(residuals, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Time (Days)')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residuals Over Time - {model_name}')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residuals Distribution - {model_name}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_data_overview(df, save_path=None):
    """
    Plot overview of the water demand dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with water demand data
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Time series of water demand
    axes[0, 0].plot(df['date'], df['water_demand'], alpha=0.7, linewidth=1)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Water Demand (Liters)')
    axes[0, 0].set_title('Water Demand Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Temperature over time
    axes[0, 1].plot(df['date'], df['temperature'], alpha=0.7, color='orange', linewidth=1)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Temperature Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Rainfall over time
    axes[1, 0].plot(df['date'], df['rainfall'], alpha=0.7, color='blue', linewidth=1)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Rainfall (mm)')
    axes[1, 0].set_title('Rainfall Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Correlation heatmap
    numeric_cols = ['temperature', 'rainfall', 'population_index', 'water_demand']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1, 1], square=True)
    axes[1, 1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def generate_all_plots(data_path=None):
    """
    Generate all visualization plots.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file
    """
    import sys
    import os
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.preprocess import preprocess_pipeline
    from src.model_train import load_sarimax_model, load_quantile_models, predict_quantiles
    
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, feature_names, df = preprocess_pipeline(data_path)
    
    # Create plots directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    plots_dir = os.path.join(project_root, 'data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Data overview
    print("\n1. Generating data overview plot...")
    plot_data_overview(df, save_path=os.path.join(plots_dir, 'data_overview.png'))
    plt.close()
    
    # Load models
    print("\n2. Loading models...")
    try:
        sarimax_model = load_sarimax_model()
        quantile_models = load_quantile_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train models first by running model_train.py")
        return
    
    # 2. SARIMAX predictions
    print("\n3. Generating SARIMAX plots...")
    train_size = len(X_train)
    test_size = len(X_test)
    sarimax_forecast = sarimax_model.forecast(steps=test_size)
    
    plot_actual_vs_predicted(y_test, sarimax_forecast, "SARIMAX", 
                            save_path=os.path.join(plots_dir, 'sarimax_actual_vs_predicted.png'))
    plt.close()
    
    plot_residuals(y_test, sarimax_forecast, "SARIMAX", 
                  save_path=os.path.join(plots_dir, 'sarimax_residuals.png'))
    plt.close()
    
    # 3. Quantile regression predictions
    print("\n4. Generating Quantile Regression plots...")
    quantile_predictions = predict_quantiles(quantile_models, X_test)
    
    # Get test dates
    test_dates = df['date'].iloc[train_size:train_size+test_size].values
    
    plot_actual_vs_predicted(y_test, quantile_predictions[0.5], "Quantile Regression", 
                            save_path=os.path.join(plots_dir, 'quantile_actual_vs_predicted.png'))
    plt.close()
    
    plot_forecast_with_uncertainty(y_test, quantile_predictions, dates=test_dates,
                                  save_path=os.path.join(plots_dir, 'quantile_forecast_uncertainty.png'))
    plt.close()
    
    plot_residuals(y_test, quantile_predictions[0.5], "Quantile Regression", 
                  save_path=os.path.join(plots_dir, 'quantile_residuals.png'))
    plt.close()
    
    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Plots saved to: {plots_dir}")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_plots()

