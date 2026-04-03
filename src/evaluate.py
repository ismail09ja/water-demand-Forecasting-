"""
Model evaluation and comparison for water demand forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def pinball_loss(y_true, y_pred, quantile):
    """
    Calculate pinball loss for quantile predictions.
    
    Parameters:
    -----------
    y_true : np.array
        True values
    y_pred : np.array
        Predicted quantile values
    quantile : float
        Quantile level (0-1)
    
    Returns:
    --------
    float
        Pinball loss
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return np.mean(loss)


def evaluate_deterministic(y_true, y_pred, model_name="Model"):
    """
    Evaluate deterministic predictions (point forecasts).
    
    Parameters:
    -----------
    y_true : np.array
        True values
    y_pred : np.array
        Predicted values
    model_name : str
        Name of the model
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return metrics


def evaluate_quantile(y_true, quantile_predictions, quantiles=[0.1, 0.5, 0.9]):
    """
    Evaluate quantile predictions.
    
    Parameters:
    -----------
    y_true : np.array
        True values
    quantile_predictions : dict
        Dictionary of predictions for each quantile
    quantiles : list
        List of quantile levels
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {}
    
    print("\nQuantile Regression Performance:")
    
    # Evaluate each quantile
    for quantile in quantiles:
        if quantile in quantile_predictions:
            y_pred = quantile_predictions[quantile]
            pinball = pinball_loss(y_true, y_pred, quantile)
            metrics[f'Pinball_{quantile}'] = pinball
            print(f"  Pinball Loss ({quantile*100}% quantile): {pinball:.2f}")
    
    # Evaluate median (0.5 quantile) as point forecast
    if 0.5 in quantile_predictions:
        median_pred = quantile_predictions[0.5]
        det_metrics = evaluate_deterministic(y_true, median_pred, "Quantile Regression (Median)")
        metrics.update({f'Median_{k}': v for k, v in det_metrics.items()})
    
    # Coverage metrics
    if 0.1 in quantile_predictions and 0.9 in quantile_predictions:
        coverage_80 = np.mean(
            (y_true >= quantile_predictions[0.1]) & 
            (y_true <= quantile_predictions[0.9])
        ) * 100
        metrics['Coverage_80'] = coverage_80
        print(f"  80% Coverage: {coverage_80:.2f}%")
    
    return metrics


def compare_models(data_path=None):
    """
    Compare SARIMAX and Quantile Regression models.
    
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
    print("MODEL EVALUATION AND COMPARISON")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, feature_names, df = preprocess_pipeline(data_path)
    
    # Load models
    print("\nLoading models...")
    try:
        sarimax_model = load_sarimax_model()
        quantile_models = load_quantile_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train models first by running model_train.py")
        return
    
    # Evaluate SARIMAX
    print("\n" + "-" * 60)
    print("Evaluating SARIMAX Model")
    print("-" * 60)
    
    # Get test period for SARIMAX
    train_size = len(X_train)
    test_size = len(X_test)
    
    # Predict with SARIMAX
    sarimax_forecast = sarimax_model.forecast(steps=test_size)
    sarimax_metrics = evaluate_deterministic(y_test, sarimax_forecast, "SARIMAX")
    
    # Evaluate Quantile Regression
    print("\n" + "-" * 60)
    print("Evaluating Quantile Regression Model")
    print("-" * 60)
    
    quantile_predictions = predict_quantiles(quantile_models, X_test)
    quantile_metrics = evaluate_quantile(y_test, quantile_predictions)
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison_df = pd.DataFrame({
        'SARIMAX': [
            sarimax_metrics.get('MAE', np.nan),
            sarimax_metrics.get('RMSE', np.nan),
            sarimax_metrics.get('R²', np.nan),
            sarimax_metrics.get('MAPE', np.nan)
        ],
        'Quantile Regression (Median)': [
            quantile_metrics.get('Median_MAE', np.nan),
            quantile_metrics.get('Median_RMSE', np.nan),
            quantile_metrics.get('Median_R²', np.nan),
            quantile_metrics.get('Median_MAPE', np.nan)
        ]
    }, index=['MAE', 'RMSE', 'R²', 'MAPE'])
    
    print("\nPoint Forecast Comparison:")
    print(comparison_df.to_string())
    
    print("\nQuantile Regression Additional Metrics:")
    print(f"  Pinball Loss (10%): {quantile_metrics.get('Pinball_0.1', np.nan):.2f}")
    print(f"  Pinball Loss (50%): {quantile_metrics.get('Pinball_0.5', np.nan):.2f}")
    print(f"  Pinball Loss (90%): {quantile_metrics.get('Pinball_0.9', np.nan):.2f}")
    print(f"  80% Coverage: {quantile_metrics.get('Coverage_80', np.nan):.2f}%")
    
    return {
        'sarimax': sarimax_metrics,
        'quantile': quantile_metrics,
        'comparison': comparison_df
    }


if __name__ == '__main__':
    compare_models()

