"""
Model training for water demand forecasting.
Includes SARIMAX and Gradient Boosting Quantile Regression models.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def train_sarimax(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    Train SARIMAX model for time series forecasting.
    
    Parameters:
    -----------
    data : pd.Series or np.array
        Time series data (water demand)
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple
        Seasonal order (P, D, Q, s)
    
    Returns:
    --------
    model
        Fitted SARIMAX model
    """
    print("Training SARIMAX model...")
    print(f"Order: {order}, Seasonal order: {seasonal_order}")
    
    try:
        model = SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, maxiter=50)
        
        print("SARIMAX model trained successfully!")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        
        return fitted_model
    except Exception as e:
        print(f"Error training SARIMAX: {e}")
        print("Trying simpler model...")
        # Try simpler model
        model = SARIMAX(
            data,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, maxiter=50)
        return fitted_model


def train_quantile_model(X_train, y_train, quantiles=[0.1, 0.5, 0.9], n_estimators=100):
    """
    Train Gradient Boosting Quantile Regression model.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training target
    quantiles : list
        List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
    n_estimators : int
        Number of boosting stages
    
    Returns:
    --------
    dict
        Dictionary of models, one for each quantile
    """
    print(f"Training Quantile Regression model for quantiles: {quantiles}...")
    
    models = {}
    
    for quantile in quantiles:
        print(f"Training model for {quantile*100}% quantile...")
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        models[quantile] = model
        print(f"Model for {quantile*100}% quantile trained successfully!")
    
    return models


def predict_quantiles(models, X):
    """
    Predict quantiles using trained models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of quantile models
    X : np.array
        Input features
    
    Returns:
    --------
    dict
        Dictionary of predictions for each quantile
    """
    predictions = {}
    for quantile, model in models.items():
        predictions[quantile] = model.predict(X)
    return predictions


def save_sarimax_model(model, file_path=None):
    """
    Save SARIMAX model.
    
    Parameters:
    -----------
    model : fitted SARIMAX model
        Model to save
    file_path : str
        Path to save the model (if None, uses default relative to project root)
    """
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        file_path = os.path.join(models_dir, 'sarimax_model.pkl')
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"SARIMAX model saved to {file_path}")


def save_quantile_models(models, file_path=None):
    """
    Save quantile regression models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of quantile models
    file_path : str
        Path to save the models (if None, uses default relative to project root)
    """
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        file_path = os.path.join(models_dir, 'quantile_models.pkl')
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(models, file_path)
    print(f"Quantile models saved to {file_path}")


def load_sarimax_model(file_path=None):
    """
    Load SARIMAX model.
    
    Parameters:
    -----------
    file_path : str
        Path to the model file (if None, uses default relative to project root)
    
    Returns:
    --------
    fitted SARIMAX model
        Loaded model
    """
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        file_path = os.path.join(project_root, 'models', 'sarimax_model.pkl')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    return joblib.load(file_path)


def load_quantile_models(file_path=None):
    """
    Load quantile regression models.
    
    Parameters:
    -----------
    file_path : str
        Path to the model file (if None, uses default relative to project root)
    
    Returns:
    --------
    dict
        Dictionary of quantile models
    """
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        file_path = os.path.join(project_root, 'models', 'quantile_models.pkl')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    return joblib.load(file_path)


def train_all_models(data_path=None):
    """
    Train all models (SARIMAX and Quantile Regression).
    
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
    
    print("=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names, df = preprocess_pipeline(data_path)
    
    # Train SARIMAX on full time series
    print("\n" + "-" * 60)
    print("SARIMAX Model")
    print("-" * 60)
    sarimax_model = train_sarimax(df['water_demand'].values)
    save_sarimax_model(sarimax_model)
    
    # Train Quantile Regression
    print("\n" + "-" * 60)
    print("Quantile Regression Model")
    print("-" * 60)
    quantile_models = train_quantile_model(X_train, y_train)
    save_quantile_models(quantile_models)
    
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    train_all_models()

