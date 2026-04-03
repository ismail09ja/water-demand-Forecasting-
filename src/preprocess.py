"""
Data preprocessing and feature engineering for water demand forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib


def load_data(file_path=None):
    """
    Load water demand data from CSV.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file (if None, uses default relative to project root)
    
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    if file_path is None:
        # Get project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        file_path = os.path.join(project_root, 'data', 'water_demand.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}. Please run generate_data.py first.")
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    # Forward fill for small gaps, then backward fill
    df = df.ffill().bfill()
    
    # If still missing, fill with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    return df


def create_features(df):
    """
    Create time-based and lag features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with 'date' column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional features
    """
    df = df.copy()
    
    # Time-based features
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Lag features
    df['demand_lag_1'] = df['water_demand'].shift(1)
    df['demand_lag_7'] = df['water_demand'].shift(7)
    df['demand_lag_30'] = df['water_demand'].shift(30)
    
    # Rolling statistics
    df['demand_rolling_mean_7'] = df['water_demand'].rolling(window=7, min_periods=1).mean()
    df['demand_rolling_mean_30'] = df['water_demand'].rolling(window=30, min_periods=1).mean()
    df['demand_rolling_std_7'] = df['water_demand'].rolling(window=7, min_periods=1).std()
    
    # Fill NaN values from lag features
    df = df.bfill().ffill()
    
    return df


def prepare_data(df, test_size=0.2, target_col='water_demand'):
    """
    Prepare data for modeling: scale features and split into train/test.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features
    test_size : float
        Proportion of data to use for testing
    target_col : str
        Name of the target column
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    # Select features (exclude date and target)
    exclude_cols = ['date', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split into train/test (time-series split)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def preprocess_pipeline(file_path=None, test_size=0.2):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler, feature_names, df)
    """
    print("Loading data...")
    df = load_data(file_path)
    
    print("Handling missing values...")
    df = handle_missing_values(df)
    
    print("Creating features...")
    df = create_features(df)
    
    print("Preparing data for modeling...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df, test_size)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names, df


if __name__ == '__main__':
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, scaler, feature_names, df = preprocess_pipeline()
    print("\nPreprocessing complete!")
    print(f"Feature names: {feature_names}")

