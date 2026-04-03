"""
Streamlit dashboard for water demand forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Water Demand Forecasting",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data(file_path=None):
    """Load water demand data."""
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        file_path = os.path.join(project_root, 'data', 'water_demand.csv')
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_resource
def load_models():
    """Load trained models."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models = {}
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Try loading SARIMAX
    try:
        from src.model_train import load_sarimax_model
        models['sarimax'] = load_sarimax_model()
    except Exception as e:
        st.warning(f"SARIMAX model not found or failed to load. (Note: Large model files might be excluded from GitHub). Error: {e}")
        models['sarimax'] = None
        
    # Try loading Quantile models
    try:
        from src.model_train import load_quantile_models
        models['quantile'] = load_quantile_models()
        scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
        models['scaler'] = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Quantile models not found or failed to load. Error: {e}")
        models['quantile'] = None
        
    if models['sarimax'] is None and models['quantile'] is None:
        return None
        
    return models


@st.cache_data
def load_feature_names():
    """Load feature names from preprocessing."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from src.preprocess import preprocess_pipeline
        _, _, _, _, _, feature_names, _ = preprocess_pipeline()
        return feature_names
    except:
        return None


def forecast_sarimax(model, steps):
    """Generate SARIMAX forecast."""
    try:
        forecast = model.forecast(steps=steps)
        return forecast
    except Exception as e:
        st.error(f"Error in SARIMAX forecast: {e}")
        return None


def forecast_quantile(models, X_last, steps, feature_names, scaler, df):
    """Generate quantile regression forecast for future steps."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocess import create_features, handle_missing_values
    
    try:
        # Create future dates
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        
        # Initialize predictions
        predictions = {q: [] for q in [0.1, 0.5, 0.9]}
        
        # Use last known values as base
        last_row = df.iloc[-1].copy()
        
        # Get historical data for lag features
        hist_demand = df['water_demand'].values
        
        for i, future_date in enumerate(future_dates):
            # Estimate seasonal temperature
            day_of_year = future_date.timetuple().tm_yday
            base_temp = 20.0
            seasonal_temp = base_temp + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            
            # Estimate rainfall (assume minimal for forecast)
            seasonal_rain = max(0, 5 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25))
            
            # Population growth
            years_from_start = (future_date - df['date'].iloc[0]).days / 365.25
            pop_growth_rate = 0.02
            population_index = 100 * (1 + pop_growth_rate) ** years_from_start
            
            # Use previous predictions for lag features, or historical if available
            if i == 0:
                demand_lag_1 = hist_demand[-1] if len(hist_demand) > 0 else last_row['water_demand']
                demand_lag_7 = hist_demand[-7] if len(hist_demand) > 7 else last_row['water_demand']
                demand_lag_30 = hist_demand[-30] if len(hist_demand) > 30 else last_row['water_demand']
                demand_rolling_mean_7 = np.mean(hist_demand[-7:]) if len(hist_demand) >= 7 else last_row['water_demand']
                demand_rolling_mean_30 = np.mean(hist_demand[-30:]) if len(hist_demand) >= 30 else last_row['water_demand']
                demand_rolling_std_7 = np.std(hist_demand[-7:]) if len(hist_demand) >= 7 else 0
                current_demand = last_row['water_demand']
            else:
                demand_lag_1 = predictions[0.5][-1]
                demand_lag_7 = predictions[0.5][-7] if len(predictions[0.5]) >= 7 else last_row['water_demand']
                demand_lag_30 = predictions[0.5][-30] if len(predictions[0.5]) >= 30 else last_row['water_demand']
                recent_preds = predictions[0.5][-7:] if len(predictions[0.5]) >= 7 else predictions[0.5]
                demand_rolling_mean_7 = np.mean(recent_preds) if len(recent_preds) > 0 else last_row['water_demand']
                recent_preds_30 = predictions[0.5][-30:] if len(predictions[0.5]) >= 30 else predictions[0.5]
                demand_rolling_mean_30 = np.mean(recent_preds_30) if len(recent_preds_30) > 0 else last_row['water_demand']
                demand_rolling_std_7 = np.std(recent_preds) if len(recent_preds) > 1 else 0
                current_demand = predictions[0.5][-1]
            
            # Create future dataframe row
            future_df = pd.DataFrame({
                'date': [future_date],
                'temperature': [seasonal_temp],
                'rainfall': [seasonal_rain],
                'population_index': [population_index],
                'water_demand': [current_demand]
            })
            
            # Create features
            future_df = create_features(future_df)
            
            # Update lag features manually to use predictions
            future_df['demand_lag_1'] = demand_lag_1
            future_df['demand_lag_7'] = demand_lag_7
            future_df['demand_lag_30'] = demand_lag_30
            future_df['demand_rolling_mean_7'] = demand_rolling_mean_7
            future_df['demand_rolling_mean_30'] = demand_rolling_mean_30
            future_df['demand_rolling_std_7'] = demand_rolling_std_7
            
            future_df = handle_missing_values(future_df)
            
            # Prepare features (exclude date and target)
            if feature_names:
                # Ensure all feature names exist
                missing_features = set(feature_names) - set(future_df.columns)
                for feat in missing_features:
                    future_df[feat] = 0  # Fill missing features with 0
                X_future = future_df[feature_names].values
            else:
                exclude_cols = ['date', 'water_demand']
                feature_cols = [col for col in future_df.columns if col not in exclude_cols]
                X_future = future_df[feature_cols].values
            
            # Scale features
            X_future_scaled = scaler.transform(X_future)
            
            # Predict
            for quantile in [0.1, 0.5, 0.9]:
                pred = models[quantile].predict(X_future_scaled)[0]
                predictions[quantile].append(max(0, pred))  # Ensure non-negative
        
        return predictions, future_dates
    except Exception as e:
        st.error(f"Error in quantile forecast: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<p class="main-header">💧 Water Demand Probability Forecasting</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Check if data exists
    df = load_data()
    if df is None:
        st.error("⚠️ Data file not found. Please run `python src/generate_data.py` first.")
        st.stop()
    
    # Load models
    models = load_models()
    if models is None:
        st.error("⚠️ Models not found. Please run `python src/model_train.py` first.")
        st.stop()
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Quantile Regression", "SARIMAX"],
        index=0
    )
    
    # Forecast horizon
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    with col3:
        st.metric("Avg Daily Demand", f"{df['water_demand'].mean():,.0f} L")
    with col4:
        st.metric("Max Daily Demand", f"{df['water_demand'].max():,.0f} L")
    
    st.markdown("---")
    
    # Forecast section
    st.header("📊 Forecast")
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            if model_type == "SARIMAX":
                forecast = forecast_sarimax(models['sarimax'], forecast_horizon)
                
                if forecast is not None:
                    # Create forecast DataFrame
                    last_date = df['date'].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_horizon,
                        freq='D'
                    )
                    
                    forecast_df = pd.DataFrame({
                        'date': future_dates,
                        'forecast': forecast
                    })
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data (last 90 days)
                    hist_df = df.tail(90)
                    ax.plot(hist_df['date'], hist_df['water_demand'], 
                           label='Historical', color='blue', alpha=0.7, linewidth=2)
                    
                    # Plot forecast
                    ax.plot(forecast_df['date'], forecast_df['forecast'],
                           label='Forecast', color='red', linestyle='--', linewidth=2)
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Water Demand (Liters)')
                    ax.set_title('Water Demand Forecast - SARIMAX')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Display forecast table
                    st.subheader("Forecast Values")
                    forecast_df['forecast'] = forecast_df['forecast'].round(2)
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Download button
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast as CSV",
                        data=csv,
                        file_name=f"water_demand_forecast_sarimax_{forecast_horizon}days.csv",
                        mime="text/csv"
                    )
            
            else:  # Quantile Regression
                feature_names = load_feature_names()
                predictions, future_dates = forecast_quantile(
                    models['quantile'], None, forecast_horizon,
                    feature_names, models['scaler'], df
                )
                
                if predictions is not None and future_dates is not None:
                    # Create forecast DataFrame
                    forecast_df = pd.DataFrame({
                        'date': future_dates,
                        'forecast_10%': predictions[0.1],
                        'forecast_50%': predictions[0.5],
                        'forecast_90%': predictions[0.9]
                    })
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data (last 90 days)
                    hist_df = df.tail(90)
                    ax.plot(hist_df['date'], hist_df['water_demand'],
                           label='Historical', color='black', alpha=0.8, linewidth=2)
                    
                    # Plot forecast
                    ax.plot(forecast_df['date'], forecast_df['forecast_50%'],
                           label='Median Forecast (50%)', color='blue', linestyle='--', linewidth=2)
                    
                    # Shade uncertainty interval
                    ax.fill_between(
                        forecast_df['date'],
                        forecast_df['forecast_10%'],
                        forecast_df['forecast_90%'],
                        alpha=0.3, color='blue', label='80% Prediction Interval'
                    )
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Water Demand (Liters)')
                    ax.set_title('Water Demand Forecast with Uncertainty - Quantile Regression')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Display forecast table
                    st.subheader("Forecast Values")
                    forecast_df = forecast_df.round(2)
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Download button
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast as CSV",
                        data=csv,
                        file_name=f"water_demand_forecast_quantile_{forecast_horizon}days.csv",
                        mime="text/csv"
                    )
    
    # Data overview section
    st.markdown("---")
    st.header("📈 Data Overview")
    
    # Time series plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['date'], df['water_demand'], alpha=0.7, linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Water Demand (Liters)')
    ax.set_title('Water Demand Over Time')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Summary Statistics")
        st.dataframe(df[['temperature', 'rainfall', 'population_index', 'water_demand']].describe())
    
    with col2:
        st.subheader("Recent Data")
        st.dataframe(df.tail(10), use_container_width=True)


if __name__ == '__main__':
    main()

