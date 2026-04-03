"""
Generate synthetic water demand data for urban areas.
Creates daily data from 2020-2024 with realistic patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_water_demand_data(start_date='2020-01-01', end_date='2024-12-31', seed=42):
    """
    Generate synthetic water demand data with realistic patterns.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, temperature, rainfall, population_index, water_demand
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Initialize arrays
    temperature = np.zeros(n_days)
    rainfall = np.zeros(n_days)
    population_index = np.zeros(n_days)
    water_demand = np.zeros(n_days)
    
    # Base parameters
    base_temp = 20.0  # Celsius
    base_demand = 500000  # Liters per day
    population_growth_rate = 0.02  # 2% annual growth
    
    for i, date in enumerate(dates):
        # Day of year for seasonal patterns
        day_of_year = date.timetuple().tm_yday
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        
        # 1. Temperature: Seasonal pattern with noise
        seasonal_temp = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)  # Peak in summer
        temp_noise = np.random.normal(0, 3)
        temperature[i] = base_temp + seasonal_temp + temp_noise
        
        # 2. Rainfall: Higher in certain seasons with random events
        seasonal_rain = 5 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)  # Peak in spring
        rain_event = np.random.exponential(2) if np.random.random() < 0.3 else 0
        rainfall[i] = max(0, seasonal_rain + rain_event + np.random.normal(0, 1))
        
        # 3. Population index: Gradual upward trend
        years_from_start = (date - dates[0]).days / 365.25
        population_index[i] = 100 * (1 + population_growth_rate) ** years_from_start
        population_index[i] += np.random.normal(0, 0.5)  # Small noise
        
        # 4. Water demand: Complex relationship
        # Base demand with population growth
        pop_factor = population_index[i] / 100
        
        # Temperature effect (hotter = more demand)
        temp_factor = 1 + 0.02 * (temperature[i] - base_temp) / 10
        
        # Rainfall effect (more rain = less demand)
        rain_factor = 1 - 0.1 * np.minimum(rainfall[i] / 10, 0.5)
        
        # Weekly pattern (higher on weekends)
        weekly_factor = 1.1 if day_of_week >= 5 else 1.0
        
        # Seasonal adjustment (summer higher demand)
        seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        
        # Random noise
        noise = np.random.normal(1, 0.05)
        
        # Calculate demand
        water_demand[i] = (base_demand * pop_factor * temp_factor * 
                          rain_factor * weekly_factor * seasonal_factor * noise)
        
        # Ensure non-negative
        water_demand[i] = max(0, water_demand[i])
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'rainfall': rainfall,
        'population_index': population_index,
        'water_demand': water_demand
    })
    
    return df


def main():
    """Generate and save water demand data."""
    print("Generating synthetic water demand data...")
    
    # Generate data
    df = generate_water_demand_data()
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(data_dir, 'water_demand.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Data generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to: {output_path}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())


if __name__ == '__main__':
    main()

