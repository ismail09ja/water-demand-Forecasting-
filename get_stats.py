import pandas as pd

df = pd.read_csv('data/water_demand.csv')
print('DATASET STATISTICS')
print('='*50)
print(f'Total Records: {len(df)}')
print(f'Date Range: {df["date"].min()} to {df["date"].max()}')
print(f'\nSummary Statistics:\n')
print(df.describe().round(2))




