import pandas as pd

# Load the dataset
df = pd.read_csv('data/protein.csv')

# Display basic information
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('\nFirst 5 rows:')
print(df.head())
print('\nMissing values per column:')
print(df.isnull().sum())