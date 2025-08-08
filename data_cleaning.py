import pandas as pd

# Load the dataset
df = pd.read_csv('data/protein.csv')

# Check for missing values
print('Missing values per column:')
print(df.isnull().sum())

# Optionally, drop rows with missing values (or you can choose to fill them)
df_clean = df.dropna()
print(f'Rows after dropping missing values: {df_clean.shape[0]}')

# Check for duplicates
duplicates = df_clean.duplicated().sum()
print(f'Duplicate rows: {duplicates}')

# Drop duplicates if any
df_clean = df_clean.drop_duplicates()
print(f'Rows after dropping duplicates: {df_clean.shape[0]}')

# Save the cleaned data
df_clean.to_csv('data/protein_clean.csv', index=False)
print('Cleaned data saved to data/protein_clean.csv')