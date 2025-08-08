import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the cleaned data
df = pd.read_csv('data/protein_clean.csv')

# Separate features and target
X = df.drop('RMSD', axis=1)
y = df['RMSD']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Save the processed features and target
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
X_poly_df['RMSD'] = y.values
X_poly_df.to_csv('data/protein_features.csv', index=False)

print('Feature engineering complete. Processed data saved to data/protein_features.csv')