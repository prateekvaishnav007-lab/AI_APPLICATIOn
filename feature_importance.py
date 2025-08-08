import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Load processed features
df = pd.read_csv('data/protein_features.csv')
X = df.drop('RMSD', axis=1)
y = df['RMSD']

# Fit Random Forest for feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Save feature importances to file
importance_df = pd.DataFrame({'feature': features, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
importance_df.to_csv('outputs/feature_importance.csv', index=False)

# Plot top 20 features
plt.figure(figsize=(10, 6))
plt.title('Top 20 Feature Importances (Random Forest)')
plt.barh(range(20), importances[indices][:20][::-1], align='center')
plt.yticks(range(20), [features[i] for i in indices][:20][::-1])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

print('Feature importance analysis complete. Results saved to outputs/feature_importance.csv')