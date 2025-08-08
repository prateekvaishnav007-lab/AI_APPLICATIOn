import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load processed features
df = pd.read_csv('data/protein_features.csv')
X = df.drop('RMSD', axis=1)
y = df['RMSD']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
gs = GridSearchCV(rf, grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
gs.fit(X_train, y_train)

print('Best parameters:', gs.best_params_)
print('Best CV score (neg MSE):', gs.best_score_)

# Evaluate on test set
y_pred = gs.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {mse:.4f}, R2: {r2:.4f}')

# Save results
with open('outputs/hyperparameter_tuning.txt', 'w') as f:
    f.write(f'Best parameters: {gs.best_params_}\n')
    f.write(f'Best CV score (neg MSE): {gs.best_score_}\n')
    f.write(f'Test MSE: {mse:.4f}, R2: {r2:.4f}\n')

# Save the best model
joblib.dump(gs.best_estimator_, 'models/RandomForest_best_tuned.pkl')
print('Best tuned Random Forest model saved to models/RandomForest_best_tuned.pkl')