import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Try to import LightGBM and XGBoost
try:
    from lightgbm import LGBMRegressor
    has_lgbm = True
except ImportError:
    has_lgbm = False
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False

# Load processed features (use only first 5000 rows for quick test)
df = pd.read_csv('data/protein_features.csv').head(5000)
X = df.drop('RMSD', axis=1)
y = df['RMSD']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate (fewer estimators for speed)
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=10, random_state=42)
}
if has_lgbm:
    models['LightGBM'] = LGBMRegressor(n_estimators=10, random_state=42)
if has_xgb:
    models['XGBoost'] = XGBRegressor(n_estimators=10, random_state=42, verbosity=0)

results = {}
best_model = None
best_score = float('inf')
best_name = ''

for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f'{name}: MSE={mse:.4f}, R2={r2:.4f}')
    if mse < best_score:
        best_score = mse
        best_model = model
        best_name = name

# Save results
df_results = pd.DataFrame(results).T
print('\nModel performance:')
print(df_results)
df_results.to_csv('outputs/model_performance.csv')

# Save the best model
joblib.dump(best_model, f'models/{best_name}_best_model.pkl')
print(f'Best model ({best_name}) saved to models/{best_name}_best_model.pkl')