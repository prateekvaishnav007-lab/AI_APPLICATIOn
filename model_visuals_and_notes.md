# Model Visuals and Interpretation Notes

## 1. Model Performance Table
**Visual:** Table of MSE and R² for Linear Regression, Random Forest, and Gradient Boosting
**Interpretation:**
- Random Forest and Gradient Boosting outperform Linear Regression, indicating non-linear relationships in the data.
- R² values suggest moderate predictive power; further tuning or feature selection may help.

## 2. Feature Importance Plot
**Visual:** Bar plot of top 20 feature importances (Random Forest)
**Interpretation:**
- Top features are mostly from F1, F2, F5 and their polynomial/interaction terms.
- Model relies heavily on a few key features, suggesting possible dimensionality reduction.

## 3. Hyperparameter Tuning Results
**Visual:** Text summary of best parameters and test performance
**Interpretation:**
- Grid search improves Random Forest performance by optimizing tree depth, number, and split criteria.
- Best model is saved for deployment.

---

**Recommendations:**
- Use the best tuned model for predictions.
- Consider further tuning or ensembling for improved results.