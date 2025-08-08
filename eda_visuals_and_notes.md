# EDA Visuals and Interpretation Notes

## 1. Feature Distributions
**Visual:** Histogram for each feature (RMSD, F1-F9)
**Interpretation:**
- Most features are right-skewed, indicating outliers or long tails.
- F3 and F4 are fractional, bounded between 0 and 1.
- RMSD is right-skewed, with most values at the lower end.

## 2. Correlation Heatmap
**Visual:** Heatmap of feature correlations
**Interpretation:**
- F1, F2, and F5 are highly correlated, suggesting redundancy.
- RMSD is moderately correlated with F1, F2, and F5, indicating their importance as predictors.

## 3. RMSD Distribution
**Visual:** Histogram of RMSD
**Interpretation:**
- Most RMSD values are low, with a few high outliers.
- Predicting high RMSD values may be challenging due to their rarity.

## 4. Feature Importance (Random Forest)
**Visual:** Bar plot of top 20 feature importances
**Interpretation:**
- The most important features for predicting RMSD are among the original and polynomial features derived from F1, F2, and F5.
- High importance of interaction terms suggests non-linear relationships.

---

**Recommendations:**
- Consider scaling and dimensionality reduction due to high feature correlation.
- Outlier handling may improve model robustness.
- Focus on the most important features for model simplicity and performance.