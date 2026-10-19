**Language:** English | [简体中文](03_supervised_learning_evaluation.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 03 | Supervised Learning: Regression, Classification, and Model Evaluation

## Unified workflow

Define the problem → explore and clean → split train/validation/test → fit preprocessing **inside training folds** → select models and hyperparameters via cross-validation → evaluate test data once → retrain on all labeled training data → export predictions.

## Regression

- Simple linear regression: `y = b0 + b1*x`; multiple regression: `y = b0 + Σ(bj*xj)`. Ridge introduces L2 penalties to stabilize coefficients.
- Random forest regression ensembles decision trees to capture nonlinear relationships, but extrapolation and missing values still require care.
- `MAE = mean(|y-ŷ|)` is interpretable and relatively robust; `RMSE = sqrt(mean((y-ŷ)^2))` penalizes extreme errors more heavily; `R² = 1-SSE/SST` can be negative.
- Skewed targets may use `log1p(y)` and convert predictions back using `expm1`; report explicitly whether each metric is measured in log units or original units.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
# X, y, and pipeline are defined in the corresponding project.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
pipeline.fit(X_train, y_train)
print('CV MAE:', -scores.mean())
print('Holdout MAE:', mean_absolute_error(y_test, pipeline.predict(X_test)))
```

## Classification

- Common classifiers: decision trees, random forests, logistic regression, and SVM. Practice datasets include Digits, fraud data (when authorized), and optional face features.
- Confusion matrix: TP/TN/FP/FN. `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`, and `F1 = 2PR/(P+R)`.
- Macro-F1 averages each class's F1 equally; weighted-F1 weights by class support. Accuracy can be misleading under class imbalance.
- For rare positive cases such as fraud: use stratified splitting, PR-AUC, recall, and false-alarm cost, not just accuracy.

## Cross-validation and generalization

- **Underfitting:** both training and validation scores are poor; revisit features and model capacity.
- **Overfitting:** training score is good, validation score is poor; try regularization, more examples, or a simpler model.
- `KFold`: general regression; `StratifiedKFold`: classification proportions; `TimeSeriesSplit`: chronological data.
- Keep grid search inside the training split. Repeated test-based hyperparameter choices turn the test set into another validation set.

## Complete experiments

1. Compare classifiers and metrics on built-in Digits data; optionally extend to downloaded MNIST with appropriate permissions.
2. Run housing regression on CSV features, cross-validate, evaluate, and export predictions.
3. Record splits, random seeds, and metrics for reproducibility.

**Projects:** [`projects/02_house_price_regression/main.py`](../projects/02_house_price_regression/main.py), [`projects/03_classification_and_faces/main.py`](../projects/03_classification_and_faces/main.py), and optional `faces.py` for authorized face-feature PCA + SVM. The numerical foundation for least squares, polynomial features, and Ridge is in [chapter 09](09_regression_optimization.md).

**Review questions:** Why must cross-validation avoid the test set? Why doesn't high precision imply high recall? When should you prefer MAE to RMSE? Remember: performance on small synthetic datasets is not evidence of real-world generalization.
