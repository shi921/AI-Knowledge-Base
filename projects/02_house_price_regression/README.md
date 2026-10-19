**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# House Price Regression: End-to-end scikit-learn Pipeline

```bash
python projects/02_house_price_regression/main.py --model ridge --cv 3
# Alternative: --model forest
```

The pipeline uses synthetic housing data: train/holdout splitting, cross-validation, numeric/categorical preprocessing, metric reporting, retraining on all labeled rows, joblib model export, and prediction files. `SalePrice` is a simulated target and should not be interpreted as a real-world monetary estimate.

**Related notes:** [Chapter 03](../../notes/03_supervised_learning_evaluation.md), [Chapter 09](../../notes/09_regression_optimization.md).
