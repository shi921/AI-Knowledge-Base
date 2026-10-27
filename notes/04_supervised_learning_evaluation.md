**Language:** English | [简体中文](04_supervised_learning_evaluation.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 04 | Supervised Learning: Regression, Classification, and Model Evaluation

> **🧭 Knowledge Map Position**: Part II (Classical ML & Math Engine) · **Chapter 04 of 15**  
> - **💡 Intuitive Analogy**: Chapter 03 taught you how a single linear ruler learns weights. Here we expand our toolkit to **the full supervised learning family (Decision Trees, Random Forests, Logistic Regression, SVMs)** and establish an honest **exam protocol**: the **Training set** is daily homework, **Cross-Validation (CV)** is a series of mock exams used to select models and hyperparameters, and the **Test set** is the final exam—taken only once at the very end without peeking!  
> - **🎯 Why Does This Matter?**:
>   1. **Two Core Tasks & Algorithm Selection**: Predict continuous quantities with **Regression** (house prices) and discrete categories with **Classification** (digits, fraud, faces).
>   2. **Honest Evaluation Metrics**: Understand why plain Accuracy is deceptive on imbalanced datasets (e.g., 99% normal, 1% fraud) and how Precision, Recall, F1, and PR-AUC measure real utility.  
> - **🔗 Learning Chain**: [03 | Linear Regression, Regularization, and Gradient Descent](03_regression_optimization.md) → This Chapter (Supervised algorithms & evaluation protocols) → **Next Chapter**: [05 | Explainable AI with SHAP](05_explainable_ai_shap.md) (Once a model is accurate, how do we look inside the black box to explain *why* it made each decision?).

---

## 1. Unified Leak-Free Supervised Workflow

Every rigorous supervised learning experiment follows a 7-step loop:
`Define the task (Regression vs. Classification) → Explore & clean (Chapter 01) → Split train/test → Fit preprocessing inside training folds only (Chapter 02) → Compare models & hyperparameters via Cross-Validation → Retrain the winning pipeline on all labeled training data → Evaluate on the held-out test set exactly once`.

### Core Supervised Algorithms Compared

| Algorithm | Tasks | Core Intuition | Strengths & Limitations |
|---|---|---|---|
| **Linear / Ridge Regression** | Regression | Weighted linear sum of features (Chapter 03) | Fast, interpretable baseline; cannot capture complex non-linearities without engineered features |
| **Logistic Regression** | Classification | Passes linear score $Xw$ through Sigmoid/Softmax to output class probabilities | Fast, well-calibrated linear classifier baseline |
| **Decision Tree** | Both | Splits data recursively via hierarchical `if-else` feature thresholds | Handles non-linear interactions without scaling; a single unconstrained tree easily overfits |
| **Random Forest** | Both | Ensemble of many decorrelated trees trained on bootstrap samples and random feature subsets | Workhorse for tabular data, robust against overfitting; black-box ensemble (pair with Chapter 05 SHAP) and cannot extrapolate beyond training ranges |
| **Support Vector Machine (SVM)** | Both | Finds the maximum-margin boundary separating classes (optionally in a kernel space) | Strong on medium-sized, high-dimensional datasets (e.g., PCA Eigenfaces); requires feature scaling |

---

## 2. Regression Tasks & Metrics

- **MAE (Mean Absolute Error)** $= \frac{1}{n}\sum |y - \hat{y}|$: Expressed in the target's natural units, easy to explain, and robust to extreme outliers.
- **RMSE (Root Mean Squared Error)** $= \sqrt{\frac{1}{n}\sum (y - \hat{y})^2}$: Squares errors before averaging, **heavily penalizing large outliers**.
- **$R^2$ (Coefficient of Determination)** $= 1 - \text{SSE}/\text{SST}$: Fraction of variance explained compared to predicting the mean ($1.0$ is perfect; can be negative on test data). **Never report regression $R^2$ as "Accuracy"!**
- **Skewed Targets**: Right-skewed targets like house prices are often transformed with `np.log1p(y)` during training and inverted via `np.expm1(pred)` at inference; always specify whether metrics are reported in log-space or original dollar units.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
pipeline.fit(X_train, y_train)
print('5-Fold CV MAE:', -scores.mean())
print('Holdout Test MAE:', mean_absolute_error(y_test, pipeline.predict(X_test)))
```

---

## 3. Classification Metrics & Imbalanced Data Pitfalls

From the **Confusion Matrix** (`TP`, `FP`, `TN`, `FN`):

| Metric | Formula | Practical Meaning | When to Use & Caveats |
|---|---|---|---|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | Overall fraction of correct predictions | **Misleading under class imbalance**: if 99% of transactions are legitimate, an "always legitimate" dummy model gets 99% accuracy while catching 0% of fraud! |
| **Precision** | $\frac{TP}{TP+FP}$ | *"Of the items flagged positive, how many are truly positive?"* | Prioritize when **False Positives (FP)** are costly (e.g., spam filter blocking important emails) |
| **Recall** | $\frac{TP}{TP+FN}$ | *"Of all true positives in the data, how many did we catch?"* | Prioritize when **False Negatives (FN)** are costly (e.g., cancer screening or fraud detection) |
| **F1-Score** | $\frac{2PR}{P+R}$ | Harmonic mean of Precision and Recall | High only when both $P$ and $R$ are high; use **Macro-F1** to weight all classes equally or **Weighted-F1** to weight by class support |
| **PR-AUC** | Area under Precision-Recall curve | Threshold-independent summary of positive-class performance | Preferred metric for rare-event classification |

---

## 4. Cross-Validation Strategies

- **`KFold`**: Standard choice for i.i.d. regression tasks.
- **`StratifiedKFold`**: Preserves class ratios in every fold—default choice for classification.
- **`TimeSeriesSplit`**: Evaluates strictly on future horizons—mandatory for temporal data.
- **Hyperparameter Tuning Discipline**: Run `GridSearchCV` inside the training split only. Tuning parameters by repeatedly checking test-set performance turns the test set into a validation set.

---

## 5. Hands-on Projects & Self-Test

- **Runnable Projects**:
  - [`projects/02_house_price_regression/main.py`](../projects/02_house_price_regression/main.py): End-to-end house price regression with CV and holdout evaluation.
  - [`projects/03_classification_and_faces/main.py`](../projects/03_classification_and_faces/main.py): Digits classification comparing multiple classifiers and metrics.
  - [`projects/03_classification_and_faces/faces.py`](../projects/03_classification_and_faces/faces.py): Eigenfaces PCA + SVM pipeline.
- **Self-Test**:
  1. Why must hyperparameter selection never use the held-out test set?
  2. Why doesn't high Precision guarantee high Recall?
  3. How do extreme outliers affect your choice between MAE and RMSE?
