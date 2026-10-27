**Language:** English | [简体中文](02_preprocessing_feature_engineering_pca.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 02 | Preprocessing, Feature Engineering, and PCA

> **🧭 Knowledge Map Position**: Part I (Data Foundations) · **Chapter 02 of 15**  
> - **💡 Intuitive Analogy**: This chapter is **washing, chopping, and condensing ingredients**. Raw features come in wildly different units (e.g., house area in thousands vs. bedrooms in single digits)—without normalization (**Scaling**), models treat large-unit columns as thousands of times more important. Text categories ("North/South") must become numeric switches (**One-Hot Encoding**), and hundreds of redundant columns can be distilled into key orthogonal directions via **PCA**.  
> - **🎯 Core Purpose & Golden Rule**: Turn raw tables into clean, uniform numeric matrices while enforcing the #1 rule of machine learning—**Prevent Data Leakage**: always `fit` scalers, imputers, and PCA **on the training split only**, and apply `transform` to validation and test splits!  
> - **🔗 Learning Chain**: [01 | Python, NumPy, and Pandas](01_python_pandas.md) → This Chapter (Feature engineering & leak-free pipelines) → **Next Chapter**: [03 | Linear Regression, Regularization, and Gradient Descent](03_regression_optimization.md) (See how machines mathematically learn weights from prepared features).

---

## 1. Why Raw Data Cannot Be Fed Directly to Models: Transformation Reference

| Transformation | Formula / Meaning | Typical Use Case | Critical Caveat |
|---|---|---|---|
| **MinMaxScaler** | $(x - \min) / (\max - \min)$ | Rescales features to `[0, 1]` (e.g., bounded pixel intensities) | Highly sensitive to outliers |
| **StandardScaler** | $(x - \mu) / \sigma$ | Zero mean, unit variance; **essential for linear models, GD, SVMs, and PCA** | $\mu$ and $\sigma$ must be computed **on the training set only** |
| **Binarizer** | `1` if $x > \text{threshold}$ else `0` | Binary rule features (e.g., "has basement") | Discards fine-grained magnitude information |
| **OneHotEncoder** | Expands $K$ categories into $K$ binary `0/1` columns | Unordered nominal categories (city, color) | Always set `handle_unknown='ignore'` for unseen test categories |
| **Rolling Mean** | Moving average over window $k$ | Smoothing noisy time series | **Use trailing windows for forecasting**; centered windows leak future values |
| **PCA** | Orthogonal projection onto directions of maximal variance | Dimensionality reduction, visualization, denoising | **Unsupervised** and scale-sensitive; largest variance $\neq$ most predictive for `y` |

---

## 2. Leak-Free Pipelines with `Pipeline` and `ColumnTransformer`

Fitting a scaler or imputer on the full dataset before `train_test_split` leaks test distribution statistics into training, inflating validation scores. Wrapping all preprocessing inside a scikit-learn `Pipeline` guarantees that `.fit()` only sees training folds:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

preprocess = ColumnTransformer([
    ('num', Pipeline([
        ('fill', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ]), ['GrLivArea', 'LotArea']),
    ('cat', Pipeline([
        ('fill', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), ['Neighborhood'])
])

model = Pipeline([
    ('preprocess', preprocess),
    ('ridge', Ridge(alpha=10))
])
# Fit only on X_train after train_test_split; test statistics never leak.
```

---

## 3. PCA Intuition & Eigenfaces

Imagine photographing a 3D teapot on a 2D camera sensor: to preserve the most visual detail, you angle the camera along the axes where the teapot varies the most. **PCA** finds orthogonal axes (principal components) that maximize projected variance:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
Z_train = pca.fit_transform(X_train_scaled)
Z_test = pca.transform(X_test_scaled)
print('Explained variance ratio:', pca.explained_variance_ratio_)
```

- **Explained Variance Ratio**: Fraction of total training variance captured by each component.
- **Eigenfaces**: High-dimensional face images (e.g., $64 \times 64 = 4096$ pixels) can be projected onto top principal components (fit on training faces only) before training an SVM classifier (see `projects/03_classification_and_faces/faces.py`).

---

## 4. Time-Series Smoothing & Common Pitfalls

1. **Trailing Rolling Mean**: `series.rolling(k).mean()` uses the current and past $k-1$ observations—safe for online feature extraction.
2. **Centered Rolling Mean**: `series.rolling(k, center=True).mean()` uses future observations—useful for retrospective charts, **forbidden in predictive features**.
3. **Top 4 Pitfalls**:
   - ❌ Fitting `StandardScaler`, `PCA`, or `SimpleImputer` before splitting train and test sets.
   - ❌ Encoding unordered input categories with `LabelEncoder` (inducing fake numerical ordering like `Tokyo(2) > Paris(1)`).
   - ❌ Running PCA without standardizing features of different units first.
   - ❌ Randomly shuffling time-series observations during cross-validation.

---

## 5. Hands-on Projects & Self-Test

- **Runnable Projects**: [`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py) and [`customer_pca.py`](../projects/01_data_foundations/customer_pca.py).
- **Self-Test**:
  1. Why must we call `scaler.transform(X_test)` instead of `scaler.fit_transform(X_test)`?
  2. What happens to PCA if one column is measured in millimeters instead of meters and we skip `StandardScaler`?
  3. Why doesn't the direction of maximum variance always separate classification labels best?
