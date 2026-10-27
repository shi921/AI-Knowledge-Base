**Language:** English | [简体中文](02_preprocessing_feature_engineering_pca.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 02 | Preprocessing, Feature Engineering, and PCA

> **🧭 Knowledge Map Position**: Stage 1 (Data Foundations) · **Step 2 of 14**  
> - **💡 Intuitive Analogy**: This chapter is **washing, chopping, and condensing ingredients**. Features come in wildly different units (e.g., house area in thousands vs. bedrooms in single digits) and must be normalized (**Scaling**); text categories ("North/South") must become numeric switches (**One-Hot Encoding**); and hundreds of redundant columns can be distilled into key directions via **PCA**.  
> - **🎯 Why Does This Matter?**: It turns raw tables into clean numeric matrices ready for algorithms while guarding against the #1 beginner mistake: **Data Leakage** (fitting scalers or imputers before splitting train and test sets).  
> - **🔗 Learning Chain**: [01 | Python & Pandas](01_python_pandas.md) → This Chapter (Clean & transform features) → **Next Step**: [09 | Gradient Descent & Optimization](09_regression_optimization.md) (Understand how machines mathematically learn weights from features).

## Which transformation should you use?

| Transformation | Meaning | Typical use | Caveat |
|---|---|---|---|
| MinMaxScaler | `(x - min) / (max - min)` | Map features to `[0, 1]` | Sensitive to extremes |
| StandardScaler | `(x - mean) / std` | Linear models, SVM, PCA | Learn statistics only from training data |
| Binarizer | Assign 1 above threshold, 0 otherwise | Binary rules | Loses information |
| OneHotEncoder | Create one indicator per category | Unordered categories | Set `handle_unknown='ignore'` |
| Rolling mean | Moving-window average | Temporal smoothing | Centered smoothing uses future observations |
| PCA | Orthogonal projections onto principal axes | Compression, visualization, denoising | Unsupervised; scale-sensitive |

## Preprocessing without leakage

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

preprocess = ColumnTransformer([
    ('num', Pipeline([('fill', SimpleImputer(strategy='median')),
                      ('scale', StandardScaler())]), ['GrLivArea', 'LotArea']),
    ('cat', Pipeline([('fill', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['Neighborhood'])
])
model = Pipeline([('preprocess', preprocess), ('ridge', Ridge(alpha=10))])
# After train_test_split, fit the whole pipeline on training data only.
```

## Core PCA ideas

For centered (and optionally standardized) matrix `X`, PCA finds orthogonal directions of maximum projected variance.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X_scaled = StandardScaler().fit_transform(X_train)
pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
```

`explained_variance_ratio_` gives the share of training variance described by each component. A principal component combines original columns; it is not itself one input feature. A component's sign may flip without changing its information content.

**Eigenfaces:** treat images as high-dimensional pixel vectors, reduce dimensionality with PCA, and classify with SVM. Fit PCA **only on training folds**, never on all images before evaluating.

## Temporal smoothing

`series.rolling(k).mean()` uses the current and previous `k-1` observations; `series.rolling(k, center=True).mean()` uses observations on both sides. Trailing averages can support online decisions; centered averages are retrospective. The project demonstrates smoothing using monthly synthetic housing sale prices.

## Common pitfalls

- Fitting `StandardScaler`, PCA, imputation, or feature selection before the train/test split leaks information.
- Encoding unordered categories using `LabelEncoder` imposes artificial magnitude relationships.
- PCA preserves variance, not necessarily information most predictive of the target.
- Randomized cross-validation may violate chronological ordering for time series; use time-aware splits.

**Project:** [`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py) and `customer_pca.py`. **Exercise:** use three PCA components and compare explained variance and plotting options.
