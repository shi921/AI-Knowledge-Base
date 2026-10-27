**Language:** English | [简体中文](01_python_pandas.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 01 | Python, NumPy, and Pandas Data Foundations

> **🧭 Knowledge Map Position**: Part I (Data Foundations) · **Chapter 01 of 15**  
> - **💡 Intuitive Analogy**: If building an AI system is like running a gourmet kitchen, this chapter is **sourcing ingredients, mastering your knives, and sorting raw produce**. Python provides general control flow, **NumPy** is a high-speed multi-blade slicer that processes thousands of numbers at once, and **Pandas** is your labeled prep table for inspecting and cleaning tabular records.  
> - **🎯 Why Must We Start Here?**: Every machine learning and deep learning model ultimately operates on **numeric matrices**. Before touching any algorithm, you must know how to load raw CSV/Excel tables into memory, represent them as arrays, and diagnose missing values, outliers, and data types.  
> - **🔗 Learning Chain**: **Start Here** → This Chapter (Python, NumPy matrices & Pandas data cleaning) → **Next Chapter**: [02 | Preprocessing, Feature Engineering, and PCA](02_preprocessing_feature_engineering_pca.md) (Transform raw columns into scaled, leak-free numeric features).

---

## 1. Core Python Types & Operations

In data workflows, native Python types handle configuration, string parsing, date manipulation, and control flow:

| Concept | Example | Role in AI Workflows |
|---|---|---|
| Numeric ops | `7 / 2`, `7 // 2`, `7 % 2`, `2 ** 5` | `/` float division, `//` floor division, `**` exponentiation |
| Strings | `'AI' + ' Service'`, `'abc'[1:]` | Text cleaning; immutable sequences supporting slicing |
| Lists & Tuples | `[1, 2]`, `(1, 2)` | `list` is mutable; `tuple` is immutable (commonly used for tensor shapes like `(32, 64)`) |
| Dicts & Sets | `{'a': 1}`, `{1, 2}` | `dict` maps keys to values (e.g., vocabulary `token -> id`); `set` removes duplicates in $O(1)$ |
| Datetimes | `datetime.now().strftime('%Y-%m')` | Extracting temporal features (year, month, day-of-week, lags) |
| Comprehensions | `[x * 2 for x in data if x > 0]` | Concise, readable data transformations |

```python
from datetime import datetime
records = [{'name': 'A', 'spend': 20}, {'name': 'B', 'spend': 32}]
mean_spend = sum(r['spend'] for r in records) / len(records)
print(datetime.now().year, mean_spend)
```

---

## 2. NumPy Mental Model: The Mathematical Engine of AI

Why do we need **NumPy (`np.ndarray`)** when Python already has `list`? Native Python loops over millions of numbers are slow. NumPy stores elements in contiguous C memory and executes **vectorized operations** and **matrix multiplications** in compiled code—forming the direct foundation for linear regression (Chapter 03) and PyTorch tensors (Chapter 08).

1. **Shape Convention**: A 2D feature matrix `X` has shape `(N, D)`, where `N` is the number of samples (rows) and `D` is the number of features (columns).
2. **Vectorization & Broadcasting**: Arithmetic operations apply element-wise across arrays without explicit `for` loops.
3. **Matrix Multiplication (`@`)**: The linear prediction formula $\hat{y} = Xw + b$ is written directly as `X @ w + b`.

```python
import numpy as np

# 3 houses, 2 features each: [area_in_100sqm, bedrooms]
X = np.array([[1.2, 2.0],
              [1.5, 3.0],
              [2.0, 4.0]])          # shape: (3, 2)
w = np.array([50.0, 10.0])          # weights shape: (2,)
b = 20.0                            # scalar bias

# Vectorized matrix product + broadcast addition
y_pred = X @ w + b                  # shape: (3,) -> [100., 125., 160.]
print('Shape:', X.shape, 'Predictions:', y_pred)
```

---

## 3. Pandas Mental Model: Labeled Tables

While NumPy requires homogeneous numeric arrays, real-world tables mix strings, numbers, dates, and missing values. **Pandas** builds on NumPy:
- `DataFrame` is a 2D table with row indices and column labels;
- `Series` is a 1D indexed column or row.

```python
import pandas as pd
from pathlib import Path

f = Path('datasets/housing/train.csv')   # Run from repository root
housing = pd.read_csv(f)
print(housing.shape, housing.dtypes.value_counts())
print(housing.loc[:, ['SalePrice', 'GrLivArea']].head())
print(housing.isna().mean().sort_values(ascending=False).head())
print(housing.groupby('Neighborhood')['SalePrice'].median().sort_values().tail())
```

- **Essential API**: `head`, `info`, `describe`, `isna`, `fillna`, `loc` (label-based), `iloc` (position-based), `groupby`, `agg`, `merge`, `sort_values`, `value_counts`, `to_csv`.
- **Key Pitfalls**: Call `.copy()` when modifying a sliced subset to avoid `SettingWithCopyWarning`; `.xls` files require `xlrd` while CSVs use `read_csv`.

---

## 4. Five-Step Data Cleaning Strategy (EDA)

1. **Understand Column Semantics**: Separate row IDs (drop before modeling), input features (`X`), target labels (`y`), and **target-leakage columns** (information that wouldn't exist at prediction time).
2. **Verify Data Types**: Parse numeric strings with `pd.to_numeric(..., errors='coerce')` and dates with `pd.to_datetime`.
3. **Handle Missing Values**: Use the **median** for skewed numeric columns and the **mode** (or a `'missing'` category) for categorical columns. **Never fit imputation statistics on the full dataset before splitting train and test sets!**
4. **Encode Categories Properly**: Use One-Hot encoding for unordered categories; reserve Ordinal encoding for genuinely ordered levels; use `LabelEncoder` only for target labels `y`.
5. **Audit Outliers**: Inspect quantiles and boxplots alongside domain knowledge before winsorizing or filtering.

---

## 5. Hands-on Projects & Self-Test

- **Runnable Projects**:
  - [`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py): Missing-value report, type counts, scaling, PCA, and rolling smoothing on synthetic housing data.
  - [`projects/01_data_foundations/customer_pca.py`](../projects/01_data_foundations/customer_pca.py): Categorical encoding, scaling, correlations, and PCA on customer records.
- **Self-Test**:
  1. Which is more sensitive to extreme outliers: `mean()` or `median()`?
  2. Why can adding two Pandas `Series` behave differently from adding two NumPy arrays? (Hint: index alignment)
  3. Why does `df[['x']]` return a 2D `DataFrame` while `df['x']` returns a 1D `Series`?
