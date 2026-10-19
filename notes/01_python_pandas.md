**Language:** English | [简体中文](01_python_pandas.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 01 | Python and Pandas Fundamentals

## Learning goals

Move from numbers, strings, datetimes, control flow, functions, and collections to NumPy arrays and Pandas tables. Be able to load data, profile quality, and compute grouped statistics as a reusable workflow.

## Python types and operations

| Concept | Example | Key point |
|---|---|---|
| Numbers | `7 / 2`, `7 // 2`, `7 % 2`, `2 ** 5` | Floating division, floor division, remainder, exponentiation |
| Strings | `'AI' + ' Service'`, `'abc'[1:]` | Indexing, slicing, immutability |
| Sequences | `[1,2]`, `(1,2)` | Lists are mutable; tuples are immutable |
| Dictionaries and sets | `{'a': 1}`, `{1,2}` | Key-value lookup and uniqueness |
| Datetime | `datetime.now().strftime('%Y-%m')` | Formatting and parsing |
| Functions | `def f(x): return x*2` | Arguments, returns, scope |

```python
from datetime import datetime
records = [{'name': 'A', 'spend': 20}, {'name': 'B', 'spend': 32}]
mean_spend = sum(r['spend'] for r in records) / len(records)
print(datetime.now().year, mean_spend)
```

## The Pandas mental model

A `DataFrame` is a two-dimensional table with row indices and labeled columns; a `Series` is a one-dimensional indexed object.

```python
import pandas as pd
from pathlib import Path
f = Path('datasets/housing/train.csv')  # Run from the repository root
housing = pd.read_csv(f)
print(housing.shape, housing.dtypes.value_counts())
print(housing.loc[:, ['SalePrice', 'GrLivArea']].head())
print(housing.isna().mean().sort_values(ascending=False).head())
print(housing.groupby('Neighborhood')['SalePrice'].median().sort_values().tail())
```

Useful methods: `head`, `info`, `describe`, `isna`, `fillna`, `loc`, `iloc`, `groupby`, `agg`, `merge`, `sort_values`, `value_counts`, `to_csv`. `loc` uses labels while `iloc` uses integer positions. `.copy()` can prevent chained-assignment problems. Reading legacy `.xls` files may require `xlrd`.

## Data cleaning strategy

1. **Understand semantics:** identify IDs, predictive features, targets, leakage columns, and missingness mechanisms.
2. **Check types:** use `pd.to_numeric(..., errors='coerce')` for numeric strings and `pd.to_datetime` for dates.
3. **Handle missing data:** median can suit skewed numeric features; mode can suit categorical features. Never fit imputation on the entire dataset before splitting.
4. **Encode categories:** one-hot for unordered groups; ordinal encoding only for genuinely ordered groups. `LabelEncoder` generally belongs to target labels.
5. **Inspect outliers:** use box plots, quantiles, and domain knowledge.

## End-to-end practice

- [`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py): synthetic housing data, missing-value reporting, types, scaling, PCA, and smoothing.
- [`projects/01_data_foundations/customer_pca.py`](../projects/01_data_foundations/customer_pca.py): optional authorized customer spreadsheet with category encoding, standardization, correlation, and PCA.

**Review questions:** Is mean or median more sensitive to outliers? Why do Pandas alignment and NumPy positional operations sometimes differ? Why is `df[['x']]` different from `df['x']`?

**Next step:** Connect file loading, validation, filtering, aggregation, and exports in one workflow, then progress to feature engineering and PCA.
