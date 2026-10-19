**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# Data Profiling, Missing Values, PCA, and Smoothing

```bash
python projects/01_data_foundations/main.py --limit 180
```

Uses `datasets/housing/train.csv` (synthetic data from `scripts/generate_demo_data.py`) to create missing-value summaries, standardized numerical features, PCA projections, and moving averages.

The optional `customer_pca.py` expects an authorized local customer Excel file; none is distributed here:

```bash
python projects/01_data_foundations/customer_pca.py --data /path/to/authorized/customers.xls
```

**Related notes:** [Chapter 01](../../notes/01_python_pandas.md), [Chapter 02](../../notes/02_preprocessing_feature_engineering_pca.md).
