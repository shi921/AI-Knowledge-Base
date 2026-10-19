**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# 数据探索、缺失值、PCA、平滑

```bash
python projects/01_data_foundations/main.py --limit 180
```

默认读取 `datasets/housing/train.csv`（由 `scripts/generate_demo_data.py` 生成的合成数据），生成缺失值分析、标准化、PCA 和时序均值平滑结果。

可选脚本 `customer_pca.py` 需要自行提供可信 Excel 客户数据；公共仓库不包含此类数据：

```bash
python projects/01_data_foundations/customer_pca.py --data /path/to/authorized/customers.xls
```

**相关知识笔记：** [第 01 章](../../notes/01_python_pandas.zh-CN.md)、[第 02 章](../../notes/02_preprocessing_feature_engineering_pca.zh-CN.md)。
