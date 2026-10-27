**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# 房价回归：完整 sklearn Pipeline

```bash
python projects/02_house_price_regression/main.py --model ridge --cv 3
# Alternative: --model forest
```

使用合成房价表格做 train/holdout 切分、交叉验证、数值/类别预处理、指标评估、全量重训、joblib 保存、预测导出。`SalePrice` 是模拟售价，其误差单位仅用于演示，不代表真实业务价格。

**相关知识笔记：** [第 03 章](../../notes/03_regression_optimization.zh-CN.md)、[第 04 章](../../notes/04_supervised_learning_evaluation.zh-CN.md)。
