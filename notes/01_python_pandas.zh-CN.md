**语言：** [English](01_python_pandas.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 01｜Python 与 Pandas 基础

> **🧭 知识脉络定位**：第一阶段「数据基石篇」· **第 1 步 / 共 14 步**  
> - **💡 零基础直觉比喻**：如果把训练 AI 比作开餐厅做菜，这一章就是**“采购食材与挑拣杂物”**。原始数据就像刚从菜市场买回来带泥的菜，有缺斤少两（缺失值）、烂叶子（异常值）和贴错的标签（类型错误）。  
> - **🎯 为什么必须学这一章？**：任何高大上的算法都不能直接吃杂乱无章的原始文件。你必须先学会用 Python 和 Pandas 把表格读进内存，看清有多少行、多少列、哪些地方是空的、平均值和中位数是多少。  
> - **🔗 前后衔接**：**起点** → 本章（把数据读进来、看清楚）→ **下一站**：[02｜预处理、特征工程与 PCA](02_preprocessing_feature_engineering_pca.zh-CN.md)（把数据洗干净、切成模型能直接吃的标准块）。

## 学习目标

从数值/字符串/日期时间、控制流、函数、数据结构，过渡到 NumPy 与 Pandas 的表格运算；最终能够独立读取数据、检查质量并进行聚合分析。将单个语言特性组织为可复用的数据处理步骤。

## Python 类型与基本操作

| 概念 | 示例 | 考点 |
|---|---|---|
| 数值 | `7 / 2`, `7 // 2`, `7 % 2`, `2 ** 5` | `/` 生成浮点结果，`//` 向下整除，`**` 幂运算 |
| 字符串 | `'AI' + ' Service'`, `'abc'[1:]` | 索引、切片、不可变 |
| 序列 | `[1,2]`, `(1,2)` | list 可变，tuple 不可变 |
| 字典、集合 | `{'a': 1}`, `{1,2}` | 键值访问、去重 |
| 时间 | `datetime.now().strftime('%Y-%m')` | 时间格式与解析 |
| 函数 | `def f(x): return x*2` | 参数、返回值、作用域 |

```python
from datetime import datetime
records = [{'name': 'A', 'spend': 20}, {'name': 'B', 'spend': 32}]
mean_spend = sum(r['spend'] for r in records) / len(records)
print(datetime.now().year, mean_spend)
```

## Pandas 心智模型

`DataFrame` 是带行索引和列标签的二维表，`Series` 是一维带索引对象。

```python
import pandas as pd
from pathlib import Path
f = Path('datasets/housing/train.csv')   # Run from the repository root
housing = pd.read_csv(f)
print(housing.shape, housing.dtypes.value_counts())
print(housing.loc[:, ['SalePrice', 'GrLivArea']].head())
print(housing.isna().mean().sort_values(ascending=False).head())
print(housing.groupby('Neighborhood')['SalePrice'].median().sort_values().tail())
```

常用 API：`head`, `info`, `describe`, `isna`, `fillna`, `loc`, `iloc`, `groupby`, `agg`, `merge`, `sort_values`, `value_counts`, `to_csv`。`loc` 使用标签，`iloc` 使用位置；`.copy()` 能避免链式赋值风险。读 `.xls` 需要 `xlrd`，CSV 用 `read_csv`。

## 数据清理策略

1. **了解语义**：区分 ID、特征、目标变量、泄漏列；识别缺失的机制。
2. **检查类型**：数字字符串需要 `pd.to_numeric(...,errors='coerce')`，日期需 `pd.to_datetime`。
3. **处理缺失值**：中位数适合偏态数值，众数可处理分类变量；不能在完整数据上拟合插补再划分训练/测试。
4. **处理类别**：无序分类常用 OneHot；有序类别才可使用 Ordinal 编码；`LabelEncoder` 通常用于目标标签。
5. **审核异常值**：箱线图、分位数检查、结合业务判断。

## 可复习完整项目

- [`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py)：合成房价数据的缺失值报告、类型统计、缩放、PCA 与平滑。
- [`projects/01_data_foundations/customer_pca.py`](../projects/01_data_foundations/customer_pca.py)：可选客户表的类别编码、标准化、相关性与 PCA。

**自测**：`mean()` 与 `median()` 对异常值哪个更敏感？对齐索引时为什么 Series 与纯 NumPy 数组可能不同？为什么 `df[['x']]` 与 `df['x']` 类型不同？


## 基础知识衔接

将数据读取、字段检查、筛选、统计和输出串成一次完整的数据流程；可先运行 [`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py)，再学习特征工程与 PCA。
