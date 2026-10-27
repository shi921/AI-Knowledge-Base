**语言：** [English](01_python_pandas.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 01｜Python、NumPy 与 Pandas 数据基础

> **🧭 知识脉络定位**：第一篇「数据科学基石」· **第 01 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：如果把训练 AI 比作开餐厅做菜，这一章就是**“采购食材、认识厨具与挑拣杂物”**。Python 是你的基本刀工，**NumPy** 是能同时切几千块食材的“高速矩阵切片机”，而 **Pandas** 则是带行列标签的“智能食材分拣台”。  
> - **🎯 为什么必须先学这一章？**：所有机器学习和深度学习模型本质上都只做一件事——**对数字矩阵做数学运算**。在接触任何算法前，你必须学会把杂乱的 CSV/Excel 表格读入内存，转成多维数组，查清缺失值、异常值与数据类型。  
> - **🔗 前后衔接**：**起点** → 本章（掌握 Python、NumPy 矩阵与 Pandas 表格清理）→ **下一章**：[02｜数据预处理、特征工程与 PCA](02_preprocessing_feature_engineering_pca.zh-CN.md)（把原始数据加工成尺度统一、无泄漏的模型输入特征）。

---

## 一、 Python 核心数据类型与基础操作

在数据科学中，Python 原生类型用于控制流、配置管理与文本/时间解析：

| 概念 | 示例 | 核心作用与考点 |
|---|---|---|
| 数值运算 | `7 / 2`, `7 // 2`, `7 % 2`, `2 ** 5` | `/` 生成浮点结果，`//` 向下整除，`**` 幂运算 |
| 字符串 | `'AI' + ' Service'`, `'abc'[1:]` | 文本清洗基础；支持索引、切片，具有不可变性 |
| 列表与元组 | `[1, 2]`, `(1, 2)` | `list` 可变（适合收集动态结果），`tuple` 不可变（适合表示张量形状如 `(32, 64)`） |
| 字典与集合 | `{'a': 1}`, `{1, 2}` | `dict` 用于键值映射（如词表 `token -> id` 或 JSON 配置），`set` 用于快速去重 |
| 日期与时间 | `datetime.now().strftime('%Y-%m')` | 时间序列特征提取（年、月、星期、时间差） |
| 函数与推导式 | `[x * 2 for x in data if x > 0]` | 将散乱的数据处理步骤封装为可复用的纯函数 |

```python
from datetime import datetime
records = [{'name': 'A', 'spend': 20}, {'name': 'B', 'spend': 32}]
mean_spend = sum(r['spend'] for r in records) / len(records)
print(datetime.now().year, mean_spend)
```

---

## 二、 NumPy 心智模型：AI 的数学计算底座

为什么有了 Python 的 `list`，还要用 **NumPy (`np.ndarray`)**？因为 Python 原生列表循环处理百万级数据时极慢，而 NumPy 在底层用连续内存和 C 语言实现了**向量化运算（Vectorization）**与**矩阵乘法**——这也是后续线性回归（第 03 章）和 PyTorch 张量（第 08 章）的直接基础。

1. **形状（Shape）约定**：在机器学习中，二维特征矩阵 `X` 的形状通常是 `(N, D)`，其中 `N` 是样本行数（多少条记录），`D` 是特征列数（每个样本有几个指标）。
2. **向量化与广播（Broadcasting）**：无需写 `for` 循环，直接对整个数组做加减乘除。
3. **矩阵乘法（`@`）**：线性模型预测 $\hat{y} = Xw$ 在代码中就是 `X @ w`。

```python
import numpy as np

# 3 个房屋样本，每个样本 2 个特征：[面积(百平米), 卧室数]
X = np.array([[1.2, 2.0],
              [1.5, 3.0],
              [2.0, 4.0]])          # shape: (3, 2)
w = np.array([50.0, 10.0])          # 权重参数 shape: (2,)
b = 20.0                            # 偏置项（基础底价）

# 向量化矩阵乘法 + 广播加法：一次算出全部 3 套房的预测价格
y_pred = X @ w + b                  # shape: (3,) -> [100., 125., 160.]
print('Shape:', X.shape, 'Predictions:', y_pred)
```

---

## 三、 Pandas 心智模型：带标签的智能表格

NumPy 擅长纯数字矩阵运算，但现实业务表格往往混合了文字、数字、日期和缺失值。**Pandas** 基于 NumPy 构建：
- `DataFrame` 是带行索引（Index）和列标签（Columns）的二维表；
- `Series` 是一维带索引对象（表格中的单列或单行）。

```python
import pandas as pd
from pathlib import Path

f = Path('datasets/housing/train.csv')   # 在仓库根目录运行
housing = pd.read_csv(f)
print(housing.shape, housing.dtypes.value_counts())
print(housing.loc[:, ['SalePrice', 'GrLivArea']].head())
print(housing.isna().mean().sort_values(ascending=False).head())
print(housing.groupby('Neighborhood')['SalePrice'].median().sort_values().tail())
```

- **高频必备 API**：`head`、`info`、`describe`、`isna`、`fillna`、`loc`（按标签索引）、`iloc`（按整数位置索引）、`groupby`、`agg`、`merge`、`sort_values`、`value_counts`、`to_csv`。
- **防坑要点**：修改切片子表前使用 `.copy()` 可避免链式赋值警告（`SettingWithCopyWarning`）；读 `.xls` 需安装 `xlrd`，读 CSV 用 `read_csv`。

---

## 四、 标准数据清理五步法（探索性数据分析 EDA）

拿到任何陌生数据集，先按这 5 步体检，绝不要直接丢进模型：

1. **理解字段语义**：区分唯一标识符（如 `ID` 列，对预测无用且会导致过拟合，必须剔除）、输入特征（`X`）、目标变量（`y`）以及**未来信息泄漏列**（如预测是否违约时混入了“催收罚款金额”）。
2. **检查并修正数据类型**：被存成字符串的数值需用 `pd.to_numeric(..., errors='coerce')` 转换；日期需用 `pd.to_datetime` 解析。
3. **诊断并处理缺失值**：偏态数值特征优先用**中位数（median）**填充（比均值更抗极端值干扰），分类变量用**众数（most_frequent）**或单独的 `'missing'` 类别填充。**铁律：绝不能在切分训练/测试集之前用全量数据计算填充统计量！**
4. **识别类别变量性质**：无序分类（如城市、颜色）用独热编码（One-Hot）；有明确等级顺序的类别才可用序数编码（Ordinal）；`LabelEncoder` 仅用于编码目标标签 `y`。
5. **审核异常值（Outliers）**：通过分位数、`describe()` 或箱线图排查录入错误或极端离群点。

---

## 五、 配套实践与自测

- **可运行项目**：
  - [`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py)：合成房价数据的缺失值报告、类型统计、缩放、PCA 与时间平滑。
  - [`projects/01_data_foundations/customer_pca.py`](../projects/01_data_foundations/customer_pca.py)：客户表的类别编码、标准化、相关性与 PCA 分析。
- **思考与自测**：
  1. 为什么在富豪与普通人混居的社区统计平均收入时，`median()`（中位数）比 `mean()`（均值）更能反映真实水平？
  2. 两个 Pandas `Series` 相加时，为什么结果可能和两个纯 NumPy 数组相加不同？（提示：思考索引对齐机制）
  3. 为什么 `df[['x']]` 返回的是二维 `DataFrame`，而 `df['x']` 返回的是一维 `Series`？
