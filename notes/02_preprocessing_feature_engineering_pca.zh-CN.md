**语言：** [English](02_preprocessing_feature_engineering_pca.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 02｜数据预处理、特征工程与 PCA

## 变换方法：什么时候使用？

| 变换 | 数学/含义 | 典型应用 | 注意 |
|---|---|---|---|
| MinMaxScaler | `(x - min) / (max - min)` | 特征映射到 `[0,1]` | 对极值敏感 |
| StandardScaler | `(x - mean) / std` | 线性模型、SVM、PCA | 学到的均值方差只能用训练集 |
| Binarizer | 大于阈值置 1，否则 0 | 二元规则特征 | 造成信息损失 |
| OneHotEncoder | 类别变为独热列 | 非有序类别 | 需 `handle_unknown='ignore'` |
| Rolling mean | 滑动窗口平均 | 时间序列平滑 | 居中平滑使用未来值，不适合实时预测 |
| PCA | 主成分正交投影 | 降维、可视化、噪声过滤 | 无监督、对量纲敏感、未必利于分类 |

## 连贯的预处理代码

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
# Fit the pipeline on training data after train_test_split; no test statistics are used.
```

## PCA 重点

令经过中心化、必要时标准化的数据为 `X`，PCA 找出使投影方差最大的正交方向。核心接口：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X_scaled = StandardScaler().fit_transform(X_train)
pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
```

**解释方差比率**表示各主成分占训练集总方差的比例；主成分与原始列之间的相关性可以用于解释结构，但一个主成分不是一个原始变量。PCA 投影的正负号可整体翻转，不影响表示的信息。

**Eigenfaces**：面部图像视为高维像素向量，用 PCA 得到低维基，再以 SVM 分类；一定要在训练集拟合 PCA，不应在全部样本上先降维后评估。

## 时间平滑

尾随平均使用当前及过去 `k-1` 个点：`series.rolling(k).mean()`；居中平均使用两侧观测：`series.rolling(k,center=True).mean()`。前者适合在线分析，后者适合回顾性趋势展示。完整项目采用**合成房价的月度成交序列**演示时间平滑。

## 常见考试误区

- **数据泄漏**：划分前整体拟合 `StandardScaler`/PCA/`SimpleImputer` 或特征选择，导致评估偏乐观。
- `LabelEncoder` 编码非有序输入类别会诱导虚假的大小关系。
- PCA 擅长保留方差，但大方差不等于预测目标最相关。
- 采样的时间数据不能随意随机打乱作 CV；时间序列宜按时间顺序切分。

**完整项目**：[`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py) 和 `customer_pca.py`。**扩展实践**：修改 PCA 维数为 3，比较 explained variance 和可视化结构。
