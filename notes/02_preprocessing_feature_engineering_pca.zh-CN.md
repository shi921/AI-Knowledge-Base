**语言：** [English](02_preprocessing_feature_engineering_pca.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 02｜数据预处理、特征工程与 PCA

> **🧭 知识脉络定位**：第一篇「数据科学基石」· **第 02 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：这一章是厨房里的**“净菜加工与浓缩提取”**。有的食材太大块（比如房屋面积动辄上千平米，而卧室数量只有两三个，量纲差几百倍），直接丢进锅里会让算法误以为“面积比卧室重要一千倍”，因此需要切成均匀大小（**标准化缩放**）；文字标签（如“朝阳区/海淀区”）机器算不了乘法，要转成数字开关（**独热编码 One-Hot**）；如果配料有几百种且高度重复，还要用 **PCA（主成分分析）** 提炼出信息量最大的几味“浓缩高汤”。  
> - **🎯 核心作用与第一铁律**：将原始表格转化为尺度一致、无缺失的数值矩阵；同时死守机器学习第一铁律——**严防数据泄漏（Data Leakage）**：所有预处理器的统计量（均值、方差、主成分方向）**只能在训练集上 `fit`**，测试集只能被动调用 `transform`！  
> - **🔗 前后衔接**：[01｜Python、NumPy 与 Pandas 基础](01_python_pandas.zh-CN.md) → 本章（特征工程与防泄漏 Pipeline）→ **下一章**：[03｜线性回归、正则化与梯度下降优化](03_regression_optimization.zh-CN.md)（看看机器拿到加工好的特征矩阵后，究竟是如何通过数学优化学会规律的）。

---

## 一、 为什么原始数据不能直接喂给模型？常用特征变换速查

| 变换方法 | 数学定义 / 含义 | 为什么需要它？（典型应用场景） | 核心防坑注意事项 |
|---|---|---|---|
| **MinMaxScaler** | $(x - \min) / (\max - \min)$ | 将特征线性压缩到 `[0, 1]` 区间（如图像像素归一化） | 对极端异常值非常敏感（一个极大值会把其余所有点挤到 0 附近） |
| **StandardScaler** | $(x - \mu) / \sigma$ | 让特征均值为 0、方差为 1；**线性模型、梯度下降、SVM、PCA 必备** | 均值 $\mu$ 和标准差 $\sigma$ **只能从训练集计算** |
| **Binarizer** | 大于阈值置 `1`，否则置 `0` | 提取二元业务规则特征（如“是否有地下室”） | 会丢弃原始连续数值的细粒度大小信息 |
| **OneHotEncoder** | 将 $K$ 个类别拆成 $K$ 列 `0/1` 开关 | 处理无内在顺序的类别变量（如区域、颜色） | 必须设置 `handle_unknown='ignore'`，防止测试集出现新类别时报错 |
| **Rolling Mean** | 过去 $k$ 个时间点的滑动平均 | 滤除时间序列的高频随机噪声，提取趋势 | **实时预测只能用历史尾随窗口**；居中窗口（`center=True`）偷看了未来数据 |
| **PCA（主成分分析）** | 寻找投影方差最大的正交方向 | 高维特征降维压缩、去除多重共线性、二维/三维可视化 | 属于**无监督降维**，必须先做 `StandardScaler`；方差最大方向不等于对分类最有用 |

---

## 二、 工业级防泄漏实践：用 `Pipeline` 与 `ColumnTransformer` 串联流程

新手最容易犯的错误是：先对整个数据集做 `StandardScaler().fit_transform(X)`，然后再调用 `train_test_split`。这会让测试集的信息提前混入训练过程（**数据泄漏**），导致本地评估分数虚高、上线后性能崩盘。

正确做法是使用 `Pipeline` 将“缺失值填充 + 特征缩放 + 类别编码 + 模型”打包成一个整体，只在 `X_train` 上调用 `.fit()`：

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
# 切分数据后，只在训练集上 fit(X_train, y_train)，测试集直接 predict(X_test)，零泄漏！
```

---

## 三、 PCA（主成分分析）直觉与 Eigenfaces 人脸降维

假设你要用一张二维照片展示一个三维茶壶，怎么拍才能保留最多细节？答案是**找茶壶延展最宽、变化最大的角度拍过去**——这就是 **PCA** 的直觉：把原始高维数据正交投影到方差最大的前 $k$ 个主方向（主成分）上。

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X_train)
pca = PCA(n_components=2)
Z_train = pca.fit_transform(X_scaled)
Z_test = pca.transform(StandardScaler().fit(X_train).transform(X_test))
print('各主成分解释方差占比:', pca.explained_variance_ratio_)
```

- **解释方差比（`explained_variance_ratio_`）**：表示每个主成分保留了原始训练集多少比例的信息（总方差）。
- **特征脸（Eigenfaces）经典应用**：一张 $64 \times 64$ 的人脸灰度图有 $4096$ 个像素特征，直接训练分类器极易过拟合。先在训练集上用 PCA 提取前 50～150 个主成分（即“特征脸”基底），再把低维坐标喂给 SVM 分类器，既大幅提速又能滤除光照噪声（详见第 04 章配套项目 `faces.py`）。

---

## 四、 时间序列平滑与常见误区汇总

1. **尾随滑动平均（Trailing Rolling Mean）**：`series.rolling(k).mean()` 仅使用当前时刻及过去 $k-1$ 个历史点，**适合在线实时预测特征构建**。
2. **居中滑动平均（Centered Rolling Mean）**：`series.rolling(k, center=True).mean()` 同时使用了过去和未来的点，**仅限事后复盘绘图，严禁用于预测任务**。
3. **四大高频误区**：
   - ❌ 划分数据集前对全表拟合 `StandardScaler` / `PCA` / `SimpleImputer`（数据泄漏）。
   - ❌ 用 `LabelEncoder` 把“北京=0、上海=1、广州=2”直接喂给线性模型（模型会荒谬地以为 `北京 + 广州 = 2 × 上海`）。
   - ❌ 忘了先做 `StandardScaler` 就直接跑 PCA（数值范围上万的列会彻底霸占第一主成分）。
   - ❌ 对具有时间先后顺序的数据随机打乱（Shuffle）做交叉验证。

---

## 五、 配套实践与自测

- **可运行项目**：[`projects/01_data_foundations/main.py`](../projects/01_data_foundations/main.py) 与 [`customer_pca.py`](../projects/01_data_foundations/customer_pca.py)。
- **思考与自测**：
  1. 为什么在测试集上必须调用 `scaler.transform(X_test)` 而不是 `scaler.fit_transform(X_test)`？
  2. 如果特征 A 的单位从“米”改成“毫米”（数值扩大 1000 倍），不做标准化直接跑 PCA 会发生什么？
  3. 为什么说“PCA 保留了最大方差方向，但该方向未必对分类预测最有帮助”？
