**语言：** [English](03_supervised_learning_evaluation.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 03｜监督学习：回归、分类、验证与调参

> **🧭 知识脉络定位**：第二阶段「传统机器学习与数学内核篇」· **第 4 步 / 共 14 步**  
> - **💡 零基础直觉比喻**：训练模型就像备战高考：**训练集（Train）**是平时的课后作业，**验证集 / 交叉验证（Validation / CV）**是高三模拟考（用来挑选学习方法和调参），而**测试集（Test）**是真正的高考——一辈子只能考一次，绝不能拿高考真题来平时刷题！  
> - **🎯 这一章解决什么核心问题？**：
>   1. **两大核心任务**：预测连续数字叫**回归**（如预测房价卖多少万），预测离散类别叫**分类**（如判断邮件是不是垃圾邮件、识别手写数字 0–9）。
>   2. **科学选指标**：为什么在抓坏人（欺诈检测，99% 正常、1% 骗子）时不能看准确率（Accuracy），而必须看精确率（Precision）、召回率（Recall）和 F1？本章帮你彻底建立防自欺欺人的评估观。  
> - **🔗 前后衔接**：[09｜梯度下降与数值优化](09_regression_optimization.zh-CN.md) → 本章（监督学习全流程与科学评估）→ **下一站**：[08｜模型可解释性与 SHAP](08_explainable_ai_shap.zh-CN.md)（模型分算准了，怎么打开黑盒看它凭什么这么判？）。

## 统一工作流

问题定义 → 探索与清洗 → 划分训练/验证/测试 → **训练折内**拟合预处理与模型 → 交叉验证选择模型/参数 → 只用测试集一次评价 → 在全体标记训练数据上重训 → 导出预测。

### 回归任务

- 简单线性回归 `y = b0 + b1*x`，多元线性回归 `y = b0 + Σ(bj*xj)`；`Ridge` 添加 L2 惩罚以缓解不稳定。
- 随机森林回归通过多个决策树集成表示非线性；对外推和缺失值处理仍有限制。
- `MAE = mean(|y-ŷ|)`：易解释、稳健性较好。`RMSE = sqrt(mean((y-ŷ)^2))`：更重罚大误差。`R² = 1 - SSE/SST`：可为负。
- 房价偏态时可对目标使用 `log1p(y)`，预测后 `expm1` 恢复美元尺度；必须标清指标是在 **log 域**还是原尺度。

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
# X, y, and pipeline are defined in the corresponding project.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
pipeline.fit(X_train, y_train)
print('CV MAE:', -scores.mean())
print('Holdout MAE:', mean_absolute_error(y_test, pipeline.predict(X_test)))
```

### 分类任务

- 模型：DecisionTree、RandomForest、LogisticRegression、SVM，常见示例包括 MNIST、Digits、信用卡欺诈和人脸。
- 混淆矩阵：TP/TN/FP/FN；`Precision = TP/(TP+FP)`；`Recall = TP/(TP+FN)`；`F1 = 2PR/(P+R)`。
- 多分类 macro-F1 是各类 F1 平均，weighted-F1 按样本数量加权；数据极不均衡时准确率可能毫无意义。
- 对欺诈这种稀有阳性问题，用分层切分、`PR-AUC`、召回率和业务误报代价；仅查看 accuracy 不够。

## 交叉验证和过/欠拟合

- **欠拟合**：训练和验证都差，考虑特征与模型表达能力。
- **过拟合**：训练好、验证差，考虑正则化、更多数据、限制复杂度。
- `KFold` 适合一般回归；`StratifiedKFold` 适合分类类比例维持；`TimeSeriesSplit` 针对时间序列。
- 网格搜索必须在训练集/内层 CV 内完成。若用测试集多次挑参数，其实测试集已变成验证集。

## 模型实验的完整流程

1. 分类：使用 Digits 数据集，对比分类器和不同评价指标。MNIST 可作为需要联网下载的可选扩展。
2. 回归：从房价 CSV 提取特征，交叉验证后评估并输出预测。
3. 为每个实验保留数据切分、指标与随机种子，便于复现。

**完整项目**：[`projects/02_house_price_regression/main.py`](../projects/02_house_price_regression/main.py)（全流程房价回归），[`projects/03_classification_and_faces/main.py`](../projects/03_classification_and_faces/main.py)（Digits 分类与 CV），`faces.py`（可选授权人脸特征 CSV → PCA → SVM）。

**自测**：为何 CV 不应使用测试集？为什么 `precision` 高不意味着 `recall` 高？在选择 MAE 与 RMSE 时如何考虑极大误差？


## 延伸：从模型评估到优化

线性回归解析解、多项式特征与 Ridge 正则化的数值原理见 [第 09 章](09_regression_optimization.zh-CN.md)。注意合成小样本上的训练表现不能代替真实数据的泛化评估。
