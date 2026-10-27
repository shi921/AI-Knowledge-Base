**语言：** [English](04_supervised_learning_evaluation.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 04｜监督学习：回归与分类算法、交叉验证与评估体系

> **🧭 知识脉络定位**：第二篇「传统机器学习与数学内核」· **第 04 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：第 03 章教你学会了怎么造一把“线性尺子”，这一章我们将武器库扩充为全套**“监督学习工具箱（决策树、随机森林、逻辑回归、SVM）”**，并建立一套严格的**“高考防作弊考评制度”**：**训练集（Train）**是平时的课后作业，**交叉验证（Cross-Validation）**是高三多轮模拟考（用来挑选最趁手的模型和超参数），而**测试集（Test）**是真正的高考——一辈子只能考最后一次，绝不能拿高考真题来平时刷题调参！  
> - **🎯 这一章解决什么核心问题？**：
>   1. **两大监督任务与算法选型**：预测连续数值用**回归**（如预测房价），预测离散类别用**分类**（如手写数字识别、欺诈检测、人脸识别）。
>   2. **科学评估指标**：为什么在抓坏人（欺诈检测，99% 正常、1% 欺诈）时看准确率（Accuracy）会彻底被骗？如何用精确率（Precision）、召回率（Recall）、F1 与 PR-AUC 衡量模型真实本领？  
> - **🔗 前后衔接**：[03｜线性回归、正则化与梯度下降优化](03_regression_optimization.zh-CN.md) → 本章（监督学习算法全景与科学验证体系）→ **下一章**：[05｜模型可解释性（XAI）：SHAP 局部与全局归因](05_explainable_ai_shap.zh-CN.md)（模型预测准了之后，如何拆解黑盒解释它为什么做出这个决定？）。

---

## 一、 监督学习统一防泄漏工作流

任何严谨的监督学习实验都遵循以下 7 步闭环：
`问题定义（回归还是分类？） → 探索与清洗（第 01 章） → 划分训练/测试集 → 在训练折（Training Folds）内部拟合预处理与模型（第 02 章） → 用交叉验证（CV）对比模型与超参数 → 在全体训练集上重训最优配置 → 仅在独立测试集上做最后一次评估并导出预测`。

### 主流监督学习算法直观对比

| 算法名称 | 适用任务 | 核心原理直觉 | 优势与局限 |
|---|---|---|---|
| **线性回归 / Ridge** | 回归 | 用加权直线/超平面拟合连续目标（见第 03 章） | 速度快、可解释强；无法直接捕捉复杂非线性关系（除非构造多项式/交叉特征） |
| **逻辑回归 (Logistic Regression)** | 分类 | 在线性打分 $Xw$ 外套一层 Sigmoid/Softmax 转化为类别概率 | 强力且快速的分类基线，输出校准良好的概率；决策边界为线性 |
| **决策树 (Decision Tree)** | 回归 / 分类 | 像玩“二十个问题”一样，按特征阈值层层二分提问（如 `面积 > 120?`） | 无需特征缩放、能处理非线性；单棵树极易过拟合（对数据微小变动敏感） |
| **随机森林 (Random Forest)** | 回归 / 分类 | 同时训练上百棵不同的决策树（随机抽样本 + 随机抽特征），最后集体投票/求平均 | 表格数据上的全能主力，抗过拟合能力强；属于黑盒集成（需配合第 05 章 SHAP 解释），不擅长超出训练范围的外推 |
| **支持向量机 (SVM)** | 分类 / 回归 | 寻找一条离两类样本边界都最远（最大间隔 Margin）的分界线，可配合核函数（Kernel）映射高维 | 在中小规模高维数据（如 PCA 降维后的人脸特征）上表现极佳；对特征缩放敏感，超大样本训练慢 |

---

## 二、 回归任务与评价指标选型

- **MAE（平均绝对误差）** $= \frac{1}{n}\sum |y - \hat{y}|$：单位与原目标一致（如相差多少美元），对极端离群值较稳健、业务最易理解。
- **RMSE（均方根误差）** $= \sqrt{\frac{1}{n}\sum (y - \hat{y})^2}$：因为先平方再开方，**对大误差惩罚极重**（适合绝不允许出现单次巨额预测偏差的场景）。
- **$R^2$（决定系数）** $= 1 - \text{SSE}/\text{SST}$：衡量模型比“直接猜平均值”好多少（满分 $1.0$，若比猜均值还差则为负数）。**注意：$R^2$ 是回归拟合优度，绝不能当成分类准确率（Accuracy）！**
- **偏态目标变换技巧**：房价、收入等右偏长尾目标常在训练时取对数 `y_log = np.log1p(y)`，预测后再用 `np.expm1(pred_log)` 还原回真实金额；汇报指标时必须标明是在 **log 域**还是**原始金额尺度**。

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

# X, y 与包含预处理的 pipeline 定义见配套项目
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
pipeline.fit(X_train, y_train)
print('5-Fold CV MAE:', -scores.mean())
print('Holdout Test MAE:', mean_absolute_error(y_test, pipeline.predict(X_test)))
```

---

## 三、 分类任务与不平衡数据评估陷阱

以二分类（如“欺诈=阳性 Positive=1，正常=阴性 Negative=0”）为例，所有指标都源于**混淆矩阵（Confusion Matrix）**：
- **TP（真阳性）**：真骗子被成功抓出；**FP（假阳性 / 误报）**：好人被冤枉成骗子；
- **TN（真阴性）**：好人被正确放行；**FN（假阴性 / 漏报）**：真骗子漏网逃脱。

| 核心指标 | 计算公式 | 直白含义（业务拷问） | 适用场景与陷阱 |
|---|---|---|---|
| **Accuracy（准确率）** | $\frac{TP+TN}{TP+TN+FP+FN}$ | 所有样本里猜对了百分之几？ | **类别极度不平衡时完全失效**：若 99% 是好人、1% 是骗子，一个永远无脑输出“好人”的模型准确率高达 99%，但一个骗子都抓不到！ |
| **Precision（精确率 / 查准率）** | $\frac{TP}{TP+FP}$ | **“抓的准不准？”** 模型报警抓出的人里，有多少比例真的是骗子？ | 误报代价高时看重（如垃圾邮件拦截，宁可放过几封广告，绝不能把老板的重要合同误判进垃圾箱） |
| **Recall（召回率 / 查全率）** | $\frac{TP}{TP+FN}$ | **“抓的全不全？”** 所有真骗子里，被模型成功揪出来了多少比例？ | 漏报代价极高时看重（如重疾筛查、金融盗刷拦截，宁可多做几次复核，绝不能漏掉真正的患者或盗刷） |
| **F1-Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | 精确率与召回率的调和平均数 | 只在 Precision 和 Recall 同时高时才高；多分类任务常用 **Macro-F1**（各类别平等平均，照顾小众类）或 **Weighted-F1** |
| **PR-AUC** | Precision-Recall 曲线下面积 | 不同判定阈值下查准与查全的综合表现 | 稀有阳性（如欺诈、罕见病）评估的首选指标 |

---

## 四、 交叉验证（Cross-Validation）选型策略

- **`KFold`**：适用于一般独立同分布的回归任务。
- **`StratifiedKFold`（分层 K 折）**：分类任务首选！确保每一折验证集中正负样本的比例与原数据集完全一致。
- **`TimeSeriesSplit`（时序切分）**：针对时间序列数据，永远只用过去预测未来，严禁随机打乱。
- **调参铁律**：网格搜索（`GridSearchCV`）必须在训练集的内层 CV 中完成。如果你根据测试集分数反复修改模型参数，测试集就已经沦为了验证集！

---

## 五、 配套实践与自测

- **可运行项目**：
  - [`projects/02_house_price_regression/main.py`](../projects/02_house_price_regression/main.py)：全流程房价回归（Pipeline 预处理 + 交叉验证 + Holdout 测试评估）。
  - [`projects/03_classification_and_faces/main.py`](../projects/03_classification_and_faces/main.py)：Digits 手写数字多分类器对比与交叉验证。
  - [`projects/03_classification_and_faces/faces.py`](../projects/03_classification_and_faces/faces.py)：人脸特征 PCA 降维 + SVM 分类流水线。
- **思考与自测**：
  1. 为什么在选模型和调超参数时绝对不能使用测试集（Test Set）？
  2. 如果把分类器的判定门槛调得极其严格（只有 99.9% 把握时才报阳性），`Precision` 和 `Recall` 会分别怎么变化？
  3. 在房价预测中，如果数据里有几套售价上亿的顶级豪宅，选择 `MAE` 还是 `RMSE` 作为损失/评估指标受豪宅影响更大？
