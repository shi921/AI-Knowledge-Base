**语言：** [English](05_explainable_ai_shap.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 05｜Explainable AI：SHAP 的局部与全局解释

> **🧭 知识脉络定位**：第二篇「传统机器学习与数学内核」· **第 05 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：在第 04 章中我们训练出了高准确率的随机森林模型，但它在外部看来像一个密不透风的“黑盒”。如果银行风控模型拒绝了某位客户的贷款，或者医疗模型提示某患者有高风险，只吐出一个冷冰冰的结果是无法通过合规审计的。**SHAP 就像是给黑盒模型拍 X 光片（或者团队分奖金时的“功劳核算表”）**——它能精确算出：在这一次预测里，“收入高”帮客户加了多少分、“历史逾期”又扣了多少分。  
> - **🎯 这一章解决什么核心问题？**：填补“预测准确”与“人类可信”之间的鸿沟。掌握如何用合作博弈论中的 **Shapley 值**对复杂模型进行**全局解释**（整体来看哪些特征最重要）与**局部解释**（针对某一个具体样本为什么判成这样），并学会避开“把模型归因当成现实因果”的常见误区。  
> - **🔗 前后衔接**：[04｜监督学习：回归与分类算法、交叉验证与评估体系](04_supervised_learning_evaluation.zh-CN.md) → 本章（拆解并审计黑盒模型决策依据）→ **下一章**：[06｜Flask 模型部署：从 sklearn 到 REST API](06_flask_model_deployment.zh-CN.md)（模型既准又通过了可解释性审计，现在将它打包上线变成对外服务接口！）。

---

## 一、 预测准确与可解释不是一回事

`model.predict(x)` 只告诉我们模型最终预测**是什么（What）**，而可解释性（Explainable AI, XAI）旨在揭示模型**是如何利用各个输入特征拼凑出这个预测结果的（How）**。
- **局部解释（Local Explanation）**：针对**单个具体样本**（如“为什么 3 号病人被判为高危？”）。
- **全局解释（Global Explanation）**：汇总**整个数据集**上所有样本的特征贡献分布（如“在所有病人中，哪三个指标对诊断影响最大？”）。二者视角不同，绝不能混为一谈。

---

## 二、 SHAP 核心原理与四大可视化图表

SHAP（SHapley Additive exPlanations）源自合作博弈论中的 Shapley 值：假设所有输入特征“组团合作”把模型输出从背景平均水平（`base_value`）推到了当前样本的最终预测值（`prediction`），SHAP 通过遍历特征加入顺序的边际贡献均值，把总变化量公平地分摊给每一个特征 $j$：

$$\text{prediction}(x) \approx \text{base\_value} + \sum_{j=1}^{d} \text{SHAP}(x_j)$$

| 核心概念 / 图表 | 作用与读图方法 |
|---|---|
| **`base_value`（基线值）** | 模型在背景样本集上的平均输出（不看当前样本任何特征时的“默认初始分”） |
| **正 / 负 SHAP 值** | 正值表示该特征将当前预测推向更高值（如推向正类 `class=1`）；负值表示拉向更低值 |
| **局部瀑布图 (`plots.waterfall`)** | 展示**单个样本**从 `base_value` 出发，每个特征一步步加分（红条）或减分（蓝条）直到最终预测值的全过程 |
| **全局蜂群汇总图 (`summary_plot`)** | 从上到下按全局重要性排列特征；每个点代表一个样本，颜色代表特征值高低（红高蓝低），横轴代表 SHAP 贡献正负 |
| **依赖散点图 (`dependence_plot`)** | 观察单一特征取值大小与其 SHAP 值的函数关系及特征交互效应 |

---

## 三、 标准代码流程（以树模型 `TreeExplainer` 为例）

```python
import shap
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=80, random_state=42).fit(X_train, y_train)
explainer = shap.TreeExplainer(clf)
result = explainer(X_test.iloc[:50])

# 新版 SHAP 对二分类可能返回三维数组 [样本数, 特征数, 类别数]，取阳性类 class=1
positive = result[:, :, 1] if result.values.ndim == 3 else result

# 1. 全局特征重要性蜂群图
shap.summary_plot(positive.values, X_test.iloc[:50], show=False)
# 2. 第 0 号样本的局部归因瀑布图
shap.plots.waterfall(positive[0], show=False)
```

---

## 四、 经典案例与四大解读误区

### 案例：Mushroom（毒蘑菇识别）与 California Housing（人机协同）
- 在毒蘑菇数据集中，仅使用 `odor`（气味）一个分类特征经 One-Hot 编码后预测是否有毒。需要注意：One-Hot 拆出的多列（如 `odor_foul=1`, `odor_almond=0`）本质上是**同一个原始变量的互斥状态**，解释时应合并理解。
- 在回归任务（如房价评估）中，SHAP 常用于构建 **Human-in-the-Loop（人机协同决策）**：先由业务专家给出初估，再对照模型预测与 SHAP 瀑布图排查是否漏看了关键因素。

### 避免误读的四大铁律
1. **SHAP 是模型归因，不是现实世界的因果效应（Causality）**：SHAP 只告诉你“模型内部是怎么算分的”，不能证明在现实中强行改变某个特征就一定能改变结果。
2. **明确输出空间单位**：分类模型的 `TreeExplainer` 可能输出**概率空间（Probability）**，也可能输出未过 Sigmoid 的**对数几率空间（Log-odds）**，汇报时必须核实清楚。
3. **高共线性特征会分摊贡献**：如果两个特征高度相关（如“房屋面积”和“房间总数”），SHAP 值可能会在二者之间分摊，不能因为某列 SHAP 变小就武断认为它在现实中无关。
4. **烂模型的解释毫无意义**：在对业务方展示 SHAP 图之前，必须先按第 04 章标准确认模型在独立测试集上具备可靠的泛化性能。

---

## 五、 配套实践与自测

- **可运行项目**：[`projects/08_explainable_ai/main.py`](../projects/08_explainable_ai/main.py)（默认使用内置乳腺癌数据集离线生成全局 Summary 图与局部 Waterfall 图；加 `--dataset mushroom` 可测试 UCI 蘑菇数据集）。
- **思考与自测**：
  1. 能否仅凭某一个样本的 `waterfall` 瀑布图，就断定某个特征在整个数据集上最重要？为什么？
  2. 某个特征的取值为正数（例如 `年龄 = 45`），它的 SHAP 值一定也是正数吗？
  3. 为什么在二分类 SHAP 解释中必须说明当前展示的是针对 `class=1` 还是 `class=0` 的输出？
