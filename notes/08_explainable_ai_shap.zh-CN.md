**语言：** [English](08_explainable_ai_shap.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 08｜Explainable AI：SHAP 的局部与全局解释

## 预测准确与可解释不是一回事

`model.predict(x)` 告诉我们输出是什么，而可解释性分析试图描述模型**如何**利用输入特征作出这个输出。局部解释针对一个样本，全局解释汇总许多样本上的模式；二者不能随意互相替代。

## SHAP 核心概念

SHAP 以合作博弈 Shapley 值为灵感，把单个预测相对于基线的变化分配到每个特征。对所选输出空间，理想上有：

`prediction ≈ base_value + Σ shap_value(feature_j)`。

- **base value**：解释器背景分布对应的参考输出。
- **正 SHAP**：将预测推向更高的输出；负 SHAP 推向更低的输出。
- **局部 waterfall**：观察单个输入样本哪些特征推动预测上升/下降。
- **全局 summary / beeswarm**：查看所有样本的影响大小与方向。
- **dependence scatter**：探索某特征大小与其 SHAP 值关系，可能体现交互效应。

## 典型流程

```python
import shap
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80, random_state=42).fit(X_train,y_train)
explainer = shap.TreeExplainer(clf)
result = explainer(X_test.iloc[:50])
# Binary classification explanations may include a class dimension.
positive = result[:,:,1] if result.values.ndim == 3 else result
shap.summary_plot(positive.values, X_test.iloc[:50], show=False)
shap.plots.waterfall(positive[0],show=False)
```

## Mushroom 数据集案例

此例仅使用 `odor` 特征预测食用/有毒类别，并要求全局 summary 和具体**有毒样本**的 waterfall。one-hot 将 odor 各类别变成可解释的二元列，但其中这些列是同一个原始变量的互斥值，因此 SHAP 的独立特征假设及相关性解释需谨慎。

California Housing 也适合用于回归解释，可展示人为预估→模型预测→SHAP 解释的“human-in-the-loop”；注意 house value 单位转换。`R²` 不是分类准确率，不能简单打印为 Accuracy。

## 避免误读

1. **SHAP 不是因果效应**，无法证明改变某个特征就能改变结果。
2. 解释器可能输出模型原始分数（如 log-odds）或概率；绘图与文字描述必须一致。
3. 高相关特征会影响贡献的分配方式；改变背景集也会改变解释。
4. 解释预测时也要先检查模型是否有足够可靠的泛化性能。

**完整项目**：[`projects/08_explainable_ai/main.py`](../projects/08_explainable_ai/main.py)。默认用内置 Breast Cancer 离线演示；`--dataset mushroom` 才使用 UCI 网络下载数据并演示仅使用 odor 的特征限制。

**自测**：单个 waterfall 可否推出全数据集重要性？正 SHAP 是否总意味着正类别？为何需要明确概率/log-odds 的输出空间？
