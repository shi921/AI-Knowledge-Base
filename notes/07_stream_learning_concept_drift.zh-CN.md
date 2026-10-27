**语言：** [English](07_stream_learning_concept_drift.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 07｜流式数据学习、概念漂移（Concept Drift）与 Page–Hinkley

> **🧭 知识脉络定位**：第三篇「模型工程化部署与流式运维」· **第 07 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：你在第 06 章把餐厅外卖窗口（API）顺利开起来了，生意起初很好。但半年后物价上涨、年轻人口味也变了，五年前定下的老菜单开始频频收到差评——这就是**概念漂移（Concept Drift，现实世界的输入输出规律随时间变了）**。你需要装一个自动烟雾报警器（**Page–Hinkley 统计检测器**），一旦发现近期预测错误率出现持续性异常抬头，立刻拉响警报，并用最近的新订单重新研发菜单（**滑动窗口重训 / 在线增量学习**）。  
> - **🎯 这一章解决什么核心问题？**：终结“模型训练一次就能用一辈子”的误区。学会区分三种线上分布偏移、遵守流式数据**“先预测评估、后更新模型（Prequential）”**的铁律，并掌握自动漂移报警与自适应重训闭环。  
> - **🔗 前后衔接**：[06｜Flask 模型部署：从 sklearn 到 REST API](06_flask_model_deployment.zh-CN.md) → 本章（线上漂移监控与动态自适应更新，完成传统机器学习闭环）→ **下一章**：[08｜神经网络、自动微分和 PyTorch](08_neural_networks_pytorch.zh-CN.md)（当特征与规律极其复杂、传统机器学习不够用时，正式迈入深度学习与神经网络篇章！）。

---

## 一、 为什么静态模型上线后会逐渐失效？三种漂移类型

真实业务系统中的用户行为、市场行情、季节气候或传感器老化都在随时间动态演化。在历史静态训练集上学会的规律，面对未来新数据时可能会遇到三类偏移：

| 漂移类型 | 数学表述 | 通俗解释与现实案例 |
|---|---|---|
| **数据漂移 / 协变量偏移 (Data Drift / Covariate Shift)** | 输入分布 $P(X)$ 改变，但条件关系 $P(y \mid X)$ 未变 | 来申请贷款的客群整体变年轻了（输入 $X$ 变了），但“年收入多少能还得起钱”的判定标准没变 |
| **概念漂移 (Concept Drift)** | **目标映射机制 $P(y \mid X)$ 本身发生改变** | 即使客群特征 $X$ 没变，但因为宏观经济危机或新规出台，原本安全的负债率现在也会违约（最棘手，旧决策边界直接作废！） |
| **标签偏移 (Label Shift)** | 目标先验分布 $P(y)$ 改变 | 流感爆发季医院发热门诊中阳性患者比例暴增 |

---

## 二、 批处理模型 vs. 在线流式模型与 Prequential 评估

1. **批处理学习（Batch Learning）**：一次性在静态历史数据上调用 `fit(X_train, y_train)`，模型参数固定不变；一旦发生概念漂移，只能在检测到衰退后用近期滑动窗口数据重新 `fit`。
2. **在线增量学习（Online / Incremental Learning）**：使用 River 库的 `learn_one(x, y)` 或 sklearn 的 `partial_fit(X_batch, y_batch)`，每到达一条或一小批新数据就增量更新一次权重。
3. **先测后训原则（Prequential / Test-then-Train Evaluation）**：
   - 在流式场景中，每当新一批样本 $(X_{\text{next}}, y_{\text{next}})$ 到达时，**必须先用当前模型做预测并记录误差，然后才能把真实标签拿去更新模型或漂移检测器**！如果先训练再预测同一批数据，就是严重的作弊（先看答案再考试）。

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss', random_state=42)
model.partial_fit(X_first, y_first, classes=[0, 1])

# 新批次到达：严格遵守 Prequential 先预测评估、后增量更新！
y_pred = model.predict(X_next)
batch_error = (y_pred != y_next).mean()
model.partial_fit(X_next, y_next)
```

---

## 三、 Page–Hinkley 漂移检测器原理与直觉

怎么判断线上错误率上升是“偶然的正常小波动”，还是“真的发生了概念漂移”？**Page–Hinkley (PH) 检验**将每个样本的误分类损失（`error = 1 - accuracy`，即预测正确记 `0.0`，预测错误记 `1.0`）转化为连续统计量，持续跟踪它相对于历史长期均值的**累积上升幅度**：

- **`delta`（容忍度）**：允许的微小波动幅度；
- **`threshold`（报警阈值 $\lambda$）**：累积偏离超过该阈值就触发漂移警报（阈值越低越灵敏，但误报 False Alarm 越多；阈值越高越稳重，但报警延迟越大）；
- **`min_instances`（最小预热样本数）**：刚开始统计不稳定时不报警。

```python
from river.drift import PageHinkley

ph = PageHinkley(min_instances=30, delta=0.005, threshold=40)
for error in (predictions != labels).astype(float):
    ph.update(float(error))      # 传入误分类损失 (0.0 或 1.0)，越大代表越差
    if ph.drift_detected:        # 注意检查 .drift_detected 属性，而非 update() 返回值！
        print('Drift alarm triggered! Retraining model on recent sliding window...')
```

---

## 四、 两类经典流式基准数据集与实验设计

| 数据集 | 仓库路径 | 数据特征与漂移任务设计 |
|---|---|---|
| **SEA 数据集** | `datasets/sea/sea.csv` | 含连续数值特征的合成流数据；前段训练基础分类器，后续批次中分界阈值发生突变，用 Page–Hinkley 检测报警并触发**最近窗口重训** |
| **STAGGER 数据集** | `datasets/stagger/Stagger_full.csv` | 含 `Size, Color, Shape` 三个类别特征，目标概念规则在序列中途发生切换（如从“小且红”突变为“大或绿”），验证在线分类器的恢复速度 |

---

## 五、 配套实践与自测

- **可运行项目**：[`projects/05_concept_drift/sea.py`](../projects/05_concept_drift/sea.py) 与 [`stagger.py`](../projects/05_concept_drift/stagger.py)（运行后在 `outputs/05_concept_drift/` 生成批次准确率变化与漂移报警点可视化曲线）。
- **思考与自测**：
  1. 为什么在流式概念漂移实验中，绝对不能在开始前对整张 CSV 表做随机打乱（`shuffle=True`）？
  2. 当第 $t$ 批次触发漂移报警需要重训模型时，为什么只能使用 $\le t$ 的近期历史窗口数据，而不能使用 $t+1$ 之后的数据？
  3. 调小 Page–Hinkley 的 `threshold` 阈值，对“报警及时性”和“误报率”分别有什么影响？
