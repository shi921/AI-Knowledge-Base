**语言：** [English](05_stream_learning_concept_drift.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 05｜流式数据学习、Concept Drift 与 Page–Hinkley

## 为什么传统一次性训练会失败？

服务系统的用户行为、环境、设备或市场可能随时间变化。静态训练集上的 `P(y|X)` 可能不再适用，导致线上准确率下降。

- **Data drift/covariate shift**：输入分布 `P(X)` 发生变化。
- **Concept drift**：目标机制 `P(y|X)` 改变；即使特征分布不变也可能出现。
- **Label shift**：目标分布 `P(y)` 变化。

这些变化可能同时发生。只看错误率能发现质量下降，但不一定能定位原因：也可能是标签噪声、系统故障或业务策略变化。

## 批处理模型 vs 在线模型

- Batch：`fit(X_train,y_train)` 训练固定模型。随后 `predict(X_future)` 检查性能衰退。
- Incremental：River 的 `learn_one(x,y)` 或 sklearn 部分模型的 `partial_fit(X_batch,y_batch)`，到达新数据后增量更新。
- **Prequential test-then-train**：先对到达样本预测、记录损失，再用已知真实标签更新检测器/模型，避免用同一批次的标签提前训练后才测试。

```python
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='log_loss', random_state=42)
model.partial_fit(X_first, y_first, classes=[0,1])
y_pred = model.predict(X_next)       # Predict the next batch first.
model.partial_fit(X_next, y_next)      # Update only afterward.
```

## Page–Hinkley 的直觉

将错误率/误分类损失转换为连续统计量，跟踪其相对于长期均值的累积上升。统计量超过阈值，就触发可能的变化报警。一般会有 `delta`（容忍小波动）、`threshold`（报警阈值）、`min_instances`（最少观测数）。阈值越低通常越灵敏，同时误报风险越大。

对于准确率 `accuracy`，使用 `1-accuracy` 作为损失：**数值越大表示越差**。检测器应接收定义清晰的误分类损失，不应混用负准确率与错误率。不要把 River `.update` 的返回值误当成报警标记；应检查 `.drift_detected`。

```python
from river.drift import PageHinkley
ph = PageHinkley(min_instances=30, delta=0.005, threshold=40)
for error in (predictions != labels).astype(float):
    ph.update(float(error))
    if ph.drift_detected:
        print('Possible drift')
```

## 两类合成流式基准

| 数据 | 文件 | 主要任务 |
|---|---|---|
| SEA | `datasets/sea/sea.csv`，18,500 条合成记录 | 用前 10,000 条训练 SVM，后续分批评估，用 Page–Hinkley 报警，滑动窗口重训 |
| STAGGER | `datasets/stagger/Stagger_full.csv` | 含 Size、Color、Shape 三个类别字段，适合流式编码和在线分类 |

可以实验调整 batch size、SVM gamma、滑动窗口和 Page–Hinkley 阈值。检测器默认每个样本更新错误，而不是每批仅更新一次；比较参数时应保持检测频率一致。

**完整项目**：[`projects/05_concept_drift/sea.py`](../projects/05_concept_drift/sea.py) 与 `stagger.py`。会保存每批准确率和漂移标记曲线。

**自测**：为什么随机打乱数据可能掩盖漂移？报警后重训为什么只用已有历史数据？稳定地改变阈值会如何影响误报、漏报？
