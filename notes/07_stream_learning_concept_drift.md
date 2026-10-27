**Language:** English | [简体中文](07_stream_learning_concept_drift.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 07 | Streaming Learning, Concept Drift, and Page–Hinkley

> **🧭 Knowledge Map Position**: Part III (Model Serving & MLOps) · **Chapter 07 of 15**  
> - **💡 Intuitive Analogy**: After opening your takeout API window (Chapter 06), business booms at first. Months later, inflation and changing customer tastes make your old menu unpopular—that is **Concept Drift** (the underlying relationship $P(y \mid X)$ changes over time). You need an automated smoke detector (**Page–Hinkley test**) that raises an alarm when prediction errors steadily climb, triggering a menu refresh using recent orders (**sliding-window retraining / online learning**).  
> - **🎯 Why Does This Matter?**: Deployed models are not static monuments. This chapter teaches you to distinguish distribution shifts, evaluate streams honestly via **prequential (test-then-train)** protocols, and build automated drift-detection and retraining loops.  
> - **🔗 Learning Chain**: [06 | Model Deployment with Flask REST APIs](06_flask_model_deployment.md) → This Chapter (Monitor drift & adapt online, completing Classical ML & MLOps) → **Next Chapter**: [08 | Neural Networks, Autograd, and PyTorch](08_neural_networks_pytorch.md) (Enter Part IV: Deep Learning and NLP!).

---

## 1. Why Static Models Fail Over Time: Three Types of Drift

User behavior, markets, seasons, and sensors evolve over time:

| Shift Type | Mathematical Change | Intuition & Example |
|---|---|---|
| **Data Drift / Covariate Shift** | Input distribution $P(X)$ changes, $P(y \mid X)$ stays fixed | Younger applicants start applying for loans ($P(X)$ shifts), though the income threshold for repayment stays the same |
| **Concept Drift** | **Target rule $P(y \mid X)$ changes** | Even for identical customer profiles $X$, an economic shock changes who defaults (the old decision boundary becomes invalid!) |
| **Label Shift** | Target prior $P(y)$ changes | Flu season sharply increases the base rate of positive patients |

---

## 2. Batch vs. Online Learning & Prequential Evaluation

1. **Batch Models**: Fit once on static historical data (`fit(X_train, y_train)`) and retrain on a sliding window when drift is flagged.
2. **Online / Incremental Models**: Update parameters continuously as new observations arrive using River's `learn_one(x, y)` or scikit-learn's `partial_fit(X_batch, y_batch)`.
3. **Prequential (Test-then-Train) Discipline**: Always **predict and record loss on each incoming batch first**, and **update the detector/model only afterward** using the revealed ground-truth labels.

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss', random_state=42)
model.partial_fit(X_first, y_first, classes=[0, 1])

# Prequential order: evaluate next batch BEFORE updating weights!
y_pred = model.predict(X_next)
model.partial_fit(X_next, y_next)
```

---

## 3. Page–Hinkley Drift Detection Intuition

The **Page–Hinkley (PH)** test converts per-sample misclassification error (`1.0` if wrong, `0.0` if right—larger means worse) into a cumulative statistic tracking sustained upward deviations above the running mean:
- `delta`: magnitude of small fluctuations tolerated;
- `threshold`: cumulative threshold triggering an alarm (lower = faster detection but more false alarms);
- `min_instances`: warmup period before alarms can fire.

```python
from river.drift import PageHinkley

ph = PageHinkley(min_instances=30, delta=0.005, threshold=40)
for error in (predictions != labels).astype(float):
    ph.update(float(error))
    if ph.drift_detected:        # Check .drift_detected, not the return value of .update()!
        print('Possible concept drift detected!')
```

---

## 4. Synthetic Streaming Benchmarks: SEA & STAGGER

| Dataset | Path | Stream Design |
|---|---|---|
| **SEA** | `datasets/sea/sea.csv` | Continuous features where the classification boundary shifts mid-stream; tests Page–Hinkley alarms and sliding-window SVM retraining |
| **STAGGER** | `datasets/stagger/Stagger_full.csv` | Categorical attributes (`Size, Color, Shape`) whose target logical rule changes abruptly across batches |

---

## 5. Hands-on Projects & Self-Test

- **Runnable Projects**: [`projects/05_concept_drift/sea.py`](../projects/05_concept_drift/sea.py) and [`stagger.py`](../projects/05_concept_drift/stagger.py).
- **Self-Test**:
  1. Why does randomly shuffling a dataset before evaluation destroy concept-drift analysis?
  2. Why must post-alarm retraining use only historical data up to the current timestamp?
  3. How does lowering the Page–Hinkley `threshold` trade off detection delay against false alarms?
