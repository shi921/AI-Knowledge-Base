**Language:** English | [简体中文](05_stream_learning_concept_drift.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 05 | Streaming Learning, Concept Drift, and Page–Hinkley

## Why do static models fail over time?

User behavior, context, markets, and devices change. As `P(y|X)` changes, a model calibrated on old observations may perform worse on new ones.

- **Data drift / covariate shift:** input distribution `P(X)` changes.
- **Concept drift:** target relationship `P(y|X)` changes, even if `P(X)` stays constant.
- **Label shift:** target distribution `P(y)` changes.

These shifts may coexist. Increasing error rates can also come from noisy labels or system failures, so detection is not automatic causal diagnosis.

## Batch training versus incremental learning

- **Batch:** fit a static model on `X_train, y_train` and periodically score future data.
- **Incremental:** update a compatible estimator through River's `learn_one(x, y)` or scikit-learn's `partial_fit(X_batch, y_batch)`.
- **Prequential test-then-train:** predict newly arrived examples and record the loss before updating on their known labels.

```python
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='log_loss', random_state=42)
model.partial_fit(X_first, y_first, classes=[0, 1])
y_pred = model.predict(X_next)       # Predict the next batch first.
model.partial_fit(X_next, y_next)     # Update only afterward.
```

## Page–Hinkley intuition

The detector tracks cumulative increases in an observed loss relative to its historical average. An alert is raised when the statistic passes a threshold. Common parameters include `delta` (tolerance), `threshold` (alarm level), and `min_instances` (warm-up). Lower thresholds improve sensitivity but can increase false positives.

Convert accuracy to error using `1-accuracy` (**larger is worse**). Feed a well-defined loss to the detector; do not confuse negative accuracy with error rate. With River, call `.update()` and inspect `.drift_detected`, not the return value of `.update()`.

```python
from river.drift import PageHinkley
ph = PageHinkley(min_instances=30, delta=0.005, threshold=40)
for error in (predictions != labels).astype(float):
    ph.update(float(error))
    if ph.drift_detected:
        print('Possible drift')
```

## Synthetic stream examples

| Stream | File | Scenario |
|---|---|---|
| SEA | `datasets/sea/sea.csv` (18,500 synthetic rows) | Fit an SVM on an initial period, evaluate subsequent batches, detect changes with Page–Hinkley, and retrain on a sliding history window |
| STAGGER | `datasets/stagger/Stagger_full.csv` | Categorical Size/Color/Shape variables, stream encoding, and incremental classification |

Try different batch sizes, SVM gamma, window widths, and detection thresholds. The default detector updates on every prediction error; use consistent update frequencies when comparing runs.

**Projects:** [`projects/05_concept_drift/sea.py`](../projects/05_concept_drift/sea.py) and `stagger.py`, exporting batch accuracy and alert plots.

**Review questions:** How can shuffling hide drift? Why must retraining after an alert use only the history currently available? How does lowering the threshold change false alarms and missed detections?
