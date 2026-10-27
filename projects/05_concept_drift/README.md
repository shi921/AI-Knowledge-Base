**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# Online Learning and Concept Drift

```bash
python projects/05_concept_drift/stagger.py --rows 5000 --batch-size 1000
python projects/05_concept_drift/sea.py --init 1000 --batch-size 500 --max-batches 8
```

`datasets/stagger/` and `datasets/sea/` contain synthetic temporal streams whose labeling rules change over time. They demonstrate test-then-train evaluation, sliding-window retraining, and Page–Hinkley detection. A simplified local detector is used if River is unavailable. Calibrate detection thresholds and latency for real applications.

**Related notes:** [Chapter 07](../../notes/07_stream_learning_concept_drift.md).
