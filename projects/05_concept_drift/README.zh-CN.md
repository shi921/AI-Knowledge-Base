**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# 流式学习与概念漂移

```bash
python projects/05_concept_drift/stagger.py --rows 5000 --batch-size 1000
python projects/05_concept_drift/sea.py --init 1000 --batch-size 500 --max-batches 8
```

`datasets/stagger` 和 `datasets/sea` 是自动生成的合成时序数据，标签决策规则会在序列中变化，用于展示先评估后更新、滚动训练及 Page–Hinkley 漂移检测。如果未安装 River，会使用简化的本地检测器。检测阈值和延迟需要针对具体数据校准。

**相关知识笔记：** [第 05 章](../../notes/05_stream_learning_concept_drift.zh-CN.md)。
