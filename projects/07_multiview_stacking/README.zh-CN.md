**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# 多视角文本分类与 Stacking

```bash
python projects/07_multiview_stacking/main.py
# Optional authorized local CSV with text, aux_text, and label columns
python projects/07_multiview_stacking/main.py --data /path/to/private_dataset.csv
```

默认用**程序自动生成的合成数据**展示两个特征视角 + Meta Classifier 的训练流程。使用互不重叠的 base/meta/test 集来避免元模型泄漏。演示数字只验证程序流程，不代表真实性能。不包含任何个人社交媒体记录。

**相关知识笔记：** [第 10 章](../../notes/10_multiview_text_classification_stacking.zh-CN.md)、[第 09 章](../../notes/09_classical_nlp_language_models.zh-CN.md)。
