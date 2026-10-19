**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# Multi-view Text Classification with Stacking

```bash
python projects/07_multiview_stacking/main.py
# Optional trusted local CSV with columns: text, aux_text, label
python projects/07_multiview_stacking/main.py --data /path/to/private_dataset.csv
```

Programmatically generated **synthetic examples** demonstrate two feature views and a meta-classifier. Disjoint base/meta/test splits prevent information leakage into the meta-model. Toy results only verify code behavior, not performance on real populations. No personal social-media records are distributed.

**Related notes:** [Chapter 07](../../notes/07_multiview_text_classification_stacking.md), [Chapter 11](../../notes/11_classical_nlp_language_models.md).
