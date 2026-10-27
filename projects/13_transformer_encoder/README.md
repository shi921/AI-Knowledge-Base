**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# Transformer Encoder: Self-attention and Text Classification

```bash
python projects/13_transformer_encoder/main.py --epochs 16
# Optional local pickle data that you own and trust
python projects/13_transformer_encoder/main.py --data-dir /path/to/trusted-data --epochs 20
```

An external data folder needs `movie_reviews.pkl` and `movie_labels.pkl`; these files are not bundled. Built-in short sentences are used by default. Model state and attention heatmaps are saved to `outputs/13_transformer_encoder/`. Attention visualization is not proof of causality.

**Related notes:** [Chapter 12](../../notes/12_transformer_attention.md), [Chapter 13](../../notes/13_llm_foundations.md).
