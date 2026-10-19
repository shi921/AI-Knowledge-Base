**Language:** English | [简体中文](README.zh-CN.md) · [Home](../README.md)

# Example datasets

- `housing/`: **programmatically generated synthetic housing tables** for data profiling and regression. Sale prices are illustrative only.
- `stagger/` and `sea/`: **synthetic streams with changing labeling rules**, for incremental learning and drift detection.
- `documents/`: three short, originally written AI articles for offline TF-IDF retrieval.
- Iris and Digits classification load examples with `sklearn.datasets`; this repository does not redistribute those files.

Run `python scripts/generate_demo_data.py` to regenerate the CSVs. Face images, personal records, WikiText files, and model weights are not bundled; acquire licensed copies independently and keep them in ignored local paths.
