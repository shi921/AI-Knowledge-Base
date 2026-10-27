**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# Transformer Encoder：自注意力与文本分类

```bash
python projects/13_transformer_encoder/main.py --epochs 16
# Optional authorized local pickle data (load only trusted files)
python projects/13_transformer_encoder/main.py --data-dir /path/to/trusted-data --epochs 20
```

数据目录需要 `movie_reviews.pkl` 和 `movie_labels.pkl`；公共仓库不附带这些文件。默认使用内置短句示例，输出模型状态与注意力热图到 `outputs/13_transformer_encoder/`。注意力可视化不能证明因果。

**相关知识笔记：** [第 12 章](../../notes/12_transformer_attention.zh-CN.md)、[第 13 章](../../notes/13_llm_foundations.zh-CN.md)。
