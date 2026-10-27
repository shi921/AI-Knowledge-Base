**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# Prompt Engineering 与 LoRA/PEFT

本地提示模板练习：

```bash
python projects/14_llm_prompting_and_lora/prompting.py "I really like this movie"
```

微调需要额外安装 `requirements-finetuning.txt`，并获取与你的 Adapter 匹配且遵守许可的基座模型。公共仓库**不提供预训练模型、Adapter 权重、Tokenizer 或训练状态**：

```bash
python projects/14_llm_prompting_and_lora/lora_pipeline.py --mode train --train-limit 60
python projects/14_llm_prompting_and_lora/lora_pipeline.py --mode infer --adapter /path/to/adapter --text "I love cats"
```

上述训练命令会下载模型与数据集，可能需要大量内存、网络与 GPU，并可能产生费用。Adapter 不是完整模型，必须配套正确的 Base Model。输出写在被忽略的 `outputs/` 中。

**相关知识笔记：** [第 13 章](../../notes/13_llm_foundations.zh-CN.md)、[第 14 章](../../notes/14_llm_prompting_finetuning_lora.zh-CN.md)、[第 15 章](../../notes/15_llm_langchain_rag.zh-CN.md)。
