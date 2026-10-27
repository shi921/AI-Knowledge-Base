**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# Prompt Engineering and LoRA/PEFT

Try the local prompt templates:

```bash
python projects/14_llm_prompting_and_lora/prompting.py "I really like this movie"
```

For fine-tuning, install `requirements-finetuning.txt` and obtain an appropriately licensed base model that matches your adapter. **No pretrained base model, adapter weights, tokenizer, or training state is distributed**:

```bash
python projects/14_llm_prompting_and_lora/lora_pipeline.py --mode train --train-limit 60
python projects/14_llm_prompting_and_lora/lora_pipeline.py --mode infer --adapter /path/to/adapter --text "I love cats"
```

Training downloads a model and dataset and may require substantial memory, a GPU, network access, or paid resources. A LoRA adapter is not a complete model; use the correct base model. Generated artifacts belong in the ignored `outputs/` directory.

**Related notes:** [Chapter 13](../../notes/13_llm_foundations.md), [Chapter 14](../../notes/14_llm_prompting_finetuning_lora.md), [Chapter 15](../../notes/15_llm_langchain_rag.md).
