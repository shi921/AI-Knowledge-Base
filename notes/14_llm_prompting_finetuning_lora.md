**Language:** English | [简体中文](14_llm_prompting_finetuning_lora.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 14 | LLM Prompting, Fine-tuning, LoRA, and PEFT

## Prompting versus retrieval versus tuning

**Zero-shot prompting** defines the task without examples; **few-shot prompting** includes demonstrations to clarify formats. Asking a model to check an answer may improve some outputs but does not guarantee correctness. Require explicit structured formats and validate the parsed result separately.

**RAG** retrieves documents at runtime and provides the passages in the prompt without changing model weights. **Fine-tuning** updates some or all model parameters and is more suitable for learning behaviors, formats, or task style than for keeping a rapidly changing knowledge base up to date.

| Technique | Trained model parameters | External information | Typical application |
|---|---|---|---|
| Prompting | None | Examples inside prompt | Fast experimentation and output constraints |
| RAG | None | Retrieved at inference time | Changing documents and grounded answers |
| Full fine-tuning | All parameters | Training dataset | Well-defined goals with sufficient resources |
| LoRA / PEFT | Low-rank adapter parameters | Training dataset | Resource-efficient task/style adaptation |

## LoRA mechanics

Freeze a pretrained linear weight matrix `W` and learn a low-rank update `ΔW = BA` with rank `r` much smaller than the full dimensions. This reduces trainable parameter count. An **adapter checkpoint is not a stand-alone model**: the base model, tokenizer, targeted layers, and versions must match.

Training data generally combines instruction, user input, and desired response. Response-only loss training commonly masks instruction/prompt token labels with `-100`; the included short demonstration instead trains on the full causal-LM sequence and is intentionally simplified.

## Project entry points

- Local templates: `python projects/14_llm_prompting_and_lora/prompting.py` (no API required by default).
- Put appropriately licensed compatible adapter weights under `projects/14_llm_prompting_and_lora/checkpoint/`; the empty folder is preserved.
- Inference: `python projects/14_llm_prompting_and_lora/lora_pipeline.py --mode infer` requires Transformers/PEFT, a matching Qwen2.5-0.5B-Instruct base model download, and sufficient memory.
- Training: `--mode train` downloads the `dvgodoy/yoda_sentences` dataset and writes an adapter to `outputs/14_lora_training/`, without overwriting the checkpoint placeholder.

## Safety and engineering

Never hardcode API keys; use environment variables. Consider prompt injection, untrusted third-party data, bias, privacy, and human review. Do not forward user comments to an external API without authorization.

**Review questions:** Is RAG or LoRA more suitable for company policies that change daily? Why can `.safetensors` adapter files not be deployed without a compatible base model?
