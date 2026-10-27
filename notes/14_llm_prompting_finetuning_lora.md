**Language:** English | [简体中文](14_llm_prompting_finetuning_lora.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 14 | LLM Prompting, Fine-tuning, LoRA, and PEFT

> **🧭 Knowledge Map Position**: Stage 5 (LLM Applications & Fine-Tuning) · **Step 14 of 14 (Capstone)**  
> - **💡 Intuitive Analogy**: To customize a general-purpose LLM for your domain, you have three escalating tools:  
>   1. **Prompting (Zero/Few-shot)**: Give clearer instructions or a couple of input-output examples in the prompt without touching model weights;  
>   2. **RAG (Chapter 06)**: Hand the model an up-to-date reference book at runtime;  
>   3. **LoRA / PEFT Fine-Tuning**: When you must change the model's intrinsic style, tone, or task habits, retraining all billions of weights is prohibitively expensive. **LoRA freezes the original brain and attaches a tiny, trainable "transparent overlay note" (low-rank matrices `ΔW = BA`)**—updating <1% of parameters while achieving near full-tuning behavior!  
> - **🎯 Why Does This Matter?**: It teaches architectural decision-making (knowing when to choose Prompting vs. RAG vs. Fine-Tuning) and walks through training and loading a HuggingFace PEFT/LoRA adapter.  
> - **🔗 Learning Chain**: [06 | LLMs, LangChain, and RAG](06_llm_langchain_rag.md) → This Chapter (Prompt engineering & LoRA fine-tuning) → 🎉 **Congratulations on completing all 5 stages! Return to the [Master Roadmap](../STUDY_ROADMAP.md) for advanced open-source benchmarks.**

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
