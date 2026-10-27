**Language:** English | [简体中文](14_llm_prompting_finetuning_lora.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 14 | LLM Customization: Prompt Engineering, Fine-Tuning Tradeoffs, and LoRA / PEFT

> **🧭 Knowledge Map Position**: Part V (LLMs, Fine-Tuning & RAG/Agents) · **Chapter 14 of 15**  
> - **💡 Intuitive Analogy**: In Chapter 13 we saw how pre-trained LLMs are built. Now, how do you customize a general-purpose LLM for your domain?  
>   1. **Prompt Engineering (Communication Skills)**: Leave model weights untouched and guide behavior via clear instructions (**Zero-shot**), input-output demonstrations (**Few-shot In-Context Learning**), and step-by-step reasoning prompts;  
>   2. **LoRA / PEFT Fine-Tuning (Plug-in Skill Adapters)**: When you need to change the model's intrinsic style, tone, or task habits, updating all billions of weights (Full Fine-Tuning) is expensive and risks catastrophic forgetting. **LoRA freezes the original base model and attaches tiny trainable low-rank matrices ($\Delta W = BA$)**—training <1% of parameters on a single GPU!  
> - **🎯 Why Does This Matter?**: It builds architectural judgment (**when to use Prompting vs. RAG vs. Fine-Tuning**) and walks through training and loading HuggingFace `peft` LoRA adapters.  
> - **🔗 Learning Chain**: [13 | Large Language Model (LLM) Foundations](13_llm_foundations.md) → This Chapter (Prompting & LoRA parameter adaptation) → **Next Chapter (Capstone)**: [15 | LLM Application Engineering: LangChain, RAG, and Agents](15_llm_langchain_rag.md).

---

## 1. Prompt Engineering Patterns

1. **System Role & Constraints**: Define persona, boundaries, and explicit output schemas (e.g., JSON).
2. **Zero-shot vs. Few-shot Prompting**: Zero-shot provides task instructions only; Few-shot includes $2\text{–}5$ input-output demonstrations in the context window to resolve format ambiguity.
3. **Chain-of-Thought (CoT) & Self-Verification**: Prompting the model to reason step by step or verify its draft improves complex reasoning, though **downstream code must still independently validate parsed outputs**.

---

## 2. Architectural Decision Matrix: Prompting vs. RAG vs. Fine-Tuning

| Technique | Modifies Weights? | Where Knowledge Lives | Best Suited For | Anti-Patterns (When NOT to Use) |
|---|---|---|---|---|
| **Prompting** | No (0%) | Inside the prompt context | **First resort**: rapid prototyping, format constraints, few-shot guidance | Stuffing thousands of changing documents into every prompt |
| **RAG<br/>(Chapter 15)** | No (0%) | External document index / vector store retrieved at runtime | **Frequently changing facts**, private documents, and **strict source citations** to curb hallucinations | Teaching a completely new speaking style or compact task skill |
| **Full Fine-Tuning** | Yes (100%) | Baked into all model weights | Large compute budget and massive domain corpus | Small datasets (overfitting / catastrophic forgetting) or volatile facts |
| **LoRA / PEFT** | Trains **<1% adapter weights** (base frozen) | Lightweight adapter files (MBs) paired with base model | **Adapting tone, style, structured output habits**, or domain tasks on modest hardware | Using LoRA to memorize daily-changing company policies, or deploying adapters without their matching base model |

---

## 3. How LoRA Works: Low-Rank Matrix Decomposition

Full fine-tuning requires storing gradients and Adam optimizer states for every parameter (often $3\text{–}4\times$ the memory of the model weights). **LoRA (Low-Rank Adaptation)** freezes the pre-trained linear layer $W_0 \in \mathbb{R}^{d \times k}$ and injects a low-rank update $\Delta W = B A$ with rank $r \ll \min(d, k)$:

$$h = W_0 x + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times k}, \; B \in \mathbb{R}^{d \times r}$$

- **Initialization Trick**: $A$ is initialized with Gaussian noise and **$B$ is initialized to zeros** so that $\Delta W = 0$ at the start of training, preserving base model behavior initially.
- **Key Hyperparameters**: `r` (rank, e.g., `8, 16`), `lora_alpha` (scaling factor), and `target_modules` (e.g., `["q_proj", "v_proj"]`).
- **Deployment Rule**: A saved LoRA checkpoint (`adapter_model.safetensors`) **is not a standalone model**—it must be loaded on top of the exact compatible base model and tokenizer.

---

## 4. Hands-on Projects & Self-Test

- **Runnable Scripts**:
  - Offline prompt templates: `python projects/14_llm_prompting_and_lora/prompting.py "I really like this movie"`.
  - LoRA training & inference (`Qwen/Qwen2.5-0.5B-Instruct`): [`projects/14_llm_prompting_and_lora/lora_pipeline.py`](../projects/14_llm_prompting_and_lora/lora_pipeline.py) (`--mode train` or `--mode infer`).
- **Self-Test**:
  1. Should you use **RAG** or **LoRA** when company policies change every day? Why?
  2. Why is matrix $B$ initialized to zero while $A$ is initialized randomly in LoRA?
  3. Why can't you deploy a `.safetensors` LoRA adapter file without also having access to the base model weights?
