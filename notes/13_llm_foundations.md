**Language:** English | [简体中文](13_llm_foundations.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 13 | Large Language Model (LLM) Foundations: Architectures, Pre-training, Alignment, and Decoding

> **🧭 Knowledge Map Position**: Part V (LLMs, Fine-Tuning & RAG/Agents) · **Chapter 13 of 15**  
> - **💡 Intuitive Analogy**: In Chapter 12 you learned how a single Transformer engine works. How do we scale that engine into a world-class assistant like ChatGPT, Claude, Llama, or Qwen? Think of **a three-stage education for a brilliant generalist**:  
>   1. **Pre-training = Reading the World's Library**: The model reads trillions of words of books, articles, and code, practicing one universal game—*predicting the next token*—to absorb grammar, world knowledge, and reasoning (producing a **Base Model** that completes text but doesn't yet know how to act as a polite assistant);  
>   2. **Supervised Fine-Tuning (SFT) = Job Training with Examples**: We train it on hundreds of thousands of high-quality `<User Instruction, Assistant Response>` pairs so it learns to follow instructions rather than just continuing the question (producing an **Instruct / Chat Model**);  
>   3. **Preference Alignment (RLHF / DPO) = Professional Ethics & Polish**: Human raters compare responses so the model learns to be **Helpful, Honest, and Harmless**.  
> - **🎯 Why Is This Chapter Essential?**: It is the **missing bridge** between low-level neural architectures (Chapters 08–12) and modern LLM engineering (Chapters 14–15). Here you will master why Decoder-only Transformers dominate, how subword Tokenization and KV Caching work, how `Temperature` and `Top-p` control generation, and why LLMs hallucinate!  
> - **🔗 Learning Chain**: [12 | Transformer Architecture & Self-Attention](12_transformer_attention.md) → This Chapter (LLM Foundations) → **Next Chapters**: [14 | LLM Customization: Prompting & LoRA Fine-Tuning](14_llm_prompting_finetuning_lora.md) and [15 | LLM Engineering: LangChain, RAG, and Agents](15_llm_langchain_rag.md).

---

## 1. Three Transformer Families & Why Decoder-Only Won

| Architecture Family | Representative Models | Attention Pattern | Primary Role in Modern AI |
|---|---|---|---|
| **1. Encoder-Only<br/>(Bidirectional Understanding)** | BERT, RoBERTa,<br/>**BGE / E5 Embeddings** | **Bidirectional self-attention** (every token sees both left and right context) | Ideal for text classification (Chapter 12), named-entity recognition, and **dense semantic Embedding / Reranker models in RAG (Chapter 15)** |
| **2. Encoder-Decoder<br/>(Sequence-to-Sequence)** | Original Transformer,<br/>T5, BART, Whisper | Bidirectional Encoder + Causal Decoder with Cross-Attention | Structured sequence transduction (translation, speech-to-text) |
| **3. Decoder-Only<br/>(Autoregressive Generation)** | **GPT-4, Claude, Llama,<br/>Qwen, DeepSeek** | **Causal Mask**: token $t$ can only attend to tokens $\le t$ (never peeking at future tokens) | **Dominant architecture of modern LLMs!** Unifies all tasks (QA, coding, summarization, reasoning) into **Next-Token Prediction** |

### Why Did Decoder-Only Become the Standard for LLMs?
1. **Maximum Training Efficiency**: Using a lower-triangular causal mask, a single forward pass over a sequence of length $L$ computes next-token loss at all $L$ positions simultaneously.
2. **In-Context Learning & Scaling Laws**: Because every task can be framed as predicting continuation tokens given a context prefix, scaling parameters and training tokens unlocks emergent zero-shot and few-shot reasoning.

---

## 2. How LLMs Read and Compute: Subword Tokenization & KV Cache

### 1. Subword Tokenization (BPE / Byte-Pair Encoding)
Splitting text by whitespace (Chapter 11) creates huge vocabularies and fails on unseen words (`<unk>`), while splitting by single characters makes sequences too long. Modern LLMs use **subword tokenizers (like BPE)**:
- Frequent words remain single tokens (`"apple"`), rare words split into meaningful morphemes (`["un", "happi", "ness"]`), and unknown characters fall back to UTF-8 bytes—**eliminating `<unk>` out-of-vocabulary errors**.
- **Rule of Thumb**: In English, $1 \text{ token} \approx 0.75 \text{ words}$. Both API pricing and **Context Windows** (e.g., 8K–128K tokens) are measured in tokens.

### 2. Prefill vs. Decode & KV Cache
Autoregressive generation happens in two distinct hardware phases:
1. **Prefill Phase (Compute-bound)**: Processes all prompt tokens in parallel to compute their $Q, K, V$ states and emit the first output token.
2. **Decode Phase (Memory-bandwidth-bound)**: Generates subsequent tokens one by one. Because causal masking guarantees that past tokens' Key ($K$) and Value ($V$) vectors never change as new tokens are appended, engines store past $K$ and $V$ tensors in GPU memory (**KV Cache**) so each new step only computes $q_{\text{new}}, k_{\text{new}}, v_{\text{new}}$—trading memory for a massive speedup!

---

## 3. The Three-Stage LLM Training Lifecycle

```mermaid
flowchart LR
    A["Trillions of Raw Web/Book/Code Tokens"] -->|Stage 1: Self-Supervised Pre-training<br/>Next-Token Prediction| B["Base Model<br/>(Document Completion Engine)"]
    B -->|Stage 2: Supervised Fine-Tuning (SFT)<br/>Instruction-Response Pairs + LoRA/Full| C["Instruct / Chat Model<br/>(Instruction-Following Assistant)"]
    C -->|Stage 3: Preference Alignment<br/>RLHF (PPO) / DPO / GRPO| D["Aligned Production LLM<br/>(Helpful, Honest, Harmless)"]
```

| Stage | Dataset Format | Training Objective | Resulting Model Behavior |
|---|---|---|---|
| **1. Pre-training** | Trillions of raw unlabelled tokens | Causal LM cross-entropy loss across all tokens | **Base Model** (e.g., `Qwen2.5-7B`): absorbs world knowledge and syntax, but merely continues text rather than answering questions directly |
| **2. Supervised Fine-Tuning (SFT)** | Curated `<System, User Instruction, Assistant Response>` demonstrations | Cross-entropy loss computed **on Assistant response tokens only** (prompt tokens masked with `-100`) | **Instruct / Chat Model** (e.g., `Qwen2.5-7B-Instruct`): follows instructions and formats (**Chapter 14 LoRA fine-tuning operates here!**) |
| **3. Alignment (RLHF / DPO)** | Preference pairs `(Prompt, Chosen, Rejected)` or verifiable rewards | **RLHF** optimizes a reward model; **DPO** directly increases the log-probability margin of Chosen over Rejected | Aligns style with human preferences, reduces toxic/unsafe outputs, and strengthens multi-step reasoning |

---

## 4. Controlling Generation: Decoding & Sampling Strategies

At each step, the LLM outputs raw **logits $z \in \mathbb{R}^{|V|}$** across the entire vocabulary. How do we pick the next token?

1. **Greedy Decoding (`do_sample=False` or `temperature=0`)**: Always picks $\arg\max P(w)$. Deterministic and ideal for code generation, math, and structured extraction, though prone to repetitive loops in open-ended writing.
2. **Temperature Scaling ($T > 0$)**: Divides logits by $T$ before softmax: $P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$.
   - **Low $T$ (`0.1–0.3`)**: Sharpens the distribution, making outputs focused and conservative.
   - **High $T$ (`0.8–1.2`)**: Flattens the distribution, increasing creativity at the risk of incoherence.
3. **Top-k & Top-p (Nucleus Sampling)**: Truncates the unreliable long tail before sampling:
   - **Top-k**: Keeps only the $k$ highest-probability tokens.
   - **Top-p (`0.9`)**: Sorts tokens by descending probability and keeps the smallest prefix whose cumulative probability exceeds $p$, dynamically adapting candidate pool size!

```python
import torch
import torch.nn.functional as F

def sample_next_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 0, top_p: float = 0.9) -> int:
    """Reference implementation of Greedy, Temperature, Top-k, and Top-p decoding."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    scaled = logits / temperature
    if top_k > 0:
        top_vals, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        scaled = torch.where(scaled < top_vals[-1], torch.tensor(float('-inf')), scaled)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float('-inf')
        scaled = torch.zeros_like(scaled).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
```

---

## 5. Four Inherent Limitations of LLMs & How Chapters 14–15 Solve Them

| Inherent LLM Limitation | Root Cause | Engineering Solution in Next Chapters |
|---|---|---|
| **1. Ambiguous Task Formats** | Models predict plausible continuations; vague instructions yield inconsistent formats | ➡️ **[Chapter 14: Prompt Engineering](14_llm_prompting_finetuning_lora.md)** (Zero/Few-shot, CoT, schema validation) |
| **2. Domain Style & Task Adaptation Cost** | Full fine-tuning of billions of parameters requires massive GPU memory | ➡️ **[Chapter 14: PEFT / LoRA](14_llm_prompting_finetuning_lora.md)** (Train low-rank adapters $\Delta W = BA$ on <1% of weights) |
| **3. Knowledge Cutoff & Hallucinations** | Weights are frozen at training time and lack private facts; models guess fluent-sounding tokens | ➡️ **[Chapter 15: Retrieval-Augmented Generation (RAG)](15_llm_langchain_rag.md)** (Inject retrieved evidence with citations) |
| **4. No Direct Action or Exact Computation** | Next-token predictors cannot query live databases or execute code on their own | ➡️ **[Chapter 15: Tool Calling & Agents](15_llm_langchain_rag.md)** (Bind external tools in a reasoning loop) |

---

## 6. Hands-on Connections & Self-Test

- **Code Connections**: Inspect how [`projects/14_llm_prompting_and_lora/lora_pipeline.py`](../projects/14_llm_prompting_and_lora/lora_pipeline.py) loads `AutoTokenizer` and `AutoModelForCausalLM` (`Qwen/Qwen2.5-0.5B-Instruct`) and runs deterministic generation (`do_sample=False`).
- **Self-Test**:
  1. Why are Encoder-only models (like BERT/BGE) preferred for document Embeddings while Decoder-only models (like GPT/Qwen) dominate text generation?
  2. What is the difference between a **Base** model and an **Instruct** model?
  3. Should you set `temperature` near `0` or near `1.0` when generating structured JSON or SQL queries? Why?
