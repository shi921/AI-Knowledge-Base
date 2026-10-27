**Language:** English | [简体中文](13_transformer_attention.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 13 | Transformers: Scaled Dot-product Attention and Encoders

> **🧭 Knowledge Map Position**: Stage 4 (Deep Learning & NLP Evolution) · **Step 12 of 14**  
> - **💡 Intuitive Analogy**: An LSTM (Chapter 12) is like a single-file game of telephone—word #1 can only reach word #100 by passing through 99 middlemen. **A Transformer's Self-Attention is like an open roundtable meeting**: every word simultaneously broadcasts three cards—**Query (`Q`, what I am looking for)**, **Key (`K`, what label I advertise)**, and **Value (`V`, the content I carry)**. Every word directly matches its `Q` against every other word's `K` in parallel, regardless of distance!  
> - **🎯 Why Does This Matter?**: This is the architectural heart of modern AI. Every major Large Language Model (BERT, GPT, Claude, Qwen, Llama) is built by stacking the exact Self-Attention, LayerNorm, Positional Encoding, and Feed-Forward blocks covered here.  
> - **🔗 Learning Chain**: [12 | Neural Language Models & LSTMs](12_neural_language_models.md) → This Chapter (Master the Transformer & Self-Attention architecture) → **Next Step**: [06 | LLMs, LangChain, and RAG](06_llm_langchain_rag.md) (How do we turn pre-trained Transformer LLMs into grounded applications and Agents?).

## Attention formula and tensor shapes

Project each token representation into `Q=XW_Q`, `K=XW_K`, and `V=XW_V`. **Scaled dot-product attention** is:

`Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k) + mask) V`.

Q and K typically have shape `[B,H,L,d_k]`, V has `[B,H,L,d_v]`, and attention scores have shape `[B,H,L,L]`. Dividing by `sqrt(d_k)` prevents large dot products from saturating softmax as dimensions grow. Masks suppress padding or disallow attending to future tokens.

- **Self-attention:** Q/K/V come from the same sequence. **Cross-attention:** decoder queries attend to encoder keys and values.
- **Multi-head attention:** multiple heads can learn distinct relationships; concatenate their outputs and project to `d_model`.
- **Positional information:** plain self-attention has no inherent word order. Sinusoidal or learned positional encodings provide position signals.
- **Encoder block:** attention → residual + normalization → feed-forward network → residual + normalization. A CLS representation can summarize a sequence for classification.

```python
import torch
import torch.nn.functional as F
q, k, v = [torch.randn(2, 5, 32) for _ in range(3)]
score = q @ k.transpose(-2, -1) / (32 ** 0.5)
weight = F.softmax(score, dim=-1)
context = weight @ v  # [2, 5, 32]
```

## Practical training and limits

[`projects/13_transformer_encoder/main.py`](../projects/13_transformer_encoder/main.py) trains a small padding-aware self-attention encoder for offline sentiment classification and renders attention heatmaps. Supply authorized `movie_reviews.pkl` and `movie_labels.pkl` using `--data-dir` if desired (**load trusted pickle only**).

Pretrained BERT or translation-model attention requires separately licensed models, dependencies, and often network access. **Attention plots are not causal explanations.**

**Review questions:** Why divide by `sqrt(d_k)`? What happens if a padding mask is missing? How does an encoder mask differ from a causal decoder mask?
