**Language:** English | [简体中文](12_transformer_attention.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 12 | Transformer Architecture: Scaled Dot-Product Attention and Encoders

> **🧭 Knowledge Map Position**: Part IV (Deep Learning & NLP Evolution) · **Chapter 12 of 15**  
> - **💡 Intuitive Analogy**: An LSTM (Chapter 11) is like a single-file game of telephone—word #1 can only reach word #100 by passing through 99 sequential steps. **A Transformer's Self-Attention is like an open roundtable meeting**: every token simultaneously holds up three cards:  
>   - **Query (`Q`, what information I am looking for)**  
>   - **Key (`K`, what label/attribute I advertise)**  
>   - **Value (`V`, the semantic content I carry)**  
>   Every token matches its `Q` against every other token's `K` in parallel via matrix multiplication, instantly retrieving relevant `V` content across any distance!  
> - **🎯 Why Does This Matter?**: Introduced in *Attention Is All You Need* (2017), the Transformer replaced recurrence with parallel self-attention—becoming the universal backbone of modern AI.  
> - **🔗 Learning Chain**: [11 | Neural Language Models & LSTMs](11_neural_language_models.md) → This Chapter (Transformer & Self-Attention mechanics) → **Next Chapter**: [13 | Large Language Model (LLM) Foundations: Architectures, Pre-training, Alignment, and Decoding](13_llm_foundations.md) (See how stacking Transformer blocks at scale creates modern LLMs like GPT, Claude, Llama, and Qwen!).

---

## 1. Scaled Dot-Product Attention & Tensor Shapes

Given token representations $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$, linear projections produce $Q = XW_Q$, $K = XW_K$, and $V = XW_V$. **Scaled Dot-Product Attention** computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{Mask}\right) V$$

- **Shapes**: With $H$ heads, $Q$ and $K$ have shape `[B, H, L, d_k]`, $V$ has shape `[B, H, L, d_v]`, and the attention score matrix $Q K^T$ has shape **`[B, H, L, L]`** (capturing all pairwise token interactions).
- **Why Divide by $\sqrt{d_k}$?**: As $d_k$ grows, dot products grow in variance, pushing `softmax` into flat saturation regions with near-zero gradients. Scaling by $\sqrt{d_k}$ keeps unit variance.

```python
import torch
import torch.nn.functional as F

q, k, v = [torch.randn(2, 5, 32) for _ in range(3)]
score = q @ k.transpose(-2, -1) / (32 ** 0.5)   # [2, 5, 5]
weight = F.softmax(score, dim=-1)
context = weight @ v                            # [2, 5, 32]
```

---

## 2. Four Pillars of a Transformer Block

| Component | Purpose |
|---|---|
| **Multi-Head Attention** | Splits embeddings into $H$ subspaces so different heads attend to syntax, coreference, or sentiment in parallel before concatenating back to $d_{\text{model}}$ |
| **Positional Encoding** | Injects token order information (sinusoidal, learned embeddings, or modern Rotary Position Embeddings **RoPE**) because raw self-attention is permutation-invariant |
| **Masking** | Adds $-\infty$ before `softmax` (zeroing weights) for:<br/>① **Padding Mask**: ignoring `<pad>` tokens;<br/>② **Causal Mask**: preventing autoregressive decoders from peeking at future tokens |
| **Residual Connections + LayerNorm + FFN** | `LayerNorm(x + SubLayer(x))` allows gradients to flow unimpeded through dozens of stacked layers |

---

## 3. Hands-on Practice: Transformer Encoder & Attention Heatmaps

[`projects/13_transformer_encoder/main.py`](../projects/13_transformer_encoder/main.py) builds a compact PyTorch Self-Attention Encoder with proper padding masks and a `<cls>` pooling token for sentiment classification, exporting an attention heatmap to `outputs/13_transformer_encoder/attention_heatmap.png`. Remember: **attention weights show token mixing ratios, not guaranteed causal explanations**.

---

## 4. Hands-on Projects & Self-Test

- **Runnable Project**: [`projects/13_transformer_encoder/main.py`](../projects/13_transformer_encoder/main.py).
- **Self-Test**:
  1. Why do we divide $Q K^T$ by $\sqrt{d_k}$ before applying `softmax`?
  2. What happens to sentence representations if you omit the padding mask on variable-length batches?
  3. How does a bidirectional Encoder mask differ from an autoregressive Decoder causal mask?
