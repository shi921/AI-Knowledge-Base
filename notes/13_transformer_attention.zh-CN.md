**语言：** [English](13_transformer_attention.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 13｜Transformer：Scaled Dot-Product Attention 与 Encoder

## 注意力公式与张量形状

对每个 token 向量做线性变换得到 `Q=XW_Q, K=XW_K, V=XW_V`。**Scaled Dot-Product Attention**：

`Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k) + mask) V`

Q/K 形状 `[B,H,L,d_k]`，V 为 `[B,H,L,d_v]`，得分矩阵 `[B,H,L,L]`。`sqrt(d_k)` 缓解维度变大导致 softmax 过于尖锐的问题。mask 用于避免关注 padding 或禁止自回归模型关注未来 token。

- **Self-attention**：同一序列产生 Q/K/V；**cross-attention**：decoder 的 Q，encoder 的 K/V。
- **Multi-head**：不同头可学习不同关系，再 concat 并线性投影回 `d_model`。
- **Positional Encoding**：没有位置信息时 self-attention 不知道输入顺序；常见方法包括 sinusoidal 编码，实际也常用可学习位置嵌入。
- **Encoder**：attention → 残差+Norm → FFN → 残差+Norm；CLS token 表示整条句子的分类信号。

```python
import torch
import torch.nn.functional as F
q,k,v = [torch.randn(2, 5, 32) for _ in range(3)]
score = q @ k.transpose(-2, -1) / (32 ** 0.5)
weight = F.softmax(score, dim=-1)
context = weight @ v   # [2, 5, 32]
```

## 训练实践与局限

本章项目 [`projects/13_transformer_encoder/main.py`](../projects/13_transformer_encoder/main.py) 使用一个正确处理 padding 的小型 self-attention Encoder，做离线情感分类并生成注意力热图。可通过 `--data-dir` 指定自有的 `movie_reviews.pkl` 和 `movie_labels.pkl`（**仅加载可信文件**）。

若需要观察预训练 BERT 或翻译模型的注意力，可以另外加载拥有使用权限的模型；这通常需要额外依赖与网络。需要记住：**attention 可视化不能代替因果解释**。

**自测**：为什么 QKᵀ 需要除以 `sqrt(d_k)`？padding mask 漏掉会怎样？encoder 与 causal decoder 的注意力 mask 有何不同？
