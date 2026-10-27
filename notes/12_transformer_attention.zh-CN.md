**语言：** [English](12_transformer_attention.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 12｜Transformer 架构：缩放点积自注意力（Self-Attention）与编码器

> **🧭 知识脉络定位**：第四篇「深度学习与自然语言处理演进」· **第 12 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：第 11 章的 LSTM 像“单列传话游戏”，第 1 个词要传话给第 100 个词必须挨个过手 99 次，既慢又容易传丢。2017 年诞生的 **Transformer 自注意力机制（Self-Attention）则像一场“全员圆桌会议”**——句子里的每个词都同时举起三张牌：  
>   - **Query（`Q`，查询牌：我现在想找什么信息？比如代词“它”想找自己指代谁）**  
>   - **Key（`K`，索引牌：我身上有什么特征标签？比如名词“小狗”亮出名词属性）**  
>   - **Value（`V`，内容牌：我实际携带的语义干货是什么？）**  
>   全句所有词同时用自己的 `Q` 去和所有词的 `K` 对暗号（算矩阵内积相似度），不管隔了多远都能一步到位直接提取对方的 `V`！  
> - **🎯 这一章解决什么核心问题？**：这是现代人工智能最重要的一块拼图！彻底搞懂 `Q/K/V` 缩放点积注意力公式、多头注意力（Multi-Head）、位置编码（Positional Encoding）、残差层归一化以及 Padding Mask 与 Causal Mask 的区别。  
> - **🔗 前后衔接**：[11｜神经语言模型与 LSTM](11_neural_language_models.zh-CN.md) → 本章（掌握 Transformer 与自注意力底层架构）→ **下一章**：[13｜大语言模型（LLM）基础：架构演进、预训练、对齐与解码生成](13_llm_foundations.zh-CN.md)（看看科学家如何把本章的 Transformer 堆叠扩展成改变世界的 GPT、Claude、Qwen 等大语言模型！）。

---

## 一、 缩放点积注意力（Scaled Dot-Product Attention）与张量形状

给定输入序列表示 $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$（$B$ 为批大小，$L$ 为序列长度 Token 数，$d_{\text{model}}$ 为隐藏维度），首先通过三个可学习线性层投影出查询、键、值矩阵：
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

**核心注意力计算公式**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{Mask}\right) V$$

1. **张量形状变化**：
   - 多头拆分后，$Q, K$ 形状为 `[B, H, L, d_k]`，$V$ 形状为 `[B, H, L, d_v]`（其中 $H$ 是注意力头数，$d_k = d_{\text{model}} / H$）。
   - 注意力得分矩阵 $Q K^T$ 形状为 **`[B, H, L, L]`** —— 直观记录了序列中第 $i$ 个词对第 $j$ 个词的关注程度！
2. **为什么要除以 $\sqrt{d_k}$（Scaling）？**
   - 当向量维度 $d_k$ 较大时，两个随机向量的点积方差会随 $d_k$ 线性增大，导致点积数值过大，经过 `softmax` 后变成接近 `1` 和 `0` 的极端尖锐分布，处于梯度几乎为 `0` 的饱和区（梯度消失）。除以 $\sqrt{d_k}$ 能把点积方差重新拉回 $1$。

```python
import torch
import torch.nn.functional as F

# B=2 个句子, L=5 个词, d_k=32 维特征
q, k, v = [torch.randn(2, 5, 32) for _ in range(3)]
scores = (q @ k.transpose(-2, -1)) / (32 ** 0.5)   # shape: [2, 5, 5]
weights = F.softmax(scores, dim=-1)                # 每行概率和为 1
context = weights @ v                              # shape: [2, 5, 32]
```

---

## 二、 Transformer 的四大关键组件

| 核心组件 | 为什么必不可少？（解决什么问题） |
|---|---|
| **多头注意力 (Multi-Head Attention)** | 把特征拆成 $H$ 个子空间并行算注意力（比如第 1 个头专门看主谓语法搭配，第 2 个头专门看代词指代，第 3 个头看情感修饰），最后拼接并线性投影回 $d_{\text{model}}$ |
| **位置编码 (Positional Encoding)** | 自注意力本身是“位置盲”的（打乱词序算出的集合交互一样）。因此必须在输入词向量上加上正弦余弦位置编码、可学习位置嵌入或现代大模型常用的 **RoPE（旋转位置编码）** |
| **掩码机制 (Masking)** | 将不该关注的位置加上 $-\infty$（过 Softmax 后权重变为 `0`）：<br/>① **Padding Mask**：屏蔽为对齐批次长度而补齐的 `<pad>` 空白符；<br/>② **Causal Mask（因果上三角掩码）**：在 GPT 等生成模型中禁止第 $t$ 个词偷看 $t+1$ 之后的未来词 |
| **残差连接 + 层归一化 + FFN** | `x = LayerNorm(x + MultiHeadAttn(x))` 与 `x = LayerNorm(x + FFN(x))`：残差连接（跳过连接）让梯度能无损直通底层，使堆叠几十上百层深层网络成为可能 |

---

## 三、 本章实战：手写 Transformer Encoder 与注意力热图可视化

配套项目 [`projects/13_transformer_encoder/main.py`](../projects/13_transformer_encoder/main.py) 使用纯 PyTorch 搭建了一个完整处理 `<pad>` 掩码与 `<cls>` 汇总 Token 的小型 Transformer Encoder：
- 在序列开头拼接特殊标记 `<cls>`，经过多层自注意力交互后，取 `<cls>` 位置的输出向量送入线性分类头完成情感分类；
- 自动提取句子的自注意力权重矩阵 `[L, L]` 并绘制保存为热图 `outputs/13_transformer_encoder/attention_heatmap.png`。
- **科学警示**：注意力热图反映了模型内部各 Token 向量加权混合的比例，但**不能直接等同于第 05 章的严格特征因果解释**。

---

## 四、 配套实践与自测

- **可运行项目**：[`projects/13_transformer_encoder/main.py`](../projects/13_transformer_encoder/main.py)（零网络依赖直接运行；亦可通过 `--data-dir` 加载自定义情感语料）。
- **思考与自测**：
  1. 为什么 $Q K^T$ 必须除以 $\sqrt{d_k}$？如果不除，当 $d_k$ 很大时 Softmax 的梯度会怎样？
  2. 如果在批处理训练时漏掉了 Padding Mask，模型会把大段 `<pad>` 填充符当成什么？
  3. 编码器（Encoder，如 BERT）的双向注意力掩码与解码器（Causal Decoder，如 GPT）的下三角掩码有什么本质区别？
