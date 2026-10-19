**语言：** [English](12_neural_language_models.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 12｜神经语言模型：Neural N-gram、LSTM、生成与词表

## 知识脉络

统计 n-gram 使用频次表；**Neural N-gram** 用 Embedding 层把上下文 token 映射为向量，拼接后通过 MLP 输出词表 logits，训练目标通常是下一个词的交叉熵。词表 `token → index` 只能用训练集构建，未见词使用 `<unk>`；对测试集禁止重新建词表。

LSTM 是循环模型：`hₜ,cₜ = LSTM(xₜ,hₜ₋₁,cₜ₋₁)`，通过门控记住历史。推理中模型逐 token 生成：先把整段 prompt 输入一次得到最后一个隐藏状态；之后只传入**新预测**的 token，与隐藏状态一同递推；遇到 `<eos>` 或达到最大 token 数停止。采样策略包括贪心、温度缩放、top-k/top-p，但不能把生成的流畅程度直接当作事实准确性。

## 复习一个前向过程

```python
# x: [batch, context_length], IDs in [0, vocab_size)
# embeddings: [batch, context_length, embedding_dim]
# flatten: [batch, context_length * embedding_dim]
# output_logits: [batch, vocab_size]
# CrossEntropyLoss(output_logits, y_next_token)
```

## 数据/模型缺失与可运行部分

**仓库不包含** `sample_text.txt`、WikiText-103 数据、`vocab.pkl`、`lstm_model_weights.pkl`。目录已在 `projects/12_neural_language_models/data/`、`data/wikitext-103/`、`data/wikitext-2/` 和 `pretrained_model/` 中预留。映射细节看 [`projects/12_neural_language_models/ASSETS.md`](../projects/12_neural_language_models/ASSETS.md)。

- 无需外部文件：运行 `python projects/12_neural_language_models/ngram_train.py`（小语料训练，可理解词表、Embedding、DataLoader、PPL）。
- 补齐词表和权重：运行 `python projects/12_neural_language_models/lstm_inference.py`；并注意历史 `torchtext` 与 pickle 兼容性及模型内存开销。**没有这些文件时不会伪称能复现预训练模型输出**。

**联系前序**：回看 11 的 n-gram/PPL；再进入 13 理解 Transformer 如何取代循环结构。

**自测**：为什么 LSTM 推理需要携带 hidden/cell state？为什么不应该把同一个 prompt 末尾 token 连续输入两次？
