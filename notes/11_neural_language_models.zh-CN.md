**语言：** [English](11_neural_language_models.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 11｜神经语言模型：词嵌入（Embedding）、LSTM 与序列生成

> **🧭 知识脉络定位**：第四篇「深度学习与自然语言处理演进」· **第 11 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：第 09 章的统计 N-gram 就像死记硬背一本厚字典：如果字典里只记过“小狗在公园跑”，没记过“小猫在花园走”，它就完全不会预测了。本章将第 08 章的神经网络与第 09 章的语言模型结合起来：  
>   1. **Neural N-gram（神经 N-gram）**：用 **Embedding 层（可学习的语义坐标表）** 把单词编号变成稠密向量——因为“猫”和“狗”向量很接近，学会了“狗在跑”就能自动泛化到“猫在跑”！  
>   2. **LSTM（长短期记忆网络）**：不再只看固定前 $n$ 个词，而是像一位**手捧笔记本按顺序读书的读者**，每读一个词就通过“遗忘门、输入门、输出门”更新手里的笔记状态 $(h_t, c_t)$，再根据笔记一个词接一个词地写出新句子。  
> - **🎯 这一章解决什么核心问题？**：打通从“离散单词 ID”到“神经网络稠密词嵌入”的桥梁，理解循环神经网络（RNN/LSTM）如何建模序列记忆、如何进行自回归逐词生成，以及**为什么循环网络最终遇到了无法并行和长文遗忘的瓶颈**。  
> - **🔗 前后衔接**：[10｜多视角文本分类与 Stacking](10_multiview_text_classification_stacking.zh-CN.md)（及 [09｜传统 NLP](09_classical_nlp_language_models.zh-CN.md)）→ 本章（词嵌入与循环神经语言模型）→ **下一章**：[12｜Transformer 架构与自注意力机制](12_transformer_attention.zh-CN.md)（彻底抛弃慢速循环结构，迎来现代大模型的基石！）。

---

## 一、 从统计频次表到 Neural N-gram（Bengio 神经语言模型）

统计 N-gram 面临两大痛点：**维度灾难**（词表 $|V|$ 越大，组合数 $|V|^n$ 指数爆炸）和**缺乏语义泛化**（词与词之间完全独立）。
**Neural N-gram** 通过三步彻底解决了这个问题：

1. **构建词表（Vocabulary）**：仅在训练集上统计词频，建立 `token → index` 映射表；测试集未见词统一映射为 `<unk>`（严禁在测试集上重建词表！）。
2. **词嵌入查表（`nn.Embedding(vocab_size, embed_dim)`）**：把前 $k$ 个上下文词的整数 ID 查表映射为 $k$ 个稠密向量，拼接成长度为 $k \times \text{embed\_dim}$ 的特征向量。
3. **MLP 预测下一个词**：经过隐藏层与激活函数，输出全词表大小 $|V|$ 的 Logits，并用 `CrossEntropyLoss` 与真实下一个词的 ID 计算损失（测试集评估指标依然是第 09 章的 $\text{PPL} = \exp(\text{CE})$）。

```python
# Neural N-gram 前向张量形状演进：
# x (输入上下文 ID): [batch, context_length], 取值范围 [0, vocab_size)
# embeddings = self.embed(x): [batch, context_length, embed_dim]
# flattened = embeddings.flatten(1): [batch, context_length * embed_dim]
# logits = self.mlp(flattened): [batch, vocab_size]
# loss = CrossEntropyLoss()(logits, y_next_token_id)
```

---

## 二、 循环神经网络（RNN）与长短期记忆网络（LSTM）

Neural N-gram 只能看固定窗口长度 $k$ 的前文。怎么让模型读任意长度的句子？
- **RNN** 引入了隐藏状态 $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$，像传话筒一样把历史信息一步步往后传；但句子一长，连乘会导致**梯度消失或梯度爆炸**（读到句尾忘了句首）。
- **LSTM（Long Short-Term Memory）** 专门设计了一条贯穿全句的“信息传送带”——**细胞状态 $c_t$（长期记忆）** 与 **隐藏状态 $h_t$（短期工作记忆）**：
  $$h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})$$
  通过三个门控开关（**遗忘门 Forget Gate** 决定擦掉哪些旧记忆、**输入门 Input Gate** 决定写入哪些新信息、**输出门 Output Gate** 决定当前输出什么）显著缓解了长距离遗忘。

### 自回归逐词生成（Autoregressive Generation）机制
在推理生成文本时，LSTM 是如何“一个词接一个词”往下写的？
1. **预热阶段（Prompt Encoding）**：先把用户给出的整段提示词（Prompt）一次性喂入 LSTM，得到最后一个词位置的隐藏状态 $(h_t, c_t)$ 和下一个词的预测分布。
2. **递推生成阶段（Step-by-Step Decoding）**：采样选出新词 $w_{t+1}$ 后，下一步**只需把这 1 个新生成的词 $w_{t+1}$ 和上一步保存的 $(h_t, c_t)$ 传入 LSTM**（切勿把前面的整段话再重复输一遍！），如此循环直到生成结束符 `<eos>` 或达到最大长度。

---

## 三、 为什么 LSTM 没能成为今天的大模型？（引出 Transformer）

尽管 LSTM 比传统 N-gram 强大得多，但它有两个致命硬伤：
1. **无法在时间步上并行计算（训练太慢）**：因为算第 $t$ 个词的 $h_t$ 必须等第 $t-1$ 个词的 $h_{t-1}$ 算完，GPU 成千上万个核心只能干等着排队，无法在海量语料上扩展到百亿参数。
2. **长距离信息依然会被压缩损耗**：无论长文有几千个词，所有历史都必须硬塞进一个固定维度的向量 $(h_t, c_t)$ 里，信息瓶颈严重。
> ➡️ **破局之道**：如果不再排队传话，而是让句子里的任意两个词直接连线对话呢？这正是下一章 **[12｜Transformer 与自注意力机制](12_transformer_attention.zh-CN.md)** 的伟大革命！

---

## 四、 配套实践与自测

- **可运行项目**：
  - 无需外部大文件即可直接运行：[`projects/12_neural_language_models/ngram_train.py`](../projects/12_neural_language_models/ngram_train.py)（在小语料上完整训练 Neural N-gram，打印每个 Epoch 的验证集 CE 与 PPL 并做下一词预测）。
  - 可选预训练 LSTM 推理：[`projects/12_neural_language_models/lstm_inference.py`](../projects/12_neural_language_models/lstm_inference.py)（需按 [`ASSETS.zh-CN.md`](../projects/12_neural_language_models/ASSETS.zh-CN.md) 说明补充词表与权重文件）。
- **思考与自测**：
  1. 为什么在 LSTM 逐词生成循环中，必须把上一步输出的 `(hidden, cell)` 状态传给下一步？
  2. 为什么 Neural N-gram 的 `nn.Embedding` 比第 09 章的 One-Hot 计数表更能处理同义词组合？
