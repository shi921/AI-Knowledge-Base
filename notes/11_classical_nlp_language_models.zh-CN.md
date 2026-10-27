**语言：** [English](11_classical_nlp_language_models.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 11｜传统 NLP：BoW、TF-IDF、Word Embeddings、N-gram 和困惑度

> **🧭 知识脉络定位**：第四阶段「深度学习与 NLP 语言模型演进篇」· **第 9 步 / 共 14 步**  
> - **💡 零基础直觉比喻**：计算机只认识数字，不认识汉字或英文单词。最早人们想出的办法是**“查字典数词频”**：如果一篇文章里频繁出现“进球、越位、点球”，那它大概率是体育新闻（**词袋模型 BoW**）；而像“的、是、the”这种每篇文章都有的废话要降权，稀有的关键词要加权（**TF-IDF**）。至于怎么让机器学会说话接龙？就统计前几个词出现后，下一个词历史上最常跟什么（**N-gram 语言模型**），并用**困惑度（Perplexity，模型对下一个词有多“犹豫不决”）**来给语言模型打分。  
> - **🎯 这一章解决什么核心问题？**：这是自然语言处理（NLP）的启蒙第一课。理解文字是如何向量化的、TF-IDF 为什么至今仍是强力的检索基线（第 06 章 RAG 会用到），以及为什么传统统计方法会遇到“新词零概率”和“丢失长距离语序”的瓶颈。  
> - **🔗 前后衔接**：[10｜神经网络与 PyTorch](10_neural_networks_pytorch.zh-CN.md) → 本章（文本特征化与统计语言模型）→ **下一站**：[07｜多视角文本分类与 Stacking](07_multiview_text_classification_stacking.zh-CN.md)（先用 TF-IDF 玩转多视角文本集成）与 [12｜神经语言模型与 LSTM](12_neural_language_models.zh-CN.md)（用神经网络突破统计 N-gram 的瓶颈）。

## 文本向量化与词向量

- **Bag of Words (BoW)**：根据训练文档构建词表，用频数表示文本；忽略了语序。
- **TF-IDF**：词频乘以逆文档频率，降低常见词权重；只在训练集 `fit`。
- **N-gram 特征**：相邻连续 2/3 个 token 的统计特征，能捕捉局部语序但维度上升。
- **分布式词向量**：词由低维稠密向量表示。GloVe 依赖大规模共现统计，Word2Vec 使用预测上下文学习表示。对文档做平均词向量会丢掉词序和否定关系。

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
classifier = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                           LogisticRegression(max_iter=1000))
# classifier.fit(train_texts, train_labels)
# classifier.predict(test_texts)
```

## 统计语言模型

令 n-gram 模型预测下一个词：`P(wₜ | wₜ₋ₙ₊₁, ..., wₜ₋₁)`。最大似然估计：`count(context,word)/count(context)`。未见过的组合概率为 0，导致整句概率为 0；**加一平滑**计算 `(count(context,word)+1)/(count(context)+|V|)`，改善零概率但并非最佳现实语言模型。

**Perplexity (PPL)**：若对真实下一个词的概率为 `p₁,...,pₙ`，则 `PPL = exp[-(1/n)∑ln pᵢ]`。若 `CE` 用自然对数，`PPL=exp(CE)`。数值计算优先对 logits 用稳定 `log_softmax`，避免直接 `exp(极大值)` 造成溢出。

### 实现中值得注意的点

- 简单手写 softmax `exp(x)/sum(exp(x))` 可能数值溢出，应先减去最大值。
- 未经平滑的概率 n-gram 模型在未知上下文可能出现除零，评估需平滑与 `<unk>` 策略。
- 词向量示例从 Gensim 在线拉取 GloVe，离线版本不自动下载；本地 `.kv` 可以通过可选脚本使用。

**完整实践**：[`projects/11_classical_nlp/main.py`](../projects/11_classical_nlp/main.py)（分类+平滑 PPL），可选 `word_embeddings_optional.py`；多视角文本分类属于下一层应用。

**自测**：为什么 TF-IDF 要在划分数据之后 fit？为什么少一个未知词就可能让未经平滑的困惑度无穷大？
