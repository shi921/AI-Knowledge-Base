**语言：** [English](11_classical_nlp_language_models.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 11｜传统 NLP：BoW、TF-IDF、Word Embeddings、N-gram 和困惑度

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
