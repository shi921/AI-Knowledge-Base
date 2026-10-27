**语言：** [English](09_classical_nlp_language_models.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 09｜传统 NLP：BoW、TF-IDF、词向量、N-gram 与困惑度（PPL）

> **🧭 知识脉络定位**：第四篇「深度学习与自然语言处理演进」· **第 09 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：前 8 章处理的都是现成的数字表格，但人类知识大多记录在文字里。计算机只认识数字，怎么教它读文章？最早科学家想出了两招：  
>   1. **文本表示（把文章变数字）**：像查户口一样数每个词出现了几次（**词袋 BoW**），并对“的、是、the”这种满大街都是的常见词降权、对罕见关键词加权（**TF-IDF**）；再进一步，给每个词分配一个“语义空间坐标”（**Word2Vec / GloVe 词向量**），让“猫”和“狗”在地图上挨在一起。  
>   2. **语言模型（教机器玩文字接龙）**：统计历史上前 $n-1$ 个词出现后，下一个词最常接什么（**N-gram 语言模型**），并用**困惑度（Perplexity, PPL——模型猜下一个词时有多犹豫不决）**给语言模型能力打分。  
> - **🎯 这一章解决什么核心问题？**：这是整个自然语言处理（NLP）与大模型知识线的起点！今天第 15 章 RAG 里的词法检索（TF-IDF/BM25）、第 13 章大模型的“预测下一个词（Next-Token Prediction）”与 PPL 评测指标，全都直接诞生于本章！  
> - **🔗 前后衔接**：[08｜深度学习基石：MLP 与 PyTorch](08_neural_networks_pytorch.zh-CN.md) → 本章（掌握文本向量化与统计语言模型）→ **下一章**：[10｜多视角文本分类与 Stacking 集成](10_multiview_text_classification_stacking.zh-CN.md)（先用 TF-IDF 与 N-gram 实战多视角文本分类），随后进入 [11｜神经语言模型与 LSTM](11_neural_language_models.zh-CN.md)。

---

## 一、 文本表示的三次演进：从稀疏词频到稠密词向量

| 表示方法 | 核心原理 | 优势 | 致命局限（为什么需要继续演进？） |
|---|---|---|---|
| **词袋模型 (Bag of Words, BoW)** | 在训练集上构建词表（大小 $|V|$），把每篇文档表示成长度为 $|V|$ 的词频计数向量 | 简单直观 | 完全丢失语序（“我爱你”=“你爱我”）；常见虚词（the, is）频次最高却毫无区分度 |
| **TF-IDF** | $\text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$：在本篇文档出现越多（TF 高）、在其他文档出现越少（IDF 高），权重越大 | 突出核心关键词，**至今仍是工业界极强的分类与检索基线** | 依然是高维稀疏向量；不懂同义词（“电脑”和“计算机”正交无关） |
| **N-gram 特征** | 把相邻连续的 2 个或 3 个词（如 `"not good"`）捆绑成一个特征列 | 捕捉局部短语搭配与否定词 | 词表维度随 $N$ 指数级爆炸 |
| **静态分布式词向量 (Word2Vec / GloVe)** | 利用大规模语料的上下文共现规律，将每个词映射为 50～300 维的低维**稠密语义向量** | 近义词余弦相似度高，具备语义泛化能力 | 一词一向量（无法区分多义词“苹果手机”与“吃苹果”）；对整篇文档词向量简单求平均会丢失语法结构 |

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 经典强基线：词+二元短语 TF-IDF（仅在训练集 fit）+ 逻辑回归分类器
classifier = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)
# classifier.fit(train_texts, train_labels)
# predictions = classifier.predict(test_texts)
```

---

## 二、 统计语言模型（Statistical Language Models）与平滑

什么是**语言模型（Language Model, LM）**？就是计算一句话有多像正常人类语言，或者给定前文预测下一个词的概率分布：
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \dots, w_{t-1})$$

1. **马尔可夫假设与 N-gram 模型**：因为历史前文太长没法统计，假设当前词只依赖前面最近的 $n-1$ 个词：
   $$P_{\text{MLE}}(w_t \mid w_{t-n+1:t-1}) = \frac{\text{count}(w_{t-n+1:t})}{\text{count}(w_{t-n+1:t-1})}$$
2. **零概率灾难与拉普拉斯平滑（Add-One / Laplace Smoothing）**：
   - 如果测试集里出现了一个训练集没见过的词组搭配，分子 `count = 0` 会导致整句话的概率乘积瞬间变成 `0`（取对数直接变成 $-\infty$）！
   - **加一平滑**：给分子加 $1$，分母加上词表大小 $|V|$，确保任何词组概率永远大于 0：
     $$P_{\text{Add-1}}(w_t \mid \text{context}) = \frac{\text{count}(\text{context}, w_t) + 1}{\text{count}(\text{context}) + |V|}$$

---

## 三、 语言模型的标尺：交叉熵（Cross-Entropy）与困惑度（Perplexity, PPL）

如何评价一个语言模型（包括今天的 GPT/Llama）预测下一个词准不准？用 **困惑度（Perplexity, PPL）**：
- 若测试集长度为 $N$，模型对每个位置真实下一个词给出的预测概率为 $p_1, p_2, \dots, p_N$，则平均负对数似然（即自然对数下的**交叉熵损失 CE**）为：
  $$\text{CE} = -\frac{1}{N} \sum_{i=1}^{N} \ln p_i$$
- **困惑度定义**：
  $$\text{PPL} = \exp(\text{CE}) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \ln p_i\right)$$
- **直觉理解**：如果 $\text{PPL} = 10$，意味着模型在预测下一个词时，平均就像在 **10 个等可能的候选词里犹豫不决**。因此：**PPL 越低，语言模型越笃定、越准确！**

### 数值实现防坑要点
- 手写 Softmax 时直接算 `np.exp(x) / np.sum(np.exp(x))` 遇到大数值会溢出为 `inf`；必须先减去最大值 `x - np.max(x)`，或直接使用稳定的 `log_softmax`。
- 词表（Vocabulary）和 `TfidfVectorizer` 只能用训练集构建，测试集遇到未登录词统一映射为 `<unk>`。

---

## 四、 配套实践与自测

- **可运行项目**：[`projects/11_classical_nlp/main.py`](../projects/11_classical_nlp/main.py)（运行 TF-IDF 文本分类与拉普拉斯平滑 N-gram 困惑度计算）；可选扩展 [`word_embeddings_optional.py`](../projects/11_classical_nlp/word_embeddings_optional.py)（词向量近义词检索与 PCA 可视化）。
- **思考与自测**：
  1. 为什么 `TfidfVectorizer` 必须在划分训练/测试集之后，只在训练集上调用 `.fit()`？
  2. 为什么测试集中只要出现一个未经平滑的零概率词组，整段测试语料的困惑度（PPL）就会变成无穷大（$\infty$）？
