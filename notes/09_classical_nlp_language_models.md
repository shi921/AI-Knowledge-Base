**Language:** English | [简体中文](09_classical_nlp_language_models.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 09 | Classical NLP: BoW, TF-IDF, Word Embeddings, N-grams, and Perplexity

> **🧭 Knowledge Map Position**: Part IV (Deep Learning & NLP Evolution) · **Chapter 09 of 15**  
> - **💡 Intuitive Analogy**: Computers only understand numbers, not words. Classical NLP solved this with two foundational ideas:  
>   1. **Text Representation (Words to Numbers)**: Count how often words appear (**Bag of Words**), down-weight ubiquitous filler words like "the/is" in favor of rare keywords (**TF-IDF**), or assign each word a coordinate on a semantic map (**Word2Vec / GloVe embeddings**) where "cat" and "dog" live near each other.  
>   2. **Language Modeling (Next-Word Prediction)**: Estimate the probability of the next word from historical word counts (**N-gram models**) and score how "hesitant or surprised" the model is using **Perplexity (PPL)**.  
> - **🎯 Why Does This Matter?**: Everything in modern NLP and LLMs traces back here: lexical RAG retrieval (Chapter 15) builds directly on TF-IDF/BM25, and GPT's next-token training objective and Perplexity metric come straight from language modeling!  
> - **🔗 Learning Chain**: [08 | Deep Learning Foundations: MLPs & PyTorch](08_neural_networks_pytorch.md) → This Chapter (Text vectorization & statistical language models) → **Next Chapter**: [10 | Multi-view Text Classification and Stacking](10_multiview_text_classification_stacking.md) (Combine multiple TF-IDF & n-gram views), followed by [11 | Neural Language Models & LSTMs](11_neural_language_models.md).

---

## 1. Evolution of Text Representations

| Representation | Mechanism | Strengths | Core Limitations |
|---|---|---|---|
| **Bag of Words (BoW)** | Counts token frequencies using a vocabulary of size $|V|$ learned on training text | Simple, fast baseline | Ignores word order; common stopwords dominate raw counts |
| **TF-IDF** | Scales term frequency by inverse document frequency $\log(N / \text{DF}(t))$ | Highlights discriminative keywords; **strong classification and lexical retrieval baseline** | High-dimensional sparse vector; treats synonyms ("car" vs. "automobile") as orthogonal |
| **N-gram Features** | Counts contiguous 2- or 3-token phrases (e.g., `"not good"`) | Captures local word order and negation | Vocabulary size explodes combinatorially with $N$ |
| **Static Word Embeddings (Word2Vec / GloVe)** | Maps each word to a low-dimensional dense vector (e.g., 100D–300D) from co-occurrence statistics | Captures semantic similarity in vector space | Single static vector per word (cannot disambiguate polysemy like "apple" fruit vs. company); averaging word vectors across a document loses syntax |

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

classifier = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)
# classifier.fit(train_texts, train_labels)
# classifier.predict(test_texts)
```

---

## 2. Statistical N-gram Language Models & Smoothing

An **N-gram language model** predicts the next token from the preceding $n-1$ tokens: $P(w_t \mid w_{t-n+1}, \dots, w_{t-1})$.
- **Maximum Likelihood Estimate (MLE)**: $\text{count}(\text{context}, w_t) / \text{count}(\text{context})$.
- **Zero-Probability Problem & Add-One (Laplace) Smoothing**: Any unseen word combination yields probability `0`, making the probability of the entire sentence zero (and $\log 0 = -\infty$). **Add-one smoothing** computes:
  $$P_{\text{Add-1}}(w_t \mid \text{context}) = \frac{\text{count}(\text{context}, w_t) + 1}{\text{count}(\text{context}) + |V|}$$

---

## 3. Evaluating Language Models: Cross-Entropy & Perplexity (PPL)

Given predicted probabilities $p_1, \dots, p_N$ assigned to the true next tokens across a held-out corpus of length $N$:
$$\text{CE} = -\frac{1}{N} \sum_{i=1}^{N} \ln p_i, \qquad \text{PPL} = \exp(\text{CE}) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \ln p_i\right)$$
- **Intuition**: A Perplexity of `10` means the model is as uncertain at each step as if it were choosing uniformly among 10 equally likely words. **Lower PPL = better next-token prediction.**
- **Numerical Stability**: Naive `exp(x) / sum(exp(x))` overflows for large logits; always subtract `max(x)` first or use `log_softmax`.

---

## 4. Hands-on Projects & Self-Test

- **Runnable Projects**: [`projects/11_classical_nlp/main.py`](../projects/11_classical_nlp/main.py) (TF-IDF classification + smoothed N-gram PPL) and optional [`word_embeddings_optional.py`](../projects/11_classical_nlp/word_embeddings_optional.py).
- **Self-Test**:
  1. Why must `TfidfVectorizer` be fit only after splitting train and test sets?
  2. Why does a single unseen token without smoothing drive unsmoothed Perplexity to infinity?
