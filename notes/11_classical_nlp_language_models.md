**Language:** English | [简体中文](11_classical_nlp_language_models.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 11 | Classical NLP: BoW, TF-IDF, Embeddings, N-grams, and Perplexity

## Text representations

- **Bag of Words (BoW):** count tokens using a vocabulary learned from training text; loses word order.
- **TF-IDF:** weight token counts by inverse document frequency, reducing the prominence of ubiquitous terms; fit only on training documents.
- **N-gram features:** count adjacent token pairs/triples to capture local order at the cost of sparsity.
- **Dense embeddings:** represent words as compact vectors. GloVe uses word co-occurrence statistics; Word2Vec predicts words/contexts. Averaging word vectors discards order and many negation relationships.

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
classifier = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                           LogisticRegression(max_iter=1000))
# classifier.fit(train_texts, train_labels)
# classifier.predict(test_texts)
```

## Statistical language modeling

An N-gram model estimates the next-token distribution `P(wₜ | wₜ₋ₙ₊₁, ..., wₜ₋₁)` from counts, such as `count(context,word)/count(context)`. Unseen sequences receive zero probability without smoothing. **Add-one smoothing** instead estimates `(count(context,word)+1)/(count(context)+|V|)`: useful pedagogically, though often not an optimal practical choice.

**Perplexity (PPL):** for true-token probabilities `p₁,...,pₙ`, `PPL = exp[-(1/n)∑ln pᵢ]`. If cross entropy uses natural logarithms, `PPL=exp(CE)`. Prefer stable `log_softmax` over directly exponentiating large logits.

## Implementation details

- Naive `exp(x)/sum(exp(x))` overflows; subtract the largest logit first.
- Unseen N-gram contexts can lead to zero denominators; define smoothing and `<unk>` handling explicitly.
- Optional embedding experiments can load authorized local Gensim `.kv` files; no automatic remote GloVe download is performed.

**Projects:** [`projects/11_classical_nlp/main.py`](../projects/11_classical_nlp/main.py) (text classification + smoothed PPL) and optional `word_embeddings_optional.py`; multi-view stacking builds on these features.

**Review questions:** Why fit TF-IDF after splitting? Why can one unknown N-gram make unsmoothed sequence perplexity infinite?
