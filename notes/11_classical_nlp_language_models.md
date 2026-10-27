**Language:** English | [简体中文](11_classical_nlp_language_models.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 11 | Classical NLP: BoW, TF-IDF, Embeddings, N-grams, and Perplexity

> **🧭 Knowledge Map Position**: Stage 4 (Deep Learning & NLP Evolution) · **Step 9 of 14**  
> - **💡 Intuitive Analogy**: Computers only understand numbers, not words. The earliest solution was **counting word frequencies**: if an article repeats "goal, penalty, offside", it's likely about sports (**Bag of Words**), while common filler words ("the, is") get down-weighted in favor of distinctive terms (**TF-IDF**). To predict the next word in a sentence, **N-gram language models** count historical word sequences, and **Perplexity (PPL)** measures how "surprised or hesitant" the model is when guessing the next token.  
> - **🎯 Why Does This Matter?**: It establishes the foundations of NLP—how text becomes vectors, why TF-IDF remains a fast retrieval baseline (used in Chapter 06 RAG), and why count-based models struggle with unseen words and word order.  
> - **🔗 Learning Chain**: [10 | Neural Networks & PyTorch](10_neural_networks_pytorch.md) → This Chapter (Text vectorization & statistical LMs) → **Next Step**: [07 | Multi-view Text Stacking](07_multiview_text_classification_stacking.md) (Combine multiple text views) and [12 | Neural Language Models & LSTMs](12_neural_language_models.md) (Overcome sparse count limits with neural networks).

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
