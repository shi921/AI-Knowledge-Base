**Language:** English | [简体中文](10_multiview_text_classification_stacking.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 10 | Multi-view Text Classification and Stacking Ensembles

> **🧭 Knowledge Map Position**: Part IV (Deep Learning & NLP Evolution) · **Chapter 10 of 15**  
> - **💡 Intuitive Analogy**: In Chapter 09 you learned how TF-IDF represents text. In real-world records (support tickets, product listings, social posts), a sample often contains both a main body text and auxiliary metadata (tags, prefixes, device codes). **Multi-view Stacking is like a panel of medical specialists**: Specialist A reads the symptom narrative (word-level TF-IDF), Specialist B analyzes shorthand lab codes (character-level n-grams), and a chief physician (**Meta-Classifier**) combines their probability vectors to make the final diagnosis.  
> - **🎯 Why Does This Matter?**: It teaches how to fuse complementary feature views via **Stacking ensembles** while enforcing the golden rule against meta-leakage: **never train the meta-classifier on in-sample predictions from the base training split!**  
> - **🔗 Learning Chain**: [09 | Classical NLP: BoW, TF-IDF, and N-grams](09_classical_nlp_language_models.md) → This Chapter (Multi-view TF-IDF & leak-free Stacking) → **Next Chapter**: [11 | Neural Language Models: Embeddings, LSTMs, and Generation](11_neural_language_models.md) (Transition from sparse count features to neural sequence models).

---

## 1. Complementary Text Feature Views

1. **Primary Text View**: `TfidfVectorizer(ngram_range=(1, 2)) → MultinomialNB` captures word- and phrase-level topical signals.
2. **Auxiliary Metadata View**: `TfidfVectorizer(analyzer='char', ngram_range=(2, 4)) → LogisticRegression` captures short character sequences, prefixes, and morphological patterns.
3. **Meta-Classifier**: Concatenates `predict_proba` outputs `[p1_1, ..., p1_K, p2_1, ..., p2_K]` from the base models and fits a logistic regression blender. (Note: this is model ensembling, not distributed computing.)

---

## 2. Leak-Free Stacking Training Order

Training the meta-model on predictions made on the same `base` split where the base models were just fit causes severe overfitting. Enforce strict separation via a **three-way split (`base / meta / test`)** or **Out-of-Fold (OOF) cross-validation**:

1. Split labeled data using stratification into **base (60%) / meta (20%) / test (20%)**.
2. Fit all text vectorizers and base classifiers **on `base` only**.
3. Generate out-of-sample probability features on **`meta`** and fit the meta-classifier on `(X_meta, y_meta)`.
4. Evaluate once on **`test`**, comparing single views, majority baseline, and the stacked ensemble.

```python
import numpy as np

# Base classifiers must share the exact same classes_ ordering.
X_meta = np.column_stack([
    text_model.predict_proba(meta['text']),
    aux_model.predict_proba(meta['aux_text']),
])
meta_model.fit(X_meta, meta['label'])
```

---

## 3. Evaluation & Data Privacy

- Compare **Macro-F1** and per-class recall against single-view baselines and a majority-class baseline.
- Never commit private personal records or unauthorized social media data to public repositories.

---

## 4. Hands-on Projects & Self-Test

- **Runnable Project**: [`projects/07_multiview_stacking/main.py`](../projects/07_multiview_stacking/main.py) (uses auto-generated synthetic multi-view text by default; pass `--data /path/to/private.csv` for authorized local datasets with `text, aux_text, label`).
- **Self-Test**:
  1. Why must the meta-classifier never be trained on base-model probabilities generated on the `base` split?
  2. Why might combining two nearly identical base models fail to improve over a single model?
  3. Why is Macro-F1 preferred over plain Accuracy under class imbalance?
