**Language:** English | [简体中文](07_multiview_text_classification_stacking.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 07 | Multi-view Text Classification and Stacking

## Learning goal

Represent the same classification problem through complementary feature views, train separate base models, and combine their probability vectors with a meta-classifier. This is model ensembling, **not** synonymous with distributed computation.

## Two feature branches

- **Main text:** `TfidfVectorizer(ngram_range=(1,2)) → MultinomialNB`, capturing words and short phrases.
- **Auxiliary text:** `TfidfVectorizer(analyzer='char', ngram_range=(2,4)) → LogisticRegression`, capturing character patterns.
- **Meta-model:** concatenate the class probabilities `[p1_1, ..., p1_K, p2_1, ..., p2_K]` and train a logistic regression on them.

## Avoiding meta-model leakage

1. Stratify and split records into **base / meta / test** (for example 60% / 20% / 20%).
2. Fit text vectorizers and base classifiers **only** on the base split.
3. Predict probabilities for **unseen meta examples** and train the meta-classifier.
4. Evaluate once on test and compare each branch, their stack, and a majority-class baseline.

Out-of-fold (OOF) stacking can use data more efficiently, but the meta-model must not be trained on in-sample predictions from the exact examples used to fit the base models.

```python
# The base classifiers must agree on classes_ ordering.
X_meta = np.column_stack([
    text_model.predict_proba(meta['text']),
    aux_model.predict_proba(meta['aux_text']),
])
meta_model.fit(X_meta, meta['label'])
```

## Metrics and privacy

For imbalanced data, inspect macro-F1, per-class recall, and majority-class performance. Do not publish personal information or text you lack permission to distribute; evaluate privacy and fairness if actual human attributes are involved.

**Project:** `python projects/07_multiview_stacking/main.py` defaults to generated synthetic text, or takes a trusted `--data /path/to/private.csv` with `text,aux_text,label` columns. Synthetic metrics validate pipeline behavior, not real-world accuracy.

**Review questions:** Why can the meta-model not train on the base models' in-sample probabilities? Why might correlated views fail to improve results? When is macro-F1 preferable to accuracy?
