**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# Classification and Optional PCA-based Face Recognition

Classify scikit-learn's built-in Digits dataset, perform cross-validation, and inspect classification metrics:

```bash
python projects/03_classification_and_faces/main.py
```

The optional PCA + SVM face example requires an authorized local CSV containing a `target` label column and pixel-feature columns; no images or face data are distributed:

```bash
python projects/03_classification_and_faces/faces.py --data /path/to/authorized/faces.csv --components 80
```

Use trustworthy inputs and observe privacy, consent, and redistribution requirements for face data.

**Related notes:** [Chapter 04](../../notes/04_supervised_learning_evaluation.md), [Chapter 02](../../notes/02_preprocessing_feature_engineering_pca.md).
