**语言：** [English](07_multiview_text_classification_stacking.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 07｜多视角文本分类与 Stacking 集成

## 学习目标

在同一任务上提取不同的信息视角（例如正文词级 TF-IDF 与短元数据字符 n-gram），分别训练基分类器，再通过元模型整合预测概率。它与分布式计算不是同一概念。

## 两个基础特征视角

- **正文**：`TfidfVectorizer(ngram_range=(1,2)) → MultinomialNB`，利用词频和短语特征。
- **辅助字段**：`TfidfVectorizer(analyzer="char",ngram_range=(2,4)) → LogisticRegression`，捕捉短字符序列。
- **元模型**：将两个基模型的 `predict_proba` 拼接为 `[p1_1, ..., p1_K, p2_1, ..., p2_K]`，输入逻辑回归预测最终类别。

## 避免数据泄漏的训练顺序

1. 将数据按照标签分层切成 **base / meta / test** 三份（例如 60% / 20% / 20%）。
2. 只在 base 上拟合文本向量器和基分类器。
3. 用基分类器对 meta 生成**未见过的样本的概率**，训练元分类器。
4. 仅在 test 上做最后一次评估。比较正文分支、辅助分支、stacking 和多数类基线。

生产系统也可以使用 out-of-fold (OOF) stacking 充分利用样本；无论哪种方法，都不能在同一训练样本的拟合内预测上训练 meta 模型，否则容易过拟合。

```python
# The base classifiers must agree on classes_ ordering.
X_meta = np.column_stack([
    text_model.predict_proba(meta['text']),
    aux_model.predict_proba(meta['aux_text']),
])
meta_model.fit(X_meta, meta['label'])
```

## 评价与隐私

分类数据不平衡时关注 macro-F1、各类别召回率和多数类基线。不要在公共仓库上传个人信息或无权分享的文本；若真实数据涉及人群属性，必须特别评估公平性和数据保护要求。

**完整项目**：`python projects/07_multiview_stacking/main.py` 使用程序生成的合成主题文本；`--data /path/to/private.csv` 可从可信的本地 CSV 输入 `text,aux_text,label` 三列数据。合成数据只验证算法流程，不表示真实效果。

**自测**：为什么 meta 模型不能使用基模型刚训练过的样本概率？为什么多个强相关分支未必优于单分支？为什么应使用 macro-F1 而非只看准确率？
