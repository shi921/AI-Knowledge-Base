**语言：** [English](07_multiview_text_classification_stacking.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 07｜多视角文本分类与 Stacking 集成

> **🧭 知识脉络定位**：第四阶段「深度学习与 NLP 语言模型演进篇」· **第 10 步 / 共 14 步**  
> - **💡 零基础直觉比喻**：就像医院里的**“多学科专家会诊”**——一位专家只看病人的主诉正文（词级 TF-IDF 特征），另一位专家专门看化验单编号与标签前缀（字符级 N-gram 特征），最后由一位主治医师（**元分类器 Meta-Classifier**）综合两位专家给出的概率，拍板做出最终诊断。  
> - **🎯 这一章解决什么核心问题？**：学会在真实业务中将不同来源、不同粒度的文本/特征视角融合起来（**Stacking 堆叠集成**）；更关键的是掌握**“绝对不能拿基模型在训练集上的预测结果去训练主治医师（元模型）”**的防泄漏铁律！  
> - **🔗 前后衔接**：[11｜传统 NLP 与统计语言模型](11_classical_nlp_language_models.zh-CN.md) → 本章（多视角文本特征工程与集成学习）→ **下一站**：[12｜神经语言模型与 LSTM](12_neural_language_models.zh-CN.md)（告别手工数词频，进入深度学习序列语言模型）。

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
