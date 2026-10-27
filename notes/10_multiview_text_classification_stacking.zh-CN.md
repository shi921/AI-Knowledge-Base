**语言：** [English](10_multiview_text_classification_stacking.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 10｜多视角文本特征融合与 Stacking 集成学习

> **🧭 知识脉络定位**：第四篇「深度学习与自然语言处理演进」· **第 10 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：在第 09 章你学会了用 TF-IDF 提取文本特征。但在真实业务（如社交媒体审核、工单分类、电商商品识别）中，一条数据往往不止一段纯正文，还带有作者简介、标签前缀、设备编码等辅助信息。**多视角 Stacking 就像医院里的“多学科专家会诊”**：专家 A 专门研读病人的主诉正文（词级 `Word TF-IDF`），专家 B 专门分析化验单与短编号规律（字符级 `Char N-gram`），最后由一位主治医师（**元分类器 Meta-Classifier**）汇总两位专家给出的概率向量，拍板做出最终诊断！  
> - **🎯 这一章解决什么核心问题？**：掌握工业界常用的**多特征视角建模（Multi-view Learning）**与**模型堆叠集成（Stacking）**；更重要的是牢记**防元模型过拟合铁律**——绝对不能用基分类器在自己训练集（Base Split）上的预测概率去训练元分类器！  
> - **🔗 前后衔接**：[09｜传统 NLP：BoW、TF-IDF 与统计语言模型](09_classical_nlp_language_models.zh-CN.md) → 本章（将多粒度 TF-IDF 特征与 Stacking 集成融会贯通）→ **下一章**：[11｜神经语言模型：Neural N-gram、LSTM 与序列生成](11_neural_language_models.zh-CN.md)（告别稀疏手工特征，进入神经网络序列语言模型）。

---

## 一、 什么是多视角（Multi-view）文本特征？

在同一分类任务中，从不同字段或不同粒度提取互补的信息视角（注意：这属于**特征与模型集成 Ensembling**，与多机分布式计算无关）：

1. **视角 1（正文语义与短语视角）**：
   - `TfidfVectorizer(ngram_range=(1, 2)) → MultinomialNB`（多项式朴素贝叶斯），擅长捕捉完整单词和双词短语的主题倾向。
2. **视角 2（辅助字段与字符形态视角）**：
   - `TfidfVectorizer(analyzer='char', ngram_range=(2, 4)) → LogisticRegression`（逻辑回归），按连续 2～4 个字符切分，对拼写变体、词缀、缩写代码和短元数据极具鲁棒性。
3. **元模型（Meta-Classifier / Blender）**：
   - 将两个基分类器输出的类别概率分布 `predict_proba` 横向拼接为 `[p1_1, ..., p1_K, p2_1, ..., p2_K]`，作为元特征输入逻辑回归，学习“在什么情况下更应该相信哪位专家”。

---

## 二、 严防数据泄漏的 Stacking 训练顺序

如果基分类器刚在数据集 $D$ 上训练完（甚至已经把 $D$ 背下来了，准确率 99%），你立刻拿它在 $D$ 上预测的概率去训练元模型，元模型就会盲目迷信那个死记硬背的基分类器，一到测试集就崩盘！

因此必须严格采用 **三阶段互不重叠切分（Three-way Split / Blending）** 或 **K 折袋外预测（Out-of-Fold OOF Stacking）**：

```text
[全量数据集] ──分层切分──► 1. Base 集 (60%) : 仅用于 fit 各个视角的 TfidfVectorizer 和基分类器
                        ├─► 2. Meta 集 (20%) : 用训练好的基分类器在“未见过的 Meta 集”上输出概率 X_meta，用于 fit 元分类器
                        └─► 3. Test 集 (20%) : 全程锁死，仅在最后评估单视角基线 vs. Stacking 最终表现
```

```python
import numpy as np

# 必须确保所有基分类器的 classes_ 类别顺序完全一致
X_meta = np.column_stack([
    text_model.predict_proba(meta['text']),
    aux_model.predict_proba(meta['aux_text']),
])
meta_model.fit(X_meta, meta['label'])
```

---

## 三、 评估准则与数据隐私伦理

- **评估基线对比**：汇报结果时应同时展示多数类基线（Majority Baseline）、单正文分支、单辅助分支以及 Stacking 融合后的 **Macro-F1** 与准确率。只有当两个视角提供**互补且不完全雷同**的信息时，Stacking 才能显著超越单分支。
- **数据合规与隐私**：切勿将包含个人身份信息（PII）或未经授权的真实社交媒体私密记录上传至公共代码库；涉及人口统计学属性的文本分类需额外审计群体公平性（Fairness）。

---

## 四、 配套实践与自测

- **可运行项目**：[`projects/07_multiview_stacking/main.py`](../projects/07_multiview_stacking/main.py)（默认使用自动生成的合成多视角文本验证完整无泄漏训练闭环；可通过 `--data /path/to/private.csv` 接入含 `text, aux_text, label` 三列的本地数据）。
- **思考与自测**：
  1. 为什么训练元模型（Meta-Classifier）时绝对不能使用基模型在 `base` 训练集上的拟合内预测概率？
  2. 如果两个基模型用了完全相同的特征和高度相似的算法（预测结果 99% 雷同），做 Stacking 还能带来明显提升吗？
  3. 当各类别样本数量不均衡时，为什么 Macro-F1 比普通 Accuracy 更能反映多分类模型的真实水平？
