# 知识点笔记目录

**语言：** [English](README.md) | 简体中文 · [首页](../README.zh-CN.md) · [全景学习路线](../STUDY_ROADMAP.zh-CN.md)

建议零基础学习者按照以下 **「五阶递进知识脉络」** 阅读笔记，理解每个知识点解决的核心问题后，再运行对应的实战项目巩固。

---

## 一、 按「五阶递进脉络」阅读（强烈推荐新手使用）

### 阶段 1：数据基石篇（看懂并洗净数据）
- **第 1 步** · [01｜Python 与 Pandas 基础](01_python_pandas.zh-CN.md) — *如何把杂乱的原始表格读入内存、检查缺失与异常并做分组统计？*
- **第 2 步** · [02｜预处理、特征工程与 PCA](02_preprocessing_feature_engineering_pca.zh-CN.md) — *如何统一特征量纲、把类别变成数字、用 PCA 降维并防止数据泄漏？*

### 阶段 2：传统机器学习与数学内核篇（机器如何从数据中学习）
- **第 3 步** · [09｜梯度下降与数值优化](09_regression_optimization.zh-CN.md) — *机器到底是怎么通过损失函数和梯度一步步自动调整参数的？*
- **第 4 步** · [03｜监督学习与模型评估](03_supervised_learning_evaluation.zh-CN.md) — *如何科学地训练回归与分类模型，并用交叉验证和合理指标避免自欺欺人？*
- **第 5 步** · [08｜模型可解释性与 SHAP](08_explainable_ai_shap.zh-CN.md) — *如何打开黑盒模型，算出每个特征对单次预测和全局决策推高或拉低了多少？*

### 阶段 3：工程落地与持续运维篇（模型上线与动态更新）
- **第 6 步** · [04｜Flask 模型部署](04_flask_model_deployment.zh-CN.md) — *如何把训练好的模型包装成 HTTP REST API 对外提供实时预测服务？*
- **第 7 步** · [05｜流式学习与概念漂移](05_stream_learning_concept_drift.zh-CN.md) — *当外部世界变化导致模型变笨时，如何用 Page–Hinkley 检测报警并在线更新模型？*

### 阶段 4：深度学习与 NLP 语言模型演进篇（机器如何一步步读懂人类文字）
- **第 8 步** · [10｜神经网络与 PyTorch](10_neural_networks_pytorch.zh-CN.md) — *如何用多层神经网络（MLP）、非线性激活和自动微分拟合复杂规律？*
- **第 9 步** · [11｜传统 NLP 与统计语言模型](11_classical_nlp_language_models.zh-CN.md) — *如何用词袋、TF-IDF、词向量表示文本，并用 N-gram 和困惑度（PPL）预测下一个词？*
- **第 10 步** · [07｜多视角文本分类与 Stacking](07_multiview_text_classification_stacking.zh-CN.md) — *如何将“正文词频”与“元数据字符特征”多个视角的模型无泄漏地融合在一起？*
- **第 11 步** · [12｜神经语言模型与 LSTM](12_neural_language_models.zh-CN.md) — *如何用词嵌入和循环记忆状态（LSTM）记住词序上下文并逐词生成新文本？*
- **第 12 步** · [13｜Transformer 与自注意力](13_transformer_attention.zh-CN.md) — *如何摆脱慢速循环结构，用 `Q/K/V` 自注意力让全句任意词直接对话（现代大模型核心基石）？*

### 阶段 5：大语言模型（LLM）应用与微调实战篇（打造现代 AI 系统）
- **第 13 步** · [06｜大语言模型、LangChain 与 RAG](06_llm_langchain_rag.zh-CN.md) — *如何给大模型外挂私有文档检索库（RAG）以减少幻觉，并赋予它调用工具（Agent）的能力？*
- **第 14 步** · [14｜大模型提示工程、微调与 LoRA](14_llm_prompting_finetuning_lora.zh-CN.md) — *如何用 Zero/Few-shot 提示引导输出，并用 PEFT/LoRA 低秩适配器低成本定制专属大模型？*

---

## 二、 按章节编号快速索引（01–14）

1. [Python 与 Pandas 基础](01_python_pandas.zh-CN.md)
2. [预处理、特征工程与 PCA](02_preprocessing_feature_engineering_pca.zh-CN.md)
3. [监督学习与模型评估](03_supervised_learning_evaluation.zh-CN.md)
4. [Flask 模型部署](04_flask_model_deployment.zh-CN.md)
5. [流式学习与概念漂移](05_stream_learning_concept_drift.zh-CN.md)
6. [大语言模型、LangChain 与 RAG](06_llm_langchain_rag.zh-CN.md)
7. [多视角文本分类与 Stacking](07_multiview_text_classification_stacking.zh-CN.md)
8. [模型可解释性与 SHAP](08_explainable_ai_shap.zh-CN.md)
9. [梯度下降与优化](09_regression_optimization.zh-CN.md)
10. [神经网络与 PyTorch](10_neural_networks_pytorch.zh-CN.md)
11. [传统 NLP 与统计语言模型](11_classical_nlp_language_models.zh-CN.md)
12. [神经语言模型与 LSTM](12_neural_language_models.zh-CN.md)
13. [Transformer 与自注意力](13_transformer_attention.zh-CN.md)
14. [大模型提示工程、微调与 LoRA](14_llm_prompting_finetuning_lora.zh-CN.md)
