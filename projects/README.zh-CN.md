# 完整实战项目目录

**语言：** [English](README.md) | 简体中文 · [首页](../README.zh-CN.md) · [全景学习路线](../STUDY_ROADMAP.zh-CN.md)

所有脚本的注释、异常提示、命令行输出统一使用英文；每个项目都有中英文两份运行说明。建议新手按以下**五阶递进顺序**动手运行：

## 一、 按「五阶递进脉络」运行

- **阶段 1：数据基石**
  - [`01_data_foundations`](01_data_foundations/README.zh-CN.md)：缺失值诊断、标准化缩放、特征相关性、PCA 降维与时间序列滑动平滑（对应第 01、02 章）。
- **阶段 2：传统机器学习与数学内核**
  - [`09_regression_and_optimization`](09_regression_and_optimization/README.zh-CN.md)：手写梯度下降、最小二乘解析解、多项式回归与 Ridge 正则化对比（对应第 09 章）。
  - [`02_house_price_regression`](02_house_price_regression/README.zh-CN.md)：端到端房价回归 Pipeline、防泄漏预处理与交叉验证评估（对应第 03 章）。
  - [`03_classification_and_faces`](03_classification_and_faces/README.zh-CN.md)：多分类器对比、混淆矩阵评估与可选 PCA+SVM 人脸识别（对应第 03 章）。
  - [`08_explainable_ai`](08_explainable_ai/README.zh-CN.md)：基于 SHAP TreeExplainer 的全局特征重要性图与单样本瀑布图归因（对应第 08 章）。
- **阶段 3：工程落地与持续运维**
  - [`04_flask_iris_api`](04_flask_iris_api/README.zh-CN.md)：模型序列化保存、Flask REST API 推理接口、参数校验与自动化测试（对应第 04 章）。
  - [`05_concept_drift`](05_concept_drift/README.zh-CN.md)：流式数据先测后训（Prequential）、Page–Hinkley 漂移检测报警与滑动窗口重训（对应第 05 章）。
- **阶段 4：深度学习与 NLP 语言模型演进**
  - [`10_pytorch_mlp`](10_pytorch_mlp/README.zh-CN.md)：PyTorch 多层感知机（MLP）完整训练循环、验证早停与状态保存（对应第 10 章）。
  - [`11_classical_nlp`](11_classical_nlp/README.zh-CN.md)：TF-IDF 文本分类、拉普拉斯平滑统计 N-gram 语言模型与困惑度 PPL 计算（对应第 11 章）。
  - [`07_multiview_stacking`](07_multiview_stacking/README.zh-CN.md)：正文词特征 + 辅助字段字符特征的双视角基分类器与防泄漏 Stacking 元模型（对应第 07 章）。
  - [`12_neural_language_models`](12_neural_language_models/ASSETS.zh-CN.md)：基于词嵌入的 Neural N-gram 训练与可选预训练 LSTM 逐词生成推理（对应第 12 章）。
  - [`13_transformer_encoder`](13_transformer_encoder/README.zh-CN.md)：手写 PyTorch 多头自注意力（Multi-Head Self-Attention）文本分类器与注意力热图可视化（对应第 13 章）。
- **阶段 5：大语言模型（LLM）应用与微调实战**
  - [`06_llm_rag`](06_llm_rag/README.zh-CN.md)：文档滑动切块、离线 TF-IDF 检索、可选 LangChain 云端问答生成与 Agent 工具调用（对应第 06 章）。
  - [`14_llm_prompting_and_lora`](14_llm_prompting_and_lora/README.zh-CN.md)：Zero/Few-shot 提示词工程模板与基于 HuggingFace PEFT 的 Qwen LoRA 微调/推理流水线（对应第 14 章）。

---

## 二、 按编号快速索引（01–14）

1. [数据处理基础](01_data_foundations/README.zh-CN.md)
2. [房价回归](02_house_price_regression/README.zh-CN.md)
3. [分类与可选人脸特征](03_classification_and_faces/README.zh-CN.md)
4. [Flask 鸢尾花 API](04_flask_iris_api/README.zh-CN.md)
5. [概念漂移](05_concept_drift/README.zh-CN.md)
6. [大模型与 RAG](06_llm_rag/README.zh-CN.md)
7. [多视角 Stacking](07_multiview_stacking/README.zh-CN.md)
8. [可解释 AI](08_explainable_ai/README.zh-CN.md)
9. [回归与优化](09_regression_and_optimization/README.zh-CN.md)
10. [PyTorch 多层感知机](10_pytorch_mlp/README.zh-CN.md)
11. [传统 NLP](11_classical_nlp/README.zh-CN.md)
12. [神经语言模型](12_neural_language_models/ASSETS.zh-CN.md)
13. [Transformer 编码器](13_transformer_encoder/README.zh-CN.md)
14. [提示工程与 LoRA](14_llm_prompting_and_lora/README.zh-CN.md)
