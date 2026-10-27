# 完整实战项目目录

**语言：** [English](README.md) | 简体中文 · [首页](../README.zh-CN.md) · [全景学习路线](../STUDY_ROADMAP.zh-CN.md)

所有脚本的注释、异常提示、命令行输出统一使用英文；每个项目都有中英文两份运行说明。建议新手按照 **`notes/` 的五篇十五章顺序**动手运行对应项目：

## 一、 按「五篇十五章知识脉络」运行

- **第一篇：数据科学基石（对应第 01–02 章）**
  - [`01_data_foundations`](01_data_foundations/README.zh-CN.md)：缺失值诊断、标准化缩放、特征相关性、PCA 降维与时间序列滑动平滑（对应 [第 01 章](../notes/01_python_pandas.zh-CN.md)、[第 02 章](../notes/02_preprocessing_feature_engineering_pca.zh-CN.md)）。
- **第二篇：传统机器学习与数学内核（对应第 03–05 章）**
  - [`09_regression_and_optimization`](09_regression_and_optimization/README.zh-CN.md)：手写梯度下降、最小二乘解析解、多项式回归与 Ridge 正则化对比（对应 [第 03 章](../notes/03_regression_optimization.zh-CN.md)）。
  - [`02_house_price_regression`](02_house_price_regression/README.zh-CN.md)：端到端房价回归 Pipeline、防泄漏预处理与交叉验证评估（对应 [第 04 章](../notes/04_supervised_learning_evaluation.zh-CN.md)）。
  - [`03_classification_and_faces`](03_classification_and_faces/README.zh-CN.md)：多分类器对比、混淆矩阵评估与可选 PCA+SVM 人脸识别（对应 [第 04 章](../notes/04_supervised_learning_evaluation.zh-CN.md)）。
  - [`08_explainable_ai`](08_explainable_ai/README.zh-CN.md)：基于 SHAP TreeExplainer 的全局特征重要性图与单样本瀑布图归因（对应 [第 05 章](../notes/05_explainable_ai_shap.zh-CN.md)）。
- **第三篇：模型工程化部署与流式运维（对应第 06–07 章）**
  - [`04_flask_iris_api`](04_flask_iris_api/README.zh-CN.md)：模型序列化保存、Flask REST API 推理接口、参数校验与自动化测试（对应 [第 06 章](../notes/06_flask_model_deployment.zh-CN.md)）。
  - [`05_concept_drift`](05_concept_drift/README.zh-CN.md)：流式数据先测后训（Prequential）、Page–Hinkley 漂移检测报警与滑动窗口重训（对应 [第 07 章](../notes/07_stream_learning_concept_drift.zh-CN.md)）。
- **第四篇：深度学习与自然语言处理演进（对应第 08–12 章）**
  - [`10_pytorch_mlp`](10_pytorch_mlp/README.zh-CN.md)：PyTorch 多层感知机（MLP）完整训练循环、验证早停与状态保存（对应 [第 08 章](../notes/08_neural_networks_pytorch.zh-CN.md)）。
  - [`11_classical_nlp`](11_classical_nlp/README.zh-CN.md)：TF-IDF 文本分类、拉普拉斯平滑统计 N-gram 语言模型与困惑度 PPL 计算（对应 [第 09 章](../notes/09_classical_nlp_language_models.zh-CN.md)）。
  - [`07_multiview_stacking`](07_multiview_stacking/README.zh-CN.md)：正文词特征 + 辅助字段字符特征的双视角基分类器与防泄漏 Stacking 元模型（对应 [第 10 章](../notes/10_multiview_text_classification_stacking.zh-CN.md)）。
  - [`12_neural_language_models`](12_neural_language_models/ASSETS.zh-CN.md)：基于词嵌入的 Neural N-gram 训练与可选预训练 LSTM 逐词生成推理（对应 [第 11 章](../notes/11_neural_language_models.zh-CN.md)）。
  - [`13_transformer_encoder`](13_transformer_encoder/README.zh-CN.md)：手写 PyTorch 多头自注意力（Multi-Head Self-Attention）文本分类器与注意力热图可视化（对应 [第 12 章](../notes/12_transformer_attention.zh-CN.md)、[第 13 章](../notes/13_llm_foundations.zh-CN.md)）。
- **第五篇：大语言模型（LLM）原理、微调与应用开发（对应第 13–15 章）**
  - [`14_llm_prompting_and_lora`](14_llm_prompting_and_lora/README.zh-CN.md)：Zero/Few-shot 提示词工程模板与基于 HuggingFace PEFT 的 Qwen LoRA 微调/推理流水线（对应 [第 13 章](../notes/13_llm_foundations.zh-CN.md)、[第 14 章](../notes/14_llm_prompting_finetuning_lora.zh-CN.md)）。
  - [`06_llm_rag`](06_llm_rag/README.zh-CN.md)：文档滑动切块、离线 TF-IDF 检索、可选 LangChain 云端问答生成与 Agent 工具调用（对应 [第 15 章](../notes/15_llm_langchain_rag.zh-CN.md)）。

---

## 二、 按项目文件夹编号索引

1. [数据处理基础 (`01_data_foundations`)](01_data_foundations/README.zh-CN.md)
2. [房价回归 (`02_house_price_regression`)](02_house_price_regression/README.zh-CN.md)
3. [分类与可选人脸特征 (`03_classification_and_faces`)](03_classification_and_faces/README.zh-CN.md)
4. [Flask 鸢尾花 API (`04_flask_iris_api`)](04_flask_iris_api/README.zh-CN.md)
5. [概念漂移 (`05_concept_drift`)](05_concept_drift/README.zh-CN.md)
6. [大模型与 RAG (`06_llm_rag`)](06_llm_rag/README.zh-CN.md)
7. [多视角 Stacking (`07_multiview_stacking`)](07_multiview_stacking/README.zh-CN.md)
8. [可解释 AI (`08_explainable_ai`)](08_explainable_ai/README.zh-CN.md)
9. [回归与优化 (`09_regression_and_optimization`)](09_regression_and_optimization/README.zh-CN.md)
10. [PyTorch 多层感知机 (`10_pytorch_mlp`)](10_pytorch_mlp/README.zh-CN.md)
11. [传统 NLP (`11_classical_nlp`)](11_classical_nlp/README.zh-CN.md)
12. [神经语言模型 (`12_neural_language_models`)](12_neural_language_models/ASSETS.zh-CN.md)
13. [Transformer 编码器 (`13_transformer_encoder`)](13_transformer_encoder/README.zh-CN.md)
14. [提示工程与 LoRA (`14_llm_prompting_and_lora`)](14_llm_prompting_and_lora/README.zh-CN.md)
