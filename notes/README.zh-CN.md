# 知识讲义目录（五篇十五章渐进体系）

**语言：** [English](README.md) | 简体中文 · [首页](../README.zh-CN.md) · [全景学习路线](../STUDY_ROADMAP.zh-CN.md)

本目录将 AI 核心知识按**从零基础到现代大语言模型（LLM）工程**的技术演进逻辑，归纳为 **「五大篇章、共 15 章」**。章节编号 `01 → 15` 即为推荐的循序渐进学习顺序，每一章均用来解决上一章留下的痛点。

---

## 第一篇：数据科学基石（看懂并洗净数据）

1. **[01｜Python、NumPy 与 Pandas 数据基础](01_python_pandas.zh-CN.md)**
   - **核心问题**：如何把杂乱的 CSV/表格读入内存、转化为多维矩阵（NumPy），并排查缺失值、异常值与数据类型？
   - **配套项目**：[`projects/01_data_foundations`](../projects/01_data_foundations/README.zh-CN.md)
2. **[02｜数据预处理、特征工程与 PCA](02_preprocessing_feature_engineering_pca.zh-CN.md)**
   - **核心问题**：如何统一不同特征的量纲（标准化）、将文字类别编码为数字、用 PCA 降维压缩，并死守“防数据泄漏”底线？
   - **配套项目**：[`projects/01_data_foundations`](../projects/01_data_foundations/README.zh-CN.md)

---

## 第二篇：传统机器学习与数学内核（机器如何从数据中学习）

3. **[03｜线性回归、正则化与梯度下降优化](03_regression_optimization.zh-CN.md)**
   - **核心问题**：机器到底是怎么“学习”的？如何定义损失函数（MSE），并通过解析解或梯度下降（GD/SGD/Adam）一步步找到最优参数，再用正则化（Ridge）防止过拟合？
   - **配套项目**：[`projects/09_regression_and_optimization`](../projects/09_regression_and_optimization/README.zh-CN.md)
4. **[04｜监督学习：回归与分类算法、交叉验证与评估体系](04_supervised_learning_evaluation.zh-CN.md)**
   - **核心问题**：如何运用线性/逻辑回归、决策树、随机森林、SVM 解决回归与分类问题，并用交叉验证（CV）和科学指标（MAE、F1、PR-AUC）避免在不平衡数据上自欺欺人？
   - **配套项目**：[`projects/02_house_price_regression`](../projects/02_house_price_regression/README.zh-CN.md)、[`projects/03_classification_and_faces`](../projects/03_classification_and_faces/README.zh-CN.md)
5. **[05｜模型可解释性（XAI）：SHAP 的局部与全局解释](05_explainable_ai_shap.zh-CN.md)**
   - **核心问题**：高准确率的黑盒模型凭什么做出这个预测？如何利用 Shapley 值量化每个特征对全局和单样本预测的正负贡献？
   - **配套项目**：[`projects/08_explainable_ai`](../projects/08_explainable_ai/README.zh-CN.md)

---

## 第三篇：模型工程化部署与流式运维（上线服务与动态更新）

6. **[06｜模型工程化部署：从 sklearn 到 Flask REST API](06_flask_model_deployment.zh-CN.md)**
   - **核心问题**：如何将训练并审计好的模型序列化保存，封装成带严格输入校验的 HTTP REST API 对外提供实时预测？
   - **配套项目**：[`projects/04_flask_iris_api`](../projects/04_flask_iris_api/README.zh-CN.md)
7. **[07｜流式数据学习、概念漂移（Concept Drift）与 Page–Hinkley](07_stream_learning_concept_drift.zh-CN.md)**
   - **核心问题**：模型上线后外部环境随时间变化导致准确率衰退怎么办？如何遵循“先测后训（Prequential）”原则，用 Page–Hinkley 统计检验实时报警并动态重训？
   - **配套项目**：[`projects/05_concept_drift`](../projects/05_concept_drift/README.zh-CN.md)

---

## 第四篇：深度学习与自然语言处理演进（机器如何读懂人类语言）

8. **[08｜深度学习基石：多层感知机（MLP）、自动微分与 PyTorch](08_neural_networks_pytorch.zh-CN.md)**
   - **核心问题**：如何通过多层网络与非线性激活突破线性边界？PyTorch 的张量、自动求导（Autograd）与通用训练四步循环是如何工作的？
   - **配套项目**：[`projects/10_pytorch_mlp`](../projects/10_pytorch_mlp/README.zh-CN.md)
9. **[09｜传统 NLP：BoW、TF-IDF、词向量、N-gram 与困惑度（PPL）](09_classical_nlp_language_models.zh-CN.md)**
   - **核心问题**：计算机如何把文字转化为数字向量（词袋、TF-IDF、Word2Vec/GloVe）？如何用统计 N-gram 预测下一个词并用困惑度（PPL）评估语言模型？
   - **配套项目**：[`projects/11_classical_nlp`](../projects/11_classical_nlp/README.zh-CN.md)
10. **[10｜多视角文本特征融合与 Stacking 集成学习](10_multiview_text_classification_stacking.zh-CN.md)**
    - **核心问题**：如何同时提取“正文词级 TF-IDF”与“辅助字段字符级 N-gram”等多个互补视角，并用无数据泄漏的 Stacking 元分类器融合预测？
    - **配套项目**：[`projects/07_multiview_stacking`](../projects/07_multiview_stacking/README.zh-CN.md)
11. **[11｜神经语言模型：词嵌入（Embedding）、LSTM 与序列生成](11_neural_language_models.zh-CN.md)**
    - **核心问题**：如何用可学习的 `nn.Embedding` 和带门控记忆的 LSTM 解决传统 N-gram 的维度爆炸与语序丢失问题，实现自回归逐词生成？
    - **配套项目**：[`projects/12_neural_language_models`](../projects/12_neural_language_models/ASSETS.zh-CN.md)
12. **[12｜Transformer 架构：缩放点积自注意力（Self-Attention）与编码器](12_transformer_attention.zh-CN.md)**
    - **核心问题**：为什么 LSTM 会遇到无法并行和长文遗忘的瓶颈？Transformer 如何通过 `Q/K/V` 自注意力机制让全句任意词直接并行对话？
    - **配套项目**：[`projects/13_transformer_encoder`](../projects/13_transformer_encoder/README.zh-CN.md)

---

## 第五篇：大语言模型（LLM）原理、微调与应用开发（站在巨人肩膀上）

13. **[13｜大语言模型（LLM）基础：架构演进、分词、预训练、对齐与解码生成](13_llm_foundations.zh-CN.md)**
    - **核心问题**：从 Transformer 到 GPT/Claude/Qwen 大模型经历了什么？为什么 Decoder-only 架构成为主流？什么是 BPE 子词分词与 KV Cache？**预训练（Pre-training）→ 指令微调（SFT）→ 偏好对齐（RLHF/DPO）**三阶段各自在做什么？`Temperature / Top-k / Top-p` 如何控制生成？大模型为什么会产生幻觉？
    - **关联项目**：[`projects/13_transformer_encoder`](../projects/13_transformer_encoder/README.zh-CN.md)、[`projects/14_llm_prompting_and_lora`](../projects/14_llm_prompting_and_lora/README.zh-CN.md)
14. **[14｜大模型定制化：提示词工程（Prompting）、微调选型与 LoRA / PEFT](14_llm_prompting_finetuning_lora.zh-CN.md)**
    - **核心问题**：面对具体业务，何时该用 Prompt（Zero/Few-shot/CoT）、何时该用 RAG、何时才该做微调？LoRA 如何通过冻结基座、仅训练 <1% 的低秩矩阵（$\Delta W = BA$）实现低成本定制？
    - **配套项目**：[`projects/14_llm_prompting_and_lora`](../projects/14_llm_prompting_and_lora/README.zh-CN.md)
15. **[15｜大模型应用工程：LangChain、检索增强生成（RAG）与智能体（Agent）](15_llm_langchain_rag.zh-CN.md)**
    - **核心问题**：如何给大模型外挂私有文档检索库（RAG：切块、向量/词法检索、溯源引用）以攻克知识截断与幻觉？如何通过工具调用（Tool Calling & Agent）让大模型安全地调用外部 API 解决复杂任务？
    - **配套项目**：[`projects/06_llm_rag`](../projects/06_llm_rag/README.zh-CN.md)
