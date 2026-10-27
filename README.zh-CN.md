**语言：** [English](README.md) | 简体中文

# AI Knowledge Base · 人工智能知识体系与项目实践

一套面向**零基础入门与算法工程实践**的中英双语 AI 知识库。全库参考国内外高星开源教程，按技术演进脉络重构为 **「五大篇章、15 章循序渐进讲义（`notes/01` → `notes/15`）」**，从 Python/NumPy/Pandas 数据清洗、梯度下降与经典机器学习，平滑过渡到模型部署与漂移监控、深度学习与 NLP 演进、Transformer 自注意力、**大语言模型（LLM）核心原理（预训练/SFT/对齐/解码）**、LoRA 微调以及 **LangChain + RAG + Agent 应用工程**。

## 快速导航

- 🧭 **[零基础全景学习路线与知识脉络](STUDY_ROADMAP.zh-CN.md)**：**强烈建议新手先看这里！** 包含通俗心智模型、五篇十五章演进图、每个知识点的“痛点与作用”速查表，以及全网高分开源 AI 知识库对标指南。
- 📚 **[知识讲义目录（01–15）](notes/README.zh-CN.md)**：15 篇按严格演进顺序排列的中英双语核心讲义，涵盖直觉比喻、数学原理、方法对比与工程防坑指南。
- 💻 **[完整实战项目](projects/README.zh-CN.md)**：14 个开箱即用的可运行 Python 项目（默认内置或自动生成离线演示数据）。
- 🛠️ **[运行验证与环境说明](ENVIRONMENT_AND_VALIDATION.zh-CN.md)**：依赖分层、已验证路径、可选外部资产与安全规范。

---

## 五篇十五章知识脉络总览（按 `01 → 15` 顺序循序渐进）

### 第一篇：数据科学基石 —— 把杂乱原始数据洗成模型能懂的“净菜”
| 章节与讲义 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套实战项目 |
|---|---|---|---|
| **[01](notes/01_python_pandas.zh-CN.md)** | Python / NumPy / Pandas 数据基础 | 掌握 NumPy 向量化矩阵运算与 Pandas 表格读取，排查缺失值、异常值与数据类型。 | [`01_data_foundations`](projects/01_data_foundations/README.zh-CN.md) |
| **[02](notes/02_preprocessing_feature_engineering_pca.zh-CN.md)** | 数据预处理 / 特征工程 / PCA | 统一特征量纲（标准化）、将类别转为独热编码、用 PCA 压缩高维特征并严防数据泄漏。 | [`01_data_foundations`](projects/01_data_foundations/README.zh-CN.md) |

### 第二篇：传统机器学习与数学内核 —— 机器是如何从数据中“总结规律”的？
| 章节与讲义 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套实战项目 |
|---|---|---|---|
| **[03](notes/03_regression_optimization.zh-CN.md)** | 线性回归 / 正则化 / 梯度下降优化 | 理解一切 AI 训练的底层数学引擎：定义损失函数（MSE），顺着梯度自动调整权重，用 Ridge 防止过拟合。 | [`09_regression_and_optimization`](projects/09_regression_and_optimization/README.zh-CN.md) |
| **[04](notes/04_supervised_learning_evaluation.zh-CN.md)** | 监督学习：回归 / 分类 / 评估体系 | 掌握主流回归与分类算法（随机森林/SVM/逻辑回归）、交叉验证（CV）与防作弊评估指标（MAE/F1/PR-AUC）。 | [`02_house_price_regression`](projects/02_house_price_regression/README.zh-CN.md)<br/>[`03_classification_and_faces`](projects/03_classification_and_faces/README.zh-CN.md) |
| **[05](notes/05_explainable_ai_shap.zh-CN.md)** | 模型可解释性（XAI） / SHAP | 拆解“黑盒模型”，利用 Shapley 值量化每个特征对全局和单次预测结果的正负贡献。 | [`08_explainable_ai`](projects/08_explainable_ai/README.zh-CN.md) |

### 第三篇：模型工程化部署与流式运维 —— 模型如何对外服务并应对环境变化？
| 章节与讲义 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套实战项目 |
|---|---|---|---|
| **[06](notes/06_flask_model_deployment.zh-CN.md)** | Flask / REST API 模型部署 | 将训练好的模型序列化保存，封装为带严格参数校验的 HTTP 接口对外提供实时预测。 | [`04_flask_iris_api`](projects/04_flask_iris_api/README.zh-CN.md) |
| **[07](notes/07_stream_learning_concept_drift.zh-CN.md)** | 流式学习与概念漂移（Concept Drift） | 当外部世界随时间变化导致旧模型失效时，利用 Page–Hinkley 统计检验实时报警并动态重训模型。 | [`05_concept_drift`](projects/05_concept_drift/README.zh-CN.md) |

### 第四篇：深度学习与自然语言处理演进 —— 机器如何一步步读懂人类文字？
| 章节与讲义 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套实战项目 |
|---|---|---|---|
| **[08](notes/08_neural_networks_pytorch.zh-CN.md)** | 深度学习基石：MLP / 自动微分 / PyTorch | 突破线性模型边界，用多层感知机（MLP）、非线性激活与 PyTorch 训练四步法拟合复杂模式。 | [`10_pytorch_mlp`](projects/10_pytorch_mlp/README.zh-CN.md) |
| **[09](notes/09_classical_nlp_language_models.zh-CN.md)** | 传统 NLP：BoW / TF-IDF / 词向量 / N-gram | 将文字转化为数字向量，用统计 N-gram 预测下一个词并用困惑度（PPL）衡量语言模型好坏。 | [`11_classical_nlp`](projects/11_classical_nlp/README.zh-CN.md) |
| **[10](notes/10_multiview_text_classification_stacking.zh-CN.md)** | 多视角文本分类与 Stacking 集成 | 将“正文词级 TF-IDF”与“元数据字符 N-gram”等多个视角的基分类器无泄漏地融合给元模型。 | [`07_multiview_stacking`](projects/07_multiview_stacking/README.zh-CN.md) |
| **[11](notes/11_neural_language_models.zh-CN.md)** | 神经语言模型：Embedding / LSTM | 解决传统统计词频丢失语序和维度爆炸的问题，用词嵌入和循环记忆状态逐词生成连贯文本。 | [`12_neural_language_models`](projects/12_neural_language_models/ASSETS.zh-CN.md) |
| **[12](notes/12_transformer_attention.zh-CN.md)** | Transformer 架构与自注意力机制 | 解决 LSTM 无法并行和长距离遗忘的痛点，让全句任意词通过 `Q/K/V` 自注意力直接并行对话。 | [`13_transformer_encoder`](projects/13_transformer_encoder/README.zh-CN.md) |

### 第五篇：大语言模型（LLM）原理、微调与应用开发 —— 站在巨人肩膀上构建智能系统
| 章节与讲义 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套实战项目 |
|---|---|---|---|
| **[13](notes/13_llm_foundations.zh-CN.md)** | **大语言模型（LLM）基础与工作原理** | 讲透 **Decoder-only 架构、BPE 分词、KV Cache、预训练→SFT→RLHF/DPO 三阶段、Temperature/Top-p 解码与幻觉成因**。 | [`13_transformer_encoder`](projects/13_transformer_encoder/README.zh-CN.md)<br/>[`14_llm_prompting_and_lora`](projects/14_llm_prompting_and_lora/README.zh-CN.md) |
| **[14](notes/14_llm_prompting_finetuning_lora.zh-CN.md)** | 大模型提示词工程 / 微调选型 / LoRA | 用 Zero/Few-shot/CoT 提示词引导输出，分清 Prompt/RAG/微调边界，并用 LoRA 低秩适配器低成本微调模型。 | [`14_llm_prompting_and_lora`](projects/14_llm_prompting_and_lora/README.zh-CN.md) |
| **[15](notes/15_llm_langchain_rag.zh-CN.md)** | 大模型应用工程：LangChain / RAG / Agent | 给大模型外挂私有文档检索库（RAG）以消除知识盲区与幻觉，并通过 Agent 工具调用执行外部函数。 | [`06_llm_rag`](projects/06_llm_rag/README.zh-CN.md) |

---

## 克隆后的最短运行路径

在**仓库根目录**执行：

```bash
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements-core.txt

python projects/01_data_foundations/main.py
python projects/09_regression_and_optimization/main.py
python projects/02_house_price_regression/main.py
python projects/11_classical_nlp/main.py
python projects/07_multiview_stacking/main.py

# Extra dependencies for neural-network examples
python -m pip install -r requirements-deep-learning.txt
python projects/10_pytorch_mlp/main.py --epochs 5
python projects/12_neural_language_models/ngram_train.py --epochs 3
python projects/13_transformer_encoder/main.py --epochs 5
```

安装 Flask、SHAP、River、LangChain 等扩展模块时再使用 `requirements-optional.txt`；LoRA 使用 `requirements-finetuning.txt`。

## 文件夹结构

```text
AI-Knowledge-Base/
├── README.md / README.zh-CN.md                  # 双语知识脉络总入口
├── STUDY_ROADMAP.md / STUDY_ROADMAP.zh-CN.md    # 零基础五篇十五章全景学习路线与高分开源库对标
├── ENVIRONMENT_AND_VALIDATION.md / .zh-CN.md    # 环境依赖与运行验证说明
├── notes/                                       # 15 篇按递进顺序排列的中英双语核心讲义 (01 -> 15)
├── projects/                                    # 14 个端到端可运行 Python 实战项目
│   ├── 12_neural_language_models/
│   │   ├── data/wikitext-103/                   # 预留语料与词表目录；内容被 Git 忽略
│   │   ├── data/wikitext-2/                     # 预留语料目录；内容被 Git 忽略
│   │   └── pretrained_model/                    # 预留 LSTM 权重目录；内容被 Git 忽略
│   └── 14_llm_prompting_and_lora/checkpoint/    # 预留 LoRA Adapter 权重目录
├── datasets/                                    # 自动生成的合成表格与原创短文档
├── outputs/                                     # 运行时生成的图表与模型产物；被 Git 忽略
├── requirements-*.txt                           # 按阶段拆分的轻量依赖清单
└── .gitignore
```
