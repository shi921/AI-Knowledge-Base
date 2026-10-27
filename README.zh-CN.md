**语言：** [English](README.md) | 简体中文

# AI Knowledge Base · 人工智能知识体系与项目实践

一套面向**零基础入门与算法工程实践**的中英双语 AI 知识库。通过 **「五阶递进式知识架构」**，将原本碎片化的 14 个核心知识点串联成一条从“数据清洗 → 机器学习内核 → 生产部署与监控 → 深度学习与 Transformer 演进 → 大模型 RAG 与 LoRA 微调”的完整技术演进脉络。

## 快速导航

- 🧭 **[零基础全景学习路线与知识脉络](STUDY_ROADMAP.zh-CN.md)**：**强烈建议新手先看这里！** 包含通俗心智模型、五阶递进路线图、14 个模块的“痛点与作用”速查表，以及全网高分开源 AI 知识库对标指南。
- 📚 **[知识笔记目录](notes/README.zh-CN.md)**：14 篇中英双语核心讲义，涵盖直觉比喻、数学原理、方法对比与工程防坑指南。
- 💻 **[完整实战项目](projects/README.zh-CN.md)**：14 个开箱即用的可运行 Python 项目（默认内置或自动生成离线演示数据）。
- 🛠️ **[运行验证与环境说明](ENVIRONMENT_AND_VALIDATION.zh-CN.md)**：依赖分层、已验证路径、可选外部资产与安全规范。

---

## 五阶递进知识脉络（为什么学 & 解决什么问题）

> 💡 **新手提示**：章节编号 `01–14` 仅为目录索引号，建议零基础读者按照以下 **5 个演进阶段（`01 → 02 → 09 → 03 → 08 → 04 → 05 → 10 → 11 → 07 → 12 → 13 → 06 → 14`）** 循序渐进学习：

### 第一阶段：数据基石篇 —— 把杂乱原始数据洗成模型能懂的“净菜”
| 建议顺序 | 章节与笔记 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套项目目录 |
|---|---|---|---|---|
| 第 1 步 | [01](notes/01_python_pandas.zh-CN.md) | Python / NumPy / Pandas | 读取杂乱表格，排查缺失值、异常值与数据类型，完成分组统计分析。 | [`01_data_foundations`](projects/01_data_foundations/README.zh-CN.md) |
| 第 2 步 | [02](notes/02_preprocessing_feature_engineering_pca.zh-CN.md) | 预处理 / 特征工程 / PCA | 统一特征量纲（标准化）、将类别转为数值编码、用 PCA 压缩高维特征并防止数据泄漏。 | [`01_data_foundations`](projects/01_data_foundations/README.zh-CN.md) |

### 第二阶段：传统机器学习与数学内核篇 —— 机器是如何从数据中“总结规律”的？
| 建议顺序 | 章节与笔记 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套项目目录 |
|---|---|---|---|---|
| 第 3 步 | [09](notes/09_regression_optimization.zh-CN.md) | 线性回归 / 正则化 / 梯度下降 | 理解机器学习的底层引擎：如何定义误差（损失函数）并顺着梯度一步步自动调整参数。 | [`09_regression_and_optimization`](projects/09_regression_and_optimization/README.zh-CN.md) |
| 第 4 步 | [03](notes/03_supervised_learning_evaluation.zh-CN.md) | 监督学习：回归 / 分类 / 评估 | 掌握预测连续值（房价）与离散类别（手写数字/人脸）的标准流水线、交叉验证与防作弊评估指标。 | [`02_house_price_regression`](projects/02_house_price_regression/README.zh-CN.md)<br/>[`03_classification_and_faces`](projects/03_classification_and_faces/README.zh-CN.md) |
| 第 5 步 | [08](notes/08_explainable_ai_shap.zh-CN.md) | 模型可解释性 / SHAP | 拆解“黑盒模型”，量化每个特征对预测结果的正负贡献，满足风控/医疗等场景的透明性要求。 | [`08_explainable_ai`](projects/08_explainable_ai/README.zh-CN.md) |

### 第三阶段：工程落地与持续运维篇 —— 模型如何对外服务并应对环境变化？
| 建议顺序 | 章节与笔记 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套项目目录 |
|---|---|---|---|---|
| 第 6 步 | [04](notes/04_flask_model_deployment.zh-CN.md) | Flask / REST API 模型部署 | 将本地训练好的模型封装为 HTTP 接口，校验外部请求并实时返回预测结果。 | [`04_flask_iris_api`](projects/04_flask_iris_api/README.zh-CN.md) |
| 第 7 步 | [05](notes/05_stream_learning_concept_drift.zh-CN.md) | 流式学习与概念漂移 | 当外部环境随时间变化导致旧模型失效时，利用 Page–Hinkley 统计检验实时报警并动态重训模型。 | [`05_concept_drift`](projects/05_concept_drift/README.zh-CN.md) |

### 第四阶段：深度学习与 NLP 语言模型演进篇 —— 机器如何一步步读懂人类文字？
| 建议顺序 | 章节与笔记 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套项目目录 |
|---|---|---|---|---|
| 第 8 步 | [10](notes/10_neural_networks_pytorch.zh-CN.md) | 神经网络 / PyTorch | 突破线性模型的表达瓶颈，用多层感知机（MLP）、非线性激活与自动求导拟合复杂模式。 | [`10_pytorch_mlp`](projects/10_pytorch_mlp/README.zh-CN.md) |
| 第 9 步 | [11](notes/11_classical_nlp_language_models.zh-CN.md) | 传统 NLP / 词向量 / N-gram | 将文字转化为数字（BoW、TF-IDF、词向量），并用统计 N-gram 与困惑度（PPL）衡量语言规律。 | [`11_classical_nlp`](projects/11_classical_nlp/README.zh-CN.md) |
| 第 10 步 | [07](notes/07_multiview_text_classification_stacking.zh-CN.md) | 多视角文本分类 / Stacking | 将“正文词频”与“元数据字符特征”等多个视角的基分类器无泄漏地融合给元模型，提升分类鲁棒性。 | [`07_multiview_stacking`](projects/07_multiview_stacking/README.zh-CN.md) |
| 第 11 步 | [12](notes/12_neural_language_models.zh-CN.md) | Neural N-gram / LSTM | 解决传统统计词频丢失语序和维度爆炸的问题，用词嵌入和循环记忆状态逐词生成连贯文本。 | [`12_neural_language_models`](projects/12_neural_language_models/ASSETS.zh-CN.md) |
| 第 12 步 | [13](notes/13_transformer_attention.zh-CN.md) | Transformer / Self-Attention | 解决 LSTM 无法并行和长距离遗忘的痛点，让全句任意词通过 `Q/K/V` 自注意力直接对话（现代大模型基石）。 | [`13_transformer_encoder`](projects/13_transformer_encoder/README.zh-CN.md) |

### 第五阶段：大语言模型（LLM）应用与微调实战篇 —— 站在巨人肩膀上构建智能系统
| 建议顺序 | 章节与笔记 | 知识主题 | 这个知识点有什么用？（解决什么核心问题） | 配套项目目录 |
|---|---|---|---|---|
| 第 13 步 | [06](notes/06_llm_langchain_rag.zh-CN.md) | LLM / LangChain / RAG / Agent | 给大模型外挂私有知识库（RAG）以减少幻觉并附带依据引用，通过工具调用（Agent）执行外部动作。 | [`06_llm_rag`](projects/06_llm_rag/README.zh-CN.md) |
| 第 14 步 | [14](notes/14_llm_prompting_finetuning_lora.zh-CN.md) | Prompting / Fine-tuning / LoRA | 通过提示词工程（Zero/Few-shot）引导输出规约，并用 LoRA 低秩适配器以极小算力定制大模型风格与能力。 | [`14_llm_prompting_and_lora`](projects/14_llm_prompting_and_lora/README.zh-CN.md) |

---

## 克隆后的最短运行路径

在**仓库根目录**执行：

```bash
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements-core.txt

python projects/01_data_foundations/main.py
python projects/02_house_price_regression/main.py
python projects/09_regression_and_optimization/main.py
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
├── STUDY_ROADMAP.md / STUDY_ROADMAP.zh-CN.md    # 零基础五阶全景学习路线与高分开源库对标
├── ENVIRONMENT_AND_VALIDATION.md / .zh-CN.md    # 环境依赖与运行验证说明
├── notes/                                       # 14 篇中英双语核心讲义（含新手导读与原理）
├── projects/                                    # 14 个端到端可运行 Python 项目
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
