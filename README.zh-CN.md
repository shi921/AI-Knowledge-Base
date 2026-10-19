**语言：** [English](README.md) | 简体中文

# AI Knowledge Base · 人工智能知识体系与项目实践

## 快速导航

- [学习路线](STUDY_ROADMAP.zh-CN.md)：14 个知识模块的依赖关系、阶段建议与实践路径。
- [知识笔记](notes/README.zh-CN.md)：14 篇中英文配对 Markdown，包含基本原理、方法对比与常见错误。
- [完整项目](projects/README.zh-CN.md)：14 个按主题整理的可运行 Python 项目。
- [运行验证与环境](ENVIRONMENT_AND_VALIDATION.zh-CN.md)：依赖、已验证项目、可选数据与安全注意事项。

## 主题与代码项目

| 章节 | 知识主题 | 项目目录 |
|---|---|---|
| 01 | Python / NumPy / Pandas | `01_data_foundations` |
| 02 | 预处理 / 特征工程 / PCA | `01_data_foundations` |
| 03 | 监督学习 / 回归 / 分类 / 评估 | `02_house_price_regression` / `03_classification_and_faces` |
| 04 | Flask / REST API 模型部署 | `04_flask_iris_api` |
| 05 | 在线学习与概念漂移 | `05_concept_drift` |
| 06 | LLM / LangChain / RAG / Agent | `06_llm_rag` |
| 07 | 多视角文本分类 / Stacking | `07_multiview_stacking` |
| 08 | 可解释性 / SHAP | `08_explainable_ai` |
| 09 | 线性回归 / 正则化 / 梯度下降 | `09_regression_and_optimization` |
| 10 | 神经网络 / PyTorch | `10_pytorch_mlp` |
| 11 | 传统 NLP / 词向量 / N-gram | `11_classical_nlp` |
| 12 | Neural N-gram / LSTM | `12_neural_language_models` |
| 13 | Transformer / Self-Attention | `13_transformer_encoder` |
| 14 | Prompting / Fine-tuning / LoRA | `14_llm_prompting_and_lora` |

## 克隆后的最短路径

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
├── README.md
├── STUDY_ROADMAP.md / STUDY_ROADMAP.zh-CN.md
├── ENVIRONMENT_AND_VALIDATION.md / ENVIRONMENT_AND_VALIDATION.zh-CN.md
├── notes/                         # 14 study chapters
├── projects/                      # 14 end-to-end projects
│   ├── 12_neural_language_models/
│   │   ├── data/wikitext-103/      # Reserved corpus and vocabulary folder; contents are ignored
│   │   ├── data/wikitext-2/        # Reserved corpus folder; contents are ignored
│   │   └── pretrained_model/       # Reserved LSTM weights; contents are ignored
│   └── 14_llm_prompting_and_lora/checkpoint/  # Reserved adapter weights
├── datasets/                      # Generated synthetic tables and original short texts
├── outputs/                       # Runtime-generated files; ignored by Git
├── requirements-*.txt
└── .gitignore
```
