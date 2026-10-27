**Language:** English | [简体中文](README.zh-CN.md)

# AI Knowledge Base · Concepts, Architecture, and Hands-on Projects

A bilingual (English / Chinese) AI knowledge base structured for **complete beginners and engineering practitioners alike**. Drawing from top-rated open-source curricula, all study notes are organized in **strictly progressive sequence from `01` to `15` across Five Cohesive Parts**—guiding you smoothly from Python/NumPy/Pandas data cleaning and mathematical optimization to MLOps, deep sequence models, Transformers, **Large Language Model (LLM) Foundations (pre-training, SFT, alignment, decoding)**, LoRA fine-tuning, and **LangChain + RAG + Agents**.

## Quick Navigation

- 🧭 **[Beginner's Master Roadmap & Knowledge Architecture](STUDY_ROADMAP.md)**: **Start here if you are new!** Includes intuitive analogies, the 5-Part 15-Chapter evolutionary flowchart, a problem-driven matrix explaining *why* every concept exists, and a curated benchmark of top-rated GitHub AI courses.
- 📚 **[Study Notes Directory (01–15)](notes/README.md)**: 15 sequentially ordered bilingual chapters covering intuition, math foundations, method comparisons, and engineering pitfalls.
- 💻 **[Hands-on Projects](projects/README.md)**: 14 runnable Python project folders (offline-first with built-in or auto-generated synthetic datasets).
- 🛠️ **[Environment and Validation](ENVIRONMENT_AND_VALIDATION.md)**: Modular dependencies, tested workflows, optional external assets, and safety guidelines.
- 📦 **[External Model and Data Assets](projects/12_neural_language_models/ASSETS.md)**: Details on optional WikiText corpora and pre-trained LSTM weight folders.

---

## 5-Part, 15-Chapter Progressive Knowledge Map (`01 → 15`)

### Part I: Data Foundations — Turning Raw Messy Tables into Clean Features
| Chapter | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|
| **[01](notes/01_python_pandas.md)** | Python / NumPy / Pandas Foundations | Master vectorized NumPy arrays and Pandas DataFrames to load tables and inspect missing values, outliers, and types. | [`01_data_foundations`](projects/01_data_foundations/README.md) |
| **[02](notes/02_preprocessing_feature_engineering_pca.md)** | Preprocessing / Feature Engineering / PCA | Standardize numeric scales, one-hot encode categories, compress dimensions with PCA, and prevent data leakage. | [`01_data_foundations`](projects/01_data_foundations/README.md) |

### Part II: Classical Machine Learning & Math Engine — How Machines Learn Patterns
| Chapter | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|
| **[03](notes/03_regression_optimization.md)** | Linear Models / Regularization / Gradient Descent | Understand the mathematical engine of ML: measuring error via MSE loss, stepping down gradients, and regularizing with Ridge. | [`09_regression_and_optimization`](projects/09_regression_and_optimization/README.md) |
| **[04](notes/04_supervised_learning_evaluation.md)** | Supervised Learning: Regression / Classification / Evaluation | Build leak-free regression and classification pipelines using Cross-Validation and honest metrics (MAE, F1, PR-AUC). | [`02_house_price_regression`](projects/02_house_price_regression/README.md)<br/>[`03_classification_and_faces`](projects/03_classification_and_faces/README.md) |
| **[05](notes/05_explainable_ai_shap.md)** | Explainable AI (XAI) / SHAP | Open the black box by attributing each prediction to individual feature contributions (global beeswarm & local waterfall plots). | [`08_explainable_ai`](projects/08_explainable_ai/README.md) |

### Part III: Model Serving & Dynamic MLOps — Deploying & Monitoring Models in Production
| Chapter | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|
| **[06](notes/06_flask_model_deployment.md)** | Model Serving with Flask / REST APIs | Serialize a trained pipeline and expose real-time predictions over HTTP with strict JSON input validation. | [`04_flask_iris_api`](projects/04_flask_iris_api/README.md) |
| **[07](notes/07_stream_learning_concept_drift.md)** | Streaming Learning / Concept Drift | Detect when real-world data distributions shift over time (Page–Hinkley test) and update models dynamically. | [`05_concept_drift`](projects/05_concept_drift/README.md) |

### Part IV: Deep Learning & NLP Evolution — How Machines Learned to Read Language
| Chapter | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|
| **[08](notes/08_neural_networks_pytorch.md)** | Deep Learning Foundations: MLPs / Autograd / PyTorch | Overcome linear boundaries by stacking layers with non-linear activations, autograd, and the 4-step PyTorch training loop. | [`10_pytorch_mlp`](projects/10_pytorch_mlp/README.md) |
| **[09](notes/09_classical_nlp_language_models.md)** | Classical NLP: BoW / TF-IDF / Embeddings / N-grams | Convert text into vectors and evaluate statistical next-word language models with Perplexity (PPL). | [`11_classical_nlp`](projects/11_classical_nlp/README.md) |
| **[10](notes/10_multiview_text_classification_stacking.md)** | Multi-view Text Classification & Stacking | Fuse predictions from complementary feature views (word TF-IDF + character n-grams) via a leak-free meta-classifier. | [`07_multiview_stacking`](projects/07_multiview_stacking/README.md) |
| **[11](notes/11_neural_language_models.md)** | Neural Language Models: Embeddings / LSTMs | Replace sparse count tables with dense embeddings and recurrent memory states to model word order and generate text. | [`12_neural_language_models`](projects/12_neural_language_models/ASSETS.md) |
| **[12](notes/12_transformer_attention.md)** | Transformers & Self-Attention | Replace sequential recurrence with parallel `Q/K/V` self-attention so every token attends directly to the full sequence. | [`13_transformer_encoder`](projects/13_transformer_encoder/README.md) |

### Part V: Large Language Models (LLMs), Fine-Tuning & RAG/Agents — Modern GenAI Engineering
| Chapter | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|
| **[13](notes/13_llm_foundations.md)** | **Large Language Model (LLM) Foundations** | Master **Decoder-only architectures, BPE tokenization, KV Cache, Pre-training → SFT → RLHF/DPO, Temperature/Top-p decoding, and hallucinations**. | [`13_transformer_encoder`](projects/13_transformer_encoder/README.md)<br/>[`14_llm_prompting_and_lora`](projects/14_llm_prompting_and_lora/README.md) |
| **[14](notes/14_llm_prompting_finetuning_lora.md)** | LLM Customization: Prompting / LoRA Fine-Tuning | Guide behavior with Zero/Few-shot/CoT prompts, choose between Prompting/RAG/Fine-Tuning, and train low-rank PEFT/LoRA adapters. | [`14_llm_prompting_and_lora`](projects/14_llm_prompting_and_lora/README.md) |
| **[15](notes/15_llm_langchain_rag.md)** | LLM Application Engineering: LangChain / RAG / Agents | Ground LLMs in private documents via Retrieval-Augmented Generation (RAG) and equip them with external tools (Agents). | [`06_llm_rag`](projects/06_llm_rag/README.md) |

---

## Quick Start

Run these commands **from the repository root**:

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

# Additional dependencies for neural-network examples
python -m pip install -r requirements-deep-learning.txt
python projects/10_pytorch_mlp/main.py --epochs 5
python projects/12_neural_language_models/ngram_train.py --epochs 3
python projects/13_transformer_encoder/main.py --epochs 5
```

Install `requirements-optional.txt` only if you use Flask, SHAP, River, or LangChain. LoRA experiments use `requirements-finetuning.txt`.

## Repository Layout

```text
AI-Knowledge-Base/
├── README.md                         # English entry point (5 Parts, 15 Chapters)
├── README.zh-CN.md                   # Chinese entry point (5 Parts, 15 Chapters)
├── STUDY_ROADMAP.md / STUDY_ROADMAP.zh-CN.md
├── ENVIRONMENT_AND_VALIDATION.md / ENVIRONMENT_AND_VALIDATION.zh-CN.md
├── notes/                            # 15 sequentially ordered bilingual chapters (01 -> 15)
├── projects/                         # 14 hands-on projects, bilingual instructions
│   ├── 12_neural_language_models/
│   │   ├── data/wikitext-103/         # Reserved for corpus and vocabulary
│   │   ├── data/wikitext-2/           # Reserved for corpus
│   │   └── pretrained_model/          # Reserved for LSTM weights
│   └── 14_llm_prompting_and_lora/checkpoint/  # Reserved for LoRA adapters
├── datasets/                         # Synthetic tables and original short texts
├── outputs/                          # Generated at runtime; not committed
├── requirements-*.txt
└── .gitignore
```