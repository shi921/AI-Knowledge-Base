**Language:** English | [简体中文](README.zh-CN.md)

# AI Knowledge Base · Concepts, Architecture, and Hands-on Projects

A bilingual (English / Chinese) AI knowledge base designed for both **beginners building a mental model from scratch** and **practitioners reviewing end-to-end workflows**. Instead of leaving 14 topics as isolated fragments, this repository organizes them into a **5-Stage Progressive Architecture** tracing how AI systems evolve from raw data cleaning and classical ML optimization to MLOps, deep sequence models, Transformers, RAG, and LoRA fine-tuning.

## Quick Navigation

- 🧭 **[Beginner's Master Roadmap & Knowledge Architecture](STUDY_ROADMAP.md)**: **Start here if you are new!** Includes an intuitive mental model, the 5-stage evolutionary flowchart, a problem-driven matrix explaining *why* each concept exists, and a curated benchmark of top-rated GitHub AI courses.
- 📚 **[Study Notes Directory](notes/README.md)**: 14 bilingual chapters featuring beginner intuition guides, mathematical foundations, method comparisons, and common pitfalls.
- 💻 **[Hands-on Projects](projects/README.md)**: 14 runnable Python project folders (offline-first with built-in or auto-generated synthetic datasets).
- 🛠️ **[Environment and Validation](ENVIRONMENT_AND_VALIDATION.md)**: Modular dependencies, tested workflows, optional external assets, and safety guidelines.
- 📦 **[External Model and Data Assets](projects/12_neural_language_models/ASSETS.md)**: Details on optional WikiText corpora and pre-trained LSTM weight folders.

---

## 5-Stage Progressive Knowledge Map (Why Each Topic Matters)

> 💡 **Beginner Tip**: Folder numbers `01–14` are topic IDs. If you are learning from scratch, follow the **5 progressive stages (`01 → 02 → 09 → 03 → 08 → 04 → 05 → 10 → 11 → 07 → 12 → 13 → 06 → 14`)** below so each concept naturally builds on the previous one:

### Stage 1: Data Foundations — Turning Raw Messy Tables into Clean Features
| Step | Chapter & Note | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|---|
| Step 1 | [01](notes/01_python_pandas.md) | Python / NumPy / Pandas | Load raw tabular data, inspect missing values, outliers, and data types, and compute grouped summaries. | [`01_data_foundations`](projects/01_data_foundations/README.md) |
| Step 2 | [02](notes/02_preprocessing_feature_engineering_pca.md) | Preprocessing / Feature Engineering / PCA | Standardize numeric scales, encode categorical variables, compress dimensions with PCA, and prevent data leakage. | [`01_data_foundations`](projects/01_data_foundations/README.md) |

### Stage 2: Core Machine Learning & Math Engine — How Machines Learn Patterns from Data
| Step | Chapter & Note | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|---|
| Step 3 | [09](notes/09_regression_optimization.md) | Linear Models / Regularization / Gradient Descent | Understand the mathematical engine of ML: measuring error via a loss function and stepping down the gradient to update weights. | [`09_regression_and_optimization`](projects/09_regression_and_optimization/README.md) |
| Step 4 | [03](notes/03_supervised_learning_evaluation.md) | Supervised Learning: Regression / Classification / Evaluation | Build leak-free regression and classification pipelines using cross-validation and task-appropriate metrics (MAE, F1, PR-AUC). | [`02_house_price_regression`](projects/02_house_price_regression/README.md)<br/>[`03_classification_and_faces`](projects/03_classification_and_faces/README.md) |
| Step 5 | [08](notes/08_explainable_ai_shap.md) | Model Explainability / SHAP | Open the black box by attributing each prediction to individual feature contributions (global importance & local waterfall plots). | [`08_explainable_ai`](projects/08_explainable_ai/README.md) |

### Stage 3: Production Serving & Dynamic Maintenance — Deploying & Monitoring Models in the Real World
| Step | Chapter & Note | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|---|
| Step 6 | [04](notes/04_flask_model_deployment.md) | Model Serving with Flask / REST APIs | Package a trained model into an HTTP service that validates JSON inputs and returns real-time predictions. | [`04_flask_iris_api`](projects/04_flask_iris_api/README.md) |
| Step 7 | [05](notes/05_stream_learning_concept_drift.md) | Online Learning / Concept Drift | Detect when real-world data distributions shift over time (Page–Hinkley test) and retrain or update models dynamically. | [`05_concept_drift`](projects/05_concept_drift/README.md) |

### Stage 4: Deep Learning & NLP Evolution — How Machines Learned to Read Human Language
| Step | Chapter & Note | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|---|
| Step 8 | [10](notes/10_neural_networks_pytorch.md) | Neural Networks / PyTorch | Overcome linear boundaries by stacking layers with non-linear activations, autograd, and the 4-step PyTorch training loop. | [`10_pytorch_mlp`](projects/10_pytorch_mlp/README.md) |
| Step 9 | [11](notes/11_classical_nlp_language_models.md) | Classical NLP / Embeddings / N-grams | Convert text into numbers (BoW, TF-IDF, word vectors) and evaluate statistical next-word models with Perplexity (PPL). | [`11_classical_nlp`](projects/11_classical_nlp/README.md) |
| Step 10 | [07](notes/07_multiview_text_classification_stacking.md) | Multi-view Text Classification / Stacking | Fuse predictions from distinct feature views (word TF-IDF + character n-grams) via a leak-free meta-classifier. | [`07_multiview_stacking`](projects/07_multiview_stacking/README.md) |
| Step 11 | [12](notes/12_neural_language_models.md) | Neural N-grams / LSTMs | Replace sparse count tables with dense embeddings and recurrent memory states to model word order and generate text. | [`12_neural_language_models`](projects/12_neural_language_models/ASSETS.md) |
| Step 12 | [13](notes/13_transformer_attention.md) | Transformers / Self-Attention | Replace sequential recurrence with parallel `Q/K/V` self-attention so every token attends to the full context (foundation of LLMs). | [`13_transformer_encoder`](projects/13_transformer_encoder/README.md) |

### Stage 5: Large Language Model (LLM) Applications & Fine-Tuning — Building Modern GenAI Systems
| Step | Chapter & Note | Topic | Why Do We Need It? (Core Problem Solved) | Project Folder |
|---|---|---|---|---|
| Step 13 | [06](notes/06_llm_langchain_rag.md) | LLMs / LangChain / RAG / Agents | Ground LLMs in private documents via Retrieval-Augmented Generation (RAG) and equip them with external tools (Agents). | [`06_llm_rag`](projects/06_llm_rag/README.md) |
| Step 14 | [14](notes/14_llm_prompting_finetuning_lora.md) | Prompting / Fine-tuning / LoRA | Guide LLM behavior with structured Zero/Few-shot prompts, and adapt model weights cheaply using low-rank PEFT/LoRA adapters. | [`14_llm_prompting_and_lora`](projects/14_llm_prompting_and_lora/README.md) |

---

## Quick Start

Run these commands **from the repository root**:

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
├── README.md                         # English entry point with 5-stage map
├── README.zh-CN.md                   # Chinese entry point with 5-stage map
├── STUDY_ROADMAP.md / STUDY_ROADMAP.zh-CN.md
├── ENVIRONMENT_AND_VALIDATION.md / ENVIRONMENT_AND_VALIDATION.zh-CN.md
├── notes/                            # 14 bilingual chapters with beginner guides
├── projects/                         # 14 projects, bilingual instructions
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