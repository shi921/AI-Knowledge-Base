**Language:** English | [简体中文](README.zh-CN.md)

# AI Knowledge Base · Concepts and Hands-on Projects

## Quick navigation

- [Study roadmap](STUDY_ROADMAP.md): prerequisites and recommended paths across 14 topics.
- [Study notes](notes/README.md): 14 in-depth topic notes covering fundamentals, comparisons, and pitfalls.
- [Hands-on projects](projects/README.md): 14 project folders containing runnable Python workflows.
- [Environment and validation](ENVIRONMENT_AND_VALIDATION.md): dependencies, tested paths, optional assets, and safety.
- [External model and data assets](projects/12_neural_language_models/ASSETS.md): empty WikiText and LSTM model folders and expected files.

## Topic-to-project map

| Chapter | Topic | Project folder |
|---|---|---|
| 01 | Python / NumPy / Pandas | `01_data_foundations` |
| 02 | Preprocessing / feature engineering / PCA | `01_data_foundations` |
| 03 | Supervised learning / regression / classification / evaluation | `02_house_price_regression`, `03_classification_and_faces` |
| 04 | Model serving with Flask / REST APIs | `04_flask_iris_api` |
| 05 | Online learning / concept drift | `05_concept_drift` |
| 06 | LLMs / LangChain / RAG / agents | `06_llm_rag` |
| 07 | Multi-view text classification / stacking | `07_multiview_stacking` |
| 08 | Model explainability / SHAP | `08_explainable_ai` |
| 09 | Linear models / regularization / gradient descent | `09_regression_and_optimization` |
| 10 | Neural networks / PyTorch | `10_pytorch_mlp` |
| 11 | Classical NLP / embeddings / N-grams | `11_classical_nlp` |
| 12 | Neural N-grams / LSTMs | `12_neural_language_models` |
| 13 | Transformers / self-attention | `13_transformer_encoder` |
| 14 | Prompting / fine-tuning / LoRA | `14_llm_prompting_and_lora` |

## Quick start

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

## Repository layout

```text
AI-Knowledge-Base/
├── README.md                         # English entry point
├── README.zh-CN.md                   # Chinese entry point
├── STUDY_ROADMAP.md / STUDY_ROADMAP.zh-CN.md
├── ENVIRONMENT_AND_VALIDATION.md / ENVIRONMENT_AND_VALIDATION.zh-CN.md
├── notes/                            # 14 bilingual chapters
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