# Study Notes Directory (5 Parts · 15 Progressive Chapters)

**Language:** English | [简体中文](README.zh-CN.md) · [Home](../README.md) · [Master Roadmap](../STUDY_ROADMAP.md)

This directory organizes core AI knowledge in a **strictly sequential, beginner-friendly progression from `01` to `15`** across **Five Cohesive Parts**—moving step by step from data cleaning and mathematical optimization to MLOps, deep sequence models, Transformers, LLM foundations, LoRA fine-tuning, and RAG/Agents.

---

## Part I: Data Foundations (Inspect & Clean Raw Data)

1. **[01 | Python, NumPy, and Pandas Data Foundations](01_python_pandas.md)**
   - **Core Question**: How do we load messy tables into memory, manipulate multidimensional NumPy arrays, and diagnose missing values, outliers, and data types?
   - **Hands-on Project**: [`projects/01_data_foundations`](../projects/01_data_foundations/README.md)
2. **[02 | Preprocessing, Feature Engineering, and PCA](02_preprocessing_feature_engineering_pca.md)**
   - **Core Question**: How do we normalize feature scales, encode categories into numbers, compress dimensions with PCA, and prevent data leakage?
   - **Hands-on Project**: [`projects/01_data_foundations`](../projects/01_data_foundations/README.md)

---

## Part II: Classical Machine Learning & Math Engine (How Machines Learn)

3. **[03 | Linear Regression, Regularization, and Gradient Descent Optimization](03_regression_optimization.md)**
   - **Core Question**: How does a machine mathematically learn weights via a loss function (MSE), closed-form least squares, or gradient descent (GD/SGD/Adam), and how does Ridge regularization prevent overfitting?
   - **Hands-on Project**: [`projects/09_regression_and_optimization`](../projects/09_regression_and_optimization/README.md)
4. **[04 | Supervised Learning: Regression, Classification, and Model Evaluation](04_supervised_learning_evaluation.md)**
   - **Core Question**: How do we build and compare regression and classification models (Logistic Regression, Decision Trees, Random Forests, SVMs) using Cross-Validation and honest metrics (MAE, F1, PR-AUC)?
   - **Hands-on Projects**: [`projects/02_house_price_regression`](../projects/02_house_price_regression/README.md), [`projects/03_classification_and_faces`](../projects/03_classification_and_faces/README.md)
5. **[05 | Explainable AI: Local and Global SHAP Explanations](05_explainable_ai_shap.md)**
   - **Core Question**: How do we open a black-box model using Shapley values to quantify each feature's global importance and single-sample attribution?
   - **Hands-on Project**: [`projects/08_explainable_ai`](../projects/08_explainable_ai/README.md)

---

## Part III: Model Serving & Dynamic MLOps (Deploy & Monitor)

6. **[06 | Model Deployment: From scikit-learn to a Flask REST API](06_flask_model_deployment.md)**
   - **Core Question**: How do we serialize a trained pipeline and expose real-time predictions over an HTTP REST API with strict input validation?
   - **Hands-on Project**: [`projects/04_flask_iris_api`](../projects/04_flask_iris_api/README.md)
7. **[07 | Streaming Learning, Concept Drift, and Page–Hinkley](07_stream_learning_concept_drift.md)**
   - **Core Question**: What happens when real-world data distributions shift over time, and how do we detect drift (Page–Hinkley) and update models online using prequential evaluation?
   - **Hands-on Project**: [`projects/05_concept_drift`](../projects/05_concept_drift/README.md)

---

## Part IV: Deep Learning & NLP Evolution (How Machines Read Language)

8. **[08 | Deep Learning Foundations: Multi-Layer Perceptrons (MLPs), Autograd, and PyTorch](08_neural_networks_pytorch.md)**
   - **Core Question**: How do non-linear activations, computation graphs, and the universal 4-step PyTorch training loop overcome linear model limits?
   - **Hands-on Project**: [`projects/10_pytorch_mlp`](../projects/10_pytorch_mlp/README.md)
9. **[09 | Classical NLP: BoW, TF-IDF, Word Embeddings, N-grams, and Perplexity](09_classical_nlp_language_models.md)**
   - **Core Question**: How do we turn words into numbers (BoW, TF-IDF, static word embeddings) and evaluate statistical next-word language models with Perplexity (PPL)?
   - **Hands-on Project**: [`projects/11_classical_nlp`](../projects/11_classical_nlp/README.md)
10. **[10 | Multi-view Text Classification and Stacking Ensembles](10_multiview_text_classification_stacking.md)**
    - **Core Question**: How do we extract complementary text views (word TF-IDF + character n-grams) and combine base classifiers via a leak-free Stacking meta-model?
    - **Hands-on Project**: [`projects/07_multiview_stacking`](../projects/07_multiview_stacking/README.md)
11. **[11 | Neural Language Models: Embeddings, LSTMs, and Sequential Generation](11_neural_language_models.md)**
    - **Core Question**: How do learnable `nn.Embedding` tables and recurrent LSTM memory states capture word order and generate text token by token?
    - **Hands-on Project**: [`projects/12_neural_language_models`](../projects/12_neural_language_models/ASSETS.md)
12. **[12 | Transformer Architecture: Scaled Dot-Product Attention and Encoders](12_transformer_attention.md)**
    - **Core Question**: Why did sequential recurrence hit a wall, and how does parallel `Q/K/V` Self-Attention allow every token to attend directly to the entire sequence?
    - **Hands-on Project**: [`projects/13_transformer_encoder`](../projects/13_transformer_encoder/README.md)

---

## Part V: Large Language Models (LLMs), Fine-Tuning, and RAG/Agents

13. **[13 | Large Language Model (LLM) Foundations: Architectures, Pre-training, Alignment, and Decoding](13_llm_foundations.md)**
    - **Core Question**: How do we go from a Transformer block to modern LLMs (GPT, Claude, Llama, Qwen)? Why did Decoder-only win? How do BPE Tokenization, KV Cache, the **Pre-training → SFT → RLHF/DPO** lifecycle, and `Temperature / Top-k / Top-p` decoding work? Why do LLMs hallucinate?
    - **Related Projects**: [`projects/13_transformer_encoder`](../projects/13_transformer_encoder/README.md), [`projects/14_llm_prompting_and_lora`](../projects/14_llm_prompting_and_lora/README.md)
14. **[14 | LLM Customization: Prompt Engineering, Fine-Tuning Tradeoffs, and LoRA / PEFT](14_llm_prompting_finetuning_lora.md)**
    - **Core Question**: When should we choose Prompting (Zero/Few-shot/CoT) vs. RAG vs. Fine-Tuning, and how does LoRA adapt LLMs by training <1% low-rank adapter weights ($\Delta W = BA$)?
    - **Hands-on Project**: [`projects/14_llm_prompting_and_lora`](../projects/14_llm_prompting_and_lora/README.md)
15. **[15 | LLM Application Engineering: LangChain, RAG, and Agents](15_llm_langchain_rag.md)**
    - **Core Question**: How do we ground LLMs in private documents via Retrieval-Augmented Generation (RAG) to curb hallucinations, and how do Agents invoke external tools safely?
    - **Hands-on Project**: [`projects/06_llm_rag`](../projects/06_llm_rag/README.md)
