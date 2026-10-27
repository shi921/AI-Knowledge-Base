# Study Notes Directory

**Language:** English | [简体中文](README.zh-CN.md) · [Home](../README.md) · [Master Roadmap](../STUDY_ROADMAP.md)

If you are new to AI, we strongly recommend following the **5-Stage Progressive Path** below so that each topic naturally answers the limitations of the previous one. After reading a note, run its linked hands-on project.

---

## 1. Progressive Learning Path (Recommended for Beginners)

### Stage 1: Data Foundations (Inspect & Clean Raw Data)
- **Step 1** · [01 | Python and Pandas Fundamentals](01_python_pandas.md) — *How do we load messy tables, inspect missing values/outliers, and compute grouped summaries?*
- **Step 2** · [02 | Preprocessing, Feature Engineering, and PCA](02_preprocessing_feature_engineering_pca.md) — *How do we normalize feature scales, encode categories, compress dimensions with PCA, and prevent data leakage?*

### Stage 2: Core Machine Learning & Math Engine (How Machines Learn Patterns)
- **Step 3** · [09 | Gradient Descent and Optimization](09_regression_optimization.md) — *How does a machine measure prediction error via a loss function and step down the gradient to learn optimal weights?*
- **Step 4** · [03 | Supervised Learning and Model Evaluation](03_supervised_learning_evaluation.md) — *How do we train regression and classification models and evaluate them honestly using Cross-Validation and proper metrics?*
- **Step 5** · [08 | Explainable AI with SHAP](08_explainable_ai_shap.md) — *How do we open the black box to quantify how much each feature pushed a prediction up or down?*

### Stage 3: Production Serving & Dynamic Maintenance (Deploy & Monitor)
- **Step 6** · [04 | Flask Model Deployment](04_flask_model_deployment.md) — *How do we serialize a trained model and serve real-time predictions over an HTTP REST API?*
- **Step 7** · [05 | Streaming Learning and Concept Drift](05_stream_learning_concept_drift.md) — *What happens when real-world data distributions shift over time, and how do we detect drift (Page–Hinkley) and update models online?*

### Stage 4: Deep Learning & NLP Evolution (How Machines Learned to Read Text)
- **Step 8** · [10 | Neural Networks and PyTorch](10_neural_networks_pytorch.md) — *How do multi-layer perceptrons (MLPs), non-linear activations, and autograd learn complex representations?*
- **Step 9** · [11 | Classical NLP and Statistical Language Models](11_classical_nlp_language_models.md) — *How do we turn words into numbers (BoW, TF-IDF, embeddings) and measure next-word prediction with N-grams and Perplexity?*
- **Step 10** · [07 | Multi-view Text Classification and Stacking](07_multiview_text_classification_stacking.md) — *How do we combine multiple feature views (e.g., body words + metadata characters) via a leak-free meta-classifier?*
- **Step 11** · [12 | Neural Language Models and LSTMs](12_neural_language_models.md) — *How do dense word embeddings and recurrent memory states (LSTMs) capture word order and generate text sequentially?*
- **Step 12** · [13 | Transformers and Self-Attention](13_transformer_attention.md) — *How does parallel `Q/K/V` self-attention replace slow recurrence so every token directly attends to every other token?*

### Stage 5: Large Language Model (LLM) Applications & Fine-Tuning (Modern GenAI)
- **Step 13** · [06 | LLMs, LangChain, and RAG](06_llm_langchain_rag.md) — *How do we ground LLMs in private documents (RAG) to reduce hallucinations and let them call external tools (Agents)?*
- **Step 14** · [14 | LLM Prompting, Fine-tuning, and LoRA](14_llm_prompting_finetuning_lora.md) — *When should we use Zero/Few-shot prompting vs. RAG vs. fine-tuning, and how does LoRA adapt LLMs cheaply?*

---

## 2. Numerical Index (01–14)

1. [Python and Pandas Fundamentals](01_python_pandas.md)
2. [Preprocessing, Feature Engineering, and PCA](02_preprocessing_feature_engineering_pca.md)
3. [Supervised Learning and Model Evaluation](03_supervised_learning_evaluation.md)
4. [Flask Model Deployment](04_flask_model_deployment.md)
5. [Streaming Learning and Concept Drift](05_stream_learning_concept_drift.md)
6. [LLMs, LangChain, and RAG](06_llm_langchain_rag.md)
7. [Multi-view Text Classification and Stacking](07_multiview_text_classification_stacking.md)
8. [Explainable AI with SHAP](08_explainable_ai_shap.md)
9. [Gradient Descent and Optimization](09_regression_optimization.md)
10. [Neural Networks and PyTorch](10_neural_networks_pytorch.md)
11. [Classical NLP and Statistical Language Models](11_classical_nlp_language_models.md)
12. [Neural Language Models and LSTMs](12_neural_language_models.md)
13. [Transformers and Self-attention](13_transformer_attention.md)
14. [LLM Prompting, Fine-tuning, and LoRA](14_llm_prompting_finetuning_lora.md)
