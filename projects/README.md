# Hands-on Project Directory

**Language:** English | [简体中文](README.zh-CN.md) · [Home](../README.md) · [Master Roadmap](../STUDY_ROADMAP.md)

All scripts use English comments, user-facing messages, and logs. Beginners are encouraged to run projects following the **5-Part, 15-Chapter progression in `notes/`**:

## 1. Projects Grouped by the 5 Progressive Parts

- **Part I: Data Foundations (Chapters 01–02)**
  - [`01_data_foundations`](01_data_foundations/README.md): Missing-value diagnostics, feature scaling, correlation inspection, PCA dimensionality reduction, and rolling time-series smoothing ([Chapter 01](../notes/01_python_pandas.md), [Chapter 02](../notes/02_preprocessing_feature_engineering_pca.md)).
- **Part II: Classical Machine Learning & Math Engine (Chapters 03–05)**
  - [`09_regression_and_optimization`](09_regression_and_optimization/README.md): From-scratch gradient descent, closed-form least squares, polynomial regression, and Ridge regularization ([Chapter 03](../notes/03_regression_optimization.md)).
  - [`02_house_price_regression`](02_house_price_regression/README.md): End-to-end house price regression pipeline with leak-free preprocessing and cross-validation ([Chapter 04](../notes/04_supervised_learning_evaluation.md)).
  - [`03_classification_and_faces`](03_classification_and_faces/README.md): Multi-classifier comparison, confusion matrices, and optional Eigenfaces PCA+SVM workflow ([Chapter 04](../notes/04_supervised_learning_evaluation.md)).
  - [`08_explainable_ai`](08_explainable_ai/README.md): Global SHAP summary plots and single-sample local waterfall explanations with TreeExplainer ([Chapter 05](../notes/05_explainable_ai_shap.md)).
- **Part III: Model Serving & Dynamic MLOps (Chapters 06–07)**
  - [`04_flask_iris_api`](04_flask_iris_api/README.md): Model serialization, Flask REST API endpoint, JSON input validation, and pytest suite ([Chapter 06](../notes/06_flask_model_deployment.md)).
  - [`05_concept_drift`](05_concept_drift/README.md): Prequential stream evaluation, Page–Hinkley drift detection alerts, and sliding-window retraining ([Chapter 07](../notes/07_stream_learning_concept_drift.md)).
- **Part IV: Deep Learning & NLP Evolution (Chapters 08–12)**
  - [`10_pytorch_mlp`](10_pytorch_mlp/README.md): PyTorch Multi-Layer Perceptron (MLP) training loop, validation tracking, and checkpoint saving ([Chapter 08](../notes/08_neural_networks_pytorch.md)).
  - [`11_classical_nlp`](11_classical_nlp/README.md): TF-IDF text classification, Laplace-smoothed statistical N-gram language models, and Perplexity (PPL) evaluation ([Chapter 09](../notes/09_classical_nlp_language_models.md)).
  - [`07_multiview_stacking`](07_multiview_stacking/README.md): Word TF-IDF + character n-gram multi-view base classifiers combined via a leak-free Stacking meta-model ([Chapter 10](../notes/10_multiview_text_classification_stacking.md)).
  - [`12_neural_language_models`](12_neural_language_models/ASSETS.md): Trainable embedding-based Neural N-gram language model and optional pre-trained LSTM text generation ([Chapter 11](../notes/11_neural_language_models.md)).
  - [`13_transformer_encoder`](13_transformer_encoder/README.md): Compact PyTorch Multi-Head Self-Attention sentiment encoder and attention heatmap visualization ([Chapter 12](../notes/12_transformer_attention.md), [Chapter 13](../notes/13_llm_foundations.md)).
- **Part V: Large Language Models (LLMs), Fine-Tuning & RAG/Agents (Chapters 13–15)**
  - [`14_llm_prompting_and_lora`](14_llm_prompting_and_lora/README.md): Zero/Few-shot prompt templates and HuggingFace PEFT/LoRA training & inference pipeline on Qwen2.5 ([Chapter 13](../notes/13_llm_foundations.md), [Chapter 14](../notes/14_llm_prompting_finetuning_lora.md)).
  - [`06_llm_rag`](06_llm_rag/README.md): Overlapping document chunking, offline TF-IDF retrieval, optional LangChain LLM generation, and Agent tool calling ([Chapter 15](../notes/15_llm_langchain_rag.md)).

---

## 2. Project Folder Index

1. [Data foundations (`01_data_foundations`)](01_data_foundations/README.md)
2. [House price regression (`02_house_price_regression`)](02_house_price_regression/README.md)
3. [Classification and faces (`03_classification_and_faces`)](03_classification_and_faces/README.md)
4. [Flask Iris API (`04_flask_iris_api`)](04_flask_iris_api/README.md)
5. [Concept drift (`05_concept_drift`)](05_concept_drift/README.md)
6. [LLM and RAG (`06_llm_rag`)](06_llm_rag/README.md)
7. [Multi-view stacking (`07_multiview_stacking`)](07_multiview_stacking/README.md)
8. [Explainable AI (`08_explainable_ai`)](08_explainable_ai/README.md)
9. [Regression and optimization (`09_regression_and_optimization`)](09_regression_and_optimization/README.md)
10. [PyTorch MLP (`10_pytorch_mlp`)](10_pytorch_mlp/README.md)
11. [Classical NLP (`11_classical_nlp`)](11_classical_nlp/README.md)
12. [Neural language models (`12_neural_language_models`)](12_neural_language_models/ASSETS.md)
13. [Transformer encoder (`13_transformer_encoder`)](13_transformer_encoder/README.md)
14. [Prompting and LoRA (`14_llm_prompting_and_lora`)](14_llm_prompting_and_lora/README.md)
