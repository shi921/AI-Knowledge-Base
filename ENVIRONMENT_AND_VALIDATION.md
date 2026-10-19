**Language:** English | [简体中文](ENVIRONMENT_AND_VALIDATION.zh-CN.md)

# Environment, Validation, and Reproducibility

## Environment

Python 3.10+ is recommended; legacy `torchtext` checkpoints may require a separate compatible environment. Install `requirements-core.txt` first. Install `requirements-optional.txt`, `requirements-deep-learning.txt`, and `requirements-finetuning.txt` only when required.

## What has been checked

After bilingual localization, the following entry points were actually executed successfully with lightweight settings: data foundations, housing regression, Digits classification, STAGGER and SEA drift, offline RAG retrieval, synthetic multi-view stacking, SHAP, numerical optimization, classical NLP, PyTorch MLP, neural N-gram, and Transformer encoder. All captured CLI messages in these runs were in English. Recheck dependencies on your own operating system.

- Multi-view stacking now uses programmatically generated synthetic records; no private social-media data is included. Run `python projects/07_multiview_stacking/main.py` from the repository root.
- Flask HTTP behavior, external model downloads, and network-dependent extensions have **not** been fully validated as production-ready end-to-end systems.

## Resources you must supply yourself

- `projects/01_data_foundations/customer_pca.py`: pass an authorized local customer spreadsheet using `--data`.
- `projects/03_classification_and_faces/faces.py`: pass an authorized face-feature CSV using `--data`.
- `projects/07_multiview_stacking/`: uses synthetic samples by default; `--data` accepts a local authorized CSV.
- `projects/12_neural_language_models/`: provide WikiText, vocabulary, and LSTM weights in the reserved folders; see `ASSETS.md`.
- `projects/13_transformer_encoder/`: uses built-in examples; `--data-dir` allows trusted local pickle files.
- `projects/14_llm_prompting_and_lora/`: no pretrained base model or LoRA checkpoint is bundled; acquire or train appropriately licensed assets separately.

## Security and reproducibility

- `.gitignore` excludes `.env`, large models, private data, and generated artifacts. Never commit live keys. Rotate any exposed secrets.
- Deserializing `.pkl`, `.pt`, or `.joblib` may execute arbitrary code; load only from trustworthy sources.
- External datasets and pretrained models can have licensing terms distinct from this repository; check them before redistribution.
- Fix random seeds, use held-out test sets, and avoid treating toy-corpus scores or synthetic-label accuracies as realistic benchmarks. Run `python scripts/generate_demo_data.py` to regenerate example CSVs.
