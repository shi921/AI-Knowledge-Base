**语言：** [English](ENVIRONMENT_AND_VALIDATION.md) | 简体中文

# 运行验证与环境说明

## 环境

推荐 Python 3.10+（某些旧版 torchtext 权重可能需要独立兼容环境）。先安装 `requirements-core.txt`，仅在需要时安装 `requirements-optional.txt`、`requirements-deep-learning.txt` 或 `requirements-finetuning.txt`。

## 运行情况说明

双语改造后，以下入口使用轻量配置进行了实际运行测试，且捕获的命令行提示均为英文。换到自己的操作系统后，仍需重新确认依赖兼容性：

- 数据探索、房价回归、Digits 分类、STAGGER 与 SEA 漂移、离线 RAG 检索、合成多视角 Stacking、SHAP、回归优化、PyTorch MLP、传统 NLP、Neural N-gram 和 Transformer 小示例。
- 多视角 Stacking 改为自动生成的合成样例，已移除非公开的社交媒体数据。首次使用请从仓库根目录运行 `python projects/07_multiview_stacking/main.py`。
- Flask API 与外部模型 / 需要额外联网的扩展路径，不能据此认为已经过完整端到端生产验证。

## 未内置的资源

- `projects/01_data_foundations/customer_pca.py`：可通过 `--data` 使用自己授权的客户表，仓库不提供客户记录。
- `projects/03_classification_and_faces/faces.py`：可通过 `--data` 使用你有权使用的人脸 CSV，仓库不提供人脸样本。
- `projects/07_multiview_stacking/`：默认模拟数据；通过 `--data` 接入授权的本地 CSV。
- `projects/12_neural_language_models/`：WikiText、词表、LSTM 权重在预留目录中补充，见 `ASSETS.md`。
- `projects/13_transformer_encoder/`：默认内置样例，使用 `--data-dir` 可加载私有可信 pickle。
- `projects/14_llm_prompting_and_lora/`：无预训练大模型和 LoRA checkpoint，需要单独下载授权模型或训练权重。

## 安全与可复现性

- `.env`、大型模型、私有数据和生成文件被 `.gitignore` 排除；不要提交凭据，若已泄露请轮换。
- `.pkl`, `.pt`, `.joblib` 等反序列化可能运行任意代码，只加载可信来源。
- 公共数据集与预训练模型的许可证可能不同于本仓库代码；公开再分发前必须检查授权。
- 用固定随机种子和独立测试集复现实验；小型玩具语料与合成标签的准确率仅供验证流程。运行 `python scripts/generate_demo_data.py` 可重建演示 CSV。
