**语言：** [English](ASSETS.md) | 简体中文 · [首页](../../README.zh-CN.md)

# 神经语言模型：外部数据和模型放置说明

仓库在以下位置预留真实空目录，并使用 `.gitkeep` 让 Git 保留目录结构。请将自己有权使用的文件放入对应目录；**不要提交大文件、私有数据、预训练权重到公共仓库**。

```text
projects/12_neural_language_models/
├── data/
│   ├── sample_text.txt               (optional custom training text)
│   ├── wikitext-103/
│   │   ├── vocab.pkl                 (required vocabulary for LSTM inference)
│   │   ├── wiki.train.tokens         (optional)
│   │   ├── wiki.valid.tokens         (optional)
│   │   └── wiki.test.tokens          (optional)
│   └── wikitext-2/                  (optional corpus directory)
├── pretrained_model/
│   └── lstm_model_weights.pkl        (compatible LSTM weights required for inference)
├── ngram_train.py
├── lstm_inference.py
└── data_processing.py
```

未添加文件时，可离线运行 `python projects/12_neural_language_models/ngram_train.py --epochs 3`（内置迷你语料）。补齐兼容权重与词表后，运行 `python projects/12_neural_language_models/lstm_inference.py --prompt "the quick brown fox"`。LSTM 架构采用 `embedding_dim=512, hidden_dim=1024, num_layers=3`，词表、Tokenizer、权重和 torchtext 版本必须匹配。`vocab.pkl` 涉及 pickle 反序列化，仅加载可信文件。

提示：`.gitkeep` 是空文件，不是数据或模型本身。

**相关知识笔记：** [第 11 章](../../notes/11_neural_language_models.zh-CN.md)。
