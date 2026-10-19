**Language:** English | [简体中文](ASSETS.zh-CN.md) · [Home](../../README.md)

# Neural Language Models: External Data and Model Assets

Empty directories are committed using `.gitkeep` so that you can add properly licensed local files later. **Do not commit large, private, or third-party model/data assets to the public repository.**

```text
projects/12_neural_language_models/
├── data/
│   ├── sample_text.txt               (optional custom text for neural N-gram training)
│   ├── wikitext-103/
│   │   ├── vocab.pkl                 (required for LSTM inference)
│   │   ├── wiki.train.tokens         (optional)
│   │   ├── wiki.valid.tokens         (optional)
│   │   └── wiki.test.tokens          (optional)
│   └── wikitext-2/                   (optional corpus directory)
├── pretrained_model/
│   └── lstm_model_weights.pkl        (compatible weights required for inference)
├── ngram_train.py
├── lstm_inference.py
└── data_processing.py
```

Without external assets you can run the built-in mini-corpus: `python projects/12_neural_language_models/ngram_train.py --epochs 3`. After adding a compatible vocabulary and weights, try `python projects/12_neural_language_models/lstm_inference.py --prompt "the quick brown fox"`.

The expected LSTM uses `embedding_dim=512`, `hidden_dim=1024`, and `num_layers=3`; vocabulary, tokenizer, checkpoint layout, and `torchtext` versions must be compatible. Unpickling `vocab.pkl` can execute code: load trusted sources only. `.gitkeep` is an empty placeholder, not a dataset or model.

**Related notes:** [Chapter 12](../../notes/12_neural_language_models.md).
