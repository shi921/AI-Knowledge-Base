**Language:** English | [简体中文](12_neural_language_models.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 12 | Neural Language Models: Neural N-grams, LSTMs, and Generation

> **🧭 Knowledge Map Position**: Stage 4 (Deep Learning & NLP Evolution) · **Step 11 of 14**  
> - **💡 Intuitive Analogy**: Statistical N-grams (Chapter 11) memorize exact word combinations like a rigid lookup table—if they've seen "the dog runs" but never "the cat walks", they fail. **Neural N-grams** map words into dense semantic coordinates (**Embeddings**, where "cat" and "dog" sit close together), while **LSTMs** act like **a reader carrying a scratchpad (Hidden/Cell State)**, updating notes after every word to generate the next token one step at a time.  
> - **🎯 Why Does This Matter?**: It bridges discrete token IDs and continuous neural networks, introducing **Embedding layers** and **autoregressive token-by-token text generation**.  
> - **🔗 Learning Chain**: [07 | Multi-view Stacking](07_multiview_text_classification_stacking.md) (and [11 | Classical NLP](11_classical_nlp_language_models.md)) → This Chapter (Embeddings & recurrent language models) → **Next Step**: [13 | Transformers & Self-Attention](13_transformer_attention.md) (Why sequential recurrence hits a bottleneck—and how Self-Attention revolutionized AI).

## From frequency tables to networks

A statistical N-gram model counts contexts; a **neural N-gram** maps context token IDs to embeddings, concatenates them, and feeds an MLP that predicts vocabulary-wide logits. Standard training minimizes cross entropy for the next token. Build token-to-index mappings from **training text only** and map unseen terms to `<unk>`; never rebuild vocabulary on test data.

An LSTM is recurrent: `hₜ,cₜ = LSTM(xₜ,hₜ₋₁,cₜ₋₁)`. Gates regulate how past context is remembered. For autoregressive inference, process the entire prompt first to obtain the final hidden state; then feed only the **newly generated token** with that state at each step. Stop on `<eos>` or at a token limit. Sampling may use greedy decoding, temperature, top-k, or top-p; fluent generation is not evidence of factual reliability.

## Forward-pass tensor shapes

```python
# x: [batch, context_length], IDs in [0, vocab_size)
# embeddings: [batch, context_length, embedding_dim]
# flattened: [batch, context_length * embedding_dim]
# output_logits: [batch, vocab_size]
# CrossEntropyLoss(output_logits, y_next_token)
```

## Offline versus externally supplied assets

WikiText corpora, `sample_text.txt`, `vocab.pkl`, and `lstm_model_weights.pkl` are **not bundled**. Reserved directories include `projects/12_neural_language_models/data/`, `data/wikitext-103/`, `data/wikitext-2/`, and `pretrained_model/`. See [asset placement instructions](../projects/12_neural_language_models/ASSETS.md).

- **No downloads needed:** `python projects/12_neural_language_models/ngram_train.py` trains on a mini-corpus, illustrating vocabulary, embeddings, data loading, and PPL.
- **With compatible weights and vocabulary:** `python projects/12_neural_language_models/lstm_inference.py`. Legacy `torchtext` and pickle compatibility or memory requirements can prevent the model from running; no pretrained inference result is claimed without the files.

**Connections:** revisit chapter 11 for N-grams/PPL and chapter 13 for Transformers as an alternative to recurrence.

**Review questions:** Why does LSTM inference carry hidden and cell states? Why should the final prompt token not be fed twice?
