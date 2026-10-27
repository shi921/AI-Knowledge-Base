**Language:** English | [简体中文](11_neural_language_models.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 11 | Neural Language Models: Embeddings, LSTMs, and Sequential Generation

> **🧭 Knowledge Map Position**: Part IV (Deep Learning & NLP Evolution) · **Chapter 11 of 15**  
> - **💡 Intuitive Analogy**: Statistical N-grams (Chapter 09) memorize exact word sequences like a rigid lookup table—if they have seen "the dog runs" but never "the cat walks", they fail. This chapter unites neural networks (Chapter 08) with language modeling:  
>   1. **Neural N-grams** map discrete token IDs into dense semantic coordinates (**Embedding layer**, where "cat" and "dog" sit close together), sharing statistical strength across similar words.  
>   2. **LSTMs (Long Short-Term Memory networks)** act like **a reader carrying a scratchpad ($h_t, c_t$)**, updating gated memory after every word to generate text token by token.  
> - **🎯 Why Does This Matter?**: It bridges discrete token IDs and continuous neural networks, teaching how `nn.Embedding`, recurrent states, and autoregressive generation work—and **why sequential recurrence ultimately hit a parallelization wall**.  
> - **🔗 Learning Chain**: [10 | Multi-view Text Stacking](10_multiview_text_classification_stacking.md) (and [09 | Classical NLP](09_classical_nlp_language_models.md)) → This Chapter (Embeddings & recurrent language models) → **Next Chapter**: [12 | Transformers and Self-Attention](12_transformer_attention.md) (Replace sequential recurrence with parallel global attention!).

---

## 1. From Frequency Tables to Neural N-grams

A statistical N-gram counts discrete word tuples; a **Neural N-gram** maps context token IDs through a learnable lookup table (`nn.Embedding(vocab_size, embed_dim)`), concatenates the context vectors, and feeds an MLP that outputs vocabulary-wide logits trained with `CrossEntropyLoss`.
- **Vocabulary Discipline**: Build `token → index` mappings from **training text only** and map unseen test tokens to `<unk>`.

```python
# Tensor shapes through a Neural N-gram forward pass:
# x: [batch, context_length], token IDs in [0, vocab_size)
# embeddings = self.embed(x): [batch, context_length, embedding_dim]
# flatten: [batch, context_length * embedding_dim]
# output_logits: [batch, vocab_size]
# loss = CrossEntropyLoss()(output_logits, y_next_token)
```

---

## 2. Recurrent Neural Networks (RNNs) & LSTMs

To handle variable-length history beyond a fixed window $k$, **LSTMs** maintain a hidden state $h_t$ and cell state $c_t$:
$$h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})$$
using forget, input, and output gates to regulate information flow.
- **Autoregressive Generation**: During inference, feed the prompt once to obtain the final state $(h_t, c_t)$; thereafter, feed **only the newly sampled token** alongside the updated state $(h_t, c_t)$ until `<eos>` or max length is reached.

---

## 3. Why LSTMs Hit a Bottleneck (Leading to Transformers)

1. **No Parallelism Across Time Steps**: Computing step $t$ must wait for step $t-1$ to finish, underutilizing modern GPUs on massive corpora.
2. **Information Compression Bottleneck**: Cramming thousands of previous words into a single fixed-size vector $(h_t, c_t)$ loses fine-grained long-range details.
> ➡️ **Next Step**: What if every token could directly attend to every other token in parallel? That is the breakthrough of **[Chapter 12 | Transformers and Self-Attention](12_transformer_attention.md)**.

---

## 4. Hands-on Projects & Self-Test

- **Runnable Projects**:
  - Runs offline out of the box: [`projects/12_neural_language_models/ngram_train.py`](../projects/12_neural_language_models/ngram_train.py).
  - Optional pre-trained LSTM generation: [`projects/12_neural_language_models/lstm_inference.py`](../projects/12_neural_language_models/lstm_inference.py) (requires external vocabulary and weights documented in [`ASSETS.md`](../projects/12_neural_language_models/ASSETS.md)).
- **Self-Test**:
  1. Why must LSTM autoregressive decoding carry `(hidden, cell)` state from step to step?
  2. Why shouldn't you feed the final token of the prompt twice when transitioning from prompt encoding to token generation?
