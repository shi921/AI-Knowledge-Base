**Language:** English | [简体中文](08_neural_networks_pytorch.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 08 | Deep Learning Foundations: Multi-Layer Perceptrons (MLPs), Autograd, and PyTorch

> **🧭 Knowledge Map Position**: Part IV (Deep Learning & NLP Evolution) · **Chapter 08 of 15**  
> - **💡 Intuitive Analogy**: A linear model $z = Wx + b$ (Chapter 03) can only draw a straight line—stacking ten linear layers without anything in between still collapses into a single straight line! A **Multi-Layer Perceptron (MLP)** inserts **non-linear activation functions (like ReLU, a hinge that zeroes out negatives)** between linear layers like LEGO bricks to bend and fit complex patterns. Meanwhile, **PyTorch Autograd** acts as an automatic bookkeeper that records every forward operation and computes all gradients in a single `.backward()` call.  
> - **🎯 Why Does This Matter?**: It transitions you from scikit-learn estimators to **PyTorch**, introducing tensors, logits, cross-entropy loss, and the universal **4-step training loop** used across all deep learning and LLM training.  
> - **🔗 Learning Chain**: [07 | Streaming Learning & Concept Drift](07_stream_learning_concept_drift.md) (and [03 | Gradient Descent](03_regression_optimization.md)) → This Chapter (PyTorch & neural network foundations) → **Next Chapter**: [09 | Classical NLP: BoW, TF-IDF, Embeddings, and N-grams](09_classical_nlp_language_models.md) (Begin the journey of teaching machines to read human language).

---

## 1. From Linear Layers to MLPs: Why Non-Linear Activations Are Essential

1. **Linear Collapse**: Without non-linear activations, $W_2(W_1 x + b_1) + b_2 = W' x + b'$—any depth of linear layers reduces to a single linear matrix multiplication.
2. **Non-Linear Composition**: Inserting **ReLU** ($\max(0, z)$) or **GELU** lets stacked layers learn hierarchical features:
   $$h = \text{ReLU}(W_1 x + b_1), \quad \text{logits} = W_2 h + b_2$$
3. **Raw Logits & `CrossEntropyLoss`**: For multiclass classification, PyTorch's `nn.CrossEntropyLoss()` expects **unnormalized raw logits** (not `Softmax` probabilities!) because it applies a numerically stable `log_softmax` internally.

---

## 2. The Universal 4-Step PyTorch Training Loop

```python
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)   # Outputs raw logits for 10 classes
)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# For each batch from DataLoader:
# 1. Clear gradients from the previous step
# opt.zero_grad()
# 2. Forward pass to compute raw logits
# logits = net(x_batch)
# 3. Compute scalar loss
# loss = loss_fn(logits, y_batch)
# 4. Backpropagate gradients & update parameters
# loss.backward()
# opt.step()
```

---

## 3. Core PyTorch Tensor Concepts & Common Pitfalls

| Concept | Role in Deep Learning | Common Beginner Mistake |
|---|---|---|
| **`tensor.shape`** | Leading dimension is batch size: `[batch, features]` or `[batch, classes]` | Forgetting the batch dimension when predicting on a single sample (use `.unsqueeze(0)`) |
| **`dtype`** | Inputs/weights use `float32`; class target indices `y` must be `torch.long` (`int64`) | Passing one-hot float vectors or `int32` labels to standard class-index `CrossEntropyLoss` |
| **`requires_grad`** | Tracks operations on parameters for automatic differentiation | Running validation/inference without `with torch.no_grad():`, wasting memory on unused computation graphs |
| **`train()` vs. `eval()`** | Switches Dropout and BatchNorm behavior between training and inference | Assuming `model.eval()` disables gradient tracking (it does not—always pair with `torch.no_grad()`) |
| **`state_dict`** | Saves/loads parameter tensors cleanly via `model.state_dict()` | Saving model weights while forgetting to save the training `StandardScaler` statistics |

---

## 4. Generalization & Engineering Discipline

Split data into stratified `train / val / test` sets; fit feature scalers on `train` only; use `val` for early stopping and hyperparameter tuning; evaluate `test` once. Save `state_dict` alongside preprocessing parameters and random seeds.

---

## 5. Hands-on Projects & Self-Test

- **Runnable Project**: [`projects/10_pytorch_mlp/main.py`](../projects/10_pytorch_mlp/main.py) (compare PyTorch MLP performance on Digits against classical classifiers in [`projects/03_classification_and_faces/main.py`](../projects/03_classification_and_faces/main.py)).
- **Self-Test**:
  1. Why must we call `optimizer.zero_grad()` before every `loss.backward()` step?
  2. Why do we use both `model.eval()` and `with torch.no_grad():` during validation and inference?
