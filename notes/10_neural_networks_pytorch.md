**Language:** English | [简体中文](10_neural_networks_pytorch.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 10 | Neural Networks, Autograd, and PyTorch

## From linear layers to MLPs

`z=Wx+b` alone expresses only linear transformations. Compose nonlinear layers using ReLU, for example `h=ReLU(W1x+b1)` followed by `logits=W2h+b2`. For multiclass classification, `CrossEntropyLoss` expects **raw logits, not softmax probabilities**, and uses a numerically stable log-softmax internally.

Call `model.train()` while training and `model.eval()` during evaluation: the latter changes Dropout/BatchNorm behavior but does **not** disable autograd. Typically wrap inference in `torch.no_grad()`.

## Four steps per training batch

```python
import torch
from torch import nn
net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
# For each DataLoader batch:
# opt.zero_grad()
# logits = net(x_batch)                 # Forward pass
# loss = loss_fn(logits, y_batch)       # Scalar loss
# loss.backward()                       # Autograd
# opt.step()                            # Parameter update
```

| Tensor concept | Purpose | Common mistake |
|---|---|---|
| `tensor.shape` | `[batch,features]` or `[batch,classes]` | Omitting batch dimension |
| `dtype` | Floating-point inputs; integer `long` class indices | One-hot targets with standard CE |
| `requires_grad` | Mark trainable parameters for gradient tracking | Accumulating unnecessary inference gradients |
| `DataLoader` | Batching, shuffling, dataset access | Training on the test set |
| `state_dict` | Save and restore weights | Forgetting accompanying preprocessing settings |

## Generalization and training engineering

Use stratified train/validation/test splits; fit scaling on training data; choose epochs and hyperparameters with validation; evaluate test only once. Save `state_dict` and the scaler or preprocessing configuration, plus versions and random seeds. `torch.save(model)` relies on pickle and is often a poor portable artifact format.

**Project:** [`projects/10_pytorch_mlp/main.py`](../projects/10_pytorch_mlp/main.py), compared against classical classifiers in [`projects/03_classification_and_faces/main.py`](../projects/03_classification_and_faces/main.py). A deeper model does not always beat simpler methods on small datasets.

**Review questions:** Why call `zero_grad()` before each update? Why use both `.eval()` and `torch.no_grad()` at evaluation time?
