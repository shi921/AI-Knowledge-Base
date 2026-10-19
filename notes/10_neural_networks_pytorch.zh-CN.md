**语言：** [English](10_neural_networks_pytorch.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 10｜神经网络、自动微分和 PyTorch

## 从线性层到 MLP

线性变换 `z=Wx+b` 只有线性表达力；使用 ReLU 等非线性激活后堆叠多个层：`h=ReLU(W1x+b1)`，`logits=W2h+b2`。多分类中的 `CrossEntropyLoss` 接收**未经过 softmax 的 logits**，内部包含稳定的 log-softmax。训练时分别调用 `model.train()` 和 `model.eval()`；后者会影响 Dropout、BatchNorm，但不自动关闭梯度，推理通常再用 `torch.no_grad()`。

## 训练循环四步

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

| 张量概念 | 作用 | 常见错误 |
|---|---|---|
| `tensor.shape` | `[batch,features]` 或 `[batch,classes]` | 忘记 batch 维 |
| `dtype` | 模型输入浮点、分类标签整数 `long` | 把标签做 one-hot 却用 CE |
| `requires_grad` | 记录需训练的参数 | 推理时积累梯度浪费内存 |
| `DataLoader` | batch / shuffle / Dataset | 用测试集训练 |
| `state_dict` | 保存/恢复模型权重 | 只保存权重却忘记预处理参数 |

## 泛化与完整训练工程

按 stratify 的 train/val/test 划分；标准化只由 train 拟合；用 val 做早停与超参数选择；最终 test 只评估一次。保存模型 `state_dict` 和 scaler 参数，必要时记录版本与随机种子。`torch.save(model)` 依赖 pickle 并非理想的跨项目工件。

**完整实践**：[`projects/10_pytorch_mlp/main.py`](../projects/10_pytorch_mlp/main.py)，和 [`projects/03_classification_and_faces/main.py`](../projects/03_classification_and_faces/main.py) 横向对比 PyTorch MLP 与传统分类器。复杂模型不必然优于小数据上的经典模型。

**自测**：为什么每批更新前必须 `zero_grad`？为什么验证时使用 `eval` 与 `no_grad`？
