**语言：** [English](08_neural_networks_pytorch.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 08｜深度学习基石：多层感知机（MLP）、自动微分与 PyTorch

> **🧭 知识脉络定位**：第四篇「深度学习与自然语言处理演进」· **第 08 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：第 03 章的线性模型 $z = Wx + b$ 就像一把只能画直线的硬尺子，哪怕你把十把直尺叠在一起，算出来的依然是一条直线！为了画出任意弯曲的复杂形状，**多层感知机（MLP / 神经网络）** 在每层线性变换之间插入了**非线性激活函数（如 ReLU，把负数折断归零的“折页关节”）**，像搭乐高积木一样层层组合出高阶特征。而 **PyTorch 的 Autograd（自动微分）** 就像一位自动记账员，把前向计算的每一步都记在计算图账本上，调用 `.backward()` 时一键算出所有层权重的梯度！  
> - **🎯 这一章解决什么核心问题？**：带你从传统 Scikit-learn 调包正式跨越到现代深度学习框架 **PyTorch**。吃透张量（Tensor）、非线性激活、交叉熵损失（Logits）以及贯穿所有深度学习与大模型训练的**“PyTorch 训练循环四步法”**。  
> - **🔗 前后衔接**：[07｜流式数据学习与概念漂移](07_stream_learning_concept_drift.zh-CN.md)（及 [03｜梯度下降优化](03_regression_optimization.zh-CN.md)）→ 本章（筑牢 PyTorch 与深度学习神经网络底座）→ **下一章**：[09｜传统 NLP：BoW、TF-IDF、词向量与统计语言模型](09_classical_nlp_language_models.zh-CN.md)（开始探索如何让计算机处理人类语言文字）。

---

## 一、 从线性层到多层感知机（MLP）：为什么必须有激活函数？

1. **纯线性堆叠的坍缩**：如果两层网络没有激活函数，第一层 $h = W_1 x + b_1$，第二层 $y = W_2 h + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W' x + b'$——无论叠多少层，它永远等价于单层线性回归！
2. **引入非线性激活（Non-linear Activation）**：
   - 使用 **ReLU**（$\text{ReLU}(z) = \max(0, z)$）或 **GELU**（大模型常用）打破线性约束：
     $$h = \text{ReLU}(W_1 x + b_1), \quad \text{logits} = W_2 h + b_2$$
3. **输出层与 `CrossEntropyLoss` 的黄金搭档**：
   - 模型最后一层线性层输出的原始未归一化得分称为 **Logits**（范围在 $(-\infty, +\infty)$）。
   - 在 PyTorch 中，多分类损失函数 `nn.CrossEntropyLoss()` **要求直接传入原始 Logits，绝不要在模型末尾自己先加一层 `Softmax`**！因为 `CrossEntropyLoss` 内部已經集成了数值稳定的 `LogSoftmax + NLLLoss`。

---

## 二、 PyTorch 训练循环四步法（背熟这段通用模板）

无论训练简单的多层感知机还是千亿参数的大语言模型，PyTorch 的每个 Batch 训练永远遵循以下 4 步：

```python
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)   # 输出 10 个类别的原始 logits
)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 遍历 DataLoader 中的每一个 mini-batch:
# 1. 清空上一轮残留梯度（否则 PyTorch 会默认累加梯度！）
# opt.zero_grad()
# 2. 前向传播算出预测 logits
# logits = net(x_batch)
# 3. 计算标量损失
# loss = loss_fn(logits, y_batch)
# 4. 反向传播自动求导 + 优化器更新参数
# loss.backward()
# opt.step()
```

---

## 三、 PyTorch 核心概念与高频踩坑对照表

| 核心概念 | 在神经网络中的作用 | 新手最容易踩的坑 |
|---|---|---|
| **`tensor.shape`** | 永远以 `[batch_size, ...]` 开头，如特征 `[B, D]`、分类输出 `[B, C]` | 单样本推理时忘了加 batch 维（需用 `.unsqueeze(0)` 变成 `[1, D]`） |
| **`dtype`（数据类型）** | 神经网络权重与输入特征通常用 `torch.float32`，而多分类目标标签 `y` 必须是 `torch.long` (`int64`) | 把 `float` 类型的标签或者 One-Hot 编码后的矩阵传给 `CrossEntropyLoss` 导致报错 |
| **`requires_grad`** | 标记需要跟踪计算图并求导的参数张量 | 在验证/测试推理时忘了加 `with torch.no_grad():`，导致计算图不断膨胀撑爆显存 |
| **`model.train()` vs `eval()`** | 切换 Dropout（训练随机丢弃、推理全保留）与 BatchNorm 的工作模式 | 以为调了 `model.eval()` 就会自动关闭梯度计算（其实不会，仍需配合 `torch.no_grad()`） |
| **`state_dict` 模型保存** | `torch.save(model.state_dict(), path)` 仅保存参数权重字典，安全且跨目录兼容性好 | 只保存了网络权重 `state_dict`，却忘了保存训练集的 `StandardScaler` 均值方差，导致推理输入尺度错乱 |

---

## 四、 完整深度学习工程规范

1. **严格划分数据集**：按类别分层（`stratify=y`）切分 `train / val / test`；`StandardScaler` 仅在 `train` 上拟合。
2. **用验证集（Val）做早停（Early Stopping）与选模**：每个 Epoch 结束后在验证集上评估，保存验证集损失最低或准确率最高那一轮的 `state_dict`，防止后期过拟合。
3. **模型复杂度认知**：在样本量很小（如仅几百条）的结构化表格数据上，多层神经网络并不一定优于随机森林或逻辑回归（可对比第 04 章 Digits 分类结果）。深度学习真正的威力体现在图像、音频与自然语言序列等高维非结构化数据上。

---

## 五、 配套实践与自测

- **可运行项目**：[`projects/10_pytorch_mlp/main.py`](../projects/10_pytorch_mlp/main.py)（完整的 PyTorch MLP 训练、验证评估与权重保存，可与 [`projects/03_classification_and_faces/main.py`](../projects/03_classification_and_faces/main.py) 的传统分类器横向对比）。
- **思考与自测**：
  1. 为什么每次调用 `loss.backward()` 之前必须先调用 `optimizer.zero_grad()`？如果不清空会发生什么？（提示：顺便思考第 14 章微调大模型时的“梯度累积 Gradient Accumulation”是怎么利用这一特性的）
  2. 为什么验证和推理阶段需要同时使用 `model.eval()` 和 `with torch.no_grad():`？二者各自负责什么？
