**语言：** [English](09_regression_optimization.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 09｜梯度下降与数值优化

## 知识链

1. 线性模型 `ŷ = Xw`：给每个样本添加偏置项列 `1`，把截距并入 `w`。
2. 平方误差 `L(w) = (1/n)||Xw-y||²`；梯度是 `∇L = (2/n)Xᵀ(Xw-y)`。
3. **解析解**：若可逆，`w* = (XᵀX)⁻¹Xᵀy`；实际通常优先 `np.linalg.lstsq` 或带稳定正则的线性求解，避免显式求逆。
4. **梯度下降**：`w ← w - η∇L(w)`；学习率 `η` 过大可能发散，过小收敛缓慢。
5. **SGD 与 mini-batch**：只用样本子集估计梯度，降低单次更新开销；Adam 根据梯度一阶、二阶矩自适应调整。

```python
import numpy as np
X = np.array([[1., 1.], [1., 2.], [1., 3.]])
y = np.array([2., 3., 4.])
w = np.zeros(2)
for _ in range(200):
    grad = (2 / len(y)) * X.T @ (X @ w - y)
    w -= 0.05 * grad
print(w)  # Approximately [1, 1]
```

## 多项式和 Ridge

多项式回归依旧是**对参数线性**：设 `φ(x)=[1,x,x²,...,xᵈ]`，再用线性回归拟合。`d` 增大会降低训练误差，但可能提高测试误差（高方差）。岭回归添加 `λΣ_{j≥1}wⱼ²`，截距通常不惩罚。若特征尺度差异大，梯度下降路径曲折；应只在训练集拟合 scaler，避免泄漏。

| 观察现象 | 检查什么 |
|---|---|
| 训练损失为 `nan` | 学习率过大，输入异常，溢出 |
| train 好、test 差 | 过拟合；查看多项式阶数、正则强度 |
| 两种解不一致 | 是否使用同一特征与目标，GD 是否收敛 |
| 矩阵求解失败 | 共线性、秩亏，使用 `lstsq` 或 Ridge |

**完整实践**：[`projects/09_regression_and_optimization/main.py`](../projects/09_regression_and_optimization/main.py)；与 [`projects/02_house_price_regression/main.py`](../projects/02_house_price_regression/main.py) 连起来复习：先理解解法，再用 Pipeline 在模拟数据上比较泛化。

**自测**：为什么 `XᵀX` 可能奇异？为什么多项式次数提高后训练 MSE 下降、测试 MSE 却上升？
