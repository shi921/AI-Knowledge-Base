**Language:** English | [简体中文](09_regression_optimization.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 09 | Gradient Descent and Numerical Optimization

## Key concept chain

1. Linear prediction: `ŷ = Xw`. Add a column of ones to incorporate the intercept in `w`.
2. Mean squared error: `L(w) = (1/n)||Xw-y||²`, with gradient `∇L = (2/n)Xᵀ(Xw-y)`.
3. **Closed-form solution:** when invertible, `w* = (XᵀX)⁻¹Xᵀy`. In code prefer `np.linalg.lstsq` or regularized solvers instead of forming the inverse.
4. **Gradient descent:** `w ← w - η∇L(w)`. Too-large learning rates can diverge; too-small ones converge slowly.
5. **SGD and mini-batches:** estimate gradients from subsets. Adam adapts parameter updates using the first and second moments of gradients.

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

## Polynomial features and Ridge

Polynomial regression remains **linear in its coefficients**: transform one value to `φ(x)=[1,x,x²,...,xᵈ]` and fit a linear model. Higher degrees often reduce training error but can increase test error (variance/overfitting). Ridge adds `λΣ_{j≥1}wⱼ²`; the intercept is usually unpenalized. Uneven feature scales slow gradient descent, so fit the scaler on training data only.

| Observation | Check |
|---|---|
| Training loss becomes `nan` | Oversized learning rate, abnormal input, overflow |
| Training good, test poor | Overfitting; polynomial degree and regularization |
| Closed-form and GD solutions disagree | Feature/target consistency and GD convergence |
| Matrix solve fails | Collinearity or rank deficiency; use `lstsq` or Ridge |

**Project:** [`projects/09_regression_and_optimization/main.py`](../projects/09_regression_and_optimization/main.py). Pair with [`projects/02_house_price_regression/main.py`](../projects/02_house_price_regression/main.py) to move from optimization mechanics to pipeline-based generalization.

**Review questions:** Why can `XᵀX` be singular? Why might increasing polynomial degree lower training MSE while increasing test MSE?
