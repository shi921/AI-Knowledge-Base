**Language:** English | [简体中文](03_regression_optimization.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 03 | Linear Regression, Regularization, and Gradient Descent Optimization

> **🧭 Knowledge Map Position**: Part II (Classical ML & Math Engine) · **Chapter 03 of 15**  
> - **💡 Intuitive Analogy**: How does a machine actually "learn" from data? Imagine standing **blindfolded on a foggy hillside trying to reach the lowest valley (minimum prediction error)**. You feel the slope under your feet (**Gradient**) and take a step downhill proportional to your stride (**Learning Rate**). Step too far and you overshoot into `NaN`; step too timidly and training takes forever.  
> - **🎯 Why Is This Chapter the Engine of All AI?**: Whether you are fitting a simple house-price line, a PyTorch neural network (Chapter 08), or a 100-billion-parameter LLM (Chapter 13), the core training loop is identical: **Prediction Formula → Loss Function → Gradient Step → Regularization**. Mastering this chapter demystifies the engine inside every model you will ever train.  
> - **🔗 Learning Chain**: [02 | Preprocessing, Feature Engineering, and PCA](02_preprocessing_feature_engineering_pca.md) → This Chapter (Mathematical optimization & regularization) → **Next Chapter**: [04 | Supervised Learning: Regression, Classification, and Evaluation](04_supervised_learning_evaluation.md) (Expand to the full supervised algorithm toolkit and honest evaluation protocols).

---

## 1. Core Concept Chain: From Linear Prediction to Gradient Descent

1. **Linear Prediction Model (Hypothesis)**: Given feature matrix $X \in \mathbb{R}^{n \times d}$ with a prepended column of ones (absorbing the intercept $b$ into weight vector $w$), predictions are computed via matrix multiplication:
   $$\hat{y} = Xw$$
2. **Loss Function (Mean Squared Error, MSE)**: Quantifies how far predictions $\hat{y}$ deviate from ground truth $y$:
   $$L(w) = \frac{1}{n} \|Xw - y\|_2^2$$
3. **Method A: Closed-Form Solution (Normal Equation)**: Setting the derivative to zero yields $w^* = (X^T X)^{-1} X^T y$. In practice, explicit matrix inversion is slow and numerically unstable when features are collinear; prefer `np.linalg.lstsq` or regularized solvers.
4. **Method B: Gradient Descent (Universal Optimization Engine)**: When models are non-linear or massive, closed-form solutions do not exist. Instead, we compute the gradient of the loss with respect to $w$:
   $$\nabla_w L(w) = \frac{2}{n} X^T (Xw - y)$$
   and iteratively step opposite the gradient:
   $$w \leftarrow w - \eta \nabla_w L(w)$$
   where $\eta$ (learning rate) controls step size.

```python
import numpy as np

# Feature matrix X with bias column of 1s; true relation y = 1 + 1 * x
X = np.array([[1., 1.], [1., 2.], [1., 3.]])
y = np.array([2., 3., 4.])

w = np.zeros(2)
lr = 0.05
for _ in range(200):
    grad = (2 / len(y)) * X.T @ (X @ w - y)
    w -= lr * grad
print('Gradient descent weights:', w)  # Converges to [1., 1.]
```

---

## 2. Optimizer Evolution: Batch GD vs. SGD vs. Mini-batch vs. Adam

| Optimizer Variant | Data Used per Step | Pros, Cons & Role in Modern AI |
|---|---|---|
| **Batch Gradient Descent** | All $n$ training samples | Exact gradient, but prohibitively slow and memory-heavy on large datasets |
| **Stochastic GD (SGD)** | $1$ random sample | Fast updates, but high variance causes noisy zig-zag trajectories |
| **Mini-batch GD** | Small batch (e.g., `32` or `64` samples) | **Standard in Deep Learning**: balances GPU matrix parallelism with stable gradients |
| **Adam** | Mini-batch + 1st & 2nd moment running averages | Adapts per-parameter learning rates automatically; default optimizer for PyTorch and LLMs |

---

## 3. Underfitting, Overfitting, and Regularization (Polynomial & Ridge)

1. **Polynomial Regression**: A linear model can fit curved relationships by expanding inputs into $\phi(x) = [1, x, x^2, \dots, x^d]$ and fitting $\hat{y} = \phi(X)w$ (still linear in parameters $w$).
2. **Bias–Variance Tradeoff**:
   - Degree $d$ too small ➡️ **Underfitting (High Bias)**: poor fit on both train and test sets.
   - Degree $d$ too large ➡️ **Overfitting (High Variance)**: memorizes training noise (tiny train MSE) but fails wildly on unseen test points.
3. **Regularization (Reining in Exploding Weights)**:
   - **Ridge Regression (L2 Regularization)** adds a squared penalty on non-intercept weights:
     $$L_{\text{Ridge}}(w) = \frac{1}{n}\|Xw - y\|_2^2 + \lambda \sum_{j \ge 1} w_j^2$$
   - **Lasso (L1 Regularization, $\lambda \sum |w_j|$)** drives irrelevant weights to exact zero for feature selection.
   - **Mandatory Scaling**: Because gradient descent trajectories and L1/L2 penalties depend on feature scales, always standardize features (`StandardScaler` fit on train only) before optimization!

---

## 4. Troubleshooting Optimization Failures

| Observed Symptom | Root Cause & Fix |
|---|---|
| Training loss becomes `nan` or `inf` | **Learning rate $\eta$ too large** (overshooting); or uncleaned `NaN`/extreme outliers in inputs |
| Gradient descent converges very slowly | Unscaled features create elongated loss valleys; apply `StandardScaler` on training split |
| Low train MSE, huge test MSE | **Overfitting**; lower polynomial degree $d$, increase Ridge `alpha`, or gather more data |
| Closed-form and GD weights disagree | Check whether both include the intercept column, use identical scaling, and ran enough GD steps |
| Matrix inversion fails (`Singular matrix`) | Collinear or rank-deficient features; use `np.linalg.lstsq` or `Ridge` |

---

## 5. Hands-on Projects & Self-Test

- **Runnable Project**: [`projects/09_regression_and_optimization/main.py`](../projects/09_regression_and_optimization/main.py) (compares closed-form least squares, gradient descent, polynomial degrees, and Ridge regularization).
- **Self-Test**:
  1. Why can $X^T X$ become singular when two features are collinear, and how does Ridge ($\lambda I$) fix it?
  2. Why does increasing polynomial degree $d$ always decrease training MSE but eventually increase test MSE?
