**Language:** English | [简体中文](05_explainable_ai_shap.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 05 | Explainable AI: Local and Global SHAP Explanations

> **🧭 Knowledge Map Position**: Part II (Classical ML & Math Engine) · **Chapter 05 of 15**  
> - **💡 Intuitive Analogy**: In Chapter 04 we trained high-accuracy ensemble models like Random Forests, which act as opaque "black boxes." If a credit model rejects a loan or a clinical model flags a patient, outputting a bare score is unacceptable. **SHAP is like an X-ray for black-box models (or a fair team bonus ledger)**—it calculates exactly how much each feature pushed a specific prediction above or below the baseline average.  
> - **🎯 Why Does This Matter?**: It bridges predictive accuracy and human trust, providing both **Global Explanations** (which features drive predictions across the entire dataset) and **Local Explanations** (why *this specific instance* received its score), while guarding against confusing model attribution with real-world causality.  
> - **🔗 Learning Chain**: [04 | Supervised Learning & Evaluation](04_supervised_learning_evaluation.md) → This Chapter (Audit & explain model decisions) → **Next Chapter**: [06 | Model Deployment with Flask REST APIs](06_flask_model_deployment.md) (Package the validated, audited model into a live HTTP service).

---

## 1. Prediction Accuracy Is Not Interpretability

`model.predict(x)` tells us **what** a model predicts, whereas Explainable AI (XAI) characterizes **how** input features contributed to that output:
- **Local Explanation**: Explains a **single instance** (e.g., *"Why was applicant #42 denied credit?"*).
- **Global Explanation**: Aggregates patterns across **many instances** (e.g., *"Which three features dominate risk scores across the portfolio?"*). Never conflate the two.

---

## 2. SHAP Core Concepts & Plots

Inspired by cooperative game theory's Shapley values, SHAP distributes the difference between the background baseline (`base_value`) and the current prediction fairly across all $d$ features:

$$\text{prediction}(x) \approx \text{base\_value} + \sum_{j=1}^{d} \text{SHAP}(x_j)$$

| Concept / Plot | Meaning & How to Read |
|---|---|
| **`base_value`** | Expected model output over the background reference distribution |
| **Positive / Negative SHAP** | Positive values push the prediction higher (e.g., toward positive class `1`); negative values push it lower |
| **Local Waterfall (`plots.waterfall`)** | Shows step-by-step how individual features push a single sample from `base_value` to `f(x)` |
| **Global Summary / Beeswarm (`summary_plot`)** | Ranks features top-to-bottom by global impact; each dot is a sample colored by feature value (high=red, low=blue) |
| **Dependence Scatter** | Plots a feature's value against its SHAP attribution to reveal non-linear effects and interactions |

---

## 3. Standard Workflow with `TreeExplainer`

```python
import shap
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=80, random_state=42).fit(X_train, y_train)
explainer = shap.TreeExplainer(clf)
result = explainer(X_test.iloc[:50])

# Binary classification explanations in newer SHAP versions have shape [N, D, 2]; slice class=1
positive = result[:, :, 1] if result.values.ndim == 3 else result

shap.summary_plot(positive.values, X_test.iloc[:50], show=False)
shap.plots.waterfall(positive[0], show=False)
```

---

## 4. Case Studies & Four Critical Pitfalls

- **Mushroom & One-Hot Caveat**: When one-hot encoding a categorical column like `odor`, the resulting binary columns (`odor_foul`, `odor_none`, etc.) are mutually exclusive states of a single variable—interpret their attributions jointly.
- **Four Rules to Avoid Misinterpretation**:
  1. **SHAP Is Attribution, Not Causality**: It describes how the fitted model computes its output, not what will happen in the real world if you intervene on a feature.
  2. **Check the Output Space**: TreeExplainer may report attributions in **probability space** or **raw log-odds/margin space**; ensure your text matches the plot scale.
  3. **Collinearity Splits Credit**: Strongly correlated features share attribution depending on how trees split on them.
  4. **Check Generalization First**: Explaining an overfitted or inaccurate model produces misleading insights.

---

## 5. Hands-on Projects & Self-Test

- **Runnable Project**: [`projects/08_explainable_ai/main.py`](../projects/08_explainable_ai/main.py) (runs offline on Breast Cancer by default; pass `--dataset mushroom` for the optional UCI Mushroom dataset).
- **Self-Test**:
  1. Can a single local waterfall plot prove which feature is most important across the entire dataset?
  2. Does a positive feature value always produce a positive SHAP value?
  3. Why must you specify whether binary classification SHAP values refer to `class=1` or `class=0` (`probability` vs. `log-odds`)?
