**Language:** English | [简体中文](08_explainable_ai_shap.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 08 | Explainable AI: Local and Global SHAP Explanations

> **🧭 Knowledge Map Position**: Stage 2 (Core ML & Math Engine) · **Step 5 of 14**  
> - **💡 Intuitive Analogy**: If a loan model rejects an applicant or a medical model flags a patient, outputting a bare number isn't enough. **SHAP is like an X-ray for black-box models (or a fair profit-sharing ledger)**—it calculates exactly how much each feature pushed a specific prediction above or below the baseline average.  
> - **🎯 Why Does This Matter?**: It bridges the gap between accuracy and trust, giving you both **Global Explanations** (which features matter most overall) and **Local Explanations** (why *this specific sample* received its score), while warning against confusing feature attribution with causal effect.  
> - **🔗 Learning Chain**: [03 | Supervised Learning & Evaluation](03_supervised_learning_evaluation.md) → This Chapter (Audit & explain model decisions) → **Next Step**: [04 | Flask Model Deployment](04_flask_model_deployment.md) (Package the validated model into a live HTTP API).

## Prediction accuracy is not interpretability

`model.predict(x)` says **what** a model predicts. Explanation methods try to characterize **how** input features contribute to a specific prediction. Local explanations concern an instance; global explanations summarize patterns across many instances. Do not conflate the two.

## SHAP fundamentals

SHAP takes inspiration from cooperative-game Shapley values and attributes a prediction's deviation from a reference output to individual features. In the relevant output space:

`prediction ≈ base_value + Σ shap_value(feature_j)`.

- **Base value:** reference prediction induced by the background distribution.
- **Positive SHAP value:** pushes the selected output higher; negative pushes it lower.
- **Local waterfall:** shows which features drive one prediction up or down.
- **Global summary / beeswarm:** compares attribution magnitude and sign across samples.
- **Dependence scatter:** relates a feature's value to its attribution and may reveal interactions.

## Typical workflow

```python
import shap
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80, random_state=42).fit(X_train, y_train)
explainer = shap.TreeExplainer(clf)
result = explainer(X_test.iloc[:50])
# Binary classification explanations may include a class dimension.
positive = result[:, :, 1] if result.values.ndim == 3 else result
shap.summary_plot(positive.values, X_test.iloc[:50], show=False)
shap.plots.waterfall(positive[0], show=False)
```

## Illustrative datasets

The optional UCI Mushroom example uses only the `odor` input to predict edible versus poisonous classes, and illustrates a global summary plus a waterfall for an individual instance. One-hot odor categories are mutually exclusive; correlated features and the explainer's dependence assumptions require care.

A housing regression model can likewise compare a human estimate, a model prediction, and SHAP explanations. Be careful with target measurement units; `R²` is a regression metric and must not be reported as classification accuracy.

## Interpretation pitfalls

1. **SHAP is not causal inference:** changing a feature does not necessarily cause the predicted change.
2. SHAP values can refer to raw scores, log-odds, or probabilities. Specify the output space.
3. Correlated inputs and background sample choices change how contributions are assigned.
4. Assess model generalization before investing confidence in its explanation.

**Project:** [`projects/08_explainable_ai/main.py`](../projects/08_explainable_ai/main.py), using the built-in breast cancer dataset by default, with optional online `--dataset mushroom`. The demo is not clinical advice.

**Review questions:** Does a local waterfall establish global feature importance? Does positive SHAP always mean the positive class? Why does the choice of probability vs. log-odds matter?
