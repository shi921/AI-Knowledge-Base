**Language:** English | [简体中文](06_flask_model_deployment.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 06 | Model Deployment: From scikit-learn to a Flask REST API

> **🧭 Knowledge Map Position**: Part III (Model Serving & MLOps) · **Chapter 06 of 15**  
> - **💡 Intuitive Analogy**: Perfecting and auditing a recipe in your home kitchen (Chapters 04 & 05) doesn't serve customers until you **open a takeout window (HTTP REST API)**. You freeze-pack the trained pipeline (`joblib` serialization), load it once when the server starts, validate customer orders (JSON inputs), and return predictions over HTTP.  
> - **🎯 Why Does This Matter?**: It bridges offline training scripts and live production software, teaching why **training** and **inference** must be separated and how strict API contracts prevent malformed inputs from crashing services.  
> - **🔗 Learning Chain**: [05 | Explainable AI with SHAP](05_explainable_ai_shap.md) → This Chapter (Serve models via HTTP REST APIs) → **Next Chapter**: [07 | Streaming Learning and Concept Drift](07_stream_learning_concept_drift.md) (What happens after deployment when real-world data distributions shift over time?).

---

## 1. Offline Training vs. Online Serving Architecture

```text
[Offline Training Phase]
Training data → Preprocessing + Model training → joblib.dump serialized artifact (.joblib)

[Online Serving Phase]
Load model once at startup → Receive HTTP POST JSON → Validate schema/types/finiteness → model.predict → Return HTTP JSON
```

- **Core Engineering Rule**: Decouple training from serving. A web server loads the serialized artifact once at startup and only runs fast forward inference (`model.predict`) per request—never retrain inside a request handler!
- **Bundle Preprocessing**: Export the full preprocessing + model pipeline (feature order, imputers, scalers, and estimator) so online inference applies the exact same transformations as training.

---

## 2. HTTP Endpoints & Status Code Contracts

- **`GET /health`**: Liveness check returning HTTP `200` and `{"status": "ok"}`.
- **`POST /predict`**: Accepts feature values in a JSON request body and returns predicted class/probabilities.
- **Status Codes**:
  - `200 OK`: Valid input and successful prediction.
  - `400 Bad Request`: Missing fields, non-numeric strings, booleans, `NaN`, or `Inf` values in client payload.
  - `500 Internal Server Error`: Unexpected server-side failure.

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get('/health')
def health():
    return jsonify(status='ok')

# See projects/04_flask_iris_api/app.py for the full implementation.
```

---

## 3. API Input/Output Contract (Iris Example)

In `projects/04_flask_iris_api/`, the four Iris morphological features map to JSON fields `sl`, `sw`, `pl`, and `pw`. Before calling `model.predict`, the endpoint verifies that all four keys exist, rejects `None`/`bool`/`NaN`/`Inf`, and assembles the features in the exact training column order `[[sl, sw, pl, pw]]`.

---

## 4. Production Deployment Pitfalls

1. **Dev Server vs. Production WSGI**: Flask's built-in development server is for local testing only. Production deployments require a WSGI server (e.g., Gunicorn) and must **never expose `debug=True` publicly**.
2. **Deserialization Security**: `joblib.load` and `pickle` can execute arbitrary code. **Only load model artifacts that you trained and verified from trusted storage.**
3. **Input Validation & Silent Unit Drift**: Validate types, shapes, and reasonable numeric ranges.
4. **Model Versioning & Monitoring**: Track model versions alongside training data snapshots and monitor live prediction distributions for drift (Chapter 07).

---

## 5. Hands-on Projects & Self-Test

- **Runnable Project**: [`projects/04_flask_iris_api/`](../projects/04_flask_iris_api/README.md) (includes `model.py`, `app.py`, `curl` commands, and automated `pytest` tests in `test_api.py`).
- **Self-Test**:
  1. Why should the serialized model be loaded once at startup rather than inside the `/predict` route on every HTTP request?
  2. Should a missing or `NaN` feature in the request JSON return HTTP `400` or `500`? Why?
