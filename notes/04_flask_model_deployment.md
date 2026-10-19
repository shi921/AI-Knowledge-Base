**Language:** English | [简体中文](04_flask_model_deployment.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 04 | Model Deployment: From scikit-learn to a Flask API

## Serving architecture

`Training data → preprocessing and model training → joblib export → Flask loads model → HTTP JSON input → validation → model.predict → JSON response`.

Keep training separate from request handling: a service should not retrain on every request. Export feature ordering, encoders, and preprocessing together with the model.

## Essential HTTP concepts

- `GET /health`: confirm that the service is reachable.
- `POST /predict`: accept JSON features and return predicted labels or probabilities; malformed, missing, or nonfinite values should return HTTP 400.
- HTTP 200 indicates success, 4xx a client issue, and 5xx a server-side failure. Keep the development example on `127.0.0.1`.

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.get('/health')
def health():
    return jsonify(status='ok')

# See projects/04_flask_iris_api/app.py for the complete implementation.
```

## API input/output contract

The iris model expects four numeric fields: `sl` (sepal length), `sw` (sepal width), `pl` (petal length), and `pw` (petal width). `POST /predict` returns a predicted class. Type, range, order, and error-handling rules are part of the contract.

## Deployment pitfalls

1. Flask's development server is not a production WSGI stack; never expose `debug=True` publicly.
2. Unpickling or calling `joblib.load` can execute code; load only trusted artifacts.
3. Validate the input type, range, and shape.
4. Plan model versioning, monitoring, drift detection, and retraining.
5. Distinguish training scores, offline validation, and live production quality.

**Project:** `projects/04_flask_iris_api/` includes training, serving, curl calls, and pytest checks.
