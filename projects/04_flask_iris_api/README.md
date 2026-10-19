**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# Flask Model Deployment: Iris Classification API

Run from the repository root:

```bash
python projects/04_flask_iris_api/model.py
python projects/04_flask_iris_api/app.py
# Test the API from another terminal
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"sl":5.1,"sw":3.5,"pl":1.4,"pw":0.2}'
```

Tests: `python -m pytest projects/04_flask_iris_api/test_api.py -q`. Demonstrates training, saving, loading, validating inputs, and returning HTTP JSON. Flask's development server is not suitable for production deployment.

**Related notes:** [Chapter 04](../../notes/04_flask_model_deployment.md).
