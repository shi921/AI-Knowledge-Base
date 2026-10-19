"""API: GET /health, POST /predict {sl, sw, pl, pw}。Load only models produced by this project."""
from flask import Flask,request,jsonify
from pathlib import Path
import joblib
import pandas as pd
from model import MODEL_PATH,FEATURES,train

def create_app():
    app=Flask(__name__)
    if not MODEL_PATH.exists():train()
    payload=joblib.load(MODEL_PATH)  # Never load untrusted pickle/joblib files.
    model,classes=payload['model'],payload['classes']
    @app.get('/health')
    def health():return jsonify(status='ok')
    @app.post('/predict')
    def predict():
        body=request.get_json(silent=True)
        if not isinstance(body,dict):return jsonify(error='The request body must be a JSON object'),400
        try:
            values=[float(body[key]) for key in ('sl','sw','pl','pw')]
            if any(v < 0 or not (v < float('inf')) for v in values):
                raise ValueError('feature must be finite and nonnegative')
        except (KeyError,TypeError,ValueError) as e:
            return jsonify(error='Provide four nonnegative finite numbers: sl, sw, pl, pw'),400
        row=pd.DataFrame([values],columns=FEATURES)
        label=int(model.predict(row)[0]);scores=model.predict_proba(row)[0]
        return jsonify(flower=classes[label],probabilities={c:round(float(v),4) for c,v in zip(classes,scores)})
    return app

app=create_app()
if __name__=='__main__':
    # Local development only. In production, use a WSGI server and do not expose debug mode.
    app.run(host='127.0.0.1',port=8000,debug=False)
