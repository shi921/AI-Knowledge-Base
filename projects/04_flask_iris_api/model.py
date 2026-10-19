"""Train an iris classifier on built-in scikit-learn data for use by the Flask API."""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=ROOT/'outputs/04_iris_api/model.joblib'
FEATURES=['sepal length','sepal width','petal length','petal width']

def train():
    sample=load_iris(as_frame=True)
    x=sample.data.copy();x.columns=FEATURES
    y=sample.target
    model=RandomForestClassifier(n_estimators=120,random_state=42).fit(x,y)
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'classes':['Iris-'+str(name) for name in sample.target_names]},MODEL_PATH)
    print('saved:',MODEL_PATH)
    return model

if __name__=='__main__':train()
