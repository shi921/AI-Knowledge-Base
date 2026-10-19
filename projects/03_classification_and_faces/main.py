"""03: Handwritten digit classification with cross-validation and confusion-matrix analysis using built-in scikit-learn data."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/03_classification'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['forest','logistic'],default='forest');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    X,y=load_digits(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    model=RandomForestClassifier(n_estimators=120,random_state=42,n_jobs=-1) if a.model=='forest' else make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500))
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    scores=cross_val_score(model,X_train,y_train,scoring='f1_macro',cv=cv)
    model.fit(X_train,y_train);pred=model.predict(X_test)
    metrics={'cv_macro_f1_mean':float(np.mean(scores)),'cv_macro_f1_std':float(np.std(scores)),
             'holdout_accuracy':float(accuracy_score(y_test,pred)),
             'classification_report':classification_report(y_test,pred,output_dict=True)}
    (OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    ConfusionMatrixDisplay.from_predictions(y_test,pred,values_format='d',cmap='Blues')
    plt.tight_layout();plt.savefig(OUT/'confusion_matrix.png',dpi=140);plt.close()
    print('Test accuracy:',round(metrics['holdout_accuracy'],3),'5-fold macro-F1:',round(metrics['cv_macro_f1_mean'],3))
    print('Output:',OUT)

if __name__=='__main__':main()
