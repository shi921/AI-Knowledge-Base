"""08: SHAP global summary + local waterfall. Uses built-in breast cancer data by default; UCI Mushroom is optional."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/08_shap'

def load_data(name):
    if name=='breast_cancer':
        b=load_breast_cancer(as_frame=True)
        return b.data,b.target
    try:from ucimlrepo import fetch_ucirepo
    except ImportError as e:raise RuntimeError('The online Mushroom dataset requires: pip install ucimlrepo') from e
    mush=fetch_ucirepo(id=73)  # Requires network access
    features=mush.data.features[['odor']].copy().fillna('missing')
    X=pd.get_dummies(features,dtype=int)
    labels=mush.data.targets.iloc[:,0].astype(str).str.lower()
    y=(labels=='p').astype(int)
    return X,y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['breast_cancer','mushroom'],default='breast_cancer')
    ap.add_argument('--samples',type=int,default=70);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    X,y=load_data(a.dataset)
    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=.2,random_state=42)
    clf=RandomForestClassifier(n_estimators=80,max_depth=6,random_state=42,n_jobs=-1)
    clf.fit(X_train,y_train);accuracy=float(accuracy_score(y_test,clf.predict(X_test)))
    print('Test accuracy:',round(accuracy,4))
    # SHAP may take time to import, so it is loaded only when needed.
    import shap
    sample=X_test.iloc[:max(1,a.samples)]
    # Newer SHAP versions return an Explanation with class dimensions; older APIs may return per-class lists.
    explainer=shap.TreeExplainer(clf)
    result=explainer(sample)
    if result.values.ndim==3:  # class=1 is positive class
        explanation=result[:,:,1]
    else:
        explanation=result
    shap.summary_plot(explanation.values,sample,show=False)
    plt.tight_layout();plt.savefig(OUT/f'{a.dataset}_summary.png',dpi=150,bbox_inches='tight');plt.close()
    shap.plots.waterfall(explanation[0],show=False,max_display=12)
    plt.tight_layout();plt.savefig(OUT/f'{a.dataset}_waterfall.png',dpi=150,bbox_inches='tight');plt.close()
    (OUT/f'{a.dataset}_metrics.json').write_text(json.dumps({'test_accuracy':accuracy,'features':list(X.columns),
        'n_explained':len(sample)},indent=2))
    print('Figures saved to:',OUT)
    print('SHAP values use the explainer output space (often probability or raw score) and do not represent causal effects.')

if __name__=='__main__':main()
