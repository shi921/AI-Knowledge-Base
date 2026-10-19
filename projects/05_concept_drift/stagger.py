"""STAGGER：Batchwise test-then-train example using pandas, one-hot encoding, and SGDClassifier.partial_fit."""
from pathlib import Path
import argparse,pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rows',type=int,default=20000);ap.add_argument('--batch-size',type=int,default=1000);a=ap.parse_args()
    df=pd.read_csv(ROOT/'datasets/stagger/Stagger_full.csv',nrows=a.rows)
    data=df.drop(columns=['Label']).astype(str).to_dict('records')
    dv=DictVectorizer(sparse=True);X=dv.fit_transform(data);y=df['Label'].astype(str).to_numpy()
    model=SGDClassifier(loss='log_loss',random_state=42)
    acc=[]
    for begin in range(0,len(y),a.batch_size):
        end=min(begin+a.batch_size,len(y))
        if begin>0:
            predicted=model.predict(X[begin:end]);acc.append((end,accuracy_score(y[begin:end],predicted)))
        model.partial_fit(X[begin:end],y[begin:end],classes=np.unique(y))
    print('Prequential accuracy (last five batches):',acc[-5:])
    print('Note: encoding depends on feature categories only; future labels are never used before evaluation.')

if __name__=='__main__':main()
