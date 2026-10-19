"""SEA concept drift: temporal batch evaluation, Page-Hinkley detection, and rolling retraining after alerts."""
from pathlib import Path
from collections import deque
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/05_drift'

class PageHinkleyFallback:
    """Educational Page-Hinkley fallback; prefer River when available. Detect sustained increases in classification loss."""
    def __init__(self,delta=.005,threshold=40,min_instances=30):
        self.delta,self.threshold,self.min_instances=delta,threshold,min_instances
        self.reset()
    def reset(self):self.n=0;self.avg=0.;self.s=0.;self.min_s=0.;self.drift_detected=False
    def update(self,x):
        self.n+=1;self.avg+=(x-self.avg)/self.n
        self.s+=x-self.avg-self.delta;self.min_s=min(self.min_s,self.s)
        self.drift_detected=self.n>=self.min_instances and self.s-self.min_s>self.threshold
        if self.drift_detected:
            self.reset();self.drift_detected=True

def make_detector():
    try:
        from river.drift import PageHinkley
        print('Detector: river.drift.PageHinkley')
        return PageHinkley(min_instances=30,delta=.005,threshold=40)
    except ImportError:
        print('River is unavailable; using the built-in educational Page-Hinkley detector.')
        return PageHinkleyFallback(min_instances=30,delta=.005,threshold=40)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--init',type=int,default=10000)
    ap.add_argument('--batch-size',type=int,default=1000);ap.add_argument('--max-batches',type=int,default=None)
    ap.add_argument('--window',type=int,default=10000);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ROOT/'datasets/sea/sea.csv')
    X=df[['feat1','feat2','feat3']].to_numpy();y=df['label'].to_numpy()
    if len(X)<a.init+100:raise ValueError('not enough rows')
    fixed=SVC(gamma=.1).fit(X[:a.init],y[:a.init])
    adaptive=SVC(gamma=.1).fit(X[:a.init],y[:a.init])
    detector=make_detector();rows=[];alerts=[]
    for idx,start in enumerate(range(a.init,len(y),a.batch_size)):
        if a.max_batches is not None and idx>=a.max_batches:break
        stop=min(len(y),start+a.batch_size)
        x_batch,y_batch=X[start:stop],y[start:stop]
        static_pred=fixed.predict(x_batch)
        pred=adaptive.predict(x_batch)
        raised=False
        for error in (pred!=y_batch).astype(float):
            detector.update(float(error))
            raised=raised or bool(detector.drift_detected)
        rows.append({'end_index':stop,'frozen_acc':accuracy_score(y_batch,static_pred),
                     'adaptive_acc':accuracy_score(y_batch,pred),'drift':int(raised)})
        if raised:
            alerts.append(stop)
            lo=max(0,stop-a.window)
            if len(set(y[lo:stop]))>=2:adaptive=SVC(gamma=.1).fit(X[lo:stop],y[lo:stop])
    stats=pd.DataFrame(rows);stats.to_csv(OUT/'batch_metrics.csv',index=False)
    ax=stats.plot(x='end_index',y=['frozen_acc','adaptive_acc'],figsize=(9,4),ylim=(0,1),title='SEA concept drift (test-then-train)')
    for x in alerts:ax.axvline(x,alpha=.25,color='red',ls='--')
    plt.tight_layout();plt.savefig(OUT/'accuracy_drift.png',dpi=140);plt.close()
    print('batches',len(stats),'alerts',len(alerts),'frozen mean acc',round(stats.frozen_acc.mean(),3),'adaptive mean acc',round(stats.adaptive_acc.mean(),3))
    print('Output:',OUT)

if __name__=='__main__':main()
