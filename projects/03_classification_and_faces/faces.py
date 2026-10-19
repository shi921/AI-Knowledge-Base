"""Optional face-recognition experiment: train PCA and SVM on an authorized local CSV."""
from pathlib import Path
import argparse,json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--components',type=int,default=80);ap.add_argument('--data',type=Path,default=ROOT/'datasets/faces/lfw_people.csv',help='CSV path (not included in public repository)');a=ap.parse_args()
    if not a.data.is_file():ap.error(f'No face dataset found at {a.data}. Supply --data with an authorized local CSV.')
    df=pd.read_csv(a.data,index_col=0)
    print('shape:',df.shape)
    y=df['target'].astype(str)
    X=df.drop(columns=['target','num_target'],errors='ignore').select_dtypes(include='number')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    n=min(a.components,X_train.shape[0]-1,X_train.shape[1])
    pipe=make_pipeline(StandardScaler(),PCA(n_components=n,whiten=True,svd_solver='randomized',random_state=42),SVC(C=10,gamma='scale'))
    pipe.fit(X_train,y_train);pred=pipe.predict(X_test)
    print('Accuracy:',round(accuracy_score(y_test,pred),3))
    print(classification_report(y_test,pred,zero_division=0))

if __name__=='__main__':main()
