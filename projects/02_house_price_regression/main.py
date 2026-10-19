"""02: House price prediction with fold-safe preprocessing, cross-validation, baselines, and model export."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'datasets/housing'
OUTPUT = ROOT/'outputs/02_housing'

def features(frame):
    """Drop non-feature columns; the test CSV does not contain SalePrice."""
    return frame.drop(columns=['SalePrice','Id'], errors='ignore')

def make_pipeline(X, name):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    prep = ColumnTransformer([
      ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num_cols),
      ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                               ('encoder',OneHotEncoder(handle_unknown='ignore'))]),cat_cols),
    ])
    est = Ridge(alpha=20) if name=='ridge' else RandomForestRegressor(n_estimators=120,
           min_samples_leaf=2, random_state=42, n_jobs=-1)
    return Pipeline([('preprocess',prep),('model',est)])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['ridge','forest'],default='ridge')
    ap.add_argument('--cv',type=int,default=3); a=ap.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA/'train.csv')
    X=features(df);y=np.log1p(df['SalePrice'])
    # Reserve the holdout set for final evaluation; perform cross-validation only on the training set.
    X_train,X_holdout,y_train,y_holdout=train_test_split(X,y,test_size=.2,random_state=42)
    pipe=make_pipeline(X_train,a.model)
    cv = cross_val_score(pipe,X_train,y_train,cv=a.cv,scoring='neg_mean_absolute_error',n_jobs=1)
    pipe.fit(X_train,y_train)
    pred_log=pipe.predict(X_holdout)
    actual=np.expm1(y_holdout.to_numpy());pred=np.expm1(pred_log)
    report={'model':a.model, 'n_train':len(X_train),'n_holdout':len(X_holdout),
      'cv_log_mae_mean':float(-cv.mean()),'cv_log_mae_std':float(cv.std()),
      'holdout_MAE_price_units':float(mean_absolute_error(actual,pred)),
      'holdout_RMSE_price_units':float(np.sqrt(mean_squared_error(actual,pred))),
      'holdout_R2_price_units':float(r2_score(actual,pred))}
    print(json.dumps(report,indent=2,ensure_ascii=False))
    (OUTPUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # Retrain on all labeled rows, then predict the simulated unlabeled test set.
    final=make_pipeline(X,a.model).fit(X,y)
    joblib.dump(final,OUTPUT/'house_price_pipeline.joblib')
    external=pd.read_csv(DATA/'test.csv')
    predictions=np.expm1(final.predict(features(external)))
    pd.DataFrame({'Id':external['Id'],'SalePrice':np.maximum(predictions,0)}).to_csv(OUTPUT/'submission.csv',index=False)
    print('Files: metrics.json, house_price_pipeline.joblib, submission.csv')

if __name__=='__main__':main()
