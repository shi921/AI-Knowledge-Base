"""Optional customer PCA experiment on an authorized local Excel file (requires xlrd)."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/01_foundations'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'datasets/customer/customers.xls');args=ap.parse_args()
    if not args.data.is_file():ap.error('No customer file. Supply --data path to a local authorized spreadsheet.')
    df=pd.read_excel(args.data)
    print('Original dimensions:',df.shape)
    # Integer encoding is only illustrative; use one-hot encoding for unordered categories in supervised learning.
    mapping={'Sex':{'F':0,'M':1},'Status':{'S':0,'M':1},
             'Car_Owner':{'N':0,'Y':1},'Paymethod':{'CC':0,'CH':1,'Auto':2},
             'LocalBilltype':{'Budget':0,'FreeLocal':1},
             'LongDistanceBilltype':{'Standard':0,'Intl_discount':1},
             'CHURN':{'Cancelled':0,'Current':1}}
    encoded=df.copy()
    for col in encoded.select_dtypes(exclude='number').columns:
        if col in mapping and encoded[col].dropna().isin(mapping[col]).all():
            encoded[col]=encoded[col].map(mapping[col])
        else:
            codes,labels=pd.factorize(encoded[col],sort=True)
            print(f'{col}: integer-encoded labels do not imply an ordinal relationship:',labels.tolist()[:10])
            encoded[col]=codes
    columns=['Age','Est_Income','Usage','LongDistance']
    values=encoded[columns].apply(pd.to_numeric,errors='coerce')
    if values.isna().any().any():values=values.fillna(values.median())
    normalized=StandardScaler().fit_transform(values)
    encoded[columns]=normalized
    pcs=PCA(n_components=4).fit_transform(normalized)
    pc_names=[f'PC{i}' for i in range(1,5)]
    pcdf=pd.DataFrame(pcs,columns=pc_names,index=encoded.index)
    corr=pd.DataFrame(np.corrcoef(normalized.T,pcs.T)[:4,4:],index=columns,columns=pc_names)
    i,j=np.unravel_index(np.nanargmax(np.abs(corr.to_numpy())),corr.shape)
    print('Maximum absolute correlation:',columns[i],pc_names[j],'correlation coefficient:',corr.iloc[i,j])
    OUT.mkdir(parents=True,exist_ok=True)
    encoded.to_csv(OUT/'customers_encoded_standardized.csv',index=False)
    pcdf.to_csv(OUT/'customers_pca.csv',index=False)
    corr.to_csv(OUT/'customers_corr.csv')
    print('Output:',OUT)

if __name__=='__main__':main()
