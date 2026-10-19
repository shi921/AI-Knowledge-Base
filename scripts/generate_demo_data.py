"""Generate small deterministic, synthetic datasets for the AI knowledge projects.
Usage: python scripts/generate_demo_data.py (from repository root).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets'

def housing():
    rng=np.random.default_rng(2026)
    n=920
    lot_area=rng.lognormal(np.log(8500),.45,n).clip(2000,30000)
    living_area=np.maximum(500,rng.normal(1600,450,n)).astype(int)
    qual=rng.integers(2,11,n)
    year=rng.integers(1950,2012,n)
    garage=rng.integers(0,4,n)
    bsmt=np.maximum(0,rng.normal(900,300,n)).astype(int)
    neighborhood=rng.choice(['North','West','Central','East'],n)
    style=rng.choice(['SingleStory','TwoStory','Townhouse'],n)
    zone=rng.choice(['Residential','Mixed'],n)
    quality=qual*21000+living_area*70+lot_area*3.0+garage*14000+bsmt*23
    quality+=(year-1950)*350+rng.normal(0,16000,n)
    df=pd.DataFrame({
      'Id':np.arange(1,n+1),'LotArea':lot_area.astype(int),'GrLivArea':living_area,
      'OverallQual':qual,'YearBuilt':year,'GarageCars':garage,'TotalBsmtSF':bsmt,
      'YrSold':rng.integers(2013,2018,n),'MoSold':rng.integers(1,13,n),
      'FullBath':rng.integers(1,4,n),'Neighborhood':neighborhood,
      'HouseStyle':style,'MSZoning':zone,
      'LotFrontage':rng.normal(80,25,n).clip(20,170),
      'SalePrice':np.maximum(60000,quality).astype(int)
    })
    df.loc[rng.choice(n,50,replace=False),'LotFrontage']=np.nan
    df.loc[rng.choice(n,20,replace=False),'Neighborhood']=np.nan
    data=DATA/'housing';data.mkdir(exist_ok=True)
    df.iloc[:780].to_csv(data/'train.csv',index=False)
    df.iloc[780:].drop(columns='SalePrice').to_csv(data/'test.csv',index=False)

def stagger():
    rng=np.random.default_rng(2026);n=16000
    sizes=rng.choice(['small','medium','large'],n)
    colors=rng.choice(['red','green','blue'],n)
    shapes=rng.choice(['circular','non-circular'],n)
    idx=np.arange(n)
    label=np.where(idx<n//2,
       (sizes=='small') | (colors=='red'),
       (colors=='green') & (shapes=='circular'))
    df=pd.DataFrame({'Size':sizes,'Color':colors,'Shape':shapes,'Label':np.where(label,'p','n')})
    data=DATA/'stagger';data.mkdir(exist_ok=True)
    df.to_csv(data/'Stagger_full.csv',index=False)

def sea():
    rng=np.random.default_rng(2026);n=18500
    x=rng.uniform(0,10,size=(n,3))
    threshold=np.where(np.arange(n)<11800,9.0,12.5)
    y=(x[:,0]+x[:,1]>threshold).astype(int)
    flip=rng.random(n)<0.04;y=np.where(flip,1-y,y)
    data=DATA/'sea';data.mkdir(exist_ok=True)
    pd.DataFrame({'feat1':x[:,0],'feat2':x[:,1],'feat3':x[:,2],'label':y}).to_csv(data/'sea.csv',index=False)

if __name__=='__main__':
    housing();stagger();sea()
    print('Generated synthetic housing, STAGGER and SEA data in',DATA)
