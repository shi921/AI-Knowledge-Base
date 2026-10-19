"""01: Data profiling, missing values, scaling, PCA, and smoothing on local synthetic housing data."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/housing/train.csv'
OUTPUT = ROOT / 'outputs/01_foundations'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='First N rows for quick debugging')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, nrows=args.limit)
    print(f'Dataset size: {df.shape[0]} rows x {df.shape[1]} cols')
    print('Data type distribution:', df.dtypes.astype(str).value_counts().to_dict())
    missing = pd.DataFrame({'missing_count': df.isna().sum(), 'missing_ratio': df.isna().mean()})
    missing.sort_values('missing_ratio', ascending=False).to_csv(OUTPUT / 'missing_report.csv')
    df.describe(include='all').transpose().to_csv(OUTPUT / 'describe.csv')
    # Exclude the target and ID. For supervised learning, fit transformations inside a Pipeline to prevent leakage.
    X = df.select_dtypes(include='number').drop(columns=['SalePrice','Id'], errors='ignore')
    X = X.loc[:, X.notna().any()]
    imputed = SimpleImputer(strategy='median').fit_transform(X)
    scaled = StandardScaler().fit_transform(imputed)
    pc = PCA(n_components=2, random_state=42).fit_transform(scaled)
    mm = MinMaxScaler().fit_transform(imputed)
    binary = Binarizer(threshold=0.5).transform(mm)
    print(f'PCA: {scaled.shape[1]} -> 2 dimensions; binarized array shape: {binary.shape}')
    pd.DataFrame(pc, columns=['PC1','PC2']).to_csv(OUTPUT / 'pca_2d.csv', index=False)
    plt.figure(figsize=(7,5))
    scatter = plt.scatter(pc[:,0], pc[:,1], c=df['SalePrice'], s=12, cmap='viridis')
    plt.colorbar(scatter, label='SalePrice');plt.xlabel('PC1');plt.ylabel('PC2');plt.tight_layout()
    plt.savefig(OUTPUT / 'pca.png', dpi=140);plt.close()
    # Aggregate sales by month and smooth the mean price. Missing months are not zero-valued sales.
    monthly = df.assign(date=pd.to_datetime({'year':df['YrSold'], 'month':df['MoSold'], 'day':1}))\
                .groupby('date')['SalePrice'].mean().sort_index().rename('mean_price').to_frame()
    monthly['moving_average_3'] = monthly['mean_price'].rolling(3, min_periods=1).mean()
    monthly['centered_average_3'] = monthly['mean_price'].rolling(3, center=True, min_periods=1).mean()
    monthly.to_csv(OUTPUT/'monthly_smoothing.csv')
    monthly.plot(title='Home prices: trailing vs centered moving average', figsize=(9,4))
    plt.tight_layout();plt.savefig(OUTPUT/'smoothing.png',dpi=140);plt.close()
    print('Output directory:', OUTPUT)

if __name__ == '__main__':
    main()
