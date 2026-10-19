"""Linear, polynomial, and ridge regression with closed-form solutions, gradient descent, train/test errors, and plots."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

OUT = Path(__file__).resolve().parents[2] / 'outputs' / '09_regression'

def fit_closed_form(x, y, ridge=0.0):
    # Ridge intercept should not be penalized
    reg = np.eye(x.shape[1]) * ridge
    reg[0, 0] = 0
    return np.linalg.solve(x.T @ x + reg + 1e-12 * np.eye(x.shape[1]), x.T @ y)

def gradient_descent(x, y, steps=800, lr=0.1):
    weights = np.zeros(x.shape[1]); history=[]
    for _ in range(steps):
        err=x @ weights-y
        weights -= lr * (2 / len(y)) * (x.T @ err)
        history.append(np.mean(err ** 2))
    return weights, history

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--degree',type=int,default=5);ap.add_argument('--ridge',type=float,default=0.03);args=ap.parse_args()
    if args.degree<1 or args.ridge<0: ap.error('degree must >=1, ridge must >=0')
    rng=np.random.default_rng(42)
    x_train=np.sort(rng.uniform(0, 1, 50));x_test=np.linspace(0,1,200)
    y_train=np.sin(2*np.pi*x_train)+rng.normal(0,0.18,len(x_train));y_test=np.sin(2*np.pi*x_test)
    poly=PolynomialFeatures(degree=args.degree,include_bias=True)
    train=poly.fit_transform(x_train.reshape(-1,1));test=poly.transform(x_test.reshape(-1,1))
    # degree-scaled polynomial values avoid difficult optimization for GD
    w_ols=fit_closed_form(train,y_train);w_ridge=fit_closed_form(train,y_train,args.ridge)
    scaler=StandardScaler();train_gd=scaler.fit_transform(x_train.reshape(-1,1))
    test_gd=scaler.transform(x_test.reshape(-1,1))
    Xtr=np.c_[np.ones(len(train_gd)),train_gd];Xte=np.c_[np.ones(len(test_gd)),test_gd]
    w_gd, losses=gradient_descent(Xtr,y_train)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,pred_train,pred_test in [('OLS',train@w_ols,test@w_ols),('Ridge',train@w_ridge,test@w_ridge),('GD(linear)',Xtr@w_gd,Xte@w_gd)]:
        print(f'{name:12} train_MSE={mean_squared_error(y_train,pred_train):.4f} test_MSE={mean_squared_error(y_test,pred_test):.4f}')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].scatter(x_train,y_train,alpha=.6,label='train samples');axes[0].plot(x_test,y_test,label='true function')
    axes[0].plot(x_test,test@w_ols,label='OLS');axes[0].plot(x_test,test@w_ridge,label='Ridge')
    axes[0].plot(x_test,Xte@w_gd,label='GD(linear)');axes[0].legend();axes[0].set_title('model complexity + regularization')
    axes[1].plot(losses);axes[1].set_xlabel('step');axes[1].set_ylabel('train MSE');axes[1].set_title('gradient descent')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=135);plt.close(fig)
    print('plot:',OUT/'comparison.png')
if __name__=='__main__':main()
