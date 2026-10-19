"""Digits MLP：Fit scaling on training data only, use validation-based early stopping, and save/reload weights."""
from pathlib import Path
import argparse, copy, json
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT=Path(__file__).resolve().parents[2]/'outputs'/'10_pytorch_mlp'
class DigitsMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x):return self.net(x)

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--epochs',type=int,default=30);pa.add_argument('--batch-size',type=int,default=128)
    args=pa.parse_args();torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(min(torch.get_num_threads(),4))
    X,y=load_digits(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=.2,random_state=42)
    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,stratify=y_train,test_size=.2,random_state=42)
    scaler=StandardScaler();X_train=scaler.fit_transform(X_train);X_val=scaler.transform(X_val);X_test=scaler.transform(X_test)
    dataset=torch.utils.data.TensorDataset(torch.from_numpy(X_train.astype('float32')),torch.tensor(y_train,dtype=torch.long))
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    model=DigitsMLP();opt=torch.optim.Adam(model.parameters(),lr=.002);loss_fn=nn.CrossEntropyLoss()
    val_x=torch.tensor(X_val,dtype=torch.float32);test_x=torch.tensor(X_test,dtype=torch.float32)
    best_loss=float('inf');best=None;stale=0
    for ep in range(args.epochs):
        model.train();loss_sum=0
        for xb,yb in loader:
            opt.zero_grad();loss=loss_fn(model(xb),yb);loss.backward();opt.step();loss_sum+=loss.item()*len(xb)
        model.eval()
        with torch.no_grad():val_loss=loss_fn(model(val_x),torch.tensor(y_val,dtype=torch.long)).item()
        if val_loss<best_loss-1e-4:best_loss=val_loss;best=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if ep==0 or (ep+1)%5==0: print('epoch',ep+1,'train_loss',round(loss_sum/len(dataset),4),'val_loss',round(val_loss,4))
        if stale>=8:break
    model.load_state_dict(best);model.eval()
    with torch.no_grad():pred=model(test_x).argmax(-1).numpy()
    print('test accuracy',round(accuracy_score(y_test,pred),4));print(classification_report(y_test,pred,zero_division=0))
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),OUT/'digits_mlp_state.pt')
    np.savez(OUT/'scaler_params.npz',mean=scaler.mean_,scale=scaler.scale_)
    (OUT/'metrics.json').write_text(json.dumps({'accuracy':accuracy_score(y_test,pred),'confusion_matrix':confusion_matrix(y_test,pred).tolist()},indent=2))
    # demonstrate restoring state_dict without pickling arbitrary model objects
    reloaded=DigitsMLP();reloaded.load_state_dict(torch.load(OUT/'digits_mlp_state.pt',weights_only=True));reloaded.eval()
    with torch.no_grad():assert np.array_equal(reloaded(test_x).argmax(-1).numpy(),pred)
    print('saved:',OUT)
if __name__=='__main__':main()
