"""Tiny Transformer encoder sentiment classifier + attention heatmap.
No network needed. Optional --data-dir explicitly trusts local pickled samples.
"""
from pathlib import Path
import argparse, pickle, re
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs'/'13_transformer_encoder'
SAMPLES=[
('the movie was exciting and wonderful',1),('brilliant acting and great script',1),
('a beautiful and uplifting film',1),('i really enjoyed the story',1),
('the music was lovely and beautiful',1),('excellent actors and great scenery',1),
('this was a happy and inspiring movie',1),('a fantastic and interesting ending',1),
('the movie was boring and awful',0),('terrible acting and poor script',0),
('a terrible and disappointing film',0),('i hated the story',0),
('the music was annoying and loud',0),('poor actors and bad scenery',0),
('this was a dull and depressing movie',0),('a dreadful and predictable ending',0),
]
def words(text):return re.findall(r'[a-z]+',text.lower())
def encode(texts,vocab,max_len=24):
    seq=[]
    for t in texts:
        ids=[vocab['<cls>']]+[vocab.get(w,vocab['<unk>']) for w in words(t)]
        ids=ids[:max_len];seq.append(ids+[vocab['<pad>']]*(max_len-len(ids)))
    return torch.tensor(seq,dtype=torch.long)
class Block(nn.Module):
    def __init__(self,dim=32,heads=4):
        super().__init__();self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n1=nn.LayerNorm(dim);self.n2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.ReLU(),nn.Linear(dim*2,dim))
    def forward(self,x,mask):
        z,weights=self.attn(x,x,x,key_padding_mask=mask,need_weights=True)
        x=self.n1(x+z)
        return self.n2(x+self.ff(x)),weights
class TinyEncoder(nn.Module):
    def __init__(self,vocab_size,max_len=24,dim=32):
        super().__init__();self.embed=nn.Embedding(vocab_size,dim,padding_idx=0)
        self.position=nn.Embedding(max_len,dim);self.block=Block(dim)
        self.classifier=nn.Linear(dim,2)
    def forward(self,tokens):
        pos=torch.arange(tokens.size(1),device=tokens.device).unsqueeze(0)
        z=self.embed(tokens)+self.position(pos)
        z,weights=self.block(z,tokens.eq(0))
        return self.classifier(z[:,0,:]),weights

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=16)
    ap.add_argument('--data-dir',type=Path,default=None,help='directory containing trusted movie_reviews.pkl and movie_labels.pkl')
    args=ap.parse_args();torch.manual_seed(42);torch.set_num_threads(min(4,torch.get_num_threads()))
    if args.data_dir is not None:
        with (args.data_dir/'movie_reviews.pkl').open('rb') as f:texts=pickle.load(f)
        with (args.data_dir/'movie_labels.pkl').open('rb') as f:labels=pickle.load(f)
        texts=list(map(str,texts));labels=[int(y) for y in labels]
    else:
        texts=[t for t,_ in SAMPLES];labels=[y for _,y in SAMPLES]
        print('Using built-in sentiment examples; --data-dir loads trusted local pickle data only')
    vocab={'<pad>':0,'<unk>':1,'<cls>':2}
    # build vocabulary on train split only to avoid leakage
    tr_idx,te_idx=train_test_split(np.arange(len(labels)),test_size=.25,stratify=labels,random_state=42)
    for i in tr_idx:
        for w in words(texts[i]):
            if w not in vocab:vocab[w]=len(vocab)
    all_x=encode(texts,vocab);Y=torch.tensor(labels,dtype=torch.long)
    model=TinyEncoder(len(vocab));optim=torch.optim.Adam(model.parameters(),lr=.008)
    for epoch in range(args.epochs):
        model.train();optim.zero_grad();logits,_=model(all_x[tr_idx]);loss=nn.functional.cross_entropy(logits,Y[tr_idx]);loss.backward();optim.step()
        if epoch==0 or (epoch+1)%5==0:print('epoch',epoch+1,'training loss',round(loss.item(),4))
    model.eval()
    with torch.no_grad():
        logits,att=model(all_x[te_idx]);pred=logits.argmax(-1)
    acc=(pred==Y[te_idx]).float().mean().item();print('held-out accuracy:',round(acc,3))
    OUT.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),OUT/'encoder_state.pt')
    sample_i=te_idx[0];valid=int((all_x[sample_i]!=0).sum().item())
    _,att1=model(all_x[sample_i].unsqueeze(0))
    attention=att1[0,:valid,:valid].detach().numpy()
    ivocab={v:k for k,v in vocab.items()};tokens=[ivocab.get(int(v),'?') for v in all_x[sample_i,:valid]]
    fig,ax=plt.subplots(figsize=(6,5));img=ax.imshow(attention,cmap='viridis')
    ax.set_xticks(range(valid),tokens,rotation=45,ha='right');ax.set_yticks(range(valid),tokens)
    ax.set_title('Average self-attention across heads');fig.colorbar(img,ax=ax);fig.tight_layout()
    fig.savefig(OUT/'attention_heatmap.png',dpi=140);plt.close(fig)
    print('saved:',OUT)
if __name__=='__main__':main()
