"""Trainable neural n-gram LM. Uses user-supplied data/sample_text.txt or demo text."""
from pathlib import Path
from collections import Counter
import argparse, math, json, re
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs'/'12_neural_language_models'
DEMO=("the quick brown fox jumps over the lazy dog . the little dog runs in the park . "
      "language models learn next token probabilities from preceding words . "
      "a quick fox runs in the beautiful park . the brown dog learns words . ")*25

def tokenize(text):return re.findall(r"[a-z]+|[.,!?]",text.lower())

class NeuralNgram(nn.Module):
    def __init__(self,vocab_size,context=4,embed_dim=32,hidden_dim=96):
        super().__init__();self.embed=nn.Embedding(vocab_size,embed_dim)
        self.predictor=nn.Sequential(nn.Linear(context*embed_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,vocab_size))
    def forward(self,indices):return self.predictor(self.embed(indices).flatten(1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--context',type=int,default=4)
    ap.add_argument('--text-file',type=Path,default=ROOT/'data'/'sample_text.txt');args=ap.parse_args()
    if args.context<1:ap.error('context must be positive')
    torch.manual_seed(42);torch.set_num_threads(min(torch.get_num_threads(),4))
    corpus=args.text_file.read_text(encoding='utf8') if args.text_file.exists() else DEMO
    if not args.text_file.exists():print('No data/sample_text.txt: using a small built-in demo corpus')
    tokens=tokenize(corpus)
    split=int(.8*len(tokens));train=tokens[:split];test=tokens[split:]
    counts=Counter(train);vocab=['<unk>']+sorted(counts);stoi={v:i for i,v in enumerate(vocab)}
    def data(tok):
        seq=[stoi.get(w,0) for w in tok]
        if len(seq)<=args.context:raise ValueError('corpus too short for this context')
        return torch.tensor([seq[i:i+args.context] for i in range(len(seq)-args.context)],dtype=torch.long),torch.tensor([seq[i+args.context] for i in range(len(seq)-args.context)],dtype=torch.long)
    xtr,ytr=data(train);xte,yte=data(test)
    dl=DataLoader(TensorDataset(xtr,ytr),batch_size=64,shuffle=True)
    model=NeuralNgram(len(vocab),args.context);opt=torch.optim.Adam(model.parameters(),lr=.01);crit=nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        for xb,yb in dl:
            opt.zero_grad();loss=crit(model(xb),yb);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            val_loss=crit(model(xte),yte).item();ppl=math.exp(min(val_loss,80))
        print(f'epoch {ep+1} heldout CE={val_loss:.4f} PPL={ppl:.2f}')
    prompt=['the','quick','brown','fox'][-args.context:]
    prompt=['<unk>']*max(0,args.context-len(prompt))+prompt
    with torch.no_grad():
        logits=model(torch.tensor([[stoi.get(w,0) for w in prompt]]));word=vocab[logits.argmax(-1).item()]
    print('next-token prediction:',word)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),OUT/'neural_ngram_state.pt')
    (OUT/'vocab.json').write_text(json.dumps({'vocab':vocab,'context':args.context},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
