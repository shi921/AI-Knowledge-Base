"""Optional LSTM checkpoint inference; requires compatible vocabulary and model weights."""
from pathlib import Path
import argparse
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
class LMLSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim=512,hidden_dim=1024,num_layers=3):
        super().__init__();self.embedding=nn.Embedding(vocab_size,embed_dim)
        self.lstm=nn.LSTM(embed_dim,hidden_dim,num_layers=num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_dim,vocab_size)
    def forward(self,x,h=None):
        out,h=self.lstm(self.embedding(x),h);return self.fc(out),h

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prompt',default='the quick brown fox')
    ap.add_argument('--max-new-tokens',type=int,default=25);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    vocab_path=ROOT/'data'/'wikitext-103'/'vocab.pkl'
    weights_path=ROOT/'pretrained_model'/'lstm_model_weights.pkl'
    missing=[p for p in (vocab_path,weights_path) if not p.is_file()]
    if missing:
        ap.error('Required local assets missing: '+', '.join(str(p.relative_to(ROOT)) for p in missing)+'; see ASSETS.md')
    try:from data_processing import load_vocab # requires compatible torchtext
    except ImportError as ex:ap.error(f'missing dependency for historical torchtext vocabulary: {ex}. Read ASSETS.md.')
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA is not available')
    # vocab.pkl is Python pickle: ONLY use a file from a trusted source
    vocab=load_vocab(str(vocab_path))
    model=LMLSTM(len(vocab)).to(args.device)
    # avoid unsafe Python-object unpickling in PyTorch weight loading
    try:weights=torch.load(weights_path,map_location=args.device,weights_only=True)
    except Exception as ex:ap.error(f'weight loading failed ({ex}); verify checkpoint is a compatible state_dict')
    model.load_state_dict(weights);model.eval()
    # Torchtext basic_english tokenizer should match the vocabulary tokenizer
    from torchtext.data.utils import get_tokenizer
    tokenizer=get_tokenizer('basic_english');start=['<sos>']+tokenizer(args.prompt)
    ids=vocab(start);tokens=list(start)
    with torch.no_grad():
        # feed the prefix ONCE, then only new tokens: no duplicated final prefix token
        x=torch.tensor([ids],device=args.device)
        logits,h=model(x)
        for _ in range(args.max_new_tokens):
            logits[0,-1,vocab['<unk>']]=float('-inf')
            next_id=int(logits[0,-1].argmax().item())
            word=vocab.get_itos()[next_id];tokens.append(word)
            if word=='<eos>':break
            logits,h=model(torch.tensor([[next_id]],device=args.device),h)
    print(' '.join(tokens))
if __name__=='__main__':main()
