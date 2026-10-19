"""Only loads an explicit locally installed Gensim KeyedVectors file. No automatic download."""
import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description='Local pretrained word embedding PCA')
    ap.add_argument('model_path',type=Path,help='trusted gensim KeyedVectors .kv model')
    ap.add_argument('--words',nargs='+',default=['king','queen','man','woman','apple','orange'])
    args=ap.parse_args()
    if not args.model_path.is_file():ap.error('model file not found; supply a .kv file downloaded separately')
    from gensim.models import KeyedVectors
    vectors=KeyedVectors.load(str(args.model_path),mmap='r') # trusted files only
    words=[w for w in args.words if w in vectors]
    if len(words)<2:ap.error('need >=2 words found in this embedding')
    xy=PCA(n_components=2).fit_transform(np.array([vectors[w] for w in words]))
    out=Path(__file__).resolve().parents[2]/'outputs'/'11_classical_nlp';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots();ax.scatter(xy[:,0],xy[:,1]);
    for (x,y),w in zip(xy,words):ax.annotate(w,(x,y))
    fig.savefig(out/'embeddings_pca.png');plt.close(fig);print(out/'embeddings_pca.png')
if __name__=='__main__':main()
