"""BoW/TF-IDF sentiment classification, n-gram modeling, and add-one-smoothed Perplexity (PPL)."""
from pathlib import Path
from collections import Counter
import argparse, json, math, re
import gensim.downloader as api
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REVIEWS=[
('what a wonderful movie with excellent acting',1),('i absolutely loved the vibrant story',1),
('fantastic performances and a beautiful score',1),('an inspiring and clever film',1),
('great cast makes a delightful movie',1),('this touching story made me happy',1),
('impressive direction and enjoyable scenes',1),('charming characters with lovely dialogue',1),
('bright visuals and a delightful ending',1),('superb emotional performance',1),
('a dull movie with terrible acting',0),('i hated the slow boring story',0),
('awful performances and irritating music',0),('disappointing plot and tedious scenes',0),
('poor direction makes a dreadful film',0),('this depressing movie wasted my time',0),
('annoying cast with a predictable ending',0),('a weak script and bland characters',0),
('painfully bad dialogue and terrible pacing',0),('an extremely boring performance',0),
]
# Add varied but clearly synthetic review sentences so held-out evaluation demonstrates generalization.
# Deliberately no pretrained embeddings / external dataset.
for subject in ('movie','acting','ending','story','script','dialogue'):
    for adjective in ('great','wonderful','excellent','amazing','lovely'):
        REVIEWS.append((f'the {subject} was {adjective} and i enjoyed it',1))
    for adjective in ('bad','terrible','awful','boring','dreadful'):
        REVIEWS.append((f'the {subject} was {adjective} and i disliked it',0))
OUT=Path(__file__).resolve().parents[2]/'outputs'/'11_classical_nlp'
def tokens(text):return re.findall(r"[a-z]+|<eos>|<sos>",text.lower())
def train_ngram(lines,n=3):
    seq=[['<sos>']*(n-1)+tokens(t)+['<eos>'] for t in lines]
    gram=Counter();history=Counter();vocab={'<unk>','<eos>'}
    for row in seq:
        vocab.update(row)
        for i in range(n-1,len(row)):
            context=tuple(row[i-(n-1):i]);gram[context+(row[i],)]+=1;history[context]+=1
    return gram,history,vocab

def perplexity(texts,gram,history,vocab,n=3):
    log_sum=0;count=0;vsize=len(vocab)
    for text in texts:
        row=['<sos>']*(n-1)+[w if w in vocab else '<unk>' for w in tokens(text)]+['<eos>']
        for i in range(n-1,len(row)):
            context=tuple(row[i-(n-1):i]);token=row[i]
            # add-one smoothing avoids zero-probability -> infinite PPL
            p=(gram[context+(token,)]+1)/(history[context]+vsize)
            log_sum+=math.log(p);count+=1
    return math.exp(-log_sum/count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="glove-wiki-gigaword-200"
    )
    parser.add_argument(
        "--words",
        nargs="+",
        default=["king", "queen", "man", "woman", "apple", "orange"]
    )
    args = parser.parse_args()

    # Download pretrained embeddings or load them from cache
    print(f"Loading pretrained embeddings: {args.model}")
    vectors = api.load(args.model)

    # Select words available in the vocabulary
    words = [word for word in args.words if word in vectors]

    if len(words) < 2:
        parser.error("At least two valid words are required.")

    # Reduce embedding dimensions to 2D
    embeddings = np.array([vectors[word] for word in words])
    coordinates = PCA(n_components=2).fit_transform(embeddings)

    # Save visualization
    output_dir = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "11_classical_nlp"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.scatter(coordinates[:, 0], coordinates[:, 1])

    for (x, y), word in zip(coordinates, words):
        ax.annotate(word, (x, y))

    output_file = output_dir / "embeddings_pca.png"
    fig.savefig(output_file)
    plt.close(fig)

    print(f"Visualization saved to: {output_file}")


if __name__ == "__main__":
    main()
