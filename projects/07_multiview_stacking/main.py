"""Multi-view text classification with leakage-aware stacking.
Uses generated synthetic texts by default. Optionally supply a trusted CSV with
columns text, aux_text, label (do not publish private records).
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/07_multiview'
TOPICS={
 'technology': [('computer','software','network','database','algorithm'), ('python','cloud','server')],
 'sports': [('soccer','tennis','fitness','team','match'), ('court','running','stadium')],
 'nature': [('forest','river','garden','wildlife','mountain'), ('plants','birds','lake')],
}

def generated_examples(n_per_class=90,seed=42):
    rng=np.random.default_rng(seed)
    rows=[]
    for label,(words,metadata) in TOPICS.items():
        for i in range(n_per_class):
            main=' '.join(rng.choice(words,size=5,replace=True))
            aux=' '.join(rng.choice(metadata,size=2,replace=True))
            if i%7==0:main+=' interesting article'
            rows.append({'text':main,'aux_text':aux,'label':label})
    return pd.DataFrame(rows).sample(frac=1,random_state=seed).reset_index(drop=True)

def read_data(path):
    df=pd.read_csv(path) if path else generated_examples()
    req={'text','aux_text','label'}
    if not req.issubset(df.columns):raise ValueError(f'CSV needs columns: {sorted(req)}')
    df=df[list(sorted(req))].dropna()
    df[['text','aux_text','label']]=df[['text','aux_text','label']].astype(str)
    df=df[(df.text.str.strip()!='')&(df.aux_text.str.strip()!='')]
    if df.label.value_counts().min()<10:raise ValueError('Each class needs at least 10 rows to split into base/meta/test sets')
    return df

def combined_probabilities(models,frame,classes):
    values=[]
    for model,column in zip(models,['text','aux_text']):
        if not np.array_equal(model.classes_,classes):raise ValueError('Class order mismatch')
        values.append(model.predict_proba(frame[column]))
    return np.hstack(values)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=None,help='Optional trusted CSV with text,aux_text,label')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    df=read_data(args.data)
    base,rest=train_test_split(df,test_size=.4,random_state=42,stratify=df.label)
    meta,test=train_test_split(rest,test_size=.5,random_state=42,stratify=rest.label)
    classes=np.unique(base.label)
    text_model=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),MultinomialNB())
    aux_model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,4)),LogisticRegression(max_iter=1500))
    text_model.fit(base.text,base.label)
    aux_model.fit(base.aux_text,base.label)
    models=[text_model,aux_model]
    stack=LogisticRegression(max_iter=1500,random_state=42)
    stack.fit(combined_probabilities(models,meta,classes),meta.label)
    pred=stack.predict(combined_probabilities(models,test,classes))
    result={
      'data_type':'custom CSV' if args.data else 'generated synthetic sample',
      'n_total':len(df),'n_base':len(base),'n_meta':len(meta),'n_test':len(test),
      'majority_baseline':float(test.label.value_counts(normalize=True).max()),
      'text_accuracy':float(accuracy_score(test.label,text_model.predict(test.text))),
      'aux_accuracy':float(accuracy_score(test.label,aux_model.predict(test.aux_text))),
      'stack_accuracy':float(accuracy_score(test.label,pred)),
      'stack_macro_f1':float(f1_score(test.label,pred,average='macro')),
      'classification_report':classification_report(test.label,pred,output_dict=True,zero_division=0)
    }
    (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='classification_report'},indent=2,ensure_ascii=False))
    print('Synthetic examples are only for pipeline demonstration; scores are not real-world benchmarks.')
if __name__=='__main__':main()
