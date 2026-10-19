"""06: Offline TF-IDF retrieval with optional LangChain LLM generation when OPENAI_API_KEY is set."""
from pathlib import Path
import argparse,os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'datasets/documents'

def chunk_text(text,chunk_size=700,overlap=80):
    if overlap>=chunk_size:raise ValueError('overlap must be < chunk_size')
    pos=0
    while pos<len(text):
        yield text[pos:pos+chunk_size]
        if pos+chunk_size>=len(text):break
        pos+=chunk_size-overlap

def get_corpus():
    chunks=[]
    for p in sorted(DOCS.glob('*.txt')):
        text=p.read_text(encoding='utf-8',errors='replace')
        chunks.extend((p.name,c) for c in chunk_text(text) if c.strip())
    return chunks

def retrieve(query,top_k=3):
    chunks=get_corpus()
    if not chunks:raise RuntimeError('No searchable documents found; check datasets/documents/')
    # Word-level TF-IDF for English documents, not embeddings. Semantic retrieval can use embeddings and a vector store.
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=30000)
    matrix=vectorizer.fit_transform([c for _,c in chunks])
    query_vec=vectorizer.transform([query]);scores=cosine_similarity(query_vec,matrix)[0]
    order=scores.argsort()[-min(top_k,len(chunks)):][::-1]
    return [(chunks[i][0],chunks[i][1],float(scores[i])) for i in order]

def answer_with_langchain(question,hits):
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('OPENAI_API_KEY is not configured; no request will be sent')
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as exc:raise RuntimeError('Install dependencies: pip install langchain-core langchain-openai') from exc
    context='\n\n'.join(f'Source: {name}\n{txt}' for name,txt,_ in hits)
    prompt=ChatPromptTemplate.from_messages([
        ('system','You are an AI knowledge base assistant. Answer only from the provided material. If the evidence is insufficient, say so. Cite source filenames at the end.\nContext:\n{context}'),
        ('human','{question}')])
    chain=prompt|ChatOpenAI(model='gpt-4o-mini',temperature=0)|StrOutputParser()
    return chain.invoke({'question':question,'context':context})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('query',nargs='?',default='What is retrieval augmented generation?')
    ap.add_argument('--backend',choices=['local','openai'],default='local')
    ap.add_argument('--top-k',type=int,default=3);a=ap.parse_args()
    hits=retrieve(a.query,top_k=a.top_k)
    print('Question:',a.query)
    print('\nTop retrievals:')
    for name,text,score in hits:print(f'[{name}, score={score:.3f}] {text[:260].replace(chr(10)," ")} ...')
    if a.backend=='openai':
        print('\nLLM answer (API usage may incur charges):\n',answer_with_langchain(a.query,hits))
    else:
        print('\nOffline mode displays retrieved passages only; it does not generate an LLM answer.')

if __name__=='__main__':main()
