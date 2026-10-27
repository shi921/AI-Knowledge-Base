**Language:** English | [简体中文](15_llm_langchain_rag.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 15 | LLM Application Engineering: LangChain, RAG, and Agents

> **🧭 Knowledge Map Position**: Part V (LLMs, Fine-Tuning & RAG/Agents) · **Chapter 15 of 15 (Master Capstone)**  
> - **💡 Intuitive Analogy**: In Chapters 12–14 you mastered Transformers, LLM pre-training/alignment, Prompting, and LoRA fine-tuning. To deploy LLMs in real-world production, two final hurdles remain: **① LLMs don't know your private, up-to-date documents and may guess when unsure (hallucinations); ② LLMs only output text—they lack "hands" to query databases, run calculations, or call APIs.**  
>   - **RAG (Retrieval-Augmented Generation)** turns a closed-book exam into an **open-book exam**: before the LLM answers, your system retrieves the most relevant passages from your document corpus (using Chapter 09 TF-IDF/BM25 or Chapter 13 dense Embeddings), injects them into a Chapter 14 Prompt template, and instructs the LLM to answer strictly from the evidence with source citations!  
>   - **Agents & Tool Calling** equip the LLM with **a toolkit and reasoning loop** so it can decide when to invoke external functions.  
> - **🎯 Why Is This the Capstone Chapter?**: It unites **Data Cleaning (01), Evaluation (04), API Engineering (06), TF-IDF & Embeddings (09), Transformers & LLM Decoding (12–13), and Prompt Engineering (14)** into the architecture powering 90% of modern enterprise AI applications!  
> - **🔗 Learning Chain**: [14 | LLM Customization: Prompting & LoRA](14_llm_prompting_finetuning_lora.md) → This Chapter (RAG & Agents Capstone) → 🎉 **Congratulations on completing all 5 Parts and 15 Chapters of the AI Knowledge Base!**

---

## 1. Core Building Blocks of LLM Applications

| Component | Purpose | Typical Implementation (LangChain) |
|---|---|---|
| **Prompt Template** | Defines system instructions, retrieved context slots, and output constraints | `ChatPromptTemplate.from_messages(...)` |
| **Chat Model** | Generates assistant messages conditioned on conversation context | `ChatOpenAI(model='gpt-4o-mini', temperature=0)` |
| **Chain / LCEL** | Composes reusable pipelines declaratively with `\|` | `prompt \| model \| StrOutputParser()` |
| **Structured Output** | Parses and validates responses against a strict Pydantic/JSON schema | `model.with_structured_output(PydanticModel)` |
| **Embeddings & Vector Store** | Maps text chunks to dense semantic vectors for similarity search | `OpenAIEmbeddings` / `BGE` + `FAISS` / `Chroma` |
| **RAG** | Retrieves relevant passages and grounds LLM generation in cited evidence | `Retrieve → Format Context → Prompt → LLM → Cite` |
| **Tool Calling** | Lets the model request structured function calls executed by host code | `@tool` + `model.bind_tools([func])` |
| **Agent** | Multi-step loop where the model decides which tools to call until finished | ReAct loop with step limits and termination rules |
| **Message History** | Maintains multi-turn state isolated by session ID | Persistent session history store |

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an AI study assistant.'),
    ('human', 'Explain {topic} in three sentences.')
])
chain = prompt | ChatOpenAI(model='gpt-4o-mini', temperature=0) | StrOutputParser()
# Configure OPENAI_API_KEY and verify network/cost budgets before invoking.
# print(chain.invoke({'topic': 'retrieval augmented generation'}))
```

---

## 2. The Six-Step RAG Architecture

1. **Document Collection & Loading**: Load trusted local documents (`datasets/documents/*.txt`) or parse PDFs/Markdown.
2. **Chunking (`chunk_size` & `overlap`)**: Split long documents into manageable passages. Overlapping windows (e.g., `chunk_size=700, overlap=80`) prevent sentences at chunk boundaries from losing context; larger chunks add noise, while tiny chunks lose coherence.
3. **Indexing & Retrieval (Lexical vs. Semantic vs. Hybrid)**:
   - **Lexical Retrieval (TF-IDF / BM25, Chapter 09)**: Matches exact terms weighted by rarity. Runs offline at zero cost and excels on exact product codes or acronyms, but misses synonyms.
   - **Dense Semantic Retrieval (Embeddings, Chapter 13)**: Encodes chunks into dense vectors to match conceptual meaning across different phrasings.
   - **Production Standard (Hybrid Search + Reranking)**: Combine lexical BM25 + dense vector search and re-score top candidates with a Cross-Encoder Reranker.
4. **Grounded Generation with Citations**: Instruct the LLM to answer **only** from the supplied context passages, abstain when evidence is insufficient, and cite source filenames at the end.
5. **Evaluating RAG Systems**: Measure **Retrieval Recall@k**, **Faithfulness / Groundedness** (absence of unsupported hallucinations), **Answer Relevance**, latency, and token cost.

---

## 3. Agents, Tool Calling, and Application Security

- **Chain vs. Agent**: A Chain runs a fixed sequence of steps; an **Agent** uses the LLM in a loop to choose which tools to invoke based on intermediate observations. Remember: the LLM only proposes tool call arguments—**your application code controls whether and how the function actually executes**.
- **Security Essentials**:
  1. **Defend Against Indirect Prompt Injection**: Treat retrieved external documents strictly as **untrusted data**, never as executable system instructions.
  2. **Validate Tool Arguments & Permissions**: Enforce schema checks, scope limits, and human confirmation for state-changing actions.
  3. **Protect API Keys & Data Privacy**: Store credentials in environment variables (ignored in `.gitignore`), never hardcoded in scripts or notebooks.

---

## 4. Hands-on Projects & Self-Test

- **Runnable Projects**:
  - Offline retrieval + optional LangChain OpenAI RAG: [`projects/06_llm_rag/main.py`](../projects/06_llm_rag/main.py).
  - Tool-calling agent example: [`projects/06_llm_rag/agent.py`](../projects/06_llm_rag/agent.py).
- **Self-Test**:
  1. Does RAG completely eliminate all hallucinations? Why doesn't high retrieval similarity guarantee a faithful answer?
  2. Why do we include a positive `overlap` (< `chunk_size`) when splitting documents into chunks?
  3. What is the fundamental difference between a static `PromptTemplate | LLM` chain and a tool-calling Agent?
