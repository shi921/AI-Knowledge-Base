**Language:** English | [简体中文](06_llm_langchain_rag.zh-CN.md) · [Home](../README.md) · [Study roadmap](../STUDY_ROADMAP.md)

# 06 | Large Language Models: Prompts, Chains, Agents, Memory, and RAG

> **🧭 Knowledge Map Position**: Stage 5 (LLM Applications & Fine-Tuning) · **Step 13 of 14**  
> - **💡 Intuitive Analogy**: A pre-trained Transformer LLM (Chapter 13) is like a **brilliant scholar locked in a closed-book exam room**—knowledgeable, but unaware of your company's private files, prone to guessing confidently when unsure (**hallucination**), and unable to use a calculator.  
>   - **RAG (Retrieval-Augmented Generation)** turns that closed-book test into an **open-book exam**: before the LLM answers, your code retrieves the most relevant passages from your documents and pastes them into the prompt as evidence;  
>   - **Agents & Tool Calling** give the scholar **a toolkit** so it can request external function calls (search, math, APIs).  
> - **🎯 Why Does This Matter?**: RAG and Agents are the backbone of modern enterprise AI engineering. This chapter covers document chunking, lexical vs. semantic retrieval, grounded generation, tool binding, and prompt-injection defense.  
> - **🔗 Learning Chain**: [13 | Transformers & Self-Attention](13_transformer_attention.md) → This Chapter (Ground LLMs with RAG & tools) → **Final Step**: [14 | LLM Prompting, Fine-Tuning, and LoRA](14_llm_prompting_finetuning_lora.md) (When prompting and RAG aren't enough, how do we adapt the model's weights on a budget?).

## Building blocks

| Concept | Purpose | Typical API |
|---|---|---|
| Prompt | Specify role, objective, constraints, and output format | `ChatPromptTemplate.from_messages(...)` |
| Chat model | Generate messages from context | `ChatOpenAI(...)` |
| Chain / LCEL | Compose reusable steps | `prompt | model | StrOutputParser()` |
| Structured output | Parse responses into validated fields/JSON | `model.with_structured_output(PydanticModel)` |
| Tool calling | Let a model request tool actions controlled by code | `@tool` and `model.bind_tools(...)` |
| Agent | Iteratively choose tools and actions | Tool loop with state and stopping rules |
| Message history | Maintain conversation state | Session-scoped message storage |
| Embeddings | Map text to semantic vectors | `OpenAIEmbeddings`, etc. |
| RAG | Retrieve evidence before generating | retrieve → prompt → LLM → cite |

## Minimal chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an AI study assistant.'),
    ('human', 'Explain {topic} in three sentences.')
])
chain = prompt | ChatOpenAI(model='gpt-4o-mini', temperature=0) | StrOutputParser()
# Configure OPENAI_API_KEY and check connectivity and costs first.
# print(chain.invoke({'topic': 'concept drift'}))
```

## Full RAG workflow

1. **Ingest:** start with `datasets/documents/`. Load web pages or PDFs only when appropriately licensed and supported.
2. **Chunk:** split long texts with suitable overlap; too-large chunks dilute relevance and tiny chunks lose context.
3. **Index:** use TF-IDF for lexical retrieval or embeddings for semantic retrieval.
4. **Retrieve:** fetch top-k passages and optionally rerank.
5. **Generate:** include retrieved passages and filenames; require evidence-based answers, and acknowledge missing evidence.
6. **Evaluate:** retrieval recall, answer faithfulness, citation precision, latency, tokens, and cost.

**Important:** TF-IDF + cosine similarity is **lexical**, not neural semantic embedding retrieval. Offline mode uses TF-IDF; `--backend openai` enables optional cloud generation.

## Security and architecture

- Treat retrieved documents as **data**, not instructions; resist prompt injection.
- Keep API keys in environment variables, never hardcoded or committed.
- Validate tool arguments, enforce privileges and limits, and require human approval for consequential actions.
- Check whether external-provider transmission is authorized and whether charges are acceptable.
- LangChain APIs evolve; the examples target `langchain_core` and `langchain_openai` building blocks.

An **agent** differs from a chain through conditional, potentially repeated tool choice; code still determines what tools actually execute. Conversation **memory** needs session isolation and persistence rather than relying on a chat window as a database.

**Projects:** [`projects/06_llm_rag/main.py`](../projects/06_llm_rag/main.py) (offline retrieval and optional LLM), `agent.py` (tool calling). Potential extensions include structured outputs, vector databases, routing, and persistent conversation state.

**Review questions:** Can RAG eliminate hallucinations? Does high retrieval recall ensure trustworthy answers? What distinguishes a prompt template from an agent?

**Connection to prompting:** compare zero-shot, few-shot, and checking prompts in [`projects/14_llm_prompting_and_lora/prompting.py`](../projects/14_llm_prompting_and_lora/prompting.py) before building RAG or tuning a model. Validate permissions and fields in code before acting on any LLM response.
