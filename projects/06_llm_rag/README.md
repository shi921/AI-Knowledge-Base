**Language:** English | [简体中文](README.zh-CN.md) · [Home](../../README.md)

# Offline Retrieval with Optional LLM-powered Answers

By default, the project uses three **original short reference documents** in `datasets/documents/`. No network access or API charges are needed:

```bash
python projects/06_llm_rag/main.py "What is retrieval augmented generation?"
python projects/06_llm_rag/agent.py
```

With `OPENAI_API_KEY` configured and `langchain-core langchain-openai` installed, use `--backend openai` to enable cloud-based generation. Retrieved passages are sent to the provider and requests may incur charges. The default TF-IDF engine performs **lexical** retrieval, not neural semantic embedding search. Add only text documents you have permission to process to `datasets/documents/`.

**Related notes:** [Chapter 06](../../notes/06_llm_langchain_rag.md), [Chapter 14](../../notes/14_llm_prompting_finetuning_lora.md).
