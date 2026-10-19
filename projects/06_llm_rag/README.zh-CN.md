**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# 离线检索 + 可选 LLM 问答

默认使用仓库中的**原创迷你知识文档** `datasets/documents/`，无网络、无 API 费：

```bash
python projects/06_llm_rag/main.py "What is retrieval augmented generation?"
python projects/06_llm_rag/agent.py
```

设置 `OPENAI_API_KEY` 且安装 `langchain-core langchain-openai` 后，可使用 `--backend openai` 开启云端生成（传输检索文档片段，会产生费用）。默认 TF-IDF 是**词汇检索**，不是神经语义向量检索。可将经过授权的纯文本文件放入 `datasets/documents/` 进行扩展。

**相关知识笔记：** [第 06 章](../../notes/06_llm_langchain_rag.zh-CN.md)、[第 14 章](../../notes/14_llm_prompting_finetuning_lora.zh-CN.md)。
