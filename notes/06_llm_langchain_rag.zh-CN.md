**语言：** [English](06_llm_langchain_rag.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 06｜大语言模型：Prompt、Chains、Agents、Memory 与 RAG

> **🧭 知识脉络定位**：第五阶段「大语言模型（LLM）应用与微调实战篇」· **第 13 步 / 共 14 步**  
> - **💡 零基础直觉比喻**：基于第 13 章 Transformer 训练出的大语言模型（LLM）就像一位**博览群书但被关在考场里的天才毕业生**——他常识极其丰富，但有三大弱点：① 不知道你们公司今天刚发的内部规章；② 记不清细节时会为了面子瞎编（**幻觉 Hallucination**）；③ 没有手脚，不能帮你查数据库或按计算器。  
>   - **RAG（检索增强生成）** 就是把“闭卷考试”变成**“开卷考试”**：用户提问时，先从公司文档库里翻出最相关的几段话塞进试卷（Prompt）里，让大模型看着参考资料回答并标明出处；  
>   - **Agent（智能体与工具调用）** 则是给这位天才配上**“手脚和工具箱”**，让他自己决定什么时候该调用计算器或外部接口。  
> - **🎯 这一章解决什么核心问题？**：这是当今企业落地大模型最核心的工程技术（90% 的企业 AI 应用都是 RAG + Agent）。掌握从文档切块、检索、提示词组装到工具调用的完整闭环，以及防范提示词注入（Prompt Injection）的安全底线。  
> - **🔗 前后衔接**：[13｜Transformer 与自注意力](13_transformer_attention.zh-CN.md) → 本章（用外挂知识库与工具武装大模型）→ **下一站（收官）**：[14｜大模型提示工程、微调与 LoRA](14_llm_prompting_finetuning_lora.zh-CN.md)（如果开卷参考书还不够，怎么低成本修改大模型自身的习惯与参数？）。

## 从简单调用扩展为应用

| 概念 | 作用 | 典型实现 |
|---|---|---|
| Prompt | 定义角色、目标、边界和输出结构 | `ChatPromptTemplate.from_messages(...)` |
| Chat Model | 根据上下文生成消息 | `ChatOpenAI(...)` |
| Chain / LCEL | 将步骤串接以便复用 | `prompt | model | StrOutputParser()` |
| Structured output | 解析为受控字段/JSON | `model.with_structured_output(PydanticModel)` |
| Tool calling | 模型提出工具调用，由程序执行 | `@tool` + `model.bind_tools(...)` |
| Agent | 模型多轮决定下一步调用哪些工具 | 工具状态循环与终止条件 |
| Message history | 维护会话状态 | history + session ID |
| Embeddings | 把文本映射为语义向量 | `OpenAIEmbeddings` 等 |
| RAG | 检索文档后依据证据生成答案 | retrieve → prompt → LLM → cite |

## 核心最短链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an AI study assistant.'),
    ('human', 'Explain {topic} in three sentences.')
])
chain = prompt | ChatOpenAI(model='gpt-4o-mini', temperature=0) | StrOutputParser()
# Configure OPENAI_API_KEY and check network access and costs first.
# print(chain.invoke({'topic':'concept drift'}))
```

## RAG 的完整流程

1. **收集文档**：使用仓库自带的短文（`datasets/documents`）；网页/PDF 可以用相应 Loader。
2. **切块**：长文分段，重叠能减少段落边界遗漏；块过大会引入无关信息，过小则损失语境。
3. **向量化与索引**：用 TF-IDF（关键词相关性）或语义 embeddings（语义相关性）建立检索库。
4. **召回**：对用户问题检索 top-k 内容，可再排序。
5. **生成**：将内容和来源放入 Prompt，要求严格依据材料回答；找不到就拒绝编造。
6. **评价**：检索召回率、回答事实一致性、引用准确性、时延、token/成本。

**重要区别**：`TF-IDF + 余弦相似度` 是词汇检索，不等同于向量语义 embedding。仓库默认离线 TF-IDF；只有 `--backend openai` 才会调用云端 LLM。

## Prompt 与应用安全

- 把外部检索文档视为**数据**，不应执行文档中的指令（prompt injection）。
- API Key 不应硬编码 Notebook；使用环境变量，在 Git 中忽略 `.env`。
- 工具调用需参数检验、范围限制、权限约束和必要的人为确认。
- 使用外部 LLM 前确认文档是否允许发送到第三方及费用预算。
- LangChain API 会随版本变化；示例采用 `langchain_core`, `langchain_openai` 的基础接口。

## Agent 与 Memory 的本质

Agent 比普通 Chain 多了**条件分支/选择工具**的循环，工具可以是检索、计算、API；最终仍由代码控制是否执行。Memory 需要定义会话隔离和持久化策略，不能把聊天窗口里的历史自动当作可靠存储。

**完整项目**：[`projects/06_llm_rag/main.py`](../projects/06_llm_rag/main.py)（离线检索+可选 LangChain LLM）、`agent.py`（工具调用）。示例选择器、结构化输出、向量数据库和会话持久化可以作为扩展方向。

**自测**：RAG 能否完全杜绝幻觉？为什么向量召回高不等于答案可信？PromptTemplate 与 Agent 的主要区别是什么？


## 提示词设计与 RAG 的关系

先通过 [`projects/14_llm_prompting_and_lora/prompting.py`](../projects/14_llm_prompting_and_lora/prompting.py) 对比 Zero-shot、Few-shot、核验提示，再使用 [`projects/06_llm_rag/main.py`](../projects/06_llm_rag/main.py) 实践检索增强，最后阅读第 14 章了解参数微调。任何触发外部动作的输出，都必须由代码验证字段、权限与操作条件。
