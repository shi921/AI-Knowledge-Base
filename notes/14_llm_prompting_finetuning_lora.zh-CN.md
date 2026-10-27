**语言：** [English](14_llm_prompting_finetuning_lora.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 14｜大模型定制化：提示词工程（Prompting）、微调选型与 LoRA / PEFT

> **🧭 知识脉络定位**：第五篇「大语言模型（LLM）原理、微调与应用开发」· **第 14 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：在第 13 章我们了解了预训练大模型（LLM）的诞生过程。现在你想把这位“通才毕业生”培养成你的专属业务专家，有两条由轻到重的定制路线：  
>   1. **Prompting（提示词工程 —— 教说话沟通技巧）**：完全不动他的脑子（不改模型权重），而是在布置任务时把角色、背景、步骤和输出格式写清楚（**Zero-shot**），或者直接给他看几个标准问答范例让他照猫画虎（**Few-shot 上下文学习**）；  
>   2. **LoRA 微调（参数高效微调 PEFT —— 贴专属技能便利贴）**：如果你需要彻底改变他的固有说话风格、行业黑话或输出规约，把几十亿参数全部重新训练一遍（Full Fine-tuning）不仅贵而且容易把原有的通用智商洗坏（灾难性遗忘）。**LoRA 的做法是：把原来的大脑参数全部冰冻锁死，只在关键层旁边挂上一块极小的“可训练附加芯片（低秩矩阵 $\Delta W = BA$）”**——只训练不到 1% 的参数，就能在一张普通消费级显卡上定制出专属大模型！  
> - **🎯 这一章解决什么核心问题？**：掌握系统化的提示词设计模式、建立架构师级别的 **「Prompt vs. RAG vs. 微调」选型决策力**，并亲手跑通基于 HuggingFace `transformers` + `peft` 的 LoRA 训练与挂载推理全流程。  
> - **🔗 前后衔接**：[13｜大语言模型（LLM）基础](13_llm_foundations.zh-CN.md) → 本章（掌握提示词引导与 LoRA 参数微调）→ **下一章（收官）**：[15｜大模型应用工程：LangChain、RAG 与 Agent](15_llm_langchain_rag.zh-CN.md)（将大模型与私有检索库、外部工具结合，构建完整企业级应用系统）。

---

## 一、 提示词工程（Prompt Engineering）核心范式

在不修改任何模型权重的前提下，通过精心构造输入上下文（Context）来激发大模型的最佳表现：

1. **角色与边界设定（System Prompt）**：明确模型的身份、任务目标、禁止事项以及期望的输出格式（如 JSON / Markdown 表格）。
2. **零样本提示（Zero-shot Prompting）**：只给任务说明，不提供示例，适合模型已经熟悉的常规任务。
3. **少样本提示（Few-shot / In-Context Learning）**：在 Prompt 中提供 $2 \sim 5$ 个高质量 `<输入, 期望输出>` 示范对，能显著消除输出格式歧义。
4. **思维链与自洽性核验（Chain-of-Thought & Self-Verification）**：
   - 引导模型“先分步推理，再给出最终结论”，或要求模型在生成答案后对照约束条件做一次自我核查。
   - **工程准则**：要求模型内部思考虽然能提升复杂逻辑题的准确率，但**不能保证 100% 绝对正确**；对于需要下游程序解析的结构化输出，必须在代码中用正则或 Pydantic Schema 做独立验证！

---

## 二、 架构师选型指南：Prompting vs. RAG vs. Fine-Tuning

新手最常犯的错误是：**“想让大模型记住公司每天更新的产品文档，于是跑去做 LoRA 微调”**。请死记以下选型决策表：

| 定制技术 | 是否修改模型权重？ | 知识/材料存放在哪？ | 最适合解决什么问题？ | 绝对不适合什么场景？ |
|---|---|---|---|---|
| **Prompting<br/>（提示词工程）** | 否（0 参数改动） | 写在单次请求的 Prompt 里 | **首选起步方案**：快速验证想法、约束输出格式、提供少量示范 | 示例太多塞不下上下文窗口，或每次请求重复发送几万字导致 Token 成本过高 |
| **RAG<br/>（检索增强，第 15 章）** | 否（0 参数改动） | 存放在外部文档库 / 向量数据库中，运行时动态检索 | **高频更新的事实知识**（公司规章、最新财报、私域知识库）、**要求严格标注文献出处**以抑制幻觉 | 试图改变模型的基础语言风格、深层推理习惯或特定代码语法 |
| **Full Fine-tune<br/>（全量微调）** | 是（修改 100% 权重） | 烘焙进全模型权重中 | 算力充足且拥有海量领域数据，需重塑模型底层领域认知 | 算力受限、数据量少（极易过拟合与灾难性遗忘）或需频繁更新事实 |
| **LoRA / PEFT<br/>（低秩微调）** | 仅训练 **<1% 的外挂适配器权重**（基座冻结） | 存放在几 MB～几十 MB 的 Adapter 权重文件中 | **低成本定制说话风格/腔调**（如特定角色扮演）、固化复杂输出格式、垂直领域指令适配 | 把微调当成“实时数据库”用来死记硬背经常变动的事实，或缺少与 Adapter 匹配的原始基座模型 |

---

## 三、 LoRA（Low-Rank Adaptation）核心数学原理与显存账本

### 1. 为什么全量微调（Full Fine-tuning）这么吃显存？
训练一个 7B（70 亿参数）的模型，除了存模型权重本身，还必须存**所有参数的梯度**以及 **Adam 优化器的一阶、二阶动量状态**（优化器状态占用的显存往往是模型权重的 $2 \sim 4$ 倍！）。

### 2. LoRA 的四两拨千斤之道：低秩矩阵分解
研究发现，在大模型微调过程中，权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$ 的变化量 $\Delta W$ 具有极低的“内在秩（Intrinsic Rank）”。于是 LoRA **冻结原始权重 $W_0$（设置 `requires_grad=False`，不计算梯度也不存优化器状态）**，仅在旁边并联两个瘦长的低秩矩阵 $A \in \mathbb{R}^{r \times k}$ 和 $B \in \mathbb{R}^{d \times r}$：

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \qquad \text{其中秩 } r \ll \min(d, k)$$

- **参数量对比举例**：若原始层维度为 $4096 \times 4096$（共 $1677$ 万参数），设 LoRA 秩 $r = 8$，则 $A$ 和 $B$ 加起来只有 $2 \times 4096 \times 8 = 6.55$ 万参数——**需要训练的参数减少了 99.6%！**
- **精妙初始化**：训练开始前，矩阵 $A$ 用高斯随机初始化，而**矩阵 $B$ 全零初始化（$B = 0$）**——这样保证训练第 0 步时 $\Delta W = BA = 0$，模型表现与原始预训练模型 100% 完全一致，不会破坏原有能力。
- **三大核心超参数**：
  - `r`（Rank 秩，常用 `8, 16, 32`）：低秩瓶颈维度，决定了适配器的容量；
  - `lora_alpha`（缩放系数，常设为 `r` 或 `2*r`）：通过 $\alpha / r$ 控制适配器输出对原模型的干预强度；
  - `target_modules`（挂载目标层）：通常挂载在自注意力层的投影矩阵（如 `["q_proj", "v_proj"]` 或全部线性层）。
- **工程铁律**：训练保存下来的 LoRA Checkpoint（如 `adapter_model.safetensors`）**只是那张“透明便利贴”，不是完整模型**！部署推理时，必须先加载版本、词表完全一致的**原始 Base Model**，再把 Adapter 挂载上去（或合并权重 `merge_and_unload()`）。

---

## 四、 本库实验入口与安全注意事项

- **本地提示词工程练习（无需联网或 API Key）**：
  - 运行 `python projects/14_llm_prompting_and_lora/prompting.py "I really like this movie"`，直观对比 Zero-shot、Few-shot 与自核验提示模板构建。
- **基于 HuggingFace PEFT 的 Qwen2.5 LoRA 微调与推理流水线**：
  - 代码见 [`projects/14_llm_prompting_and_lora/lora_pipeline.py`](../projects/14_llm_prompting_and_lora/lora_pipeline.py)。
  - **训练模式**：`python projects/14_llm_prompting_and_lora/lora_pipeline.py --mode train --train-limit 60`（显式执行时才会联网拉取 `Qwen/Qwen2.5-0.5B-Instruct` 与 `yoda_sentences` 数据集，将训练好的 LoRA 适配器保存至 `outputs/14_lora_training/adapter/`）。
  - **推理模式**：`python projects/14_llm_prompting_and_lora/lora_pipeline.py --mode infer --adapter /path/to/adapter --text "I love learning AI"`。

---

## 五、 思考与自测

1. 如果你们公司的报销政策每周都在修改，应该用 **RAG** 还是 **LoRA 微调** 来让大模型回答报销问题？为什么？
2. 为什么在 LoRA 初始化时，要把矩阵 $A$ 设为随机高斯分布、把矩阵 $B$ 设为全 `0`？如果两个都设为全 `0` 会发生什么？
3. 为什么只把几 MB 的 `adapter_model.safetensors` 发给同事，同事在没有下载原始基座模型的情况下无法独立运行推理？
