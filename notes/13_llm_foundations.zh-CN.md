**语言：** [English](13_llm_foundations.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 13｜大语言模型（LLM）基础：架构演进、分词、预训练、对齐与解码生成

> **🧭 知识脉络定位**：第五篇「大语言模型（LLM）原理、微调与应用开发」· **第 13 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：在第 12 章，你学会了如何造一台精巧的“Transformer 发动机”。但怎么把这台小发动机变成像 ChatGPT、Claude、Qwen、DeepSeek 这样无所不知的超级大脑？这就好比**“培养一位人类顶尖通才的三部曲”**：  
>   1. **预训练（Pre-training）＝ 闭关通读人类图书馆**：让他把互联网上几万亿字的书籍、百科、代码全读一遍，做无数道“猜下一个词”的完形填空题，从而掌握人类语言规律、常识与逻辑（此时叫 **Base 基座模型**，懂很多知识，但只会顺着你的话往下续写，不懂怎么礼貌回答问题）；  
>   2. **指令微调（SFT）＝ 岗前对话培训**：给他看几十万道`<人类提问, 专家标准回答>`的示范卷，教会他“看到问题要回答，而不是接着把问题抄下去”（此时变成 **Instruct / Chat 对话模型**）；  
>   3. **人类偏好对齐（RLHF / DPO）＝ 职业道德与情商打磨**：人类考官对他的多个回答打分排序，教他说话要**有用（Helpful）、诚实（Honest）、无害（Harmless）**。  
> - **🎯 为什么必须有这一章？**：它是连接底层神经网络（第 08–12 章）与上层大模型微调/应用（第 14–15 章）的**核心枢纽**！读完本章，你将彻底明白：为什么现代大模型全都是 Decoder-only 架构？什么是 Token 和 KV Cache？`Temperature` 和 `Top-p` 到底怎么控制生成？以及为什么大模型天生会有“幻觉”，从而必须引入提示工程、LoRA 微调与 RAG！  
> - **🔗 前后衔接**：[12｜Transformer 架构与自注意力机制](12_transformer_attention.zh-CN.md) → 本章（系统建立大语言模型全局认知）→ **下一章**：[14｜大模型定制化：提示词工程与 LoRA 微调](14_llm_prompting_finetuning_lora.zh-CN.md) 与 [15｜大模型应用工程：LangChain、RAG 与 Agent](15_llm_langchain_rag.zh-CN.md)。

---

## 一、 从 Transformer 到现代大模型：三大架构路线演进

2017 年原始 Transformer 包含编码器（Encoder）和解码器（Decoder）两部分。随后 AI 界分化出了三条技术路线，并在今天形成了明确分工：

| 架构家族 | 代表模型 | 注意力机制特点 | 核心专长与在今天 AI 系统中的角色 |
|---|---|---|---|
| **1. Encoder-only<br/>（仅编码器 / 双向理解型）** | BERT、RoBERTa、<br/>**BGE / E5（向量模型）** | **双向自注意力**（每个词都能同时看见它左边和右边的所有词） | 不擅长逐字聊天生成；但极其擅长理解整句话的语义，是文本分类（第 12 章）和 **RAG 语义向量检索（Embedding 模型 / Reranker 重排器，第 15 章）** 的主力！ |
| **2. Encoder-Decoder<br/>（编码器-解码器）** | 原始 Transformer、<br/>T5、BART | Encoder 双向理解输入，Decoder 通过交叉注意力（Cross-Attention）自回归生成输出 | 适合明确的序列到序列转换任务（如早期机器翻译、语音转文字 Whisper） |
| **3. Decoder-only<br/>（仅解码器 / 自回归生成型）** | **GPT 系列、Claude、<br/>LLaMA、Qwen（千问）、<br/>DeepSeek** | **单向因果自注意力（Causal Mask）**：第 $t$ 个词只能看 $\le t$ 的前文，严禁偷看未来词 | **当今大语言模型（LLM）的绝对统治架构！** 把翻译、问答、写代码、推理全部统一为同一个极简任务——**“给定前文，预测下一个 Token（Next-Token Prediction）”** |

### 为什么 Decoder-only 架构最终一统江湖？
1. **训练效率最高**：在一段长度为 $L$ 的文本上，Decoder-only 可以利用下三角因果掩码（Causal Mask），在一次前向传播中同时对所有 $L$ 个位置计算“预测下一个词”的损失，无需像 BERT 那样只遮盖 15% 的词，数据利用效率极高。
2. **零样本与少样本泛化能力最强（In-Context Learning）**：由于万物皆可写成“上文 → 下文”的续写形式，当参数量与数据量突破临界点（**Scaling Laws 缩放定律**），模型会涌现出惊人的上下文学习与逻辑推理能力。

---

## 二、 大模型如何读写文字：子词分词（Tokenization）与 KV Cache

### 1. 什么是 Token？为什么不用“整词”或“单字母”？（BPE 子词分词）
在第 11 章中我们按空格切分完整单词，但这会导致词表巨大且遇到新造词就变成 `<unk>`；若按单个英文字母切分，序列又会变得极长且单个字母毫无语义。
现代大模型普遍采用 **BPE（Byte-Pair Encoding，字节对编码）** 等**子词分词器（Subword Tokenizer）**：
- **核心思想**：高频词保留为完整 Token（如 `"apple"`），罕见长词拆分为有意义的词根词缀（如 `"unhappiness"` 拆成 `["un", "happi", "ness"]`），连没见过的生僻字也能退化到 UTF-8 字节表示，**彻底消灭了 `<unk>` 未知词问题**！
- **工程常识**：英文中 $1 \text{ Token} \approx 0.75$ 个单词；中文里 $1$ 个汉字通常对应 $0.6 \sim 1.5$ 个 Token（视词表而定）。大模型 API 的计费和 **上下文窗口（Context Window，如 8K / 128K Tokens）** 都是按 Token 数量计算的。

### 2. 推理生成的两个阶段与 KV Cache（键值缓存加速）
当你给大模型发一段话并让它生成回复时，内部发生两个截然不同的阶段：
1. **Prefill（预填充阶段）**：一次性并行计算你输入的整段 Prompt 所有 Token 的 $Q, K, V$，算出第 1 个新生成 Token。这一步是**计算密集型（Compute-bound）**。
2. **Decode（逐词解码阶段）**：每生成出一个新 Token，都要把它拼到历史序列末尾去算下一个 Token。
   - **痛点**：因为有因果掩码，前面所有历史 Token 的 $K$ 和 $V$ 向量在后续步骤中其实是完全不变的！如果每生成一个字都把前几千个字的 $K, V$ 重新算一遍，会慢得无法忍受。
   - **KV Cache 破局**：把历史 Token 算好的 $K$ 和 $V$ 矩阵缓存在显存里，每步新 Token 只需算自己的 $q_{\text{new}}, k_{\text{new}}, v_{\text{new}}$ 并与缓存拼接即可——用**显存空间换取了数十倍的推理生成速度**！

---

## 三、 大语言模型诞生的三阶段训练全周期

为什么刚预训练完的模型不能直接拿来当聊天助手？理解下面这三个阶段，你就彻底明白了第 14 章微调到底在干什么：

```mermaid
flowchart LR
    A["海量无标注互联网文本<br/>(数万亿 Tokens)"] -->|阶段 1：自监督预训练<br/>Next-Token Prediction| B["Base 基座模型<br/>(博览群书的续写机)"]
    B -->|阶段 2：监督指令微调 SFT<br/>高质量问答/指令数据 + LoRA/全量| C["Instruct / Chat 模型<br/>(听懂指令的对话助手)"]
    C -->|阶段 3：人类偏好对齐<br/>RLHF (PPO) / DPO / GRPO| D["Aligned 生产级大模型<br/>(有用、诚实、无害、会推理)"]
```

| 训练阶段 | 训练数据形态 | 损失函数与训练目标 | 产出模型的特点 |
|---|---|---|---|
| **1. 预训练<br/>(Pre-training)** | TB 级无标注网页、书籍、论文、代码（万亿 Tokens） | 全序列自回归交叉熵损失（预测每一个位置的下一个 Token） | **Base Model**（如 `Qwen2.5-7B`）：注入了海量世界知识，但只会顺着上文续写（问它“怎么学 Python？”，它可能续写“怎么学 Java？”） |
| **2. 监督微调<br/>(SFT / Instruction Tuning)** | 万～百万条 `<System, User 指令, Assistant 回答>` 格式的高质量示范数据 | 仅对 **Assistant 回答部分**计算交叉熵损失（Prompt 部分将 label 设为 `-100` 屏蔽损失） | **Instruct / Chat Model**（如 `Qwen2.5-7B-Instruct`）：学会遵循人类指令格式、多轮对话与特定领域任务（**第 14 章 LoRA 微调正是处于这一阶段！**） |
| **3. 偏好对齐<br/>(Alignment: RLHF / DPO)** | 同一问题下的好回答 vs. 差回答成对比较数据 `(Prompt, Chosen, Rejected)` | **RLHF** 先训奖励模型再用强化学习更新；**DPO（直接偏好优化）** 直接拉大 Chosen 与 Rejected 的生成概率比 | 抑制有害/违规输出，减少顺从性胡编，让回答风格更符合人类偏好，或通过强化学习（如 GRPO）激发深度长思维链推理 |

---

## 四、 控制大模型创造力的旋钮：解码与采样策略（Decoding Strategies）

大模型前向传播的最后一层输出的是词表上所有 Token（例如 15 万个词）的 **Logits 向量 $z$**。我们该如何从中挑出下一个词？

1. **贪心解码（Greedy Search，`do_sample=False` 或 `temperature=0`）**：
   - 永远无脑挑概率最大的第 1 名 Token（$\arg\max P(w)$）。
   - **特点**：结果 100% 确定可复现，适合写代码、数学计算、信息抽取与严肃问答；但写长文章时容易陷入车轱辘话重复循环。
2. **温度采样（Temperature Scaling，$T > 0$）**：
   - 在做 Softmax 之前，先将 Logits 除以温度系数 $T$：
     $$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   - **$T \to 0$（低温，如 `0.1~0.3`）**：让高概率词的优势被无限放大，分布变得极度尖锐，回答严谨保守。
   - **$T = 1.0$**：保持模型原始输出概率分布。
   - **$T > 1.0$（高温，如 `1.2~1.5`）**：把概率分布压平，原本低概率的冷门词也有机会被抽中，回答天马行空有创意，但极易胡言乱语。
3. **Top-k 与 Top-p（核采样 Nucleus Sampling）—— 给随机采样加安全护栏**：
   - 单纯随机采样哪怕概率只有 `0.0001` 的离谱错别字也有可能被抽中。
   - **Top-k**：只保留概率最高的前 $k$ 个候选词（如 $k=50$），把其余词全部淘汰清零后重新归一化采样。
   - **Top-p（核采样，如 `top_p=0.9`）**：将所有词按概率从高到低排序，从头累加，一旦累积概率达到 $90\%$ 就截断淘汰后面的长尾词——**当模型很笃定时只留两三个词，当模型犹豫时自动放宽候选池**，比固定 $k$ 更智能！

```python
import torch
import torch.nn.functional as F

def sample_next_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 0, top_p: float = 0.9) -> int:
    """直观演示大模型解码采样逻辑：Greedy / Temperature / Top-k / Top-p"""
    if temperature <= 0:
        return int(torch.argmax(logits).item())   # 贪心解码
    scaled = logits / temperature
    if top_k > 0:
        top_vals, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        scaled = torch.where(scaled < top_vals[-1], torch.tensor(float('-inf')), scaled)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 淘汰累积概率超过 top_p 的尾部 token（至少保留第 1 名）
        mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float('-inf')
        scaled = torch.zeros_like(scaled).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

# 示例：5 个候选词的原始 logits
demo_logits = torch.tensor([3.2, 2.5, 1.1, -0.5, -2.0])
print('Greedy token ID:', sample_next_token(demo_logits, temperature=0))
```

---

## 五、 大语言模型的四大固有局限与破局路线图（承上启下）

理解了大模型“靠概率预测下一个 Token”的本质，你就明白为什么它不是万能的神，而是存在以下四大固有缺陷——而本知识库最后两章的所有工程技术，正是为了逐一攻克这些缺陷而诞生的：

| 大模型的固有痛点 | 为什么会发生？（底层原因） | 工程破局方案（对应后续章节） |
|---|---|---|
| **1. 任务指令歧义与输出格式失控** | 模型只能根据上文续写，若指令含糊，它会按互联网平均习惯自由发挥 | ➡️ **[第 14 章：提示词工程 (Prompt Engineering)](14_llm_prompting_finetuning_lora.zh-CN.md)**：用 Zero/Few-shot 示例与结构化约束锁定输出 |
| **2. 垂直行业风格与专有规约不服水土** | 通用预训练模型没学过你的特定领域任务模式，而全量重训百亿参数显存爆炸 | ➡️ **[第 14 章：参数高效微调 (PEFT / LoRA)](14_llm_prompting_finetuning_lora.zh-CN.md)**：冻结基座，仅训练 <1% 的低秩适配器权重 |
| **3. 知识截断与“一本正经胡说八道”（幻觉 Hallucination）** | 权重冻结在训练结束那天，不知道你的私有文档；且遇到盲区时仍会挑通顺的词强行续写 | ➡️ **[第 15 章：检索增强生成 (RAG)](15_llm_langchain_rag.zh-CN.md)**：先检索私有文档片段拼入 Prompt，让模型看着证据作答并附引用 |
| **4. 缺乏行动力与精确计算能力** | 概率接龙不擅长多位算术，也无法主动查询实时天气或操作外部业务系统 | ➡️ **[第 15 章：工具调用与智能体 (Tool Calling & Agents)](15_llm_langchain_rag.zh-CN.md)**：给大模型绑定外部函数与执行循环 |

---

## 六、 配套实践与自测

- **关联项目体验**：
  - 回顾 [`projects/12_neural_language_models/ngram_train.py`](../projects/12_neural_language_models/ngram_train.py) 与 [`projects/13_transformer_encoder/main.py`](../projects/13_transformer_encoder/main.py) 理解底层 Token 词表与自注意力；
  - 衔接 [`projects/14_llm_prompting_and_lora/`](../projects/14_llm_prompting_and_lora/README.zh-CN.md) 观察 `Qwen/Qwen2.5-0.5B-Instruct` 的 Tokenizer、Causal LM 训练与 `model.generate()` 解码参数。
- **思考与自测**：
  1. 为什么做语义向量检索（Embedding）通常选用 BERT 这类 Encoder-only 模型，而做聊天生成普遍选用 GPT/Qwen 这类 Decoder-only 模型？
  2. Base 基座模型（如 `Llama-3-8B`）与 Instruct 对话模型（如 `Llama-3-8B-Instruct`）在实际使用时有什么本质区别？
  3. 编写需要严格输出 JSON 或执行代码的程序时，应该把 `temperature` 设高还是设低（接近 `0`）？为什么？
