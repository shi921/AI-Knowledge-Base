**语言：** [English](14_llm_prompting_finetuning_lora.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 14｜LLM 的 Prompting、Fine-tuning、LoRA 与 PEFT

## Prompting 与 RAG 的不同层次

**Zero-shot** 给任务说明不提供示例；**few-shot** 给输入输出示范以减少格式歧义；**思考与核验提示** 引导模型检查答案，但要求内部推理并不能保证正确。结构化响应应使用明确输出格式、独立验证解析结果。

**RAG** 不修改模型权重：检索业务文档、拼入上下文、生成答案。**微调** 修改一部分或全部模型权重，更适合学习格式、风格或某类行为模式；不应把微调当成持续更新事实库的唯一机制。

| 技术 | 对参数的影响 | 外部材料 | 适用场景 |
|---|---|---|---|
| Prompting | 无 | prompt 中示例 | 快速试验和格式约束 |
| RAG | 无 | 运行时检索 | 经常变更的文档、来源归属 |
| Full Fine-tune | 全模型 | 训练集 | 资源充足且目标明确 |
| LoRA / PEFT | 只训练低秩适配器 | 训练集 | 资源较受限的任务/风格适配 |

## LoRA 核心思想

对冻结的线性层 `W` 只训练低秩增量 `ΔW = BA`；当 rank `r` 远小于原始维度时，需训练的参数显著减少。adapter checkpoint **不是完整模型**，加载必须获得原始基座模型，且适配器目标层、tokenizer、模型版本须兼容。

实际数据预处理须把指令、输入和期望响应拼成训练样本。若训练目标是“仅学习响应”，常需要对 prompt 区域的 labels 使用 `-100` 进行 loss masking；本库的短教学训练脚本使用全序列 causal LM loss 并明确属于简化方案。

## 本库实验入口

- 本地模板：`python projects/14_llm_prompting_and_lora/prompting.py`，默认不调用 API。
- 自行准备有权使用、与基座匹配的 Adapter 权重，并放入 `projects/14_llm_prompting_and_lora/checkpoint/`（空目录已保留）。
- LoRA 推理：`python projects/14_llm_prompting_and_lora/lora_pipeline.py --mode infer`，需额外安装 Transformers/PEFT、网络下载 Qwen2.5-0.5B-Instruct 基座及足够内存。
- LoRA 训练：`--mode train` 下载 `dvgodoy/yoda_sentences`，把新 adapter 保存到 `outputs/14_lora_training/`，不会覆盖其他 adapter 目录。

**安全与工程注意事项**：禁止把 API Key 写在 Notebook；通过环境变量提供凭据。prompt 注入、外部数据可信度、模型偏见、人工审核都要考虑；别把用户评论未经授权传给外部 API。

**自测**：RAG 与 LoRA 哪种适合解决“公司政策每天变化”？为什么只复制 `.safetensors` adapter 不能独立部署模型？
