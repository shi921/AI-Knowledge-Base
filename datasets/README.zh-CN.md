**语言：** [English](README.md) | 简体中文 · [首页](../README.zh-CN.md)

# 仓库演示数据

- `housing/`：**程序生成的合成房价表格**，供数据探索与回归流水线使用。售价只具有演示意义。
- `stagger/` 与 `sea/`：**程序生成的合成概念漂移序列**，供在线学习与漂移检测使用。
- `documents/`：三篇简短的原创 AI 技术介绍，用于 TF-IDF 离线检索。
- 鸢尾花 / Digits 分类使用 `sklearn.datasets` 提供的数据加载器，不在仓库内再分发数据文件。

可运行 `python scripts/generate_demo_data.py` 重建 CSV。其他人脸、个人信息、WikiText、模型权重不提供，请自行取得适当授权并存放于本地排除路径。
