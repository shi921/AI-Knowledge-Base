**语言：** [English](04_flask_model_deployment.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 04｜模型部署：从 sklearn 到 Flask API

## 服务架构

`训练数据 → 预处理+训练模型 → joblib 序列化 → Flask 加载 → HTTP JSON 输入 → 参数验证 → model.predict → JSON 输出`。训练和服务最好分开：请求不应每次重训。部署时模型还要携带特征顺序、编码映射和预处理步骤。

## HTTP 常用概念

- `GET /health`：检查服务在线。
- `POST /predict`：发送 JSON 特征，获得类别/概率；缺失、非数值、无穷值等返回 HTTP 400。
- 200 表示成功，4xx 表示请求问题，5xx 通常表示服务异常；本地演示 `127.0.0.1` 而不是暴露公网。

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.get('/health')
def health():
    return jsonify(status='ok')

# See projects/04_flask_iris_api/app.py for the complete implementation.
```

## API 输入与输出契约

将鸢尾花的四个输入特征映射为 JSON 字段 `sl`、`sw`、`pl`、`pw`，通过 `POST /predict` 返回预测类别。数据校验、特征顺序和异常处理都是部署接口的一部分。

## 部署陷阱

1. Flask 开发服务器不是高可用生产服务；不要将 `debug=True` 暴露公网。
2. `joblib.load`/pickle 可能执行任意代码，**只加载自己可信地训练的模型文件**。
3. HTTP 输入需要类型、范围和形状验证。
4. 模型数据分布变化时，应结合版本控制、服务监控和重新训练计划。
5. 评估应区分训练集指标、离线验证指标与线上质量。

**完整项目**：`projects/04_flask_iris_api/`，包含训练脚本、Flask API、curl 调用和 pytest 检验。
