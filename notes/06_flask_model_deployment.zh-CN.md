**语言：** [English](06_flask_model_deployment.md) | 简体中文 · [首页](../README.zh-CN.md) · [学习路线](../STUDY_ROADMAP.zh-CN.md)

# 06｜模型工程化部署：从 sklearn 到 Flask REST API

> **🧭 知识脉络定位**：第三篇「模型工程化部署与流式运维」· **第 06 章 / 共 15 章**  
> - **💡 零基础直觉比喻**：你在自家厨房研发出了一道高分且配方透明的好菜（第 04、05 章训练并审计好的模型），但顾客不可能跑到你家电脑里运行 Python 脚本。你必须**开一家外卖营业窗口（HTTP REST API）**：把训练好的模型封存打包成文件（`joblib` 序列化），当手机 App 或网页发来点单请求（HTTP JSON 输入）时，窗口先检查参数是否合法，再调用模型秒级返回预测结果（JSON 输出）。  
> - **🎯 这一章解决什么核心问题？**：打通“离线实验脚本”到“在线生产服务”的工程闭环。理解为什么**离线训练（Training）**与**在线推理（Serving/Inference）**必须彻底解耦，以及如何设计健壮的 API 契约与输入校验。  
> - **🔗 前后衔接**：[05｜模型可解释性（XAI）：SHAP 局部与全局归因](05_explainable_ai_shap.zh-CN.md) → 本章（将模型封装为在线 HTTP API 服务）→ **下一章**：[07｜流式数据学习、概念漂移与 Page–Hinkley](07_stream_learning_concept_drift.zh-CN.md)（服务上线运营一段时间后，外部环境变了、线上准确率下滑怎么办？）。

---

## 一、 离线训练 vs. 在线推理服务架构

```text
[离线阶段 Offline Training]
原始训练数据 → 预处理 + 模型训练 (Pipeline) → joblib.dump 序列化保存模型工件 (.joblib)

[在线阶段 Online Serving]
服务启动加载模型一次 → 接收 HTTP POST JSON 请求 → 严格校验字段/类型/范围 → model.predict → 返回 HTTP JSON 响应
```

- **核心工程铁律**：Web 服务在启动时将模型加载进内存，每次收到 HTTP 请求**只做轻量推理（`predict`），绝不能在请求处理函数里重新训练模型**！
- **完整工件打包**：部署保存的不应只是裸分类器权重，而应将**特征顺序、缺失值填充器、标准化缩放器（`StandardScaler`）与分类器**一起打包保存，确保线上推理与离线训练的特征变换 100% 一致。

---

## 二、 HTTP 接口设计与状态码契约

- **`GET /health`（健康检查接口）**：供负载均衡器或容器编排系统（如 Kubernetes）探测服务是否存活，正常返回 HTTP `200` 与 `{"status": "ok"}`。
- **`POST /predict`（推理预测接口）**：客户端通过请求体（Request Body）发送 JSON 格式的特征数据，服务端返回预测类别或概率。
- **HTTP 状态码规范**：
  - `200 OK`：请求合法且推理成功；
  - `400 Bad Request`：客户端请求格式错误、缺少必填字段、包含非数值字符串、`NaN` 或无穷大（`Inf`）；
  - `500 Internal Server Error`：服务端内部异常（如模型文件未找到或运行时崩溃）。

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get('/health')
def health():
    return jsonify(status='ok')

# 完整参数校验与推理实现见 projects/04_flask_iris_api/app.py
```

---

## 三、 API 输入与输出契约示例（鸢尾花分类）

在配套项目中，我们将鸢尾花的四个形态特征映射为明确的 JSON 字段契约：
- **输入 JSON**：`{"sl": 5.1, "sw": 3.5, "pl": 1.4, "pw": 0.2}`（花萼长宽、花瓣长宽）
- **服务端防线**：在喂给 `model.predict` 之前，必须逐一验证：① 请求体是否为合法 JSON 对象；② 四个键是否齐全；③ 值是否可转为有限浮点数（拒绝 `None`、布尔值、`NaN`、`Inf`）；④ 按训练时的严格列顺序组装二维数组 `[[sl, sw, pl, pw]]`。

---

## 四、 生产环境部署的五大陷阱

1. **开发服务器 vs. 生产 WSGI 服务器**：Flask 自带的 `app.run()` 仅限本地开发调试，单进程且非高可用；生产环境必须使用 `Gunicorn` / `uWSGI` 配合 Nginx，且**严禁在公网开启 `debug=True`**（会暴露远程代码执行漏洞）。
2. **反序列化安全（Pickle / Joblib 风险）**：`joblib.load` 和 `pickle.load` 在加载文件时可触发任意代码执行。**绝对不要加载来源不明或未经签名校验的模型文件**。
3. **特征漂移与静默失败**：如果上游业务系统把某个特征的单位从“厘米”改成了“毫米”，HTTP 类型校验不会报错，但模型会给出完全荒谬的预测——因此线上必须配置特征范围检验与分布监控（见第 07 章）。
4. **模型版本管理**：模型重训上线时应保留版本号、训练数据快照与离线评估指标，以便线上效果异常时一键回滚。

---

## 五、 配套实践与自测

- **可运行项目**：[`projects/04_flask_iris_api/`](../projects/04_flask_iris_api/README.zh-CN.md)（包含离线训练脚本 `model.py`、Flask 服务 `app.py`、`curl` 调用示例与 `pytest` 自动化接口测试 `test_api.py`）。
- **思考与自测**：
  1. 为什么不能在 `@app.post('/predict')` 函数内部每次收到请求时才调用 `joblib.load()` 读取磁盘？
  2. 如果客户端传来 `{"sl": "abc", "sw": 3.5, "pl": 1.4, "pw": 0.2}`，服务端应该返回 HTTP `400` 还是 `500`？为什么？
