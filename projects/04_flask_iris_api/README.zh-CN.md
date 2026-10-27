**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# Flask 模型部署：鸢尾花分类 API

在仓库根目录运行：

```bash
python projects/04_flask_iris_api/model.py
python projects/04_flask_iris_api/app.py
# Test from another terminal
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"sl":5.1,"sw":3.5,"pl":1.4,"pw":0.2}'
```

可通过 `python -m pytest projects/04_flask_iris_api/test_api.py -q` 测试。项目展示训练、保存、加载、输入验证与 HTTP 返回，开发服务器不适于直接用于生产环境。

**相关知识笔记：** [第 06 章](../../notes/06_flask_model_deployment.zh-CN.md)。
