**语言：** [English](README.md) | 简体中文 · [首页](../../README.zh-CN.md)

# 分类模型与可选 PCA 人脸识别

使用 scikit-learn 的 Digits 数据进行分类、交叉验证和指标评估：

```bash
python projects/03_classification_and_faces/main.py
```

可选的人脸 PCA + SVM 演示需要自行提供授权的本地人脸 CSV（包含 `target` 标签列和特征像素列），不在公开仓库分发：

```bash
python projects/03_classification_and_faces/faces.py --data /path/to/authorized/faces.csv --components 80
```

仅加载可信数据，注意真实人脸样本涉及个人隐私、公开使用和再分发权。

**相关知识笔记：** [第 03 章](../../notes/03_supervised_learning_evaluation.zh-CN.md)、[第 02 章](../../notes/02_preprocessing_feature_engineering_pca.zh-CN.md)。
