# 第14题眼底医学图像分类 Demo

这是一个用于综合实训第14题流程演示的最小闭环项目：生成合成眼底样本、检查与划分数据、训练/评价二分类模型，并通过 Django 上传图片显示预测结果。

## 重要说明

当前数据由 `src/generate_demo_data.py` 生成，仅用于演示数据流和页面功能，不是真实眼底医学图像，也不能替代 IDRiD 等真实数据的实验结果，更不能用于医学诊断。

标签固定为 `0 = normal`、`1 = disease`；图像为 RGB、`224 x 224`。

## 快速开始

```powershell
python -m pip install -r requirements.txt
./run_demo.ps1 -SamplesPerClass 20 -SkipWeb
# 如果 Django 已安装，再执行：
python web/manage.py migrate
python web/manage.py runserver 127.0.0.1:8000
```

或者逐步执行：

```powershell
python -m src.generate_demo_data --output-dir data/demo --samples-per-class 20 --seed 42
python -m src.check_data data/demo/labels.csv
python -m src.make_splits data/demo/labels.csv --output-dir data/demo/splits --seed 42
python -m src.train --train data/demo/splits/train.csv --val data/demo/splits/val.csv --test data/demo/splits/test.csv --artifact artifacts/model.json
python -m src.evaluate --model artifacts/model.json --test data/demo/splits/test.csv --output artifacts/evaluation
python -m src.predict --model artifacts/model.json --image data/demo/images/normal_0000.png
```

合成数据默认仍使用无重型依赖的 RGB 特征质心演示模型；真实 IDRiD 数据使用 `src/torch_modeling.py` 的 ResNet-18 迁移学习后端，公共 `src.modeling.predict_image` 会自动按 artifact 路由到对应模型。

## IDRiD 训练

当前已完成一次 GPU 训练：351 张 train、62 张 validation，ImageNet 预训练 ResNet-18，AdamW，batch size 16，learning rate `1e-4`，weight decay `1e-4`，8 个 epoch，按 validation F1 保存最佳权重。复现实验命令：

```powershell
D:\python\python.exe -m src.train_resnet `
  --train data/real/idrid/processed/splits/train.csv `
  --val data/real/idrid/processed/splits/val.csv `
  --artifact artifacts/idrid_resnet_model.json `
  --checkpoint artifacts/idrid_resnet_best.pt `
  --history artifacts/idrid_resnet_training_history.json `
  --epochs 8 --batch-size 16 --learning-rate 1e-4 `
  --weight-decay 1e-4 --device cuda
```

最佳 epoch 为 3。官方 103 张测试集结果保存在 `artifacts/idrid_resnet_evaluation/metrics.json`：Accuracy 0.7573、Precision 0.8333、Recall 0.7971、F1 0.8148、Specificity 0.6765、ROC-AUC 0.8201，混淆矩阵为 `[[23, 11], [14, 55]]`。这些结果仅用于课程项目研究，不构成医学诊断。

## Web 端

启动最终 Web：

```powershell
D:\python\python.exe web/manage.py migrate
D:\python\python.exe web/manage.py runserver 127.0.0.1:8000
```

浏览器访问 <http://127.0.0.1:8000/>；默认读取 `artifacts/idrid_resnet_model.json`，上传接口会调用 `artifacts/idrid_resnet_best.pt`，历史记录位于 SQLite 的 `web/demo.sqlite3`。

## 发布到 GitHub 和公网

仓库不会提交 IDRiD 原始数据、parquet 文件、SQLite 数据库或用户上传图片；压缩为 FP16 的 ResNet-18 权重（约 22 MB）和模型配置会随代码提交，云端可直接推理。Render 使用 `requirements-render.txt` 的 CPU-only PyTorch wheel，避免安装 CUDA 运行库。

### 1. 创建并推送 GitHub 仓库

在 GitHub 新建一个空仓库（例如 `eye-medical-demo`），然后在本目录执行：

```powershell
git init
git add .
git commit -m "feat: publish IDRiD eye disease demo"
git branch -M main
git remote add origin https://github.com/<你的用户名>/eye-medical-demo.git
git push -u origin main
```

### 2. Render 部署 Django Web

打开 <https://render.com/>，选择 **New + -> Blueprint**，连接刚创建的 GitHub 仓库并确认 `render.yaml`。Blueprint 会自动安装依赖、执行数据库迁移和静态文件收集，并用 Gunicorn 启动 Django。部署完成后 Render 会生成一个 `onrender.com` 网址，可直接分享访问。

也可以在 Render 手动创建 Web Service：

```text
Build Command: pip install -r requirements.txt && python web/manage.py migrate --noinput && python web/manage.py collectstatic --noinput
Start Command: gunicorn --chdir web eye_demo.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

生产环境使用 `DJANGO_DEBUG=false` 和 Render 自动生成的 `DJANGO_SECRET_KEY`。免费实例的 SQLite 和上传目录是临时磁盘，实例重启后历史记录/上传文件可能被清理；模型代码和权重不受影响。

### 3. 访问限制

这是课程项目演示，不是医疗诊断工具。模型只做 `normal / disease` 二分类，真实测试集指标见上文；公网部署后请勿上传包含个人身份信息的临床资料。

## 真实数据资源

正式实验推荐使用 IDRiD（Indian Diabetic Retinopathy Image Dataset）的 `B. Disease Grading` 子集：516 张彩色眼底图（官方 train 413 / test 103）和对应 CSV 分级标签。二分类映射为 `等级 0 -> normal`、`等级 1-4 -> disease`。原始入口：<https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>；需要免费的 IEEE DataPort 账号登录下载，IEEE 会员资格不是必需的。本项目已从公开镜像 <https://zenodo.org/records/17219542> 下载并校验 Hugging Face 版本（<https://huggingface.co/datasets/amin-nejad/idrid-disease-grading>），文件哈希与公开记录一致。

保留官方 103 张测试集作为最终测试集，再从官方 413 张训练集划分 validation；随机种子为 42。测试集只用于最终评价，不用于调参。

已下载文件位于 `data/real/idrid/`。如需从 parquet 重新导出 JPG 和 manifest：

```powershell
python -m pip install pyarrow
python -m src.prepare_idrid data/real/idrid --output-dir data/real/idrid/processed
```

生成的 manifest 是 `data/real/idrid/processed/labels_train.csv` 和 `labels_test.csv`，图片分别位于 `processed/images/train/`、`processed/images/test/`；当前本地校验结果为 train 413 张、test 103 张，缺失/损坏文件均为 0。

## 技术选型

- Python 3.11+
- Pillow：图像读写和基础处理
- NumPy：合成数据与轻量回退
- PyTorch/torchvision：首选 ResNet-18 迁移学习训练
- scikit-learn：分类指标和混淆矩阵
- Django + SQLite：上传、结果和简易历史记录

## 与 PDF 第14题的对应

| PDF 要求 | Demo 对应 |
|---|---|
| 公开眼部医学图像数据、清洗、统一尺寸和归一化 | `src/generate_demo_data.py`、`src/check_data.py`、`src/data.py`（当前用合成数据演示） |
| 迁移学习、损失函数、优化器和超参数 | `src/torch_modeling.py`、`src/train_resnet.py`；使用 ImageNet ResNet-18、加权 CrossEntropyLoss 和 AdamW |
| 准确率、精确率、召回率、F1、混淆矩阵和误判分析 | `src/evaluate.py` 生成 `metrics.json`、`predictions.json`、`confusion_matrix.csv` |
| Django 接收图片并调用模型 | `web/predictor/views.py`、`templates/predictor/` |
| 简单历史记录查询 | SQLite `Prediction` 模型和 `/history/` |

## 需要的真实资源

正式实验时需要你提供或下载：

1. IDRiD `B. Disease Grading.zip` 解压后的图像目录和两个标签 CSV；建议放到 `data/real/idrid/`，不要提交到公开仓库。
2. 可用的 Python 环境和磁盘空间；真实 ResNet-18 训练建议使用 NVIDIA GPU。
3. PyTorch/torchvision 已在本机安装为 `torch 2.5.1+cu121`、`torchvision 0.20.1+cu121`；无 GPU 时可将 `--device` 改为 `cpu`。
4. 老师提供的第一周 PPT/论文模板（如有），用于后续整理正式提交材料。

## 限制与后续工作

- 合成图像是人工生成的 fundus-like 纹理，不是医学数据；演示指标不具备临床意义。
- 当前未实现 EfficientNet-B0、Grad-CAM、复杂图像分割和硬件采集。
- 真实数据接入后，应按患者分组（若存在患者编号）划分数据，并只在最终模型确定后报告测试集。
- Web 页面明确标注“模型预测结果，仅用于课程项目展示，不构成医学诊断”。
