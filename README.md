# 基于 IDRiD 的糖尿病视网膜病变分类系统

一个可部署的眼底图像二分类系统。项目使用公开的 IDRiD（Indian Diabetic Retinopathy Image Dataset）数据集训练 ResNet-18 迁移学习模型，并提供 Django Web 端用于上传眼底图像、返回分类概率和查询历史记录。

> **项目定位**：课程/科研项目级的研究应用，可用于算法验证、流程演示和部署学习。当前版本不是医疗器械，不构成医学诊断、筛查或治疗建议。请勿将系统输出作为临床决策依据。

## 项目能力

- 接收 JPG/JPEG/PNG 眼底图像，单张文件限制 8 MB。
- 输出 `normal`（未见明显糖尿病视网膜病变）与 `disease`（糖尿病视网膜病变）二分类结果。
- 展示两类概率、预测置信度和推理耗时。
- 保存最近预测记录，并在 `/history/` 页面查询。
- 使用已训练的 FP16 模型权重，可在 CPU-only Render 实例上直接部署。
- 提供数据检查、划分、训练、评估和单图预测命令，便于复现实验。

## 技术架构

```text
浏览器上传图像
        |
        v
Django + WhiteNoise + Gunicorn
        |
        v
图像校验与预处理（RGB、224 x 224、ImageNet 归一化）
        |
        v
ResNet-18 迁移学习推理（PyTorch）
        |
        +--> 分类概率 / 置信度 / 推理耗时
        +--> SQLite 历史记录
```

主要技术：Python 3.11、PyTorch/torchvision、ResNet-18、Django、SQLite、Pillow、NumPy、scikit-learn、Gunicorn 和 WhiteNoise。

## 数据集与标签

本项目使用 IDRiD 的 **B. Disease Grading** 子集：

- 官方划分：413 张训练图像、103 张测试图像，共 516 张彩色眼底图。
- 标签映射：等级 `0` -> `normal`；等级 `1-4` -> `disease`。
- 官方 103 张测试集只用于最终评价；从 413 张训练集按随机种子 `42` 划出 validation 集。
- 原始数据不提交到 GitHub。下载和准备方法见[真实数据资源](#真实数据资源)。

数据集入口：[IEEE DataPort - IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)。下载需要免费的 IEEE DataPort 账号。项目也记录了已校验的公开镜像来源，便于复现。

## 模型训练

模型采用 ImageNet 预训练 ResNet-18，将最后的全连接层替换为二分类头，使用加权 `CrossEntropyLoss` 和 AdamW 优化器。当前已完成的一次训练配置如下：

- 训练集：351 张
- 验证集：62 张
- batch size：16
- learning rate：`1e-4`
- weight decay：`1e-4`
- epoch：8
- 最佳模型选择：validation F1
- 训练设备：NVIDIA CUDA（推理支持 CPU）

训练入口：[`src/train_resnet.py`](src/train_resnet.py)。原始全精度权重仅保留在本地，仓库提交的是用于部署的 FP16 权重 [`artifacts/idrid_resnet_best_fp16.pt`](artifacts/idrid_resnet_best_fp16.pt)。

## 测试结果

结果来自官方 103 张测试集，记录在 [`artifacts/idrid_resnet_evaluation/metrics.json`](artifacts/idrid_resnet_evaluation/metrics.json)：

| 指标 | 结果 |
|---|---:|
| Accuracy | 0.7573 |
| Precision | 0.8333 |
| Recall | 0.7971 |
| F1 | 0.8148 |
| Specificity | 0.6765 |
| ROC-AUC | 0.8201 |
| 测试样本数 | 103 |

混淆矩阵（行是真实类别，列是预测类别；类别顺序为 `normal`, `disease`）：

```text
[[23, 11],
 [14, 55]]
```

这些指标是单一公开数据集上的研究结果，不能直接代表真实临床场景的泛化能力。正式研究还应进行患者级划分、外部验证、置信区间估计和偏差分析。

## 本地运行

### 1. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 2. 启动 Web 服务

```powershell
python web/manage.py migrate
python web/manage.py runserver 127.0.0.1:8000
```

打开 <http://127.0.0.1:8000/>，上传一张 JPG/PNG 图像即可查看预测结果。默认模型配置为 `artifacts/idrid_resnet_model.json`，其中引用部署用的 FP16 权重。

### Windows 跨设备一键部署

在另一台 Windows 设备上克隆仓库后，可以直接运行根目录下的 [`setup_windows.ps1`](setup_windows.ps1)。脚本会自动创建或复用 `.venv`、安装 CPU 推理依赖、检查模型文件、执行 Django 数据库迁移和配置检查，然后启动 Web 服务。

```powershell
git clone https://github.com/Lovingfore/idrid-diabetic-retinopathy-classifier.git
cd idrid-diabetic-retinopathy-classifier
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

首次运行需要联网安装依赖，且设备需要安装 Python 3.10 或更高版本（推荐 Python 3.11）。模型权重已经随仓库提交，不需要下载 IDRiD 原始训练数据即可进行 Web 推理。

常用参数：

```powershell
# 已经安装过依赖时跳过 pip 安装
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipInstall

# 不自动打开浏览器
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipBrowser

# 允许局域网其他设备访问（还需要配置 Windows 防火墙）
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -BindAddress 0.0.0.0 -Port 8000 -SkipBrowser
```

默认服务地址为 <http://127.0.0.1:8000/>。脚本启动的是本地 Django 开发服务器，按 `Ctrl+C` 停止；公网部署仍请使用 Render 配置。

### 3. 运行测试与配置检查

```powershell
python -m pytest -q
python web/manage.py check
python web/manage.py collectstatic --noinput
```

### 4. 单图命令行预测

```powershell
python -m src.predict \
  --model artifacts/idrid_resnet_model.json \
  --image path/to/fundus.jpg
```

## 训练与评估复现

准备 IDRiD 图像和标签后，先生成处理后的清单，再训练和评估：

```powershell
python -m src.prepare_idrid data/real/idrid --output-dir data/real/idrid/processed
python -m src.make_splits data/real/idrid/processed/labels_train.csv `
  --output-dir data/real/idrid/processed/splits --seed 42
python -m src.train_resnet `
  --train data/real/idrid/processed/splits/train.csv `
  --val data/real/idrid/processed/splits/val.csv `
  --artifact artifacts/idrid_resnet_model.json `
  --checkpoint artifacts/idrid_resnet_best.pt `
  --history artifacts/idrid_resnet_training_history.json `
  --epochs 8 --batch-size 16 --learning-rate 1e-4 `
  --weight-decay 1e-4 --device cuda
python -m src.evaluate `
  --model artifacts/idrid_resnet_model.json `
  --test data/real/idrid/processed/labels_test.csv `
  --output artifacts/idrid_resnet_evaluation
```

Windows PowerShell 使用反引号换行；Linux/macOS 请改用反斜杠或单行命令。没有 CUDA 时，将 `--device cuda` 改为 `--device cpu`。

## Render 公网部署

仓库已经包含 [`render.yaml`](render.yaml)、[`Procfile`](Procfile) 和 [`requirements-render.txt`](requirements-render.txt)，可直接创建 Render Blueprint：

1. 在 Render 控制台选择 **New + -> Blueprint**。
2. 连接 GitHub 仓库并确认 `render.yaml`。
3. 等待构建、数据库迁移和静态文件收集完成。
4. 使用 Render 生成的 `onrender.com` 地址访问系统。

Render 的生产启动命令为：

```text
gunicorn --chdir web eye_demo.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

生产环境通过 `DJANGO_SECRET_KEY`、`DJANGO_DEBUG=false` 和 `RENDER_EXTERNAL_HOSTNAME` 配置安全参数。免费实例使用临时磁盘，SQLite 历史记录和用户上传文件可能在实例重启或重新部署后清理；模型代码和权重随仓库发布。

## 项目目录

```text
.
├── artifacts/
│   ├── idrid_resnet_model.json          # 模型配置
│   ├── idrid_resnet_best_fp16.pt        # Render 推理权重
│   └── idrid_resnet_evaluation/         # 测试指标和混淆矩阵
├── src/
│   ├── prepare_idrid.py                 # IDRiD 数据准备
│   ├── make_splits.py                   # 训练/验证划分
│   ├── train_resnet.py                  # ResNet-18 训练入口
│   ├── evaluate.py                      # 指标与误判明细
│   └── modeling.py                      # 统一推理接口
├── web/
│   ├── eye_demo/                        # Django 项目配置
│   └── predictor/                       # 上传、预测、历史记录
├── tests/                               # 自动化测试
├── render.yaml                          # Render Blueprint 配置
├── requirements-render.txt               # CPU 部署依赖
└── README.md
```

## 真实数据资源

- IDRiD 官方页面：[IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
- 已记录的公开镜像：[Zenodo 17219542](https://zenodo.org/records/17219542)
- Hugging Face 数据集镜像：[amin-nejad/idrid-disease-grading](https://huggingface.co/datasets/amin-nejad/idrid-disease-grading)

原始图像和 parquet 文件应只保存在本地 `data/` 目录，不要上传到公开仓库。使用临床图像前必须确认授权、去标识化和数据使用范围。

## 使用边界与后续工作

当前版本只支持 `normal / disease` 二分类，且训练样本量有限。后续正式研究建议：

- 按患者编号进行数据划分，避免同一患者图像泄漏到不同集合。
- 使用多中心外部数据验证，并报告置信区间、校准曲线和亚组表现。
- 增加 Grad-CAM 等可解释性分析与人工复核流程。
- 评估更强的模型、类别阈值和图像质量检测。
- 为生产环境接入持久化数据库、对象存储、访问控制、日志审计和 HTTPS。

**重要：本系统输出仅供课程/科研项目使用，不构成医疗诊断或医疗建议。**
