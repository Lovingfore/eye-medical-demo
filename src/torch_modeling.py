"""PyTorch/ResNet-18 训练与推理实现。

本模块对应项目的真实训练路线：

* 技术栈：Python、PyTorch、torchvision、Pillow、NumPy、scikit-learn；
* 模型：ImageNet 预训练 ResNet-18，替换最后全连接层进行二分类迁移学习；
* 数据集：公开 IDRiD B. Disease Grading。原始等级 0 映射为 normal，
  等级 1~4 合并为 disease；
* 输入：CSV manifest 中的 image_path、label、class_name 三个字段；
* 输出：可部署的 JSON artifact 与 PyTorch checkpoint，供 CLI 和 Django Web 共用。

公开数据集的原始图像不随仓库提交；本模块只读取已经整理好的本地 manifest。
"""

from __future__ import annotations

import json
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

try:
    from .data import load_manifest, resolve_image_path
except ImportError:  # pragma: no cover
    from data import load_manifest, resolve_image_path  # type: ignore[no-redef]


CLASS_NAMES = ["normal", "disease"]
IMAGE_SIZE = (224, 224)
# 【技术栈/输入标准】torchvision 负责图像变换，Pillow 负责读取 JPG/PNG，
# PyTorch 张量采用 CHW 布局；224×224 是 ResNet-18 的统一输入尺寸。
# 采用 ImageNet 的通道均值和标准差，是因为 backbone 使用了 ImageNet 预训练权重。
# 训练和线上推理必须使用同一组值，否则输入分布会发生偏移。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18(*, pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    # 【模型结构/迁移学习】加载 ImageNet 预训练 ResNet-18，保留卷积主干的
    # 通用视觉特征，把原 ImageNet 分类头替换成 normal/disease 二分类头。
    """构建 ResNet-18，并把 ImageNet 分类头替换为本项目的分类头。

    ``pretrained=True`` 时复用 ImageNet 特征提取能力；最后的全连接层输入
    维度保持不变，输出维度改成 normal/disease 两类。
    """

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int]]):
    # 【数据集读取】Dataset 根据 CSV manifest 逐行读取 IDRiD（或格式兼容的新增数据），
    # 将 image_path 解析成图片，将 label 转换为 PyTorch 的整数类别标签。
    def __init__(self, manifest_path: str | Path, *, train: bool) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = load_manifest(self.manifest_path)
        self.transform = _make_transform(train=train)
        for row in self.rows:
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Expected binary labels 0/1, got {label}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = resolve_image_path(row["image_path"], self.manifest_path)
        with Image.open(image_path) as image:
            # Dataset 在训练和验证阶段共用；差异由 _make_transform(train=...) 控制。
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"])


def _make_transform(*, train: bool) -> transforms.Compose:
    # 【图像预处理/数据增强】训练集使用随机裁剪、翻转和颜色扰动增加样本变化；
    # 验证集、测试集和 Web 推理使用确定性 CenterCrop，保证评价和线上输入一致。
    """返回训练或评估变换。

    训练增强只作用于训练集，避免验证/测试结果受随机变换影响。验证和线上
    推理使用确定性的 CenterCrop，保证评估与 Web 端输入处理一致。
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                # 在 256 边长上随机裁剪 224，模拟轻微构图变化。
                transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                # 轻微改变亮度/对比度/饱和度，降低设备和曝光差异的影响。
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _metrics(y_true: list[int], y_pred: list[int], probabilities: list[float]) -> dict[str, float]:
    # 【多指标评价】训练/验证阶段同时统计 Accuracy、Precision、Recall、F1、
    # Specificity 和 ROC-AUC；模型选择主要依据验证集 F1，避免只看单一准确率。
    """计算训练/验证阶段的二分类指标。

    disease 的概率用于 ROC-AUC；阈值分类结果由 logits.argmax 得到。训练时
    记录这些指标用于观察收敛，但模型选择仍以验证集 F1 为主。
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) == 2 else 0.5,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    # 【训练循环/验证循环】optimizer 不为空时执行训练模式（前向、损失、反向传播、
    # 参数更新）；optimizer 为空时执行验证模式，只推理和统计指标，不更新权重。
    """执行一个训练或验证 epoch。

    ``optimizer`` 不为空代表训练：启用梯度、反向传播和参数更新；为空代表
    验证：关闭梯度并只统计损失与指标。
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                # 典型的 PyTorch 更新顺序：清梯度 -> 前向损失 -> 反向 -> 更新。
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            probabilities.extend(probs.detach().cpu().tolist())
    metrics = _metrics(y_true, y_pred, probabilities)
    metrics.update({"loss": total_loss / max(len(loader.dataset), 1), "num_samples": len(y_true)})
    return metrics


def save_torch_artifact(
    artifact_path: str | Path,
    checkpoint_path: str | Path,
    *,
    best_epoch: int,
    best_val_f1: float,
) -> dict[str, Any]:
    """保存可部署 JSON 配置，并记录 checkpoint 的相对路径。

    使用相对路径是为了让同一份仓库可以从 Windows 本地训练目录迁移到
    Linux/Render；真正的权重文件仍由 checkpoint_path 指向。
    """
    destination = Path(artifact_path)
    checkpoint = Path(checkpoint_path)
    # Store a repository-relative path so an artifact trained on Windows also
    # works when the same checkout is deployed on Linux.
    try:
        checkpoint_reference = os.path.relpath(
            checkpoint.resolve(), destination.parent.resolve()
        ).replace(os.sep, "/")
    except ValueError:
        checkpoint_reference = str(checkpoint.resolve())
    artifact = {
        "format_version": 2,
        "backend": "pytorch",
        "model_type": "resnet18",
        "architecture": "resnet18",
        "class_names": CLASS_NAMES,
        "image_size": list(IMAGE_SIZE),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "checkpoint_path": checkpoint_reference,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "warning": "Research/course project model; not for medical diagnosis.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep the location only in the in-memory object returned to callers.  It
    # is intentionally omitted from the JSON file committed to the repository.
    artifact["_artifact_path"] = str(destination.resolve())
    return artifact


def train_resnet_model(
    train_manifest: str | Path,
    val_manifest: str | Path,
    artifact_path: str | Path,
    checkpoint_path: str | Path,
    *,
    history_path: str | Path | None = None,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    pretrained: bool = True,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """训练 ResNet-18 并返回训练摘要。

    当前实现是单次固定超参数实验，不包含网格搜索、贝叶斯优化或交叉验证。
    ``pretrained`` 可以关闭，便于做随机初始化对照实验。
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    _set_seed(seed)
    target_device = _select_device(device)
    train_data = ManifestImageDataset(train_manifest, train=True)
    val_data = ManifestImageDataset(val_manifest, train=False)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=target_device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=target_device.type == "cuda",
    )
    labels = torch.tensor([int(row["label"]) for row in train_data.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=2).float()
    # 类别权重 w_k=N/(K*n_k)：样本少的 normal 类获得更高损失权重，缓解不均衡。
    class_weights = labels.numel() / (2.0 * counts.clamp(min=1.0))
    model = build_resnet18(pretrained=pretrained, num_classes=2).to(target_device)
    # 【损失函数】带类别权重的交叉熵，类别权重 w_k=N/(K*n_k)，用于缓解
    # normal 与 disease 样本数量不均衡对训练的影响。
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(target_device))
    # 【优化器】AdamW 将权重衰减与 Adam 的梯度更新解耦，适合小样本迁移学习的稳定微调。
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        # 【每轮训练】先用训练集更新参数，再用未增强的验证集计算泛化指标；
        # 只保存验证集 F1 最好的 checkpoint，降低最后几轮过拟合权重被部署的风险。
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, target_device)
        val_metrics = _run_epoch(model, val_loader, criterion, None, target_device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        score = float(val_metrics["f1"])
        previous_loss = float(history[best_epoch - 1]["val"]["loss"]) if best_epoch else float("inf")
        is_better = score > best_f1 or (score == best_f1 and float(val_metrics["loss"]) < previous_loss)
        if is_better:
            best_f1 = score
            best_epoch = epoch
            # 只保存验证集 F1 最好的状态，避免最后一个 epoch 过拟合的权重被部署。
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "resnet18",
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_config": {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "pretrained": pretrained,
                        "seed": seed,
                    },
                },
                checkpoint,
            )
        print(
            f"epoch {epoch:02d}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f}"
        )
    if best_epoch == 0:
        raise RuntimeError("Training produced no checkpoint")
    artifact = save_torch_artifact(artifact_path, checkpoint, best_epoch=best_epoch, best_val_f1=best_f1)
    history_payload = {
        "device": str(target_device),
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "class_counts": {"normal": int(counts[0]), "disease": int(counts[1])},
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "history": history,
    }
    if history_path is None:
        history_path = Path(artifact_path).parent / "training_history.json"
    history_destination = Path(history_path)
    history_destination.parent.mkdir(parents=True, exist_ok=True)
    history_destination.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "artifact_path": str(Path(artifact_path).resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "history_path": str(history_destination.resolve()),
        "device": str(target_device),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
    }


def _checkpoint_for(artifact: dict[str, Any]) -> Path:
    """将模型 JSON 中的相对 checkpoint 路径解析为本地绝对路径。"""
    path = Path(str(artifact["checkpoint_path"]))
    if path.is_absolute():
        return path
    artifact_path = artifact.get("_artifact_path") or artifact.get("artifact_path")
    if artifact_path:
        return Path(str(artifact_path)).parent / path
    # This fallback keeps hand-authored artifacts usable from the project root.
    project_root = Path(__file__).resolve().parents[1]
    return project_root / path


@lru_cache(maxsize=4)
def _load_cached_model(checkpoint_path: str, device_name: str) -> tuple[nn.Module, torch.device]:
    """加载并缓存推理模型，避免每次上传都重复构建和读取权重。"""
    device = torch.device(device_name)
    model = build_resnet18(pretrained=False, num_classes=2)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device)
    model.eval()
    return model, device


def predict_torch_image(
    image_path: str | Path,
    artifact: dict[str, Any],
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """对单张图像执行与验证阶段一致的预处理和 ResNet 推理。"""
    target_device = _select_device(device)
    model, target_device = _load_cached_model(str(_checkpoint_for(artifact).resolve()), str(target_device))
    transform = _make_transform(train=False)
    started = time.perf_counter()
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(target_device)
    with torch.no_grad():
        probabilities_tensor = torch.softmax(model(tensor), dim=1)[0]
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    probabilities = {
        CLASS_NAMES[index]: float(probabilities_tensor[index].item()) for index in range(2)
    }
    label = int(probabilities_tensor.argmax().item())
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": probabilities[CLASS_NAMES[label]],
        "probabilities": probabilities,
        "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
