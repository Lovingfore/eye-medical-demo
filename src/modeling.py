"""轻量 RGB 质心回退分类器与统一推理接口。

# 【技术栈】Python 标准库 + Pillow/ImageStat；该实现不依赖 PyTorch，主要用于
# 合成小样本流程演示和极简环境回退，不代表真实医学模型。
# 【数据集】默认使用 data/demo 中的内置合成 normal/disease 图像；当 artifact
# 标记 backend=pytorch 或 model_type=resnet18 时，统一接口会转发到真实 IDRiD 模型。

The preferred production backend is PyTorch/ResNet-18.  The demo deliberately
keeps a dependency-free fallback so that the complete data -> model -> web
flow runs on a fresh Python installation.  Both backends expose the same
artifact and prediction schema; this module implements the lightweight
centroid model and can be replaced by a torch adapter without changing callers.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, ImageStat

CLASS_NAMES = ["normal", "disease"]
IMAGE_SIZE = (224, 224)
FEATURE_NAMES = ["mean_red", "mean_green", "mean_blue", "brightness_std"]


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _resolve_image(csv_path: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    options = [csv_path.parent / candidate]
    # Split manifests normally retain paths relative to the dataset root. Walk
    # ancestors to support both ``splits/train.csv`` and ``dataset/train.csv``.
    options.extend(parent / candidate for parent in csv_path.parents)
    for option in options:
        if option.exists():
            return option
    # A split file may be written beside (rather than inside) the dataset
    # directory.  Resolve the manifest's relative image path by looking below
    # its nearby ancestors; this keeps generated ``splits/train.csv`` usable
    # without adding absolute paths to a portable CSV.
    pattern = candidate.as_posix()
    for ancestor in csv_path.parents:
        try:
            matches = list(ancestor.rglob(pattern))
        except OSError:
            matches = []
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Image referenced by {csv_path} was not found: {image_path}")


def _rows(manifest_path: str | Path) -> list[dict[str, str]]:
    path = _as_path(manifest_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def _feature(image_path: str | Path) -> list[float]:
    # 【特征提取】把图片缩放到 32×32，计算 RGB 通道均值和亮度标准差，作为
    # 无 PyTorch 回退模型的四维输入；真实 ResNet-18 不使用这些手工特征。
    """提取四个简单 RGB 特征，作为无 PyTorch 环境的轻量回退模型输入。"""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((32, 32))
        channels = image.split()
        means = [ImageStat.Stat(channel).mean[0] / 255.0 for channel in channels]
        pixel_data = image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata()
        brightness = [sum(pixel) / 3.0 for pixel in pixel_data]
    mean_brightness = sum(brightness) / len(brightness)
    variance = sum((value - mean_brightness) ** 2 for value in brightness) / len(brightness)
    return [*means, math.sqrt(variance) / 255.0]


def _centroid(features: Iterable[list[float]]) -> list[float]:
    values = list(features)
    if not values:
        return [0.0] * len(FEATURE_NAMES)
    return [sum(row[index] for row in values) / len(values) for index in range(len(FEATURE_NAMES))]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def train_model(
    split_paths: Mapping[str, str | Path],
    artifact_path: str | Path,
    *,
    seed: int = 42,
    epochs: int = 5,
    backend: str = "auto",
) -> dict[str, object]:
    """Train the deterministic fallback classifier and save a JSON artifact.

    ``split_paths`` is the dictionary returned by :func:`make_splits`; only
    the train split is used for fitting. Validation/test data are intentionally
    not consulted for model selection.
    """
    # 质心模型没有随机优化过程，因此 seed 只为保持与训练接口兼容。
    del seed
    train_path = split_paths.get("train")
    if train_path is None:
        raise ValueError("split_paths must contain a train manifest")
    train_manifest = _as_path(train_path)
    grouped: dict[int, list[list[float]]] = {0: [], 1: []}
    for row in _rows(train_manifest):
        label = int(row["label"])
        if label not in grouped:
            raise ValueError(f"Unsupported binary label: {label}")
        grouped[label].append(_feature(_resolve_image(train_manifest, row["image_path"])))
    # 每一类的质心就是该类训练样本特征的均值，预测时比较欧氏距离。
    centroids = {str(label): _centroid(values) for label, values in grouped.items()}
    artifact = {
        "format_version": 1,
        "backend": "stdlib" if backend in {"auto", "stdlib", "numpy", "pytorch"} else backend,
        "model_type": "rgb_centroid",
        "class_names": CLASS_NAMES,
        "image_size": list(IMAGE_SIZE),
        "feature_names": FEATURE_NAMES,
        "centroids": centroids,
        "epochs": int(epochs),
        "num_train_samples": sum(len(values) for values in grouped.values()),
        "warning": "Synthetic-data demo model; not for medical diagnosis.",
    }
    destination = _as_path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"artifact_path": str(destination), "backend": artifact["backend"], "num_train_samples": artifact["num_train_samples"]}


def load_model(artifact_path: str | Path) -> dict[str, object]:
    """读取 JSON artifact，并附加来源路径供相对 checkpoint 解析使用。"""
    if isinstance(artifact_path, Mapping):
        return dict(artifact_path)
    path = _as_path(artifact_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(artifact, dict):
        # Used by the PyTorch adapter to resolve relative checkpoint paths.
        artifact["_artifact_path"] = str(path.resolve())
    return artifact


def predict_image(image_path: str | Path, model: str | Path | Mapping[str, object]) -> dict[str, object]:
    """返回 CLI 和 Django 共用的预测结构。

    当前 IDRiD 配置会路由到 PyTorch/ResNet-18；轻量 demo artifact 才会走
    RGB 质心回退模型。两条路径都返回类别、置信度、概率和耗时。
    """
    artifact = load_model(model) if not isinstance(model, Mapping) else dict(model)
    # 用 artifact 元数据路由后端，Web 层不需要知道具体模型实现。
    if artifact.get("backend") == "pytorch" or artifact.get("model_type") == "resnet18":
        try:
            from .torch_modeling import predict_torch_image
        except ImportError:  # pragma: no cover - supports direct script imports
            from torch_modeling import predict_torch_image  # type: ignore[no-redef]
        return predict_torch_image(image_path, artifact)
    started = time.perf_counter()
    features = _feature(image_path)
    centroids = artifact["centroids"]
    distances = {label: _distance(features, list(centroids[str(label)])) for label in (0, 1)}
    # 【回退模型推理】比较输入特征与两类质心的欧氏距离，再对负距离做 softmax，
    # 得到依赖无关的相对概率；它不是医学意义上的校准概率。
    scores = {label: math.exp(-min(distances[label], 50.0)) for label in (0, 1)}
    total = scores[0] + scores[1]
    probabilities = {CLASS_NAMES[label]: scores[label] / total for label in (0, 1)}
    label = 0 if probabilities["normal"] >= probabilities["disease"] else 1
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": float(probabilities[CLASS_NAMES[label]]),
        "probabilities": probabilities,
        "inference_ms": round(elapsed_ms, 3),
    }
