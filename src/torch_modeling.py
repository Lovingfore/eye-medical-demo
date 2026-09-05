"""ResNet-18 transfer-learning backend for the IDRiD classifier.

The public API deliberately mirrors ``src.modeling`` so evaluation and the
Django upload flow can consume either the lightweight demo artifact or this
real PyTorch checkpoint.
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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18(*, pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """Build ResNet-18 and replace its classifier with a binary head."""

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int]]):
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
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"])


def _make_transform(*, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
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
    class_weights = labels.numel() / (2.0 * counts.clamp(min=1.0))
    model = build_resnet18(pretrained=pretrained, num_classes=2).to(target_device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(target_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
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
