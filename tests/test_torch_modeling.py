"""Contract tests for the real PyTorch/ResNet backend."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from src.modeling import predict_image  # noqa: E402
from src.torch_modeling import build_resnet18, save_torch_artifact, predict_torch_image  # noqa: E402


def test_resnet18_builder_replaces_head_with_two_classes() -> None:
    model = build_resnet18(pretrained=False, num_classes=2)
    output = model(torch.zeros(1, 3, 224, 224))
    assert tuple(output.shape) == (1, 2)
    assert model.fc.out_features == 2


def test_torch_artifact_prediction_has_web_schema(tmp_path: Path) -> None:
    model = build_resnet18(pretrained=False, num_classes=2)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    artifact = save_torch_artifact(
        tmp_path / "model.json",
        checkpoint,
        best_epoch=1,
        best_val_f1=0.5,
    )
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (80, 60), (120, 80, 40)).save(image_path)

    result = predict_torch_image(image_path, artifact, device="cpu")

    assert result["label"] in (0, 1)
    assert result["class_name"] in ("normal", "disease")
    assert set(result["probabilities"]) == {"normal", "disease"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_common_prediction_api_routes_pytorch_artifact(tmp_path: Path) -> None:
    model = build_resnet18(pretrained=False, num_classes=2)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    artifact = save_torch_artifact(tmp_path / "model.json", checkpoint, best_epoch=1, best_val_f1=0.5)
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (80, 60), (120, 80, 40)).save(image_path)

    result = predict_image(image_path, artifact)

    assert result["class_name"] in ("normal", "disease")
