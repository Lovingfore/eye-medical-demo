"""Contract tests for training, evaluation, and single-image prediction.

The tests intentionally exercise the public API rather than a particular ML
backend.  They must pass both with the preferred PyTorch stack and with the
deterministic standard-library fallback used by the lightweight demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_model  # noqa: E402
from src.generate_demo_data import generate_demo_dataset  # noqa: E402
from src.make_splits import make_splits  # noqa: E402
from src.modeling import load_model, predict_image, train_model  # noqa: E402


def test_train_model_creates_loadable_artifact_and_prediction_schema(tmp_path: Path) -> None:
    generated = generate_demo_dataset(tmp_path / "dataset", samples_per_class=3, seed=42)
    splits = make_splits(generated["manifest_path"], tmp_path / "splits", seed=42)

    result = train_model(splits["paths"], tmp_path / "artifacts" / "model.json", seed=42, epochs=2)
    artifact_path = Path(result["artifact_path"])

    assert artifact_path.exists()
    assert result["backend"] in {"pytorch", "numpy", "stdlib"}

    artifact = load_model(artifact_path)
    prediction = predict_image(tmp_path / "dataset" / "images" / "normal_0000.png", artifact)
    assert set(("label", "class_name", "confidence", "probabilities", "inference_ms")) <= set(prediction)
    assert prediction["label"] in (0, 1)
    assert prediction["class_name"] in ("normal", "disease")
    assert set(prediction["probabilities"]) == {"normal", "disease"}
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-5


def test_evaluate_model_returns_required_metrics_and_confusion_matrix(tmp_path: Path) -> None:
    generated = generate_demo_dataset(tmp_path / "dataset", samples_per_class=4, seed=7)
    splits = make_splits(generated["manifest_path"], tmp_path / "splits", seed=42)
    trained = train_model(splits["paths"], tmp_path / "model.json", seed=42, epochs=1)

    metrics = evaluate_model(trained["artifact_path"], splits["paths"]["test"], tmp_path / "evaluation")

    required = {"accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "confusion_matrix"}
    assert required <= set(metrics)
    assert len(metrics["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in metrics["confusion_matrix"])
    assert (tmp_path / "evaluation" / "metrics.json").exists()
    assert (tmp_path / "evaluation" / "confusion_matrix.png").exists()
