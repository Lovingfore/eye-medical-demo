"""Metrics and command-line evaluation for a saved demo model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

try:
    from .modeling import _resolve_image, _rows, load_model, predict_image
except ImportError:  # pragma: no cover - supports ``python src/evaluate.py``
    from modeling import _resolve_image, _rows, load_model, predict_image  # type: ignore[no-redef]


def _binary_metrics(y_true: list[int], y_pred: list[int], scores: list[float]) -> dict[str, object]:
    # Prefer sklearn's well-tested implementations in a full environment,
    # while keeping the exact same output contract for the minimal demo.
    try:
        from sklearn.metrics import (  # type: ignore[import-not-found]
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        try:
            auc = 0.5 if len(set(y_true)) < 2 else float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = 0.5
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "roc_auc": auc,
            "confusion_matrix": matrix,
            "num_samples": len(y_true),
        }
        result["f1_score"] = result["f1"]
        result["roc_auc_score"] = result["roc_auc"]
        return result
    except ImportError:
        pass

    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    total = max(len(y_true), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Tie-aware rank AUC. A single-class set has no defined AUC; return 0.5.
    positives = [score for score, label in zip(scores, y_true) if label == 1]
    negatives = [score for score, label in zip(scores, y_true) if label == 0]
    if not positives or not negatives:
        auc = 0.5
    else:
        auc = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives) / (len(positives) * len(negatives))
    result = {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "roc_auc": auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "num_samples": len(y_true),
    }
    result["f1_score"] = result["f1"]
    result["roc_auc_score"] = result["roc_auc"]
    return result


def evaluate_model(artifact_path: str | Path, manifest_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    artifact = load_model(artifact_path)
    manifest = Path(manifest_path)
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[float] = []
    predictions: list[dict[str, object]] = []
    for row in _rows(manifest):
        result = predict_image(_resolve_image(manifest, row["image_path"]), artifact)
        y_true.append(int(row["label"]))
        y_pred.append(int(result["label"]))
        scores.append(float(result["probabilities"]["disease"]))
        predictions.append({"image_path": row["image_path"], "true_label": int(row["label"]), **result})
    metrics = _binary_metrics(y_true, y_pred, scores)
    metrics["mean_inference_ms"] = sum(float(item["inference_ms"]) for item in predictions) / max(len(predictions), 1)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    with (destination / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows([["", "pred_normal", "pred_disease"], ["true_normal", *metrics["confusion_matrix"][0]], ["true_disease", *metrics["confusion_matrix"][1]]])
    _save_confusion_matrix_png(metrics["confusion_matrix"], destination / "confusion_matrix.png")
    return metrics


def _save_confusion_matrix_png(matrix: list[list[int]], path: Path) -> None:
    """Save a dependency-light 2x2 heatmap for reports and slides."""
    width = height = 260
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    max_value = max(max(row) for row in matrix) if matrix else 1
    max_value = max(max_value, 1)
    cell = 80
    origin_x, origin_y = 70, 60
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(235 - 180 * (value / max_value))
            x0 = origin_x + col_index * cell
            y0 = origin_y + row_index * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=(intensity, 80, 80), outline="black", width=1)
            text = str(value)
            draw.text((x0 + cell // 2 - 4 * len(text), y0 + cell // 2 - 5), text, fill="white")
    draw.text((origin_x + 16, 28), "predicted", fill="black")
    draw.text((12, origin_y + cell - 4), "true", fill="black")
    image.save(path, format="PNG")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved eye-image demo model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="artifacts/evaluation")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_model(args.model, args.test, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
