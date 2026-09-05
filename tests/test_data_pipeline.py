"""Tests for image preprocessing, data checks, and deterministic splits."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.check_data import check_dataset  # noqa: E402
from src.data import preprocess_image  # noqa: E402
from src.generate_demo_data import generate_demo_dataset  # noqa: E402
from src.make_splits import make_splits  # noqa: E402


def test_preprocess_image_converts_grayscale_to_rgb_and_resizes(tmp_path: Path) -> None:
    source = tmp_path / "gray.png"
    Image.new("L", (31, 19), color=127).save(source)

    array = preprocess_image(source, image_size=(16, 12))

    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    assert array.shape == (3, 12, 16)
    assert float(array.min()) >= 0.0
    assert float(array.max()) <= 1.0
    np.testing.assert_allclose(array[0], array[1])
    np.testing.assert_allclose(array[1], array[2])


def test_check_dataset_reports_missing_files_and_class_counts(tmp_path: Path) -> None:
    result = generate_demo_dataset(tmp_path, samples_per_class=2, seed=42)
    manifest = Path(result["manifest_path"])
    missing_relative = "images/missing.png"
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "class_name"])
        writer.writerow({"image_path": missing_relative, "label": "1", "class_name": "disease"})

    report = check_dataset(manifest)

    assert report["total_rows"] == 5
    assert report["valid_rows"] == 4
    assert report["missing_files"] == [missing_relative]
    assert report["corrupt_files"] == []
    assert report["class_counts"] == {"normal": 2, "disease": 2}


def test_make_splits_is_deterministic_and_uses_70_15_15_counts(tmp_path: Path) -> None:
    result = generate_demo_dataset(tmp_path / "dataset", samples_per_class=10, seed=42)
    split_dir = tmp_path / "splits"

    first = make_splits(result["manifest_path"], split_dir, seed=42)
    second = make_splits(result["manifest_path"], tmp_path / "splits_again", seed=42)

    assert first["counts"] == {"train": 14, "val": 3, "test": 3}
    assert first["counts"] == second["counts"]
    for split_name in ("train", "val", "test"):
        first_rows = Path(first["paths"][split_name]).read_bytes()
        second_rows = Path(second["paths"][split_name]).read_bytes()
        assert first_rows == second_rows

    split_rows = {}
    for split_name, split_path in first["paths"].items():
        with Path(split_path).open(newline="", encoding="utf-8") as handle:
            split_rows[split_name] = {row["image_path"] for row in csv.DictReader(handle)}
    assert not (split_rows["train"] & split_rows["val"])
    assert not (split_rows["train"] & split_rows["test"])
    assert not (split_rows["val"] & split_rows["test"])

    # Split manifests keep dataset-root-relative image paths and remain
    # checkable when consumed independently by downstream stages.
    train_report = check_dataset(first["paths"]["train"])
    assert train_report["valid_rows"] == 14

    # A binary demo split must retain both classes in every non-empty partition
    # so that Precision/Recall/F1/ROC-AUC are meaningful in the smoke run.
    for split_name, split_path in first["paths"].items():
        with Path(split_path).open(newline="", encoding="utf-8") as handle:
            labels = {int(row["label"]) for row in csv.DictReader(handle)}
        assert labels == {0, 1}
    assert train_report["missing_files"] == []
