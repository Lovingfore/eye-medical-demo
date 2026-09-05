"""Tests for the deterministic synthetic fundus data generator.

These tests intentionally describe the small public API used by the rest of
the demo before the implementation exists (TDD red phase).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, IMAGE_SIZE  # noqa: E402
from src.generate_demo_data import generate_demo_dataset  # noqa: E402


def test_generator_creates_expected_sample_counts_and_manifest(tmp_path: Path) -> None:
    result = generate_demo_dataset(tmp_path, samples_per_class=3, seed=42)

    image_paths = sorted((tmp_path / "images").glob("*.png"))
    assert len(image_paths) == 6
    assert result["num_samples"] == 6
    assert Path(result["manifest_path"]).exists()

    with Path(result["manifest_path"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert {row["class_name"] for row in rows} == set(CLASS_NAMES)
    assert {row["label"] for row in rows} == {"0", "1"}
    assert all((tmp_path / row["image_path"]).exists() for row in rows)


def test_generated_images_are_rgb_and_use_configured_shape(tmp_path: Path) -> None:
    generate_demo_dataset(tmp_path, samples_per_class=1, seed=42)

    for image_path in (tmp_path / "images").glob("*.png"):
        with Image.open(image_path) as image:
            assert image.mode == "RGB"
            assert image.size == IMAGE_SIZE


def test_generation_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    generate_demo_dataset(first_dir, samples_per_class=2, seed=123)
    generate_demo_dataset(second_dir, samples_per_class=2, seed=123)

    first_images = sorted((first_dir / "images").glob("*.png"))
    second_images = sorted((second_dir / "images").glob("*.png"))
    assert [path.name for path in first_images] == [path.name for path in second_images]
    assert [path.read_bytes() for path in first_images] == [path.read_bytes() for path in second_images]

