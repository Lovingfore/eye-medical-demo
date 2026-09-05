"""Shared constants for the Topic 14 eye-image demo."""

from __future__ import annotations

from pathlib import Path

CLASS_NAMES = ("normal", "disease")
IMAGE_SIZE = (224, 224)
DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_CLASS = 20
MANIFEST_NAME = "labels.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "demo"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
