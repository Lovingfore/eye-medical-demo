"""Shared image loading and preprocessing utilities.

The command-line trainer and the Django predictor both use this module so that
the online path cannot silently diverge from the training preprocessing.  A
preprocessed image is returned as a CHW ``float32`` NumPy array in ``[0, 1]``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

try:  # Keep module importable enough to provide a useful dependency message.
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in minimal installs
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in minimal installs
    Image = None  # type: ignore[assignment,misc]

try:
    from .config import IMAGE_SIZE
except ImportError:  # Support ``python src/data.py`` style imports.
    from config import IMAGE_SIZE  # type: ignore[no-redef]


def _require_image_dependencies() -> None:
    """在真正处理图像前给出清晰的依赖缺失提示。"""
    if np is None or Image is None:
        raise ImportError(
            "Image preprocessing requires NumPy and Pillow. "
            "Install the demo requirements with `pip install -r requirements.txt`."
        )


def resolve_image_path(image_path: str | Path, manifest_path: str | Path | None = None) -> Path:
    """Resolve a manifest path relative to the manifest directory.

    Absolute paths are preserved.  Relative paths are interpreted relative to
    the directory containing the manifest when one is supplied; otherwise they
    are interpreted relative to the current working directory.
    """

    path = Path(image_path)
    if path.is_absolute() or manifest_path is None:
        return path
    manifest = Path(manifest_path).resolve()
    # Generated split manifests live in ``<dataset>/splits`` while their
    # ``image_path`` values remain relative to ``<dataset>``. Walk ancestors
    # so both the original labels.csv and train/val/test.csv resolve identically.
    # Check the manifest directory and its immediate parent.  The split
    # writer stores paths relative to the split directory, so this handles
    # normal portable layouts without recursively scanning large directories.
    for parent in (manifest.parent, *list(manifest.parents)[1:2]):
        candidate = parent / path
        if candidate.exists():
            return candidate
    # Preserve a useful deterministic path for missing-file diagnostics.
    return manifest.parent / path


def load_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    """Load and lightly validate a CSV manifest.

    The expected columns are ``image_path``, ``label`` and ``class_name``.
    Extra columns are retained, which keeps this helper compatible with future
    metadata columns.
    """

    path = Path(manifest_path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        # 只强制检查训练/评价所需字段；像 original_grade 等额外元数据会保留。
        required = {"image_path", "label", "class_name"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def preprocess_image(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> "np.ndarray":
    """Read, RGB-convert, resize, normalize, and transpose one image.

    ``image_size`` follows Pillow's ``(width, height)`` convention.  The
    returned array has shape ``(3, height, width)`` and values in ``[0, 1]``.
    """

    _require_image_dependencies()
    with Image.open(image_path) as image:  # type: ignore[union-attr]
        # RGB 转换统一通道数；缩放和 [0,1] 归一化让不同尺寸图像进入同一输入接口。
        image = image.convert("RGB")
        image = image.resize(tuple(image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    # ``convert('RGB')`` guarantees three channels, but this assertion gives a
    # clearer error if a non-standard Pillow backend ever violates that.
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image array, got shape {array.shape}")
    # PyTorch 常用 CHW 布局，而 Pillow/NumPy 读取的是 HWC。
    return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def preprocess_pil_image(image: Any, image_size: tuple[int, int] = IMAGE_SIZE) -> "np.ndarray":
    """Preprocess an already-open Pillow image (used by web upload handlers)."""

    _require_image_dependencies()
    image = image.convert("RGB").resize(tuple(image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def iter_manifest_images(
    manifest_path: str | Path,
    rows: Iterable[dict[str, str]] | None = None,
):
    """Yield ``(absolute_path, label, row)`` tuples for a manifest."""

    manifest = Path(manifest_path)
    for row in rows if rows is not None else load_manifest(manifest):
        yield resolve_image_path(row["image_path"], manifest), int(row["label"]), row


# Friendly aliases used by downstream code and notebooks.
read_manifest = load_manifest
preprocess = preprocess_image
