"""Dataset integrity checks for the demo manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment,misc]

try:
    from .data import load_manifest, resolve_image_path
except ImportError:  # pragma: no cover
    from data import load_manifest, resolve_image_path  # type: ignore[no-redef]


def _check_image(path: Path) -> str | None:
    """检查文件是否能被 Pillow 验证并完整解码。

    ``verify`` 主要检查文件结构；重新打开并 ``load`` 是为了实际读取像素，
    避免只通过容器校验但在训练时才发现像素损坏。
    """
    if Image is None:
        raise ImportError(
            "Dataset checks require Pillow. Install the demo requirements with "
            "`pip install -r requirements.txt`."
        )
    try:
        with Image.open(path) as image:
            image.verify()
        # verify() may not decode all pixel data, so reopen and load it.
        with Image.open(path) as image:
            image.load()
    except Exception as exc:  # Pillow uses several exception classes by format.
        return f"{type(exc).__name__}: {exc}"
    return None


def check_dataset(manifest_path: str | Path) -> dict[str, Any]:
    """Check manifest paths and image readability.

    ``valid_rows`` excludes missing and corrupt files.  Class counts likewise
    count only valid rows, making the report directly useful before training.
    Paths in the report remain exactly as written in the manifest.
    """

    manifest = Path(manifest_path)
    rows = load_manifest(manifest)
    missing_files: list[str] = []
    corrupt_files: list[str] = []
    class_counts: Counter[str] = Counter()

    for row in rows:
        relative = row["image_path"]
        path = resolve_image_path(relative, manifest)
        if not path.exists():
            # 缺失文件不会进入 valid_rows，也不会参与类别计数。
            missing_files.append(relative)
            continue
        error = _check_image(path)
        if error is not None:
            # 这里的“清洗”仅指文件级完整性过滤，不代表医学质量筛选。
            corrupt_files.append(relative)
            continue
        class_counts[row["class_name"]] += 1

    return {
        "manifest_path": str(manifest.resolve()),
        "total_rows": len(rows),
        "valid_rows": len(rows) - len(missing_files) - len(corrupt_files),
        "missing_files": missing_files,
        "corrupt_files": corrupt_files,
        "class_counts": dict(class_counts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check an eye-image CSV manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(check_dataset(args.manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
