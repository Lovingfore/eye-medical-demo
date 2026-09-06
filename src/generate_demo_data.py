"""生成确定性的合成眼底风格 RGB 图像，用于演示数据/训练/Web 流程。

The generated images are intentionally artificial and are only suitable for
demonstrating the data/training/UI pipeline. They are not medical data and
must not be used to claim clinical performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from .config import (
    CLASS_NAMES,
    DEFAULT_DATA_DIR,
    DEFAULT_SAMPLES_PER_CLASS,
    DEFAULT_SEED,
    IMAGE_SIZE,
    MANIFEST_NAME,
)

# 【数据集】本文件生成内置合成 normal/disease 小样本，不是公开医学数据集，也不用于
# 训练结果或临床性能声明；真实模型训练使用 prepare_idrid.py 整理的 IDRiD 数据。
# 【技术栈】Pillow、ImageDraw/ImageFilter 和 Python 标准库，用确定性随机种子生成图片。


def _seed_for_sample(seed: int, label: int, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(seed: int):
    # Keep the generator dependency-free: use a tiny deterministic LCG.
    state = seed & ((1 << 64) - 1)

    def random() -> float:
        nonlocal state
        state = (6364136223846793005 * state + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return random


def _make_image(label: int, sample_seed: int) -> Image.Image:
    width, height = IMAGE_SIZE
    random = _rng(sample_seed)
    image = Image.new("RGB", IMAGE_SIZE, (8, 10, 14))
    pixels = image.load()
    cx, cy = width / 2.0, height / 2.0
    radius = min(width, height) * 0.46

    # Smooth circular retinal field with a mild radial illumination gradient.
    for y in range(height):
        for x in range(width):
            dx, dy = x - cx, y - cy
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > radius:
                pixels[x, y] = (4, 5, 8)
                continue
            light = max(0.0, 1.0 - distance / radius)
            noise = (random() - 0.5) * 7.0
            red = int(max(0, min(255, 42 + 80 * light + noise)))
            green = int(max(0, min(255, 9 + 26 * light + noise * 0.35)))
            blue = int(max(0, min(255, 8 + 16 * light)))
            pixels[x, y] = (red, green, blue)

    draw = ImageDraw.Draw(image, "RGBA")
    # Optic disc and vessel-like spokes provide a recognizable fundus-like cue.
    disc_x = int(width * 0.67 + (random() - 0.5) * 12)
    disc_y = int(height * 0.46 + (random() - 0.5) * 14)
    disc_r = int(width * 0.085)
    draw.ellipse((disc_x - disc_r, disc_y - disc_r, disc_x + disc_r, disc_y + disc_r), fill=(255, 185, 130, 185))
    for branch in range(8):
        angle = (branch / 8.0) * math.tau + (random() - 0.5) * 0.25
        end_x = cx + math.cos(angle) * radius * 0.82
        end_y = cy + math.sin(angle) * radius * 0.82
        draw.line((cx, cy, end_x, end_y), fill=(175, 38, 35, 125), width=2)

    if label == 1:
        # Disease class: deterministic bright lesions and small dark spots.
        for _ in range(5):
            lesion_x = int(width * (0.28 + random() * 0.44))
            lesion_y = int(height * (0.28 + random() * 0.44))
            lesion_r = int(2 + random() * 4)
            draw.ellipse(
                (lesion_x - lesion_r, lesion_y - lesion_r, lesion_x + lesion_r, lesion_y + lesion_r),
                fill=(255, 226, 125, 220),
            )
        for _ in range(3):
            spot_x = int(width * (0.30 + random() * 0.40))
            spot_y = int(height * (0.30 + random() * 0.40))
            spot_r = int(2 + random() * 3)
            draw.ellipse(
                (spot_x - spot_r, spot_y - spot_r, spot_x + spot_r, spot_y + spot_r),
                fill=(75, 6, 9, 170),
            )

    return image.filter(ImageFilter.GaussianBlur(radius=0.35)).convert("RGB")


def generate_demo_dataset(
    output_dir: str | Path = DEFAULT_DATA_DIR,
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create a deterministic balanced dataset and return its metadata."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be >= 1")

    root = Path(output_dir)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    rows: list[dict[str, str]] = []

    for label, class_name in enumerate(CLASS_NAMES):
        for index in range(samples_per_class):
            filename = f"{class_name}_{index:04d}.png"
            relative_path = Path("images") / filename
            _make_image(label, _seed_for_sample(seed, label, index)).save(root / relative_path, format="PNG")
            rows.append({"image_path": relative_path.as_posix(), "label": str(label), "class_name": class_name})

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "root_dir": str(root),
        "manifest_path": str(manifest_path),
        "num_samples": len(rows),
        "class_counts": {name: samples_per_class for name in CLASS_NAMES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic eye-image demo data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = generate_demo_dataset(args.output_dir, args.samples_per_class, args.seed)
    print(f"Generated {result['num_samples']} samples at {result['root_dir']}")
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
