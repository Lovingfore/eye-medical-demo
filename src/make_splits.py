"""生成确定性的 train/validation/test CSV 划分。

# 【技术栈】Python csv、random 和 pathlib；使用固定 seed=42 保证同一 manifest
# 可以复现实验划分，并按二分类标签做分层抽样。
# 【数据集】默认接收 prepare_idrid.py 生成的 IDRiD 训练 manifest，也可接收格式
# 相同的新数据。当前实现是图像级分层，正式研究应改为患者级划分以避免数据泄漏。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

try:
    from .data import load_manifest
except ImportError:  # pragma: no cover
    from data import load_manifest  # type: ignore[no-redef]


def _split_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """按最大余数法把总样本数转换成整数划分数量。

    直接相乘会产生小数；先取整数部分，再把剩余样本分配给小数部分最大
    的划分，从而使总数保持不变。
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    # Largest remainder allocation gives sensible non-zero validation/test
    # partitions for small demo sets while retaining the requested proportions.
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _class_split_counts(
    class_total: int,
    split_counts: tuple[int, int, int],
    total: int,
    tie_order: tuple[int, int, int] = (1, 2, 0),
) -> tuple[int, int, int]:
    """按全局划分比例分配一个类别，尽量保持 train/val/test 的类别比例。"""
    if total <= 0:
        return (0, 0, 0)
    raw = [class_total * count / total for count in split_counts]
    counts = [int(value) for value in raw]
    remainder = class_total - sum(counts)
    rank = {index: position for position, index in enumerate(tie_order)}
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -rank[i]), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def make_splits(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    # 【数据划分】按 label 分层后生成互不重叠的 train/val/test 清单；训练只读取
    # train，验证用于选最佳模型，独立 test 应留到最终评价阶段。
    """用固定随机种子生成互不重叠的 train/val/test CSV。

    这里是按图像标签分层，而不是按患者编号分组。若数据包含患者元数据，
    正式研究应改成患者级划分，以避免同一患者图像跨集合造成数据泄漏。
    """

    if any(ratio < 0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Split ratios must be non-negative")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0")

    manifest = Path(manifest_path)
    rows = load_manifest(manifest)
    train_count, val_count, test_count = _split_counts(
        len(rows), train_ratio, val_ratio, test_ratio
    )
    split_counts = (train_count, val_count, test_count)
    # 按二分类标签分组后分别打乱，保证每个划分尽量包含两个类别。
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    rng = random.Random(seed)
    partitions = {"train": [], "val": [], "test": []}
    split_names = ("train", "val", "test")
    for class_index, label in enumerate(sorted(grouped)):
        class_rows = grouped[label]
        rng.shuffle(class_rows)
        tie_order = (1, 2, 0) if class_index % 2 == 0 else (2, 1, 0)
        class_counts = _class_split_counts(len(class_rows), split_counts, len(rows), tie_order)
        cursor = 0
        for split_name, count in zip(split_names, class_counts):
            partitions[split_name].extend(class_rows[cursor : cursor + count])
            cursor += count
    for split_rows in partitions.values():
        rng.shuffle(split_rows)

    destination = Path(output_dir) if output_dir is not None else manifest.parent / "splits"
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    fieldnames = ["image_path", "label", "class_name"]
    # Preserve extra manifest columns in split files when present.
    if rows:
        fieldnames = list(rows[0].keys())
    for split_name, split_rows in partitions.items():
        split_path = destination / f"{split_name}.csv"
        portable_rows: list[dict[str, str]] = []
        for row in split_rows:
            portable = dict(row)
            # 原始 manifest 的路径相对数据集根目录；这里改写为相对 split CSV
            # 的路径，使 train.csv/val.csv/test.csv 被单独移动后仍可使用。
            source = manifest.parent / row["image_path"]
            portable["image_path"] = os.path.relpath(source, destination).replace(os.sep, "/")
            portable_rows.append(portable)
        with split_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(portable_rows)
        paths[split_name] = str(split_path.resolve())

    return {
        "manifest_path": str(manifest.resolve()),
        "output_dir": str(destination.resolve()),
        "seed": seed,
        "counts": {name: len(partitions[name]) for name in ("train", "val", "test")},
        "paths": paths,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create deterministic dataset splits")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = make_splits(args.manifest, args.output_dir, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
