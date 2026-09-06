"""将公开 IDRiD parquet 镜像转换为项目统一的 JPG + CSV manifest。

# 【数据集】本项目使用 IDRiD B. Disease Grading（Indian Diabetic Retinopathy
# Image Dataset）。原始标签为 0~4 级，本项目的二分类规则是 0=normal、1~4=disease。
# 【技术栈】Python 标准库 csv/pathlib + PyArrow parquet 读取；本脚本不做医学质量评分，
# 只负责二进制图像导出、路径整理和标签映射。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq


# IDRiD 原始标签是 0~4 级。本项目为了演示二分类流程，将 0 级保留为
# normal，将 1~4 级统一合并为 disease；原始等级仍写入 CSV，便于追溯。
LABEL_NAMES = {
    0: "normal",
    1: "disease",
    2: "disease",
    3: "disease",
    4: "disease",
}


def _export_split(parquet_path: Path, output_root: Path, split: str) -> dict[str, int]:
    """把一个 parquet 划分导出成独立图像文件和 CSV manifest。

    parquet 中的 image 字段包含原始文件名和二进制内容。导出成普通 JPG
    后，后续检查、划分、训练和 Web 推理都只依赖 manifest 中的相对路径。
    """
    table = pq.read_table(parquet_path).to_pydict()
    image_dir = output_root / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    counts = {"normal": 0, "disease": 0}
    for item, original_label in zip(table["image"], table["label"]):
        original_label = int(original_label)
        class_name = LABEL_NAMES[original_label]
        filename = Path(item.get("path") or f"{split}_{len(rows):04d}.jpg").name
        destination = image_dir / filename
        # 这里只做格式/路径转换，不在此处删除低质量医学图像；质量筛选需要
        # 额外的图像质量指标或人工复核，目前项目尚未实现。
        destination.write_bytes(item["bytes"])
        rows.append(
            {
                "image_path": destination.relative_to(output_root).as_posix(),
                "label": str(int(class_name == "disease")),
                "class_name": class_name,
                "original_grade": str(original_label),
            }
        )
        counts[class_name] += 1
    manifest = output_root / f"labels_{split}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return counts


def prepare(dataset_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, object]:
    """准备训练集和官方测试集，并返回生成文件的位置与类别统计。"""
    dataset = Path(dataset_dir)
    destination = Path(output_dir) if output_dir else dataset / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": _export_split(dataset / "idrid_train.parquet", destination, "train"),
        "test": _export_split(dataset / "idrid_test.parquet", destination, "test"),
    }
    return {
        "output_dir": str(destination.resolve()),
        "manifests": {
            "train": str((destination / "labels_train.csv").resolve()),
            "test": str((destination / "labels_test.csv").resolve()),
        },
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare IDRiD parquet files")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    import json

    print(json.dumps(prepare(args.dataset_dir, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
