"""Convert the public IDRiD parquet mirror into portable JPG manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq


LABEL_NAMES = {
    0: "normal",
    1: "disease",
    2: "disease",
    3: "disease",
    4: "disease",
}


def _export_split(parquet_path: Path, output_root: Path, split: str) -> dict[str, int]:
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
