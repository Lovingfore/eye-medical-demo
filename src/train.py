"""内置合成数据演示模型的训练入口。

# 【技术栈】Python argparse/JSON + src.modeling 的 Pillow RGB 特征和质心分类器。
# 【数据集】默认配合 src.generate_demo_data 生成的合成 normal/disease 小样本，
# 仅用于界面与流程演示；正式 IDRiD 训练请使用 train_resnet.py。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .modeling import train_model
except ImportError:  # pragma: no cover - supports ``python src/train.py``
    from modeling import train_model  # type: ignore[no-redef]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the eye-image demo classifier")
    parser.add_argument("--train", required=True, help="train.csv path")
    parser.add_argument("--val", help="validation.csv path (not used for fitting)")
    parser.add_argument("--test", help="test.csv path (not used for fitting)")
    parser.add_argument("--artifact", default="artifacts/model.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    splits = {key: value for key, value in {"train": args.train, "val": args.val, "test": args.test}.items() if value}
    result = train_model(splits, args.artifact, seed=args.seed, epochs=args.epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
