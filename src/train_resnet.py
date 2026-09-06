"""IDRiD ResNet-18 迁移学习训练命令行入口。

# 【技术栈】Python argparse/JSON + src.torch_modeling（PyTorch、torchvision、
# Pillow、NumPy、scikit-learn）。本文件只负责读取命令行参数，具体数据加载、
# 预处理、损失函数、优化器和 checkpoint 保存均在 torch_modeling.py 实现。
# 【数据集】默认输入 IDRiD 二分类 manifest；新增数据需先转换成相同 CSV 格式。
"""

from __future__ import annotations

import argparse
import json

from .torch_modeling import train_resnet_model


def main(argv: list[str] | None = None) -> int:
    """训练命令行入口；参数最终传给 src.torch_modeling.train_resnet_model。"""
    parser = argparse.ArgumentParser(description="Train ResNet-18 on IDRiD binary labels")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--artifact", default="artifacts/idrid_resnet_model.json")
    parser.add_argument("--checkpoint", default="artifacts/idrid_resnet_best.pt")
    parser.add_argument("--history", default="artifacts/idrid_resnet_training_history.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)
    result = train_resnet_model(
        args.train,
        args.val,
        args.artifact,
        args.checkpoint,
        history_path=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
