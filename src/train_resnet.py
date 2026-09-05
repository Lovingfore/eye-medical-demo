"""CLI for IDRiD ResNet-18 transfer-learning training."""

from __future__ import annotations

import argparse
import json

from .torch_modeling import train_resnet_model


def main(argv: list[str] | None = None) -> int:
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
