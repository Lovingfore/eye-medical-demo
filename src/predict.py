"""Command-line single-image prediction."""

from __future__ import annotations

import argparse
import json

try:
    from .modeling import predict_image
except ImportError:  # pragma: no cover - supports ``python src/predict.py``
    from modeling import predict_image  # type: ignore[no-redef]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict one eye image")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(predict_image(args.image, args.model), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
