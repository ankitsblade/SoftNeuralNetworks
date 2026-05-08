from __future__ import annotations

import argparse

from src.evaluation.evaluate import evaluate_and_save
from src.utils.config import load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10H checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num-workers", type=int, help="Override dataset.num_workers for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.num_workers is not None:
        config.setdefault("dataset", {})["num_workers"] = args.num_workers
    seed_everything(int(config.get("seed", 42)))
    evaluate_and_save(config, args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
