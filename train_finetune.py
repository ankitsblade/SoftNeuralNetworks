from __future__ import annotations

import argparse

from src.training.finetune_cifar10h import run_finetuning
from src.utils.config import load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune on CIFAR-10H soft human label distributions.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    best_path = run_finetuning(config)
    print(f"Best fine-tuning checkpoint: {best_path}")


if __name__ == "__main__":
    main()
