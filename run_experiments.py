from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple CIFAR-10H fine-tuning experiments with uv.")
    parser.add_argument("configs", nargs="+", help="Config files to run.")
    parser.add_argument("--python", default="python", help="Python executable used by uv run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config in args.configs:
        config_path = Path(config)
        command = ["uv", "run", args.python, "train_finetune.py", "--config", str(config_path)]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
