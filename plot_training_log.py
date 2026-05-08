from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.training_curves import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a training-curve figure from an existing CSV log.")
    parser.add_argument("--log", required=True, help="Path to a CSV log in outputs/logs.")
    parser.add_argument("--output", help="Output PNG path. Defaults to outputs/figures/<log-stem>_curves.png.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    output_path = Path(args.output) if args.output else Path("outputs/figures") / f"{log_path.stem}_curves.png"
    plot_training_curves(log_path, output_path)
    print(f"Saved training curve to {output_path}")


if __name__ == "__main__":
    main()
