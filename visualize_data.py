from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.data_viz import generate_data_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CIFAR-10H data sanity figures.")
    parser.add_argument("--root", default="data", help="CIFAR-10 root directory.")
    parser.add_argument(
        "--cifar10h-path",
        default="data/cifar10h/cifar10h-probs.npy",
        help="Path to CIFAR-10H 10000x10 soft target file.",
    )
    parser.add_argument("--output-dir", default="outputs/figures/data", help="Where to save figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_data_visualizations(
        root=args.root,
        cifar10h_path=args.cifar10h_path,
        output_dir=Path(args.output_dir),
    )
    print(f"Saved data figures to {args.output_dir}")


if __name__ == "__main__":
    main()
