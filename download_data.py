from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets

from src.datasets.cifar10h_dataset import CIFAR10H_URLS, download_cifar10h_file, load_cifar10h_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 and CIFAR-10H data.")
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument(
        "--cifar10h-dir",
        default="data/cifar10h",
        help="Directory for CIFAR-10H files.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also download raw annotator-level CIFAR-10H zip and counts.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download files that already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    cifar10h_dir = Path(args.cifar10h_dir)
    root.mkdir(parents=True, exist_ok=True)
    cifar10h_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CIFAR-10 train/test through torchvision...")
    datasets.CIFAR10(root=str(root), train=True, download=True)
    datasets.CIFAR10(root=str(root), train=False, download=True)

    files = ["cifar10h-probs.npy"]
    if args.include_raw:
        files.extend(["cifar10h-counts.npy", "cifar10h-raw.zip"])

    print("Downloading CIFAR-10H files...")
    for filename in files:
        destination = cifar10h_dir / filename
        print(f"  {filename} <- {CIFAR10H_URLS[filename]}")
        download_cifar10h_file(destination, filename=filename, overwrite=args.overwrite)

    probs_path = cifar10h_dir / "cifar10h-probs.npy"
    targets = load_cifar10h_targets(probs_path)
    print(f"Verified CIFAR-10H targets: {tuple(targets.shape)} at {probs_path}")


if __name__ == "__main__":
    main()
