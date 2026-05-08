from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.datasets.splits import make_cifar10h_split_indices, subset_from_indices

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10H_URLS = {
    "cifar10h-probs.npy": "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-probs.npy",
    "cifar10h-counts.npy": "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy",
    "cifar10h-raw.zip": "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-raw.zip",
}


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def image_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def download_cifar10h_file(
    destination: str | Path = "data/cifar10h/cifar10h-probs.npy",
    filename: str = "cifar10h-probs.npy",
    overwrite: bool = False,
) -> Path:
    if filename not in CIFAR10H_URLS:
        raise ValueError(f"Unknown CIFAR-10H file: {filename}. Valid files: {sorted(CIFAR10H_URLS)}")
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        return destination_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(CIFAR10H_URLS[filename], destination_path)
    except URLError as exc:
        raise RuntimeError(f"Failed to download CIFAR-10H from {CIFAR10H_URLS[filename]}") from exc
    return destination_path


def _extract_array_from_npz(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("probs", "soft_labels", "targets", "labels", "cifar10h"):
        if key in npz and np.asarray(npz[key]).ndim == 2:
            return np.asarray(npz[key])
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.ndim == 2 and arr.shape[-1] == 10:
            return arr
    raise ValueError("No 2D array with 10 class columns found in NPZ file.")


def _load_raw_targets(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=True))
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            return _extract_array_from_npz(npz)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep)
        class_cols = [c for c in CIFAR10_CLASSES if c in frame.columns]
        if len(class_cols) == 10:
            return frame[class_cols].to_numpy()
        numeric = frame.select_dtypes(include="number")
        if numeric.shape[1] < 10:
            raise ValueError("CSV/TSV file must contain at least 10 numeric target columns.")
        return numeric.iloc[:, -10:].to_numpy()
    if suffix in {".txt", ".dat"}:
        return np.loadtxt(path)
    raise ValueError(f"Unsupported CIFAR-10H target file extension: {path.suffix}")


def load_cifar10h_targets(path: str | Path, download: bool = False) -> torch.Tensor:
    target_path = Path(path)
    if not target_path.exists():
        if download:
            download_cifar10h_file(target_path, filename=target_path.name)
        else:
            raise FileNotFoundError(
                f"CIFAR-10H target file not found: {target_path}. "
                "Run `uv run python download_data.py`, or pass dataset.download: true in YAML."
            )
    if not target_path.exists():
        raise FileNotFoundError(
            f"CIFAR-10H target file not found: {target_path}. "
            "Place a 10000x10 probability/count file at this path, for example "
            "data/cifar10h/cifar10h-probs.npy, or pass dataset.cifar10h_path in YAML."
        )

    raw = np.asarray(_load_raw_targets(target_path), dtype=np.float32)
    if raw.shape != (10_000, 10):
        raise ValueError(f"CIFAR-10H targets must have shape (10000, 10), got {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError("CIFAR-10H targets contain NaN or infinite values.")
    if (raw < 0).any():
        raise ValueError("CIFAR-10H targets must be non-negative.")

    row_sums = raw.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Every CIFAR-10H target row must have positive mass.")
    targets = raw / row_sums
    tensor = torch.from_numpy(targets)
    sums = tensor.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise ValueError("CIFAR-10H target rows do not sum to one after normalization.")
    return tensor


class CIFAR10HDataset(Dataset):
    """CIFAR-10 test images aligned with CIFAR-10H soft distributions."""

    def __init__(
        self,
        root: str | Path,
        cifar10h_path: str | Path,
        transform: Any | None = None,
        download: bool = True,
    ) -> None:
        self.cifar10 = datasets.CIFAR10(
            root=str(root),
            train=False,
            transform=transform,
            download=download,
        )
        self.soft_targets = load_cifar10h_targets(cifar10h_path, download=download)
        if len(self.cifar10) != len(self.soft_targets):
            raise ValueError(
                f"CIFAR-10 test length ({len(self.cifar10)}) does not match "
                f"CIFAR-10H targets ({len(self.soft_targets)})."
            )

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, index: int):
        image, hard_label = self.cifar10[index]
        soft_target = self.soft_targets[index]
        if not torch.allclose(soft_target.sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError(f"Soft target at index {index} does not sum to one.")
        return image, soft_target, hard_label, index


def build_cifar10_pretrain_loaders(
    root: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_size: int = 5_000,
    download: bool = True,
    device: str | torch.device = "cpu",
):
    dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        transform=train_transform(),
        download=download,
    )
    val_dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        transform=test_transform(),
        download=download,
    )
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_subset = subset_from_indices(dataset, train_indices)
    val_subset = subset_from_indices(val_dataset, val_indices)
    pin_memory = torch.device(device).type == "cuda"
    return (
        DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


def build_cifar10h_loaders(
    root: str | Path,
    cifar10h_path: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_size: int = 6_000,
    val_size: int = 2_000,
    test_size: int = 2_000,
    download: bool = True,
    device: str | torch.device = "cpu",
):
    split = make_cifar10h_split_indices(
        total_size=10_000,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    root = Path(root)
    train_dataset = CIFAR10HDataset(root, cifar10h_path, train_transform(), download)
    eval_dataset = CIFAR10HDataset(root, cifar10h_path, test_transform(), download)
    pin_memory = torch.device(device).type == "cuda"
    return (
        DataLoader(
            subset_from_indices(train_dataset, split.train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            subset_from_indices(eval_dataset, split.val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            subset_from_indices(eval_dataset, split.test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )
