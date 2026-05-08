from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Subset


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


def make_cifar10h_split_indices(
    total_size: int = 10_000,
    train_size: int = 6_000,
    val_size: int = 2_000,
    test_size: int = 2_000,
    seed: int = 42,
) -> SplitIndices:
    if train_size + val_size + test_size != total_size:
        raise ValueError("CIFAR-10H split sizes must sum to total_size.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_end = train_size
    val_end = train_end + val_size
    return SplitIndices(
        train=indices[:train_end],
        val=indices[train_end:val_end],
        test=indices[val_end:],
    )


def subset_from_indices(dataset, indices: list[int]) -> Subset:
    return Subset(dataset, indices)
