from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def per_sample_kl(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    predicted = predicted.clamp_min(eps)
    target = target.clamp_min(eps)
    return torch.sum(target * (torch.log(target) - torch.log(predicted)), dim=1)


def per_sample_jsd(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    midpoint = 0.5 * (predicted + target)
    return 0.5 * per_sample_kl(midpoint, target, eps) + 0.5 * per_sample_kl(midpoint, predicted, eps)


def per_sample_cosine(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(predicted, target, dim=1)


def summarize_distribution_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    kl = per_sample_kl(predicted, target)
    jsd = per_sample_jsd(predicted, target)
    cosine = per_sample_cosine(predicted, target)
    return {
        "kl_mean": float(kl.mean().item()),
        "kl_std": float(kl.std(unbiased=False).item()),
        "jsd_mean": float(jsd.mean().item()),
        "jsd_std": float(jsd.std(unbiased=False).item()),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_std": float(cosine.std(unbiased=False).item()),
    }


def confusion_style_matrix(target: np.ndarray, majority_labels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.float64)
    for label, distribution in zip(majority_labels, target, strict=True):
        matrix[int(label)] += distribution
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums
