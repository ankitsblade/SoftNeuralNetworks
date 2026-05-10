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


def per_sample_soft_cross_entropy(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    predicted = predicted.clamp_min(eps)
    return -torch.sum(target * torch.log(predicted), dim=1)


def per_sample_brier(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum((predicted - target).pow(2), dim=1)


def per_sample_total_variation(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.abs(predicted - target), dim=1)


def majority_top1_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    pred_labels = predicted.argmax(dim=1)
    target_labels = target.argmax(dim=1)
    return float((pred_labels == target_labels).float().mean().item())


def soft_calibration_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 15,
) -> dict[str, float]:
    """ECE/MCE where correctness is the human mass assigned to the predicted class."""

    confidences, pred_labels = predicted.max(dim=1)
    soft_correctness = target.gather(1, pred_labels[:, None]).squeeze(1)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=predicted.device)
    ece = torch.tensor(0.0, device=predicted.device)
    mce = torch.tensor(0.0, device=predicted.device)
    n = predicted.shape[0]
    for idx in range(n_bins):
        lower = boundaries[idx]
        upper = boundaries[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        gap = torch.abs(confidences[mask].mean() - soft_correctness[mask].mean())
        ece = ece + (mask.float().mean() * gap)
        mce = torch.maximum(mce, gap)
    return {
        "soft_ece": float(ece.item()),
        "soft_mce": float(mce.item()),
    }


def summarize_distribution_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    kl = per_sample_kl(predicted, target)
    jsd = per_sample_jsd(predicted, target)
    cosine = per_sample_cosine(predicted, target)
    soft_cross_entropy = per_sample_soft_cross_entropy(predicted, target)
    brier = per_sample_brier(predicted, target)
    total_variation = per_sample_total_variation(predicted, target)
    calibration = soft_calibration_errors(predicted, target)
    return {
        "kl_mean": float(kl.mean().item()),
        "kl_std": float(kl.std(unbiased=False).item()),
        "jsd_mean": float(jsd.mean().item()),
        "jsd_std": float(jsd.std(unbiased=False).item()),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_std": float(cosine.std(unbiased=False).item()),
        "soft_cross_entropy_mean": float(soft_cross_entropy.mean().item()),
        "soft_cross_entropy_std": float(soft_cross_entropy.std(unbiased=False).item()),
        "brier_mean": float(brier.mean().item()),
        "brier_std": float(brier.std(unbiased=False).item()),
        "total_variation_mean": float(total_variation.mean().item()),
        "total_variation_std": float(total_variation.std(unbiased=False).item()),
        "majority_top1_acc": majority_top1_accuracy(predicted, target),
        **calibration,
    }


def confusion_style_matrix(target: np.ndarray, majority_labels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.float64)
    for label, distribution in zip(majority_labels, target, strict=True):
        matrix[int(label)] += distribution
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums
