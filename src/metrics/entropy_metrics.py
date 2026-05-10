from __future__ import annotations

import numpy as np
import torch
from scipy import stats


def entropy_bits_tensor(distribution: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dist = distribution.clamp_min(eps)
    return -torch.sum(dist * torch.log2(dist), dim=1)


def entropy_bits_np(distribution: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dist = np.clip(distribution, eps, 1.0)
    return -np.sum(dist * np.log2(dist), axis=1)


def entropy_correlations(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_entropy = entropy_bits_tensor(predicted).detach().cpu().numpy()
    true_entropy = entropy_bits_tensor(target).detach().cpu().numpy()
    pearson = stats.pearsonr(true_entropy, pred_entropy).statistic
    spearman = stats.spearmanr(true_entropy, pred_entropy).statistic
    return {
        "entropy_pearson": float(np.nan_to_num(pearson)),
        "entropy_spearman": float(np.nan_to_num(spearman)),
    }


def entropy_error_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_entropy = entropy_bits_tensor(predicted)
    true_entropy = entropy_bits_tensor(target)
    error = pred_entropy - true_entropy
    return {
        "entropy_mae": float(torch.mean(torch.abs(error)).item()),
        "entropy_rmse": float(torch.sqrt(torch.mean(error.pow(2))).item()),
        "entropy_bias": float(torch.mean(error).item()),
    }
