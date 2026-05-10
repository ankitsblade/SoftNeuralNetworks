from __future__ import annotations

import numpy as np
import torch

from src.metrics.distribution_metrics import (
    per_sample_cosine,
    per_sample_jsd,
    per_sample_kl,
    per_sample_total_variation,
)
from src.metrics.entropy_metrics import entropy_bits_tensor
from src.metrics.precision_at_k import precision_at_many


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[sorter]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[sorter[start:end]] = average_rank
        start = end
    return ranks


def _ci(values: list[float], alpha: float) -> tuple[float, float]:
    lower = 100.0 * alpha / 2.0
    upper = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(values, lower)), float(np.percentile(values, upper))


def bootstrap_metric_cis(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap confidence intervals for report headline metrics."""

    predicted = predicted.detach().cpu()
    target = target.detach().cpu()
    n = predicted.shape[0]
    rng = np.random.default_rng(seed)

    per_sample = {
        "kl_mean": per_sample_kl(predicted, target).numpy(),
        "jsd_mean": per_sample_jsd(predicted, target).numpy(),
        "cosine_mean": per_sample_cosine(predicted, target).numpy(),
        "total_variation_mean": per_sample_total_variation(predicted, target).numpy(),
    }
    true_entropy = entropy_bits_tensor(target).numpy()
    pred_entropy = entropy_bits_tensor(predicted).numpy()
    entropy_error = pred_entropy - true_entropy

    samples: dict[str, list[float]] = {
        "entropy_pearson": [],
        "entropy_spearman": [],
        "entropy_mae": [],
        "entropy_rmse": [],
        "p_at_100": [],
        "p_at_200": [],
        "p_at_500": [],
    }
    samples.update({name: [] for name in per_sample})

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        for name, values in per_sample.items():
            samples[name].append(float(values[indices].mean()))
        boot_true = true_entropy[indices]
        boot_pred = pred_entropy[indices]
        boot_error = entropy_error[indices]
        samples["entropy_pearson"].append(_safe_corr(boot_true, boot_pred, "pearson"))
        samples["entropy_spearman"].append(_safe_corr(boot_true, boot_pred, "spearman"))
        samples["entropy_mae"].append(float(np.mean(np.abs(boot_error))))
        samples["entropy_rmse"].append(float(np.sqrt(np.mean(boot_error**2))))
        for key, value in precision_at_many(boot_true, boot_pred).items():
            samples[key].append(value)

    cis: dict[str, float] = {}
    for name, values in samples.items():
        low, high = _ci(values, alpha)
        cis[f"{name}_ci_low"] = low
        cis[f"{name}_ci_high"] = high
    return cis
