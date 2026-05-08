from __future__ import annotations

import numpy as np


def precision_at_k(true_scores: np.ndarray, predicted_scores: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(true_scores) != len(predicted_scores):
        raise ValueError("true_scores and predicted_scores must have the same length.")
    k = min(k, len(true_scores))
    true_top = set(np.argsort(-true_scores)[:k].tolist())
    predicted_top = set(np.argsort(-predicted_scores)[:k].tolist())
    return len(true_top & predicted_top) / k


def precision_at_many(true_scores: np.ndarray, predicted_scores: np.ndarray, ks=(100, 200, 500)) -> dict[str, float]:
    return {f"p_at_{k}": precision_at_k(true_scores, predicted_scores, k) for k in ks}
