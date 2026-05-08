from __future__ import annotations

import torch

from src.losses.kl_loss import kl_divergence_loss


def jsd_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    predicted = predicted.clamp_min(eps)
    target = target.clamp_min(eps)
    midpoint = 0.5 * (predicted + target)
    return 0.5 * kl_divergence_loss(midpoint, target, eps) + 0.5 * kl_divergence_loss(midpoint, predicted, eps)
