from __future__ import annotations

import torch


def kl_divergence_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    predicted = predicted.clamp_min(eps)
    target = target.clamp_min(eps)
    return torch.sum(target * (torch.log(target) - torch.log(predicted)), dim=1).mean()
