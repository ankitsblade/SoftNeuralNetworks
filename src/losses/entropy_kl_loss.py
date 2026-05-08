from __future__ import annotations

import torch

from src.losses.kl_loss import kl_divergence_loss


def entropy_bits(distribution: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dist = distribution.clamp_min(eps)
    return -torch.sum(dist * torch.log2(dist), dim=1)


def entropy_kl_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lambda_entropy: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    kl = kl_divergence_loss(predicted, target, eps=eps)
    entropy_penalty = (entropy_bits(predicted, eps) - entropy_bits(target, eps)).pow(2).mean()
    return kl + lambda_entropy * entropy_penalty
