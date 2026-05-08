from __future__ import annotations

import torch
from torchvision.transforms import functional as TF


def apply_corruption(images: torch.Tensor, corruption: str, severity: float) -> torch.Tensor:
    if corruption == "gaussian_noise":
        return (images + severity * torch.randn_like(images)).clamp(-3.0, 3.0)
    if corruption == "contrast":
        mean = images.mean(dim=(2, 3), keepdim=True)
        return mean + (1.0 - severity) * (images - mean)
    if corruption == "gaussian_blur":
        kernel = max(3, int(2 * round(severity) + 1))
        return torch.stack([TF.gaussian_blur(image, kernel_size=kernel) for image in images])
    raise ValueError(f"Unknown corruption: {corruption}")
