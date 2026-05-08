from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GradCAM:
    model: nn.Module
    target_layer: nn.Module

    def __post_init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image, return_logits=True)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[:, class_index].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)
