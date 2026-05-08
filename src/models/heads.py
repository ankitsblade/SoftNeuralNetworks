from __future__ import annotations

import math

import torch
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(features), dim=-1)


class MLPHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(features), dim=-1)


class TemperatureMLPHead(MLPHead):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.30,
        init_temperature: float = 1.5,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__(feature_dim, num_classes, hidden_dim, dropout)
        if init_temperature <= 0:
            raise ValueError("Temperature must be positive.")
        log_value = torch.tensor(math.log(init_temperature), dtype=torch.float32)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_value)
        else:
            self.register_buffer("log_temperature", log_value)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp_min(1e-4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(features) / self.temperature, dim=-1)
