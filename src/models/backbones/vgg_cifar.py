from __future__ import annotations

import torch
from torch import nn


class VGG13BN(nn.Module):
    feature_dim = 512

    def __init__(self) -> None:
        super().__init__()
        config: list[int | str] = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ]
        layers: list[nn.Module] = []
        in_channels = 3
        for item in config:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(item)
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def vgg13_bn() -> VGG13BN:
    return VGG13BN()
