from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        inter_channels = 4 * growth_rate
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.net(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNetBC(nn.Module):
    def __init__(
        self,
        depth: int = 40,
        growth_rate: int = 12,
        compression: float = 0.5,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("DenseNet-BC depth must satisfy depth = 6n + 4.")
        layers_per_block = (depth - 4) // 6
        channels = 2 * growth_rate
        self.conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

        self.block1 = DenseBlock(layers_per_block, channels, growth_rate)
        channels = self.block1.out_channels
        compressed = int(channels * compression)
        self.trans1 = Transition(channels, compressed)
        channels = compressed

        self.block2 = DenseBlock(layers_per_block, channels, growth_rate)
        channels = self.block2.out_channels
        compressed = int(channels * compression)
        self.trans2 = Transition(channels, compressed)
        channels = compressed

        self.block3 = DenseBlock(layers_per_block, channels, growth_rate)
        channels = self.block3.out_channels
        self.bn = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = channels
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = self.pool(x)
        return torch.flatten(x, 1)


def densenet_bc_40_12() -> DenseNetBC:
    return DenseNetBC(depth=40, growth_rate=12, compression=0.5)
