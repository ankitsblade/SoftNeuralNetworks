from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy depth = 6n + 4.")
        blocks_per_group = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.feature_dim = channels[-1]
        self.in_channels = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.block1 = self._make_group(channels[1], blocks_per_group, stride=1, dropout=dropout)
        self.block2 = self._make_group(channels[2], blocks_per_group, stride=2, dropout=dropout)
        self.block3 = self._make_group(channels[3], blocks_per_group, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(channels[3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _make_group(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = [WideBasicBlock(self.in_channels, out_channels, stride, dropout)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(WideBasicBlock(self.in_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = self.pool(x)
        return torch.flatten(x, 1)


def wideresnet28_2(dropout: float = 0.0) -> WideResNet:
    return WideResNet(depth=28, widen_factor=2, dropout=dropout)
