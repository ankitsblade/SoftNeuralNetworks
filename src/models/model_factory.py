from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models.backbones import cifar_resnet18, densenet_bc_40_12, vgg13_bn, wideresnet28_2
from src.models.heads import LinearHead, MLPHead, TemperatureMLPHead


class DistributionModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        features = self.backbone(x)
        if return_logits:
            return self.head.logits(features)
        return self.head(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_logits=False)


def build_backbone(backbone_name: str, dropout: float = 0.0) -> nn.Module:
    name = backbone_name.lower()
    if name == "resnet18":
        return cifar_resnet18()
    if name == "wideresnet28_2":
        return wideresnet28_2(dropout=dropout)
    if name == "densenet_bc_40_12":
        return densenet_bc_40_12()
    if name == "vgg13_bn":
        return vgg13_bn()
    raise ValueError(f"Unknown backbone: {backbone_name}")


def build_head(
    head_name: str,
    feature_dim: int,
    num_classes: int = 10,
    dropout: float = 0.30,
    temperature_type: str = "learnable",
    temperature_init: float = 1.5,
) -> nn.Module:
    name = head_name.lower()
    if name == "linear":
        return LinearHead(feature_dim, num_classes)
    if name == "mlp":
        return MLPHead(feature_dim, num_classes, dropout=dropout)
    if name == "temperature_mlp":
        return TemperatureMLPHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            init_temperature=temperature_init,
            learnable_temperature=temperature_type == "learnable",
        )
    raise ValueError(f"Unknown head: {head_name}")


def build_model(
    backbone_name: str,
    head_name: str,
    num_classes: int = 10,
    pretrained_checkpoint: str | Path | None = None,
    dropout: float = 0.30,
    wrn_dropout: float = 0.0,
    temperature_type: str = "learnable",
    temperature_init: float = 1.5,
    map_location: str | torch.device = "cpu",
) -> DistributionModel:
    backbone = build_backbone(backbone_name, dropout=wrn_dropout)
    feature_dim = int(getattr(backbone, "feature_dim"))
    head = build_head(
        head_name,
        feature_dim,
        num_classes=num_classes,
        dropout=dropout,
        temperature_type=temperature_type,
        temperature_init=temperature_init,
    )
    model = DistributionModel(backbone, head)
    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location=map_location)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        non_head_unexpected = [key for key in unexpected if not key.startswith("head.")]
        if non_head_unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {non_head_unexpected}")
        print(f"Loaded checkpoint with missing keys: {missing}")
        if unexpected:
            print(f"Ignored incompatible head keys: {unexpected}")
    return model
