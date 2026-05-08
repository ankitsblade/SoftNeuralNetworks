from __future__ import annotations

from pathlib import Path

from src.datasets.cifar10h_dataset import build_cifar10_pretrain_loaders
from src.models.model_factory import build_model
from src.training.trainer import get_device, train_pretrain
from src.utils.config import deep_get


def run_pretraining(config: dict) -> Path:
    device = get_device(config.get("device", "auto"))
    dataset = config.get("dataset", {})
    model_cfg = config.get("model", {})
    loaders = build_cifar10_pretrain_loaders(
        root=dataset.get("root", "data"),
        batch_size=int(dataset.get("batch_size", 128)),
        num_workers=int(dataset.get("num_workers", 4)),
        seed=int(config.get("seed", 42)),
        val_size=int(dataset.get("pretrain_val_size", 5000)),
        download=bool(dataset.get("download", True)),
        device=device,
    )
    model = build_model(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        head_name=model_cfg.get("head", "temperature_mlp"),
        num_classes=int(model_cfg.get("num_classes", 10)),
        dropout=float(model_cfg.get("dropout", 0.30)),
        wrn_dropout=float(model_cfg.get("wrn_dropout", 0.0)),
        temperature_type=deep_get(model_cfg, "temperature.type", "learnable"),
        temperature_init=float(deep_get(model_cfg, "temperature.init", 1.5)),
    )
    return train_pretrain(model, loaders[0], loaders[1], config, device, config["logging"]["save_dir"])
