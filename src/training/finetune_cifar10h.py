from __future__ import annotations

from pathlib import Path

from src.datasets.cifar10h_dataset import build_cifar10h_loaders
from src.models.model_factory import build_model
from src.training.trainer import get_device, train_finetune
from src.utils.config import deep_get
from src.visualization.data_viz import generate_data_visualizations


def run_finetuning(config: dict) -> Path:
    device = get_device(config.get("device", "auto"))
    dataset = config.get("dataset", {})
    model_cfg = config.get("model", {})
    logging_cfg = config.get("logging", {})
    if bool(logging_cfg.get("save_data_figures", True)):
        generate_data_visualizations(
            root=dataset.get("root", "data"),
            cifar10h_path=dataset.get("cifar10h_path", "data/cifar10h/cifar10h-probs.npy"),
            output_dir=Path(logging_cfg.get("save_dir", "outputs")) / "figures" / "data",
        )
    loaders = build_cifar10h_loaders(
        root=dataset.get("root", "data"),
        cifar10h_path=dataset.get("cifar10h_path", "data/cifar10h/cifar10h-probs.npy"),
        batch_size=int(dataset.get("batch_size", 128)),
        num_workers=int(dataset.get("num_workers", 4)),
        seed=int(config.get("seed", 42)),
        train_size=int(dataset.get("train_size", 6000)),
        val_size=int(dataset.get("val_size", 2000)),
        test_size=int(dataset.get("test_size", 2000)),
        download=bool(dataset.get("download", True)),
        device=device,
    )
    model = build_model(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        head_name=model_cfg.get("head", "temperature_mlp"),
        num_classes=int(model_cfg.get("num_classes", 10)),
        pretrained_checkpoint=model_cfg.get("pretrained_checkpoint"),
        dropout=float(model_cfg.get("dropout", 0.30)),
        wrn_dropout=float(model_cfg.get("wrn_dropout", 0.0)),
        temperature_type=deep_get(model_cfg, "temperature.type", "learnable"),
        temperature_init=float(deep_get(model_cfg, "temperature.init", 1.5)),
        map_location=device,
    )
    return train_finetune(model, loaders[0], loaders[1], config, device, config["logging"]["save_dir"])
