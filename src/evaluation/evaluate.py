from __future__ import annotations

import time
from pathlib import Path

import torch

from src.datasets.cifar10h_dataset import build_cifar10h_loaders
from src.metrics.distribution_metrics import summarize_distribution_metrics
from src.metrics.entropy_metrics import entropy_bits_tensor, entropy_correlations
from src.metrics.precision_at_k import precision_at_many
from src.models.model_factory import build_model
from src.training.trainer import count_parameters, get_device
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config import deep_get
from src.utils.logging import write_json
from src.visualization.entropy_scatter import plot_entropy_scatter


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predicted: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, soft_targets, _, _ in loader:
        images = images.to(device, non_blocking=True)
        predicted.append(model(images).detach().cpu())
        targets.append(soft_targets.float().cpu())
    return torch.cat(predicted), torch.cat(targets)


def evaluate_checkpoint(config: dict, checkpoint_path: str | Path, split: str = "test") -> dict[str, float]:
    device = get_device(config.get("device", "auto"))
    dataset = config.get("dataset", {})
    model_cfg = config.get("model", {})
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
    split_to_loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}
    model = build_model(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        head_name=model_cfg.get("head", "temperature_mlp"),
        num_classes=int(model_cfg.get("num_classes", 10)),
        dropout=float(model_cfg.get("dropout", 0.30)),
        wrn_dropout=float(model_cfg.get("wrn_dropout", 0.0)),
        temperature_type=deep_get(model_cfg, "temperature.type", "learnable"),
        temperature_init=float(deep_get(model_cfg, "temperature.init", 1.5)),
    )
    load_model_checkpoint(checkpoint_path, model, map_location=device, strict=False)
    model.to(device)
    started = time.time()
    predicted, target = collect_predictions(model, split_to_loader[split], device)
    metrics = summarize_distribution_metrics(predicted, target)
    metrics.update(entropy_correlations(predicted, target))
    true_entropy = entropy_bits_tensor(target).numpy()
    pred_entropy = entropy_bits_tensor(predicted).numpy()
    metrics.update(precision_at_many(true_entropy, pred_entropy))
    metrics["params"] = count_parameters(model)
    metrics["evaluation_time_sec"] = time.time() - started
    return metrics


def evaluate_and_save(config: dict, checkpoint_path: str | Path, split: str = "test") -> dict[str, float]:
    device = get_device(config.get("device", "auto"))
    dataset = config.get("dataset", {})
    model_cfg = config.get("model", {})
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
    split_to_loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}
    model = build_model(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        head_name=model_cfg.get("head", "temperature_mlp"),
        num_classes=int(model_cfg.get("num_classes", 10)),
        dropout=float(model_cfg.get("dropout", 0.30)),
        wrn_dropout=float(model_cfg.get("wrn_dropout", 0.0)),
        temperature_type=deep_get(model_cfg, "temperature.type", "learnable"),
        temperature_init=float(deep_get(model_cfg, "temperature.init", 1.5)),
    )
    load_model_checkpoint(checkpoint_path, model, map_location=device, strict=False)
    model.to(device)
    started = time.time()
    predicted, target = collect_predictions(model, split_to_loader[split], device)
    metrics = summarize_distribution_metrics(predicted, target)
    metrics.update(entropy_correlations(predicted, target))
    true_entropy = entropy_bits_tensor(target).numpy()
    pred_entropy = entropy_bits_tensor(predicted).numpy()
    metrics.update(precision_at_many(true_entropy, pred_entropy))
    metrics["params"] = count_parameters(model)
    metrics["evaluation_time_sec"] = time.time() - started

    output_dir = Path(config["logging"]["save_dir"])
    table_path = output_dir / "tables" / f"{config['experiment_name']}_{split}_metrics.json"
    figure_path = output_dir / "figures" / f"{config['experiment_name']}_{split}_entropy_scatter.png"
    write_json(table_path, metrics)
    plot_entropy_scatter(true_entropy, pred_entropy, figure_path)
    print(metrics)
    return metrics
