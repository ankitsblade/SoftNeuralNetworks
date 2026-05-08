from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from src.losses import entropy_kl_loss, jsd_loss, kl_divergence_loss
from src.metrics.distribution_metrics import summarize_distribution_metrics
from src.metrics.entropy_metrics import entropy_correlations
from src.training.early_stopping import EarlyStopping
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import append_csv_row
from src.visualization.training_curves import plot_training_curves


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def build_soft_loss(name: str, lambda_entropy: float = 0.1) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name == "kl":
        return kl_divergence_loss
    if name == "jsd":
        return jsd_loss
    if name in {"custom", "entropy_kl", "kl_entropy"}:
        return lambda predicted, target: entropy_kl_loss(
            predicted,
            target,
            lambda_entropy=lambda_entropy,
        )
    raise ValueError(f"Unknown soft-label loss: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_optimizer(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.head.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def run_pretrain_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images, return_logits=True)
                loss = F.cross_entropy(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += images.size(0)
    return total_loss / total_seen, total_correct / total_seen


def run_soft_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    use_amp: bool,
) -> tuple[float, dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_seen = 0
    all_predicted: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, soft_targets, _, _ in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                predicted = model(images)
                loss = loss_fn(predicted, soft_targets)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * images.size(0)
            total_seen += images.size(0)
            all_predicted.append(predicted.detach().float().cpu())
            all_targets.append(soft_targets.detach().float().cpu())
    predicted_tensor = torch.cat(all_predicted)
    target_tensor = torch.cat(all_targets)
    metrics = summarize_distribution_metrics(predicted_tensor, target_tensor)
    metrics.update(entropy_correlations(predicted_tensor, target_tensor))
    return total_loss / total_seen, metrics


def train_pretrain(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    output_dir: str | Path,
) -> Path:
    training = config.get("training", {})
    epochs = int(training.get("epochs", 100))
    optimizer = make_optimizer(
        model,
        lr_backbone=float(training.get("lr_backbone", training.get("lr", 3e-4))),
        lr_head=float(training.get("lr_head", training.get("lr", 3e-4))),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    output_dir = Path(output_dir)
    log_path = output_dir / "logs" / f"{config['experiment_name']}_pretrain.csv"
    best_path = output_dir / "checkpoints" / f"{config['experiment_name']}_pretrain_best.pt"
    final_path = output_dir / "checkpoints" / f"{config['experiment_name']}_pretrain_final.pt"
    stopper_cfg = training.get("early_stopping", {})
    stopper = EarlyStopping(
        patience=int(stopper_cfg.get("patience", 10)),
        mode=stopper_cfg.get("mode", "min"),
    )
    best_metric = float("inf")
    use_amp = device.type == "cuda"
    started = time.time()

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_pretrain_epoch(model, train_loader, optimizer, device, use_amp)
        val_loss, val_acc = run_pretrain_epoch(model, val_loader, None, device, use_amp)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started,
        }
        append_csv_row(log_path, row)
        print(row)
        if val_loss < best_metric:
            best_metric = val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, row, config)
        if stopper.step(val_loss):
            break

    save_checkpoint(final_path, model, optimizer, scheduler, epoch, row, config)
    plot_training_curves(log_path, output_dir / "figures" / f"{config['experiment_name']}_pretrain_curves.png")
    return best_path


def train_finetune(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    output_dir: str | Path,
) -> Path:
    training = config.get("training", {})
    loss_cfg = config.get("loss", {})
    epochs = int(training.get("epochs", 100))
    optimizer = make_optimizer(
        model,
        lr_backbone=float(training.get("lr_backbone", 1e-4)),
        lr_head=float(training.get("lr_head", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = build_soft_loss(
        str(loss_cfg.get("name", "kl")),
        lambda_entropy=float(loss_cfg.get("lambda_entropy", 0.1)),
    )
    output_dir = Path(output_dir)
    log_path = output_dir / "logs" / f"{config['experiment_name']}_finetune.csv"
    best_path = output_dir / "checkpoints" / f"{config['experiment_name']}_best.pt"
    final_path = output_dir / "checkpoints" / f"{config['experiment_name']}_final.pt"
    stopper_cfg = training.get("early_stopping", {})
    stopper = EarlyStopping(
        patience=int(stopper_cfg.get("patience", 10)),
        mode=stopper_cfg.get("mode", "min"),
    )
    best_val_kl = float("inf")
    use_amp = device.type == "cuda"
    started = time.time()

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_soft_epoch(model, train_loader, optimizer, device, loss_fn, use_amp)
        val_loss, val_metrics = run_soft_epoch(model, val_loader, None, device, loss_fn, use_amp)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_kl": val_metrics["kl_mean"],
            "val_jsd": val_metrics["jsd_mean"],
            "val_cosine": val_metrics["cosine_mean"],
            "val_entropy_pearson": val_metrics["entropy_pearson"],
            "val_entropy_spearman": val_metrics["entropy_spearman"],
            "train_kl": train_metrics["kl_mean"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "params": count_parameters(model),
            "elapsed_sec": time.time() - started,
        }
        append_csv_row(log_path, row)
        print(row)
        if val_metrics["kl_mean"] < best_val_kl:
            best_val_kl = val_metrics["kl_mean"]
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, row, config)
        if stopper.step(val_metrics["kl_mean"]):
            break

    save_checkpoint(final_path, model, optimizer, scheduler, epoch, row, config)
    plot_training_curves(log_path, output_dir / "figures" / f"{config['experiment_name']}_finetune_curves.png")
    return best_path
