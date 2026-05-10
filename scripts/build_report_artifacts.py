from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from torch import nn

from src.datasets.cifar10h_dataset import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, build_cifar10h_loaders
from src.evaluation.evaluate import compute_report_metrics
from src.evaluation.robustness import apply_corruption
from src.metrics.entropy_metrics import entropy_bits_tensor
from src.models.model_factory import build_model
from src.training.trainer import count_parameters, get_device
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config import deep_get, load_config
from src.visualization.gradcam import GradCAM


DEFAULT_REPORT_CONFIGS = [
    "configs/densenet_bc_kl.yaml",
    "configs/resnet18_kl.yaml",
    "configs/vgg13_bn_kl.yaml",
    "configs/wrn28_2_kl.yaml",
    "configs/resnet18_jsd.yaml",
    "configs/resnet18_custom.yaml",
    "configs/wrn28_2_jsd.yaml",
    "configs/wrn28_2_custom.yaml",
    "configs/resnet18_linear_kl.yaml",
    "configs/resnet18_mlp_kl.yaml",
    "configs/wrn28_2_linear_kl.yaml",
    "configs/wrn28_2_mlp_kl.yaml",
    "configs/resnet18_soft_only_kl.yaml",
    "configs/wrn28_2_soft_only_kl.yaml",
]

REPORT_FIGURE_DIR = Path("outputs/figures/report")
REPORT_TABLE_DIR = Path("outputs/tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tables and figures used by the LaTeX report.")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_REPORT_CONFIGS)
    parser.add_argument("--analysis-config", default="configs/wrn28_2_kl.yaml")
    parser.add_argument("--analysis-checkpoint", default="outputs/checkpoints/wrn28_2_kl_best.pt")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def checkpoint_for(config: dict) -> Path:
    output_dir = Path(config.get("logging", {}).get("save_dir", "outputs"))
    return output_dir / "checkpoints" / f"{config['experiment_name']}_best.pt"


def metric_path_for(config: dict, split: str) -> Path:
    output_dir = Path(config.get("logging", {}).get("save_dir", "outputs"))
    return output_dir / "tables" / f"{config['experiment_name']}_{split}_metrics.json"


def load_rows(config_paths: list[str], split: str) -> pd.DataFrame:
    rows = []
    for config_path in config_paths:
        path = Path(config_path)
        if not path.exists():
            continue
        config = load_config(path)
        metrics_path = metric_path_for(config, split)
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        model_cfg = config.get("model", {})
        loss_cfg = config.get("loss", {})
        pretrain_checkpoint = model_cfg.get("pretrained_checkpoint")
        pretrain_path = Path(config["logging"]["save_dir"]) / "checkpoints" / f"{config['experiment_name']}_pretrain_best.pt"
        rows.append(
            {
                "experiment_name": config["experiment_name"],
                "backbone": model_cfg.get("backbone", "resnet18"),
                "head": model_cfg.get("head", "temperature_mlp"),
                "loss": loss_cfg.get("name", "kl"),
                "data_strategy": "cifar10_pretrain_finetune"
                if pretrain_checkpoint or pretrain_path.exists()
                else "soft_only_random_init",
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["backbone", "loss", "head", "experiment_name"])
    return frame


def write_metrics_tables(frame: pd.DataFrame) -> None:
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORT_TABLE_DIR / "report_metrics.csv", index=False)
    main = frame[(frame["loss"] == "kl") & (frame["head"] == "temperature_mlp")]
    main = main[~main["experiment_name"].str.contains("soft_only", regex=False)]
    loss = frame[
        frame["experiment_name"].isin(
            ["resnet18_kl", "resnet18_jsd", "resnet18_custom", "wrn28_2_kl", "wrn28_2_jsd", "wrn28_2_custom"]
        )
    ]
    head = frame[
        frame["experiment_name"].isin(
            [
                "resnet18_linear_kl",
                "resnet18_mlp_kl",
                "resnet18_kl",
                "wrn28_2_linear_kl",
                "wrn28_2_mlp_kl",
                "wrn28_2_kl",
            ]
        )
    ]
    strategy = frame[
        frame["experiment_name"].isin(["resnet18_kl", "resnet18_soft_only_kl", "wrn28_2_kl", "wrn28_2_soft_only_kl"])
    ]
    _write_latex_table(
        main,
        REPORT_TABLE_DIR / "report_main_metrics.tex",
        "Main backbone comparison on the CIFAR-10H test split",
        "tab:main-metrics",
    )
    _write_latex_table(
        loss,
        REPORT_TABLE_DIR / "report_loss_comparison.tex",
        "Loss-function ablation for ResNet18 and WRN-28-2",
        "tab:loss-ablation",
    )
    _write_latex_table(
        head,
        REPORT_TABLE_DIR / "report_head_ablation.tex",
        "Prediction-head ablation with KL fine-tuning",
        "tab:head-ablation",
    )
    _write_latex_table(
        strategy,
        REPORT_TABLE_DIR / "report_data_strategy.tex",
        "Training-data strategy ablation",
        "tab:data-strategy",
    )


def _format_ci(row: pd.Series, metric: str, precision: int = 3) -> str:
    value = row.get(metric)
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if pd.isna(value):
        return "--"
    if pd.isna(low) or pd.isna(high):
        return f"{value:.{precision}f}"
    return f"{value:.{precision}f} [{low:.{precision}f},{high:.{precision}f}]"


def _write_latex_table(frame: pd.DataFrame, destination: Path, caption: str, label: str) -> None:
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        f"\\caption{{{caption}. Brackets show bootstrap 95\\% confidence intervals. Lower is better for KL/JSD/ECE; higher is better for cosine, entropy Spearman, and P@500.}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llllrrrrrr@{}}",
        "\\toprule",
        "Experiment & Backbone & Head & Loss & KL $\\downarrow$ & JSD $\\downarrow$ & Cos. $\\uparrow$ & $\\rho_H$ $\\uparrow$ & P@500 $\\uparrow$ & ECE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        values = [
            row["experiment_name"].replace("_", "\\_"),
            row["backbone"].replace("_", "\\_"),
            row["head"].replace("_", "\\_"),
            row["loss"].replace("_", "\\_"),
            _format_ci(row, "kl_mean"),
            _format_ci(row, "jsd_mean"),
            _format_ci(row, "cosine_mean"),
            _format_ci(row, "entropy_spearman"),
            _format_ci(row, "p_at_500"),
            f"{row.get('soft_ece', np.nan):.3f}" if not pd.isna(row.get("soft_ece", np.nan)) else "--",
        ]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    destination.write_text("\n".join(lines))


def plot_metric_bars(frame: pd.DataFrame) -> None:
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        return
    main = frame[(frame["loss"] == "kl") & (frame["head"] == "temperature_mlp")]
    main = main[~main["experiment_name"].str.contains("soft_only", regex=False)]
    if not main.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.barplot(data=main, x="backbone", y="kl_mean", ax=axes[0], color="#537895")
        sns.barplot(data=main, x="backbone", y="entropy_spearman", ax=axes[1], color="#d99058")
        axes[0].set_ylabel("KL divergence")
        axes[1].set_ylabel("Entropy Spearman")
        for ax in axes:
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(REPORT_FIGURE_DIR / "main_backbone_metrics.png", dpi=180)
        plt.close(fig)

    loss = frame[
        frame["experiment_name"].isin(
            ["resnet18_kl", "resnet18_jsd", "resnet18_custom", "wrn28_2_kl", "wrn28_2_jsd", "wrn28_2_custom"]
        )
    ]
    if not loss.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=loss, x="loss", y="jsd_mean", hue="backbone")
        plt.ylabel("JSD")
        plt.xlabel("Loss")
        plt.tight_layout()
        plt.savefig(REPORT_FIGURE_DIR / "loss_comparison_jsd.png", dpi=180)
        plt.close()

    head = frame[
        frame["experiment_name"].isin(
            [
                "resnet18_linear_kl",
                "resnet18_mlp_kl",
                "resnet18_kl",
                "wrn28_2_linear_kl",
                "wrn28_2_mlp_kl",
                "wrn28_2_kl",
            ]
        )
    ]
    if not head.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=head, x="head", y="entropy_mae", hue="backbone")
        plt.ylabel("Entropy MAE")
        plt.xlabel("Head")
        plt.tight_layout()
        plt.savefig(REPORT_FIGURE_DIR / "head_ablation_entropy_mae.png", dpi=180)
        plt.close()


def build_model_for_config(config: dict, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model_cfg = config.get("model", {})
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
    model.eval()
    return model


def build_loaders(config: dict, split: str, num_workers: int, device: torch.device):
    dataset = config.get("dataset", {})
    loaders = build_cifar10h_loaders(
        root=dataset.get("root", "data"),
        cifar10h_path=dataset.get("cifar10h_path", "data/cifar10h/cifar10h-probs.npy"),
        batch_size=int(dataset.get("batch_size", 128)),
        num_workers=num_workers,
        seed=int(config.get("seed", 42)),
        train_size=int(dataset.get("train_size", 6000)),
        val_size=int(dataset.get("val_size", 2000)),
        test_size=int(dataset.get("test_size", 2000)),
        download=bool(dataset.get("download", True)),
        device=device,
    )
    return {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]


@torch.no_grad()
def collect_for_analysis(model: nn.Module, loader, device: torch.device):
    images_all = []
    predicted_all = []
    target_all = []
    hard_all = []
    index_all = []
    for images, targets, hard_labels, indices in loader:
        images = images.to(device, non_blocking=True)
        predicted = model(images).detach().cpu()
        images_all.append(images.detach().cpu())
        predicted_all.append(predicted)
        target_all.append(targets.float().cpu())
        hard_all.append(hard_labels.cpu())
        index_all.append(indices.cpu())
    return (
        torch.cat(images_all),
        torch.cat(predicted_all),
        torch.cat(target_all),
        torch.cat(hard_all),
        torch.cat(index_all),
    )


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def run_class_conditional(config: dict, checkpoint_path: Path, split: str, num_workers: int) -> pd.DataFrame:
    device = get_device(config.get("device", "auto"))
    model = build_model_for_config(config, checkpoint_path, device)
    loader = build_loaders(config, split, num_workers, device)
    _, predicted, target, hard_labels, _ = collect_for_analysis(model, loader, device)
    rows = []
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        mask = hard_labels == class_idx
        if not mask.any():
            continue
        metrics = compute_report_metrics(predicted[mask], target[mask], seed=int(config.get("seed", 42)))
        rows.append({"class": class_name, "n": int(mask.sum().item()), **metrics})
    frame = pd.DataFrame(rows)
    destination = REPORT_TABLE_DIR / f"class_conditional_{config['experiment_name']}.csv"
    frame.to_csv(destination, index=False)
    plt.figure(figsize=(9, 4))
    sns.barplot(data=frame, x="class", y="entropy_mae", color="#b6624b")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Entropy MAE")
    plt.xlabel("Majority class")
    plt.tight_layout()
    plt.savefig(REPORT_FIGURE_DIR / f"class_conditional_{config['experiment_name']}.png", dpi=180)
    plt.close()
    return frame


@torch.no_grad()
def run_corruption_response(config: dict, checkpoint_path: Path, split: str, num_workers: int) -> pd.DataFrame:
    device = get_device(config.get("device", "auto"))
    model = build_model_for_config(config, checkpoint_path, device)
    loader = build_loaders(config, split, num_workers, device)
    corruptions = {
        "gaussian_noise": [0.0, 0.05, 0.10, 0.20, 0.30],
        "gaussian_blur": [0.0, 1.0, 2.0, 3.0, 4.0],
        "contrast": [0.0, 0.20, 0.40, 0.60, 0.80],
    }
    rows = []
    for corruption, severities in corruptions.items():
        for severity in severities:
            pred_entropy = []
            true_entropy = []
            for images, targets, _, _ in loader:
                images = images.to(device, non_blocking=True)
                if severity > 0:
                    images = apply_corruption(images, corruption, severity)
                predicted = model(images).detach().cpu()
                pred_entropy.append(entropy_bits_tensor(predicted))
                true_entropy.append(entropy_bits_tensor(targets.float()))
            pred_entropy_tensor = torch.cat(pred_entropy)
            true_entropy_tensor = torch.cat(true_entropy)
            rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "pred_entropy_mean": float(pred_entropy_tensor.mean().item()),
                    "pred_entropy_std": float(pred_entropy_tensor.std(unbiased=False).item()),
                    "true_entropy_mean": float(true_entropy_tensor.mean().item()),
                    "entropy_delta_from_truth": float((pred_entropy_tensor.mean() - true_entropy_tensor.mean()).item()),
                }
            )
    frame = pd.DataFrame(rows)
    destination = REPORT_TABLE_DIR / f"corruption_response_{config['experiment_name']}.csv"
    frame.to_csv(destination, index=False)
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=frame, x="severity", y="pred_entropy_mean", hue="corruption", marker="o")
    plt.ylabel("Mean predicted entropy")
    plt.xlabel("Corruption severity")
    plt.tight_layout()
    plt.savefig(REPORT_FIGURE_DIR / f"corruption_response_{config['experiment_name']}.png", dpi=180)
    plt.close()
    return frame


def write_failure_cases(config: dict, checkpoint_path: Path, split: str, num_workers: int) -> None:
    device = get_device(config.get("device", "auto"))
    model = build_model_for_config(config, checkpoint_path, device)
    loader = build_loaders(config, split, num_workers, device)
    images, predicted, target, hard_labels, indices = collect_for_analysis(model, loader, device)
    true_entropy = entropy_bits_tensor(target)
    pred_entropy = entropy_bits_tensor(predicted)
    error = torch.abs(pred_entropy - true_entropy)
    selected = torch.argsort(error, descending=True)[:12]
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    rgb_images = denormalize(images[selected]).permute(0, 2, 3, 1).numpy()
    for ax, image, row_idx in zip(axes.ravel(), rgb_images, selected, strict=True):
        idx = int(indices[row_idx].item())
        top_true = int(target[row_idx].argmax().item())
        top_pred = int(predicted[row_idx].argmax().item())
        ax.imshow(image)
        ax.set_title(
            f"idx {idx}\\ntrue {CIFAR10_CLASSES[top_true]}, pred {CIFAR10_CLASSES[top_pred]}\\n"
            f"Ht={true_entropy[row_idx]:.2f}, Hp={pred_entropy[row_idx]:.2f}",
            fontsize=8,
        )
        ax.axis("off")
        rows.append(
            {
                "cifar10h_index": idx,
                "majority_class": CIFAR10_CLASSES[int(hard_labels[row_idx].item())],
                "true_top_class": CIFAR10_CLASSES[top_true],
                "pred_top_class": CIFAR10_CLASSES[top_pred],
                "true_entropy": float(true_entropy[row_idx].item()),
                "pred_entropy": float(pred_entropy[row_idx].item()),
                "entropy_abs_error": float(error[row_idx].item()),
                "failure_hypothesis": "manual review required",
            }
        )
    fig.tight_layout()
    fig.savefig(REPORT_FIGURE_DIR / f"failure_cases_{config['experiment_name']}.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REPORT_TABLE_DIR / f"failure_cases_{config['experiment_name']}.csv", index=False)


def select_target_layer(model: nn.Module) -> nn.Module:
    backbone = model.backbone
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1].conv2
    if hasattr(backbone, "block3"):
        block3 = backbone.block3
        if hasattr(block3[-1], "conv2"):
            return block3[-1].conv2
        if hasattr(block3[-1], "net"):
            return block3[-1].net[-1]
    if hasattr(backbone, "features"):
        conv_layers = [module for module in backbone.features.modules() if isinstance(module, nn.Conv2d)]
        return conv_layers[-1]
    raise ValueError("Could not infer Grad-CAM target layer for this backbone.")


def write_gradcam_grid(config: dict, checkpoint_path: Path, split: str, num_workers: int) -> None:
    device = get_device(config.get("device", "auto"))
    model = build_model_for_config(config, checkpoint_path, device)
    loader = build_loaders(config, split, num_workers, device)
    images, predicted, target, _, indices = collect_for_analysis(model, loader, device)
    true_entropy = entropy_bits_tensor(target)
    ordered = torch.argsort(true_entropy)
    selected = torch.cat([ordered[:3], ordered[len(ordered) // 2 - 1 : len(ordered) // 2 + 2], ordered[-3:]])
    cam = GradCAM(model, select_target_layer(model))
    try:
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        rgb_images = denormalize(images[selected]).permute(0, 2, 3, 1).numpy()
        for ax, row_idx, image in zip(axes.ravel(), selected, rgb_images, strict=True):
            input_image = images[row_idx : row_idx + 1].to(device)
            class_index = int(target[row_idx].argmax().item())
            heat = cam(input_image, class_index=class_index).detach().cpu().squeeze().numpy()
            ax.imshow(image)
            ax.imshow(heat, cmap="magma", alpha=0.45)
            ax.set_title(
                f"idx {int(indices[row_idx].item())}, {CIFAR10_CLASSES[class_index]}\\nH={true_entropy[row_idx]:.2f}",
                fontsize=8,
            )
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(REPORT_FIGURE_DIR / f"gradcam_{config['experiment_name']}.png", dpi=180)
        plt.close(fig)
    finally:
        cam.close()


def write_manual_disagreement_sheet(config: dict, checkpoint_path: Path, split: str, num_workers: int) -> None:
    device = get_device(config.get("device", "auto"))
    model = build_model_for_config(config, checkpoint_path, device)
    loader = build_loaders(config, split, num_workers, device)
    images, predicted, target, hard_labels, indices = collect_for_analysis(model, loader, device)
    true_entropy = entropy_bits_tensor(target)
    selected = torch.argsort(true_entropy, descending=True)[:24]
    rows = []
    for row_idx in selected:
        top_two = torch.topk(target[row_idx], k=2)
        rows.append(
            {
                "cifar10h_index": int(indices[row_idx].item()),
                "majority_class": CIFAR10_CLASSES[int(hard_labels[row_idx].item())],
                "top_human_class": CIFAR10_CLASSES[int(top_two.indices[0].item())],
                "second_human_class": CIFAR10_CLASSES[int(top_two.indices[1].item())],
                "top_probability": float(top_two.values[0].item()),
                "second_probability": float(top_two.values[1].item()),
                "true_entropy": float(true_entropy[row_idx].item()),
                "suggested_categories": "ambiguous object identity; poor image quality; multi-object; class boundary; occlusion; background confusion; other",
                "manual_category": "",
                "manual_notes": "",
            }
        )
    pd.DataFrame(rows).to_csv(REPORT_TABLE_DIR / "manual_disagreement_source_sheet.csv", index=False)
    rgb_images = denormalize(images[selected]).permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))
    for ax, image, row_idx in zip(axes.ravel(), rgb_images, selected, strict=True):
        ax.imshow(image)
        ax.set_title(f"idx {int(indices[row_idx].item())}\\nH={true_entropy[row_idx]:.2f}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(REPORT_FIGURE_DIR / "manual_disagreement_source_sheet.png", dpi=180)
    plt.close(fig)


def write_architecture_diagram() -> None:
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.94,
        "CIFAR-10H Disagreement Modeling Pipeline",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#1f3340",
    )

    def box(x: float, y: float, w: float, h: float, text: str, face: str, edge: str = "#253746") -> None:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color="#17212b", linespacing=1.25)

    def arrow(start: tuple[float, float], end: tuple[float, float], dashed: bool = False) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color="#253746",
                linestyle="--" if dashed else "-",
                shrinkA=3,
                shrinkB=3,
            )
        )

    box(0.16, 0.76, 0.19, 0.14, "CIFAR-10 train\nhard labels", "#dceef8")
    box(0.40, 0.76, 0.23, 0.14, "Hard-label pretraining\nlearn visual features", "#f8ead7")
    box(0.16, 0.45, 0.19, 0.14, "CIFAR-10H image\n32x32 RGB", "#dceef8")
    box(0.40, 0.45, 0.23, 0.17, "CIFAR CNN backbone\nResNet18 / WRN-28-2\nDenseNet / VGG", "#f8ead7")
    box(0.62, 0.45, 0.18, 0.16, "Distribution head\nLinear / MLP /\ntemperature MLP", "#f8ead7")
    box(0.84, 0.45, 0.20, 0.16, "Predicted distribution\n$q_\\theta(y|x)$\n10 probabilities", "#dff0df")
    box(0.84, 0.20, 0.20, 0.15, "Human distribution\n$p_{hum}(y|x)$\nCIFAR-10H labels", "#dff0df")
    box(0.62, 0.20, 0.20, 0.15, "Training objective\nKL / JSD / custom\n+ report metrics", "#eadff2")

    arrow((0.255, 0.76), (0.285, 0.76))
    arrow((0.40, 0.69), (0.40, 0.545), dashed=True)
    arrow((0.255, 0.45), (0.285, 0.45))
    arrow((0.515, 0.45), (0.53, 0.45))
    arrow((0.71, 0.45), (0.74, 0.45))
    arrow((0.76, 0.385), (0.69, 0.265))
    arrow((0.76, 0.20), (0.72, 0.20))
    arrow((0.54, 0.25), (0.46, 0.37), dashed=True)

    ax.text(0.48, 0.63, "initialize / reuse", ha="left", va="center", fontsize=8.5, color="#536878")
    ax.text(0.47, 0.31, "fine-tune", ha="center", va="center", fontsize=8.5, color="#536878")
    ax.text(0.73, 0.29, "compare", ha="center", va="center", fontsize=8.5, color="#536878")
    fig.tight_layout()
    fig.savefig(REPORT_FIGURE_DIR / "architecture_diagram.png", dpi=180)
    plt.close(fig)


def write_parameter_table(config_paths: list[str]) -> None:
    rows = []
    for config_path in config_paths:
        path = Path(config_path)
        if not path.exists():
            continue
        config = load_config(path)
        if config["experiment_name"] not in {"resnet18_kl", "wrn28_2_kl", "densenet_bc_kl", "vgg13_bn_kl"}:
            continue
        model_cfg = config.get("model", {})
        model = build_model(
            backbone_name=model_cfg.get("backbone", "resnet18"),
            head_name=model_cfg.get("head", "temperature_mlp"),
            num_classes=int(model_cfg.get("num_classes", 10)),
            dropout=float(model_cfg.get("dropout", 0.30)),
            wrn_dropout=float(model_cfg.get("wrn_dropout", 0.0)),
            temperature_type=deep_get(model_cfg, "temperature.type", "learnable"),
            temperature_init=float(deep_get(model_cfg, "temperature.init", 1.5)),
        )
        rows.append(
            {
                "experiment_name": config["experiment_name"],
                "backbone": model_cfg.get("backbone", "resnet18"),
                "head": model_cfg.get("head", "temperature_mlp"),
                "params": count_parameters(model),
            }
        )
    pd.DataFrame(rows).to_csv(REPORT_TABLE_DIR / "parameter_counts.csv", index=False)


def main() -> None:
    args = parse_args()
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    frame = load_rows(args.configs, args.split)
    if not frame.empty:
        write_metrics_tables(frame)
        plot_metric_bars(frame)
    write_parameter_table(args.configs)
    write_architecture_diagram()

    analysis_config = load_config(args.analysis_config)
    analysis_checkpoint = Path(args.analysis_checkpoint)
    if analysis_checkpoint.exists():
        run_class_conditional(analysis_config, analysis_checkpoint, args.split, args.num_workers)
        run_corruption_response(analysis_config, analysis_checkpoint, args.split, args.num_workers)
        write_failure_cases(analysis_config, analysis_checkpoint, args.split, args.num_workers)
        write_gradcam_grid(analysis_config, analysis_checkpoint, args.split, args.num_workers)
        write_manual_disagreement_sheet(analysis_config, analysis_checkpoint, args.split, args.num_workers)


if __name__ == "__main__":
    main()
