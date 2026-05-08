from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.datasets.cifar10h_dataset import CIFAR10_CLASSES, CIFAR10HDataset, image_transform
from src.metrics.distribution_metrics import confusion_style_matrix
from src.metrics.entropy_metrics import entropy_bits_np


def plot_entropy_histogram(targets: np.ndarray, output_path: str | Path) -> None:
    entropies = entropy_bits_np(targets)
    plt.figure(figsize=(7, 4))
    sns.histplot(entropies, bins=40)
    plt.xlabel("Entropy (bits)")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_per_class_entropy(targets: np.ndarray, hard_labels: np.ndarray, output_path: str | Path) -> None:
    entropies = entropy_bits_np(targets)
    values = [entropies[hard_labels == class_idx].mean() for class_idx in range(10)]
    plt.figure(figsize=(9, 4))
    sns.barplot(x=list(CIFAR10_CLASSES), y=values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Average entropy (bits)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_matrix(targets: np.ndarray, hard_labels: np.ndarray, output_path: str | Path) -> None:
    matrix = confusion_style_matrix(targets, hard_labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        cmap="mako",
        vmin=0,
        vmax=1,
    )
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_entropy_examples(
    root: str | Path,
    cifar10h_path: str | Path,
    output_path: str | Path,
    n_each: int = 8,
) -> None:
    dataset = CIFAR10HDataset(root, cifar10h_path, transform=image_transform(), download=True)
    targets = dataset.soft_targets.numpy()
    entropies = entropy_bits_np(targets)
    indices = np.concatenate([np.argsort(entropies)[:n_each], np.argsort(-entropies)[:n_each]])
    fig, axes = plt.subplots(2, n_each, figsize=(2.2 * n_each, 4.8))
    for ax, index in zip(axes.ravel(), indices, strict=True):
        image, target, _, _ = dataset[int(index)]
        ax.imshow(torch.permute(image, (1, 2, 0)).numpy())
        top = int(target.argmax().item())
        ax.set_title(f"{CIFAR10_CLASSES[top]}\nH={entropies[index]:.2f}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def generate_data_visualizations(root: str | Path, cifar10h_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CIFAR10HDataset(root, cifar10h_path, transform=image_transform(), download=True)
    targets = dataset.soft_targets.numpy()
    hard_labels = np.asarray(dataset.cifar10.targets)
    plot_entropy_histogram(targets, output_dir / "entropy_histogram.png")
    plot_per_class_entropy(targets, hard_labels, output_dir / "per_class_entropy.png")
    plot_confusion_matrix(targets, hard_labels, output_dir / "annotator_confusion_matrix.png")
    plot_entropy_examples(root, cifar10h_path, output_dir / "entropy_examples.png")
