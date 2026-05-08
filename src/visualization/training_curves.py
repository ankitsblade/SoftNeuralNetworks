from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(log_csv: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(log_csv)
    plt.figure(figsize=(8, 4))
    for column in ("train_loss", "val_loss", "val_kl", "val_jsd"):
        if column in frame:
            plt.plot(frame["epoch"], frame[column], label=column)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
