from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def plot_entropy_scatter(true_entropy: np.ndarray, predicted_entropy: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(true_entropy, predicted_entropy, s=12, alpha=0.5)
    limit = max(true_entropy.max(), predicted_entropy.max())
    plt.plot([0, limit], [0, limit], color="black", linewidth=1)
    plt.xlabel("True entropy (bits)")
    plt.ylabel("Predicted entropy (bits)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
