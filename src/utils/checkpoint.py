from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
        },
        checkpoint_path,
    )


def load_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint
