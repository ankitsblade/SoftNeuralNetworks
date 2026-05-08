from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "experiment_name",
    "backbone",
    "loss",
    "kl",
    "jsd",
    "cosine",
    "pearson_entropy",
    "spearman_entropy",
    "p_at_100",
    "p_at_200",
    "p_at_500",
    "params",
    "training_time_sec",
]


def estimate_training_time(output_dir: str | Path, experiment_name: str) -> float | None:
    total = 0.0
    found = False
    for stage in ("pretrain", "finetune"):
        log_path = Path(output_dir) / "logs" / f"{experiment_name}_{stage}.csv"
        if not log_path.exists():
            continue
        frame = pd.read_csv(log_path)
        if frame.empty or "elapsed_sec" not in frame:
            continue
        total += float(frame["elapsed_sec"].iloc[-1])
        found = True
    return total if found else None


def build_summary_row(config: dict, metrics: dict[str, float]) -> dict[str, object]:
    experiment_name = str(config["experiment_name"])
    output_dir = config["logging"]["save_dir"]
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    return {
        "experiment_name": experiment_name,
        "backbone": model_cfg.get("backbone", "resnet18"),
        "loss": loss_cfg.get("name", "kl"),
        "kl": metrics.get("kl_mean"),
        "jsd": metrics.get("jsd_mean"),
        "cosine": metrics.get("cosine_mean"),
        "pearson_entropy": metrics.get("entropy_pearson"),
        "spearman_entropy": metrics.get("entropy_spearman"),
        "p_at_100": metrics.get("p_at_100"),
        "p_at_200": metrics.get("p_at_200"),
        "p_at_500": metrics.get("p_at_500"),
        "params": metrics.get("params"),
        "training_time_sec": estimate_training_time(output_dir, experiment_name),
    }


def collect_metric_tables(output_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(output_dir).glob("tables/*_test_metrics.json"):
        rows.append(pd.read_json(path, typ="series").to_dict() | {"run": path.stem})
    return pd.DataFrame(rows)


def write_summary_table(output_dir: str | Path, rows: list[dict[str, object]] | None = None) -> Path:
    frame = pd.DataFrame(rows) if rows is not None else collect_metric_tables(output_dir)
    if frame.empty:
        raise ValueError("No experiment rows available to summarize.")
    for column in SUMMARY_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[SUMMARY_COLUMNS].sort_values(["backbone", "loss", "experiment_name"], na_position="last")
    destination = Path(output_dir) / "tables" / "summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
