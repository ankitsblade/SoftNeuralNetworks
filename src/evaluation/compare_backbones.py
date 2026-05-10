from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "experiment_name",
    "backbone",
    "head",
    "loss",
    "data_strategy",
    "kl_mean",
    "kl_std",
    "kl_mean_ci_low",
    "kl_mean_ci_high",
    "jsd_mean",
    "jsd_std",
    "jsd_mean_ci_low",
    "jsd_mean_ci_high",
    "cosine_mean",
    "cosine_std",
    "cosine_mean_ci_low",
    "cosine_mean_ci_high",
    "soft_cross_entropy_mean",
    "soft_cross_entropy_std",
    "brier_mean",
    "brier_std",
    "total_variation_mean",
    "total_variation_std",
    "entropy_pearson",
    "entropy_pearson_ci_low",
    "entropy_pearson_ci_high",
    "entropy_spearman",
    "entropy_spearman_ci_low",
    "entropy_spearman_ci_high",
    "entropy_mae",
    "entropy_mae_ci_low",
    "entropy_mae_ci_high",
    "entropy_rmse",
    "entropy_bias",
    "p_at_100",
    "p_at_200",
    "p_at_500",
    "p_at_500_ci_low",
    "p_at_500_ci_high",
    "soft_ece",
    "soft_mce",
    "majority_top1_acc",
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
    pretrain_checkpoint = model_cfg.get("pretrained_checkpoint")
    has_local_pretrain = (Path(output_dir) / "checkpoints" / f"{experiment_name}_pretrain_best.pt").exists()
    data_strategy = "cifar10_pretrain_finetune" if pretrain_checkpoint or has_local_pretrain else "soft_only_random_init"
    return {
        "experiment_name": experiment_name,
        "backbone": model_cfg.get("backbone", "resnet18"),
        "head": model_cfg.get("head", "temperature_mlp"),
        "loss": loss_cfg.get("name", "kl"),
        "data_strategy": data_strategy,
        "kl_mean": metrics.get("kl_mean"),
        "kl_std": metrics.get("kl_std"),
        "kl_mean_ci_low": metrics.get("kl_mean_ci_low"),
        "kl_mean_ci_high": metrics.get("kl_mean_ci_high"),
        "jsd_mean": metrics.get("jsd_mean"),
        "jsd_std": metrics.get("jsd_std"),
        "jsd_mean_ci_low": metrics.get("jsd_mean_ci_low"),
        "jsd_mean_ci_high": metrics.get("jsd_mean_ci_high"),
        "cosine_mean": metrics.get("cosine_mean"),
        "cosine_std": metrics.get("cosine_std"),
        "cosine_mean_ci_low": metrics.get("cosine_mean_ci_low"),
        "cosine_mean_ci_high": metrics.get("cosine_mean_ci_high"),
        "soft_cross_entropy_mean": metrics.get("soft_cross_entropy_mean"),
        "soft_cross_entropy_std": metrics.get("soft_cross_entropy_std"),
        "brier_mean": metrics.get("brier_mean"),
        "brier_std": metrics.get("brier_std"),
        "total_variation_mean": metrics.get("total_variation_mean"),
        "total_variation_std": metrics.get("total_variation_std"),
        "entropy_pearson": metrics.get("entropy_pearson"),
        "entropy_pearson_ci_low": metrics.get("entropy_pearson_ci_low"),
        "entropy_pearson_ci_high": metrics.get("entropy_pearson_ci_high"),
        "entropy_spearman": metrics.get("entropy_spearman"),
        "entropy_spearman_ci_low": metrics.get("entropy_spearman_ci_low"),
        "entropy_spearman_ci_high": metrics.get("entropy_spearman_ci_high"),
        "entropy_mae": metrics.get("entropy_mae"),
        "entropy_mae_ci_low": metrics.get("entropy_mae_ci_low"),
        "entropy_mae_ci_high": metrics.get("entropy_mae_ci_high"),
        "entropy_rmse": metrics.get("entropy_rmse"),
        "entropy_bias": metrics.get("entropy_bias"),
        "p_at_100": metrics.get("p_at_100"),
        "p_at_200": metrics.get("p_at_200"),
        "p_at_500": metrics.get("p_at_500"),
        "p_at_500_ci_low": metrics.get("p_at_500_ci_low"),
        "p_at_500_ci_high": metrics.get("p_at_500_ci_high"),
        "soft_ece": metrics.get("soft_ece"),
        "soft_mce": metrics.get("soft_mce"),
        "majority_top1_acc": metrics.get("majority_top1_acc"),
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
