from __future__ import annotations

from pathlib import Path
from typing import Any


def _sanitize(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return str(value)


def _wandb_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("logging", {}).get("wandb", {})


def wandb_enabled(config: dict[str, Any]) -> bool:
    return bool(_wandb_config(config).get("enabled", False))


def init_wandb_run(config: dict[str, Any], stage: str, model: Any | None = None) -> Any | None:
    if not wandb_enabled(config):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging is enabled, but the `wandb` package is not installed. "
            "Install dependencies with `uv sync` or set `logging.wandb.enabled: false`."
        ) from exc

    wandb_cfg = _wandb_config(config)
    output_dir = Path(config.get("logging", {}).get("save_dir", "outputs")).resolve()
    run_name = wandb_cfg.get("name") or f"{config.get('experiment_name', 'experiment')}_{stage}"
    tags = list(wandb_cfg.get("tags", []))
    if stage not in tags:
        tags.append(stage)

    run = wandb.init(
        entity=wandb_cfg.get("entity") or None,
        project=wandb_cfg.get("project"),
        dir=str(Path(wandb_cfg.get("dir", output_dir)).resolve()),
        id=wandb_cfg.get("id"),
        name=run_name,
        notes=wandb_cfg.get("notes"),
        tags=tags or None,
        config=_sanitize(config),
        group=wandb_cfg.get("group") or config.get("experiment_name"),
        job_type=wandb_cfg.get("job_type") or stage,
        mode=wandb_cfg.get("mode"),
        resume=wandb_cfg.get("resume"),
        save_code=wandb_cfg.get("save_code"),
        reinit="create_new",
    )
    if model is not None and bool(wandb_cfg.get("watch_model", False)):
        run.watch(
            model,
            log=wandb_cfg.get("watch_log", "gradients"),
            log_freq=int(wandb_cfg.get("watch_log_freq", 100)),
        )
    run.summary["stage"] = stage
    run.summary["experiment_name"] = config.get("experiment_name", "experiment")
    return run


def log_wandb_metrics(run: Any | None, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    payload = {str(key): _sanitize(value) for key, value in metrics.items()}
    run.log(payload, step=step)


def finish_wandb_run(run: Any | None, summary: dict[str, Any] | None = None) -> None:
    if run is None:
        return
    if summary:
        for key, value in summary.items():
            run.summary[str(key)] = _sanitize(value)
    run.finish()
