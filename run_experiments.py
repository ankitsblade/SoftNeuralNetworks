from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from src.evaluation.compare_backbones import build_summary_row, write_summary_table
from src.evaluation.evaluate import evaluate_and_save
from src.training.finetune_cifar10h import run_finetuning
from src.training.pretrain_cifar10 import run_pretraining
from src.utils.config import load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comparative CIFAR-10H experiments with optional pretraining and evaluation."
    )
    parser.add_argument("configs", nargs="+", help="Config files to run.")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip CIFAR-10 hard-label pretraining.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip CIFAR-10H evaluation after fine-tuning.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    return parser.parse_args()


def prepare_finetune_config(config: dict, pretrained_checkpoint: str | Path | None) -> dict:
    finetune_config = deepcopy(config)
    if pretrained_checkpoint is not None:
        finetune_config.setdefault("model", {})["pretrained_checkpoint"] = str(pretrained_checkpoint)
    return finetune_config


def main() -> None:
    args = parse_args()
    rows_by_output_dir: dict[Path, list[dict[str, object]]] = {}
    for config in args.configs:
        config_path = Path(config)
        base_config = load_config(config_path)
        output_dir = Path(base_config["logging"]["save_dir"])
        pretrained_checkpoint: str | Path | None = None

        if not args.skip_pretrain:
            seed_everything(int(base_config.get("seed", 42)))
            print(f"[{base_config['experiment_name']}] Pretraining backbone on CIFAR-10...")
            pretrained_checkpoint = run_pretraining(deepcopy(base_config))

        finetune_config = prepare_finetune_config(base_config, pretrained_checkpoint)
        seed_everything(int(finetune_config.get("seed", 42)))
        print(f"[{finetune_config['experiment_name']}] Fine-tuning on CIFAR-10H...")
        best_checkpoint = run_finetuning(finetune_config)

        if args.skip_eval:
            continue

        seed_everything(int(finetune_config.get("seed", 42)))
        print(f"[{finetune_config['experiment_name']}] Evaluating on {args.split} split...")
        metrics = evaluate_and_save(finetune_config, best_checkpoint, split=args.split)
        rows_by_output_dir.setdefault(output_dir, []).append(build_summary_row(finetune_config, metrics))

    for output_dir, rows in rows_by_output_dir.items():
        summary_path = write_summary_table(output_dir, rows=rows)
        print(f"Wrote summary table: {summary_path}")


if __name__ == "__main__":
    main()
