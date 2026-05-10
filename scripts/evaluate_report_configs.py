from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.compare_backbones import build_summary_row, write_summary_table
from src.evaluation.evaluate import evaluate_and_save
from src.utils.config import load_config
from src.utils.seed import seed_everything


DEFAULT_CONFIGS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all report checkpoints and rebuild the summary table.")
    parser.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def checkpoint_for(config: dict) -> Path:
    return Path(config["logging"]["save_dir"]) / "checkpoints" / f"{config['experiment_name']}_best.pt"


def main() -> None:
    args = parse_args()
    rows_by_output_dir: dict[Path, list[dict[str, object]]] = {}
    for config_path in args.configs:
        path = Path(config_path)
        if not path.exists():
            print(f"Skipping missing config: {path}")
            continue
        config = load_config(path)
        config.setdefault("dataset", {})["num_workers"] = args.num_workers
        checkpoint_path = checkpoint_for(config)
        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {checkpoint_path}")
            continue
        seed_everything(int(config.get("seed", 42)))
        metrics = evaluate_and_save(config, checkpoint_path, split=args.split)
        output_dir = Path(config["logging"]["save_dir"])
        rows_by_output_dir.setdefault(output_dir, []).append(build_summary_row(config, metrics))
    for output_dir, rows in rows_by_output_dir.items():
        summary_path = write_summary_table(output_dir, rows=rows)
        print(f"Wrote summary table: {summary_path}")


if __name__ == "__main__":
    main()
