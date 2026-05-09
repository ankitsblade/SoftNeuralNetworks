# CIFAR-10H Human Disagreement Pipeline

This project implements the comparative CNN backbone pipeline from `human_disagreement_comparative_backbone_pipeline.md`.

## Setup

Use `uv` for all commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

This project currently supports Python `3.11` through `3.13`. Python `3.14` is not yet usable here because the `wandb`/`pydantic` stack in the training pipeline breaks on it.

For CUDA-specific PyTorch wheels, install PyTorch with the CUDA index first, then sync the rest:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

## Data

Download CIFAR-10 and CIFAR-10H:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python download_data.py
```

This creates the CIFAR-10H target file at:

```text
data/cifar10h/cifar10h-probs.npy
```

The file must be a `10000 x 10` probability or count array aligned with CIFAR-10 test images.

## Commands

Pretrain a backbone on CIFAR-10 hard labels:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python train_pretrain.py --config configs/resnet18_kl.yaml
```

Fine-tune on CIFAR-10H soft labels:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python train_finetune.py --config configs/resnet18_kl.yaml
```

If `outputs/checkpoints/resnet18_kl_pretrain_best.pt` already exists, the fine-tuning script will pick it up automatically for the matching experiment.

Weights & Biases tracking is now wired into the training loop. The default configs enable it with:

```yaml
logging:
  wandb:
    enabled: true
    project: softneuralnetworks
    entity:
    mode: online
```

If you have not authenticated yet:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run wandb login
```

For offline or restricted-network runs, switch `logging.wandb.mode` to `offline`. To disable W&B for a run, set `logging.wandb.enabled: false`.

Evaluate a checkpoint:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python evaluate.py --config configs/resnet18_kl.yaml --checkpoint outputs/checkpoints/resnet18_kl_best.pt
```

If multiprocessing workers are blocked in your environment, use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python evaluate.py --config configs/resnet18_kl.yaml --checkpoint outputs/checkpoints/resnet18_kl_best.pt --num-workers 0
```

Generate data sanity figures:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python visualize_data.py
```

Generate curves from an existing training log:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python plot_training_log.py --log outputs/logs/resnet18_kl_finetune.csv
```

Run the four KL backbone experiments:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python run_experiments.py configs/resnet18_kl.yaml configs/wrn28_2_kl.yaml configs/densenet_bc_kl.yaml configs/vgg13_bn_kl.yaml
```

This command now runs the document's main pipeline for each config:

```text
CIFAR-10 pretraining -> CIFAR-10H fine-tuning -> test evaluation -> outputs/tables/summary.csv
```
