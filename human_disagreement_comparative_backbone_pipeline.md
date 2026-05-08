# Predicting Human Annotator Disagreement — Comparative CNN Backbone Pipeline

## 0. Project Goal

The goal is to build a deep neural network that predicts the **full human annotator label distribution** for a CIFAR-10 image, instead of predicting only one hard class.

For an input image \(x\), the model predicts:

\[
q_\theta(y|x) \in \mathbb{R}^{10}
\]

where the 10 values correspond to:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

The target is the CIFAR-10H human annotator distribution:

\[
p_{\text{hum}}(y|x) \in \mathbb{R}^{10}
\]

The model is successful if:

\[
q_\theta(y|x) \approx p_{\text{hum}}(y|x)
\]

This is **not** a standard CIFAR-10 classifier. The model should learn which classes humans confuse with each other and how much disagreement an image is likely to produce.

---

## 1. Dataset Strategy

### 1.1 Datasets

Use two datasets:

| Dataset | Size | Label Type | Use |
|---|---:|---|---|
| CIFAR-10 train | 50,000 images | hard labels | backbone pretraining |
| CIFAR-10H | 10,000 images | soft human distributions | disagreement prediction |

### 1.2 CIFAR-10H Split

Use a fixed seed and split CIFAR-10H as:

| Split | Images |
|---|---:|
| Train | 6,000 |
| Validation | 2,000 |
| Test | 2,000 |

Example:

```python
SEED = 42
```

### 1.3 Critical Rule

Do **not** treat CIFAR-10 hard labels as soft-label disagreement targets.

Correct usage:

```text
CIFAR-10 hard labels → pretraining / representation learning
CIFAR-10H soft labels → final training objective
```

---

## 2. Data Pipeline

The code should implement the following pipeline.

### 2.1 Load CIFAR-10

Load CIFAR-10 train and test images using torchvision.

```python
torchvision.datasets.CIFAR10(...)
```

### 2.2 Load CIFAR-10H

Load CIFAR-10H annotator distributions from file.

Each CIFAR-10H target should be a 10-dimensional vector:

```python
soft_target.shape == (10,)
soft_target.sum() ≈ 1.0
```

### 2.3 Alignment Check

CIFAR-10H corresponds to the CIFAR-10 **test set** images.

Required sanity checks:

```python
assert len(cifar10h_targets) == 10000
assert image_index_i in CIFAR10_test aligns with cifar10h_distribution_i
assert torch.allclose(soft_target.sum(), torch.tensor(1.0), atol=1e-5)
```

### 2.4 Data Augmentation

Use light CIFAR-safe augmentations:

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

For validation and test:

```python
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

Avoid augmentations that alter class semantics too strongly.

---

## 3. Data-Stage Visualizations

Before training, generate:

1. Histogram of true entropy values.
2. Per-class average entropy plot.
3. Confusion-style matrix from annotator distributions.
4. Example grid of low-entropy and high-entropy images with their distributions.

Entropy:

\[
H(p) = - \sum_{c=1}^{10} p_c \log_2(p_c)
\]

---

## 4. Comparative Backbone Study

The project will compare four CNN backbones under the same prediction-head and training protocol.

The four backbones are:

1. CIFAR-ResNet18
2. WideResNet-28-2
3. DenseNet-BC
4. VGG-13-BN

The prediction head and loss setup should remain consistent so that the comparison mainly reflects backbone differences.

---

# 5. Backbone 1 — CIFAR-ResNet18

## 5.1 Architecture

Use a ResNet-18 adapted for CIFAR-10.

Do **not** use the ImageNet stem.

### CIFAR Stem

```text
Conv3x3, stride 1, padding 1, 64 channels
BatchNorm
ReLU
No maxpool
```

### Feature Stages

| Stage | Blocks | Channels | Output Size |
|---|---:|---:|---|
| Stem | - | 64 | 32×32 |
| Layer 1 | 2 residual blocks | 64 | 32×32 |
| Layer 2 | 2 residual blocks | 128 | 16×16 |
| Layer 3 | 2 residual blocks | 256 | 8×8 |
| Layer 4 | 2 residual blocks | 512 | 4×4 |
| Pool | Global average pooling | 512 | 1×1 |

## 5.2 Justification

CIFAR-ResNet18 is the main baseline backbone because it is simple, well understood, and strong for small images. Residual connections stabilize optimization and allow deeper feature learning without making the model unnecessarily large.

The CIFAR-specific stem preserves spatial information in 32×32 images. A standard ImageNet ResNet stem would downsample too aggressively and may remove fine ambiguity cues such as texture, object boundary, blur, or small parts.

## 5.3 Expected Behavior

ResNet18 should provide a strong balance between:

```text
capacity
training stability
interpretability
compute cost
```

It is the safest final model candidate.

---

# 6. Backbone 2 — WideResNet-28-2

## 6.1 Architecture

Use a CIFAR-style WideResNet with depth 28 and widen factor 2.

Suggested configuration:

```text
WideResNet-28-2
depth = 28
widen_factor = 2
dropout = 0.0 or 0.1
```

Typical stage structure:

| Stage | Channels | Output Size |
|---|---:|---|
| Initial Conv3x3 | 16 | 32×32 |
| Wide Block 1 | 32 | 32×32 |
| Wide Block 2 | 64 | 16×16 |
| Wide Block 3 | 128 | 8×8 |
| Global Average Pool | 128 | 1×1 |

## 6.2 Justification

WideResNet is included because widening residual blocks often works very well on CIFAR-style datasets. Compared to simply making a network deeper, width can improve representation quality while keeping optimization manageable.

This is useful for disagreement prediction because the model may need richer features to distinguish subtle ambiguity patterns, such as:

```text
cat vs dog
deer vs horse
truck vs automobile
airplane vs ship
```

WideResNet-28-2 is chosen instead of a very large WideResNet-28-10 because CIFAR-10H has only 6,000 soft-label training examples. WRN-28-10 may be too parameter-heavy unless strong regularization and pretraining are used.

## 6.3 Expected Behavior

WideResNet may perform best on distribution-matching metrics if the additional width helps model class-confusion structure. However, it may overfit more than ResNet18 if regularization is weak.

---

# 7. Backbone 3 — DenseNet-BC

## 7.1 Architecture

Use a compact CIFAR-style DenseNet-BC.

Suggested configuration:

```text
DenseNet-BC
depth = 40 or 100
growth_rate = 12
compression = 0.5
```

For practical compute, start with:

```text
DenseNet-BC-40-12
```

If compute allows, compare with:

```text
DenseNet-BC-100-12
```

DenseNet uses dense connectivity:

```text
each layer receives feature maps from all earlier layers in the same dense block
```

## 7.2 Justification

DenseNet is included because feature reuse is useful when the dataset is small. Since each layer has access to earlier features, DenseNet can combine low-level texture information and higher-level semantic information efficiently.

This matters for human disagreement because ambiguity can arise from both:

```text
low-level causes: blur, low resolution, color, texture
high-level causes: object identity, pose, class boundary
```

DenseNet may therefore be especially useful for modeling the source of ambiguity.

## 7.3 Expected Behavior

DenseNet may generalize well under limited soft-label supervision because of feature reuse and parameter efficiency. It may be strong on entropy correlation and high-disagreement image ranking.

---

# 8. Backbone 4 — VGG-13-BN

## 8.1 Architecture

Use a VGG-style CNN with batch normalization.

Suggested configuration:

```text
VGG-13-BN adapted for CIFAR-10
```

Structure:

```text
Conv blocks with 3x3 convolutions
BatchNorm
ReLU
MaxPool between stages
Global Average Pooling or compact classifier head
```

Avoid the original huge VGG classifier designed for ImageNet. Use a compact CIFAR classifier/head.

## 8.2 Justification

VGG-13-BN is included as a plain convolutional baseline without residual or dense skip connections. This gives the comparative study a useful contrast.

If ResNet, WideResNet, and DenseNet outperform VGG, we can argue that skip connections and feature reuse help model human uncertainty. If VGG performs competitively, then the task may depend more on local visual features and calibration than on advanced connectivity.

Batch normalization is important because plain VGG without BN can be harder to optimize.

## 8.3 Expected Behavior

VGG-13-BN is expected to be stable but may underperform residual and dense models on the final metrics. It is useful as a simpler CNN baseline for the report.

---

# 9. Shared Prediction Head

To keep the backbone comparison fair, use the same prediction head for all backbones.

Each backbone should output a feature vector:

```python
features = backbone(x)
```

Then use:

```text
Linear(feature_dim → 256)
BatchNorm1d(256)
ReLU
Dropout(0.30)
Linear(256 → 10)
Temperature-scaled Softmax
```

## 9.1 Head Output

The final output is:

\[
q_\theta(y|x) = \text{softmax}(z/T)
\]

where:

```text
z = 10-dimensional logits
T = temperature parameter
```

Use either:

1. fixed temperature, e.g. \(T = 1.5\), or
2. learnable temperature, constrained to be positive.

Recommended implementation:

```python
temperature = torch.exp(log_temperature)
q = torch.softmax(logits / temperature, dim=-1)
```

## 9.2 Justification

The MLP head is used because predicting human disagreement is more complex than standard hard classification. The model must learn how to distribute probability mass across visually plausible classes.

The temperature-scaled softmax helps calibrate the sharpness of the predicted distribution. This is important because human labels are often softer than one-hot labels.

---

# 10. Backbone Initialization Strategy

Run the comparative study under the same initialization strategy.

## Recommended Main Strategy

Use:

```text
CIFAR-10 hard-label pretraining
→ CIFAR-10H soft-label fine-tuning
```

### Stage 1: Hard-label pretraining

Train each backbone on the 50,000 CIFAR-10 training images using ordinary cross-entropy.

Objective:

\[
\mathcal{L}_{CE} = - \log q_\theta(y_{\text{hard}}|x)
\]

Purpose:

```text
learn general CIFAR visual features
```

### Stage 2: Soft-label fine-tuning

Fine-tune the pretrained backbone and prediction head on CIFAR-10H using soft-label losses.

Purpose:

```text
learn human disagreement structure
```

## Optional Ablation

Compare:

1. random initialization,
2. CIFAR-10 hard-label pretraining,
3. ImageNet pretraining where available.

But the main comparison should use the same initialization strategy for all four backbones.

---

# 11. Loss Functions

The assignment requires at least:

1. KL divergence,
2. one additional standard loss,
3. one task-specific custom/composite loss.

Use the same losses for all backbones.

---

## 11.1 Loss 1 — KL Divergence

Primary baseline.

\[
\mathcal{L}_{KL}
=
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{c=1}^{10}
p_i(c)
\log
\frac{p_i(c)}{q_i(c)}
\]

Plain meaning:

```text
penalizes the model when its predicted distribution differs from the human distribution
```

This is the main distribution-matching objective.

---

## 11.2 Loss 2 — Jensen-Shannon Divergence

Standard additional loss.

\[
m_i = \frac{1}{2}(p_i + q_i)
\]

\[
\mathcal{L}_{JSD}
=
\frac{1}{2} KL(p_i || m_i)
+
\frac{1}{2} KL(q_i || m_i)
\]

Plain meaning:

```text
a symmetric and bounded version of KL divergence
```

JSD is useful because it is less harsh than KL when the model assigns very small probability to a class that humans selected.

---

## 11.3 Loss 3 — KL + Entropy Matching

Task-specific custom/composite loss.

\[
\mathcal{L}_{custom}
=
KL(p_i || q_i)
+
\lambda
\left(H(p_i) - H(q_i)\right)^2
\]

where:

\[
H(p_i) = - \sum_{c=1}^{10} p_i(c) \log_2 p_i(c)
\]

Recommended:

```python
lambda_entropy = 0.1
```

Plain meaning:

```text
match the full distribution and also match the amount of human disagreement
```

This is task-specific because the assignment evaluates entropy prediction quality using Pearson and Spearman correlation.

---

# 12. Training Protocol

Use a consistent protocol across all backbone-loss combinations.

## 12.1 Optimizer

Recommended:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
```

For fine-tuning, optionally use a smaller learning rate for the backbone:

```python
backbone_lr = 1e-4
head_lr = 3e-4
```

## 12.2 Scheduler

Use cosine annealing or ReduceLROnPlateau.

Recommended simple choice:

```python
torch.optim.lr_scheduler.CosineAnnealingLR(...)
```

## 12.3 Batch Size

Recommended:

```python
batch_size = 128
```

If GPU memory is limited:

```python
batch_size = 64
```

## 12.4 Epochs

Pretraining:

```text
100 to 200 epochs on CIFAR-10 hard labels
```

Soft-label fine-tuning:

```text
50 to 100 epochs on CIFAR-10H
```

Use early stopping on validation KL/JSD.

## 12.5 Early Stopping

Monitor:

```text
validation KL divergence
```

Suggested patience:

```python
patience = 10
```

## 12.6 Logging

Log the following every epoch:

```text
train loss
validation loss
validation KL
validation JSD
validation cosine similarity
validation entropy Pearson correlation
validation entropy Spearman correlation
learning rate
```

Save:

```text
best checkpoint by validation KL
final checkpoint
training curves
```

---

# 13. Evaluation Metrics

Evaluate each trained model on the held-out 2,000-image CIFAR-10H test set.

## 13.1 Distribution Matching

Report mean ± standard deviation:

```text
KL divergence
Jensen-Shannon divergence
Cosine similarity
```

## 13.2 Entropy Prediction Quality

For every test image compute:

```text
true entropy H(p)
predicted entropy H(q)
```

Report:

```text
Pearson correlation
Spearman correlation
```

## 13.3 Precision@K

Rank images by true entropy and predicted entropy.

Report:

```text
Precision@100
Precision@200
Precision@500
```

This measures whether the model can identify the most disagreement-prone images.

## 13.4 Summary Table

Create one table:

```text
Backbone × Loss Function
```

Columns:

```text
KL ↓
JSD ↓
Cosine ↑
Pearson entropy ↑
Spearman entropy ↑
P@100 ↑
P@200 ↑
P@500 ↑
Params
Training time
```

---

# 14. Required Experiments

## 14.1 Main Comparative Study

Train all four backbones with the same final prediction head and same training protocol.

Backbones:

```text
CIFAR-ResNet18
WideResNet-28-2
DenseNet-BC-40-12
VGG-13-BN
```

Losses:

```text
KL
JSD
KL + Entropy Matching
```

This gives:

```text
4 backbones × 3 losses = 12 main runs
```

If time is limited, do:

```text
4 backbones × KL
best 2 backbones × JSD
best 2 backbones × custom loss
```

## 14.2 Prediction Head Ablation

For the best backbone, compare:

1. Linear + Softmax
2. MLP + Softmax
3. MLP + Temperature-scaled Softmax

## 14.3 Initialization Ablation

For the best backbone, compare:

1. random initialization,
2. CIFAR-10 hard-label pretraining,
3. ImageNet pretraining if available.

## 14.4 Training Data Strategy Ablation

Compare:

1. soft-label training only,
2. hard-label pretraining followed by soft-label fine-tuning.

---

# 15. Robustness Checks

Complete at least two.

## 15.1 Annotator Subsampling

Recompute soft-label distributions using fewer annotators if raw annotator labels are available.

Compare:

```text
5 annotators
10 annotators
25 annotators
full annotator set
```

Measure stability of predicted distributions.

## 15.2 Corruption Response

Apply corruptions to test images:

```text
Gaussian noise
Gaussian blur
contrast reduction
```

For each severity, measure predicted entropy.

Expected behavior:

```text
predicted entropy should generally increase as corruption severity increases
```

If it does not, analyze why.

## 15.3 Class-Conditional Performance

Compute metrics per majority class.

Report whether the model is better at predicting disagreement for some classes than others.

---

# 16. Explainability and Analysis

## 16.1 Grad-CAM

For the best model, generate Grad-CAM heatmaps for:

```text
low-disagreement images
medium-disagreement images
high-disagreement images
```

Analyze what the model focuses on.

## 16.2 Failure Case Analysis

Identify images where:

\[
|H(p) - H(q)|
\]

is large.

For each failure case, show:

```text
image
true distribution
predicted distribution
true entropy
predicted entropy
hypothesis for failure
```

## 16.3 Manual Disagreement Source Analysis

Manually inspect high-entropy images and categorize likely reasons:

```text
ambiguous object identity
poor image quality
multi-object or multi-label content
class boundary case
occlusion
background confusion
other
```

---

# 17. Code Structure to Generate

The codebase should be generated with the following structure.

```text
human-disagreement-cifar/
│
├── configs/
│   ├── resnet18_kl.yaml
│   ├── wrn28_2_kl.yaml
│   ├── densenet_bc_kl.yaml
│   ├── vgg13_bn_kl.yaml
│   ├── best_jsd.yaml
│   └── best_custom.yaml
│
├── data/
│   ├── cifar10h/
│   └── README.md
│
├── src/
│   ├── datasets/
│   │   ├── cifar10h_dataset.py
│   │   └── splits.py
│   │
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── cifar_resnet.py
│   │   │   ├── wide_resnet.py
│   │   │   ├── densenet_cifar.py
│   │   │   └── vgg_cifar.py
│   │   ├── heads.py
│   │   └── model_factory.py
│   │
│   ├── losses/
│   │   ├── kl_loss.py
│   │   ├── jsd_loss.py
│   │   └── entropy_kl_loss.py
│   │
│   ├── metrics/
│   │   ├── distribution_metrics.py
│   │   ├── entropy_metrics.py
│   │   └── precision_at_k.py
│   │
│   ├── training/
│   │   ├── pretrain_cifar10.py
│   │   ├── finetune_cifar10h.py
│   │   ├── trainer.py
│   │   └── early_stopping.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── compare_backbones.py
│   │   └── robustness.py
│   │
│   ├── visualization/
│   │   ├── data_viz.py
│   │   ├── training_curves.py
│   │   ├── entropy_scatter.py
│   │   ├── comparison_plots.py
│   │   └── gradcam.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── checkpoint.py
│       ├── logging.py
│       └── config.py
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── tables/
│
├── notebooks/
│   ├── 01_data_sanity_checks.ipynb
│   ├── 02_entropy_analysis.ipynb
│   └── 03_failure_case_analysis.ipynb
│
├── train_pretrain.py
├── train_finetune.py
├── evaluate.py
├── run_experiments.py
├── requirements.txt
└── README.md
```

---

# 18. Model Factory Requirements

The generated code should support:

```python
model = build_model(
    backbone_name="resnet18",
    head_name="temperature_mlp",
    num_classes=10,
    pretrained_checkpoint=None,
)
```

Valid backbone names:

```python
"resnet18"
"wideresnet28_2"
"densenet_bc_40_12"
"vgg13_bn"
```

Valid head names:

```python
"linear"
"mlp"
"temperature_mlp"
```

---

# 19. Experiment Configuration Format

Each experiment should be controlled by YAML.

Example:

```yaml
experiment_name: resnet18_kl

seed: 42

dataset:
  name: cifar10h
  train_size: 6000
  val_size: 2000
  test_size: 2000
  batch_size: 128
  num_workers: 4

model:
  backbone: resnet18
  head: temperature_mlp
  num_classes: 10
  dropout: 0.30
  temperature:
    type: learnable
    init: 1.5

training:
  stage: finetune
  epochs: 100
  optimizer: adamw
  lr_backbone: 0.0001
  lr_head: 0.0003
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping:
    monitor: val_kl
    patience: 10
    mode: min

loss:
  name: kl

logging:
  save_dir: outputs/
  save_best: true
```

---

# 20. Final Recommendation

Use the following four-backbone comparative study:

| Backbone | Role in Study | Main Reason |
|---|---|---|
| CIFAR-ResNet18 | Main baseline | balanced, stable, easy to justify |
| WideResNet-28-2 | stronger residual model | tests whether width improves human-distribution modeling |
| DenseNet-BC-40-12 | feature reuse model | useful for small data and mixed low/high-level ambiguity |
| VGG-13-BN | plain CNN baseline | tests whether skip connections are actually useful |

The expected final winner is likely one of:

```text
CIFAR-ResNet18
WideResNet-28-2
DenseNet-BC-40-12
```

The final selection should be based on the validation and test metrics, especially:

```text
KL divergence
JSD
entropy Spearman correlation
Precision@K for high-disagreement images
```

The most reportable outcome would be:

```text
WideResNet or DenseNet achieves best raw distribution matching,
while ResNet18 gives the best performance-compute tradeoff.
```

The code should be generated around this document as the implementation specification.
