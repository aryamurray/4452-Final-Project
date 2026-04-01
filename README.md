# Handwritten Chinese Character Recognition with Neurosymbolic Constraints

Final project for CS 4452. Explores offline handwritten Chinese character recognition (HCCR) on the CASIA-HWDB1.1 dataset across a progression of architectures — from a plain CNN baseline to a neurosymbolic model that injects structural priors (radicals, stroke counts, spatial layout) as fixed lookup-table constraints during training.

---

## Results Summary

| Model | Val Top-1 | Val Top-5 | Val Top-10 | Params |
|---|---|---|---|---|
| TinyCNN 64×64 (baseline) | 85.0% | — | — | 1.4M |
| TinyCNN 96×96 + CBAM (abandoned) | 68.0%† | — | — | 1.4M |
| TinyCNN 96×96 + 512-dim + aug | 83.1% | 94.2% | 96.2% | 2.7M |
| MiniResNetJoint — phase 1 (multi-task) | **95.3%** | 99.1% | 99.5% | ~7.7M |
| MiniResNetJoint — phase 2 (+ symbolic) | 95.2% | 99.1% | 99.5% | ~7.7M |
| MiniResNetJoint — phase 3 (reranker) | 94.9% | — | — | ~7.7M |

†Killed at epoch 14 due to 256-dim bottleneck with richer input. See `experiments/02_tinycnn_96_cbam_256dim.md`.

---

## Architecture

### Neurosymbolic Model (`hccr/models/mini_restnet_join.py`)

The flagship model is `MiniResNetJoint`, a multi-head ResNet trained in three phases:

```
Input: (B, 1, 64×64)
  ↓ MiniResNetBackbone
    Conv(1→64) → Conv(64→128) → Conv(128→256) → 8×8 spatial + 256-dim global
  ↓ 7 task heads
    char           → CE loss       (3755-class classification)
    radical        → BCE loss      (multi-label radical presence)
    stroke_count   → MSE loss      (regression)
    stroke_types   → BCE loss      (multi-label)
    structure      → CE loss       (13-class spatial layout)
    density        → MSE loss      (3×3 grid density profile)
    quadrant_radical → CE loss     (per-quadrant radical assignment)
  ↓ SymbolicConstraintLayer (0 learned params)
    Fixed lookup tables → re-ranking logits
  ↓ SymbolicReranker MLP
    Blends neural + symbolic scores
```

**3-phase training** via `PhaseManager`:
- **Phase 1** — all heads trained together (backbone + all heads)
- **Phase 2** — symbolic tables loaded; reranker MLP trained alongside heads
- **Phase 3** — backbone frozen; only reranker MLP fine-tuned

### Older baseline (`hccr/models/mini_resnet.py`)

4-head model (char, radical, stroke\_count, structure) at 96×96 input / 192-dim features. Used for ablation.

---

## Repository Layout

```
hccr/
  data/           dataset, label map, transforms
  models/         mini_resnet.py, mini_restnet_join.py, backbone.py, ...
  training/       trainer.py, losses.py, early_stopping.py
  evaluation/     metrics.py, confusion.py, benchmark.py
  structural/     radical_table.py, bigram.py, radical_filter.py
resources/
  label_map.json          char → class index (3755 classes)
  radical_table.json      IDS decomposition per character
  radical_mask.pt         binary radical presence mask
  stroke_types.json       stroke type signatures
  ids.txt                 IDS decomposition source data
  strokeorder.csv         stroke ordering data
experiments/
  01_tinycnn_baseline_64.md
  02_tinycnn_96_cbam_256dim.md
  03_tinycnn_96_512dim.md
  04_tinycnn_joint_phased.md
scripts/
  train_*.py              training entry points (one per model variant)
  eval_*.py               evaluation scripts
  build_*.py              resource generation scripts
  run_all.py              full pipeline orchestrator
main.py                   CLI dispatcher
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

**Dataset**: Download CASIA-HWDB1.1 and place `.gnt` files under `data/HWDB1.1/train/` and `data/HWDB1.1/test/`.

**Build resources** (required before training):

```bash
uv run python scripts/build_label_map.py
uv run python scripts/build_radical_table.py
uv run python scripts/build_constraint_tensors.py
```

---

## Training

### Neurosymbolic model (recommended)

```bash
uv run python scripts/train_neurosymbolic.py \
    --train-dir data/HWDB1.1/train \
    --epochs-phase1 50 --epochs-phase2 30 --epochs-phase3 10 \
    --output-dir outputs/neurosymbolic
```

With a pretrained backbone (speeds up phase 1):

```bash
uv run python scripts/train_neurosymbolic.py \
    --pretrained-backbone outputs/mini_resnet/best_model_classification.pt \
    --wandb --wandb-project hccr-neurosymbolic
```

### MiniResNet classification baseline

```bash
uv run python scripts/train_mini_resnet.py \
    --train-dir data/HWDB1.1/train \
    --output-dir outputs/mini_resnet
```

### TinyCNN variants

```bash
# 64×64 baseline
uv run python scripts/train_tinycnn.py --output-dir outputs/tinycnn

# 96×96 + 512-dim + augmentation
uv run python scripts/train_tinycnn.py \
    --backbone-dim 512 \
    --output-dir outputs/tinycnn_512_aug \
    --epochs 30
```

Or run the full pipeline end-to-end:

```bash
uv run python scripts/run_all.py --data-dir data --output-dir outputs
```

---

## Evaluation

```bash
# CLIP zero-shot baseline
uv run python scripts/eval_clip_zeroshot.py \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json

# Structural post-processing (radical filtering + bigram re-ranking)
uv run python scripts/eval_structural.py \
    --test-dir data/HWDB1.1/test

# Efficiency benchmarks (latency, size, quantization)
uv run python scripts/run_benchmarks.py \
    --test-dir data/HWDB1.1/test
```

---

## CLI

`main.py` dispatches to all subcommands:

```bash
uv run python main.py train-neurosymbolic --help
uv run python main.py eval-structural --help
uv run python main.py run-benchmarks --help
```
