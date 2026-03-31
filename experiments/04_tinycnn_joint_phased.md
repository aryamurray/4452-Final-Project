# Experiment 03: TinyCNNJoint Phased Training (96x96, CBAM)

## Overview

Flagship model with multi-task learning via three-phase training strategy.
Uses pretrained backbone from Experiment 02 to avoid the problem of easier auxiliary
tasks (500 radicals, stroke count) dominating gradients and hurting the harder primary
task (3,755-class character classification).

## Phased Training Strategy

### Phase 1: Load Pretrained Backbone
- Load backbone + char_head weights from Experiment 02 checkpoint
- No training in this phase — just weight initialization

### Phase 2: Train Auxiliary Heads (backbone + char_head frozen)
- **Frozen**: backbone (396,610 params), char_head (963,580 params)
- **Trainable**: radical_head + structure_head + stroke_head (132,098 params)
- **Epochs**: 10
- **LR**: 1e-3
- **Loss**: `1.0*BCE(radical) + 0.5*CE(structure) + 0.1*MSE(stroke)` (+ frozen char CE for monitoring)
- **Rationale**: Aux heads learn to read radical/stroke information from already-good features

### Phase 3: Joint Fine-tune (all params unfrozen)
- **Trainable**: All 1,493,743 params
- **Epochs**: 5
- **LR**: 1e-4 (1/10th of Phase 2)
- **Loss**: `1.0*CE(char) + 0.3*BCE(radical) + 0.1*CE(structure) + 0.1*MSE(stroke)`
- **Rationale**: Backbone slightly adjusts to support aux tasks without forgetting char classification

## Model Architecture

**TinyCNNJoint + CBAM**:

```
Input: (B, 1, 96, 96)
Backbone:
  Conv2d(1, 32, 3) -> BN -> ReLU -> MaxPool(2)    # 96 -> 48
  Conv2d(32, 64, 3) -> BN -> ReLU -> MaxPool(2)    # 48 -> 24
  Conv2d(64, 128, 3) -> BN -> ReLU -> MaxPool(2)   # 24 -> 12
  Conv2d(128, 256, 3) -> BN -> ReLU -> MaxPool(2)  # 12 -> 6
  CBAM(256)                                          # 6x6 attention
  AdaptiveAvgPool2d(1) -> Flatten                    # -> 256-dim

Heads (from 256-dim features):
  Dropout(0.5) ->
    char_head:      Linear(256, 3755)  # character classification
    radical_head:   Linear(256, N_rad) # multi-label radical prediction (sigmoid)
    structure_head: Linear(256, 13)    # structural layout classification
    stroke_head:    Linear(256, 1)     # stroke count regression
```

| Component | Parameters |
|-----------|-----------|
| Backbone + CBAM | 396,610 |
| char_head | 963,580 |
| radical_head | ~128,500 (depends on N_rad) |
| structure_head | 3,341 |
| stroke_head | 257 |
| **Total** | **~1,493,743** |

## Training Parameters

### Phase 2

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (aux heads only) |
| LR | 1e-3 |
| Epochs | 10 |
| lambda_radical | 1.0 |
| lambda_structure | 0.5 |
| lambda_stroke | 0.1 |
| Other | Same as Exp 02 (batch=256, AMP, etc.) |

### Phase 3

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (all params) |
| LR | 1e-4 |
| Epochs | 5 |
| lambda_radical | 0.3 |
| lambda_structure | 0.1 |
| lambda_stroke | 0.1 |
| Other | Same as Exp 02 |

## Results

### Phase 2 (Aux Head Training)

| Epoch | Train Loss | Val Loss | Val Acc | Radical Loss | Structure Loss | Stroke Loss |
|-------|-----------|----------|---------|-------------|----------------|-------------|
| | | | | | | |

### Phase 3 (Joint Fine-tune)

| Epoch | Train Loss | Val Loss | Val Acc | Char Loss | Radical Loss | Structure Loss | Stroke Loss |
|-------|-----------|----------|---------|-----------|-------------|----------------|-------------|
| | | | | | | | |

*(Not yet run — waiting for Experiment 02 checkpoint)*

## Expected Outcome

- Phase 2: Val accuracy stays near Exp 02 level (char_head frozen), aux losses decrease
- Phase 3: Val accuracy improves 1-3% over Exp 02 due to radical task regularization
- Target: 88-92% top-1 accuracy

## Command

```bash
uv run python scripts/train_tinycnn_joint.py \
  --pretrained-checkpoint outputs/tinycnn/best_model_classification.pt \
  --batch-size 256
```

(Defaults: --image-size 96, --use-cbam, --phase2-epochs 10, --phase3-epochs 5)

## Checkpoint

```
outputs/tinycnn_joint/phase3/best_model_joint.pt
```

## Paper Reporting

For fair isolated character recognition comparison (vs Xiao et al.):
- Report Exp 01 (baseline, 64x64): establishes lower bound
- Report Exp 02 (improved baseline, 96x96+CBAM): shows architecture improvement
- Report Exp 03 (joint + structural): shows multi-task regularization benefit
- All without bigram context — bigrams reported separately for word-level tasks
