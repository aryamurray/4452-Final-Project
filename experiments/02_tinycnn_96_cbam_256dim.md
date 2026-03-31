# Experiment 02: TinyCNN 96x96 + CBAM + 256-dim (ABANDONED)

## Overview

Attempted improvement over Experiment 01 with two changes:
1. **Input resolution**: 64x64 -> 96x96 (2.25x more pixels, feature map 6x6 before GAP vs 4x4)
2. **CBAM attention**: Channel + Spatial attention module before GAP (~8K extra params)

**Result: Significantly underperformed the 64x64 baseline.** Killed at epoch 14 with 68% val
accuracy vs 85% for the 64x64 run at the same epoch. The 256-dim bottleneck cannot compress
the richer 96x96 spatial information effectively. CBAM's random gating also slowed early
convergence.

## Hardware

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce GTX 1060 6GB |
| CUDA | 12.6 (driver), PyTorch 2.10.0+cu126 |

## Dataset

Same as Experiment 01 (HWDB1.1, 3,755 classes, ~750K train, ~188K val).

Data loading: Preloaded as uint8 `[N, 1, 96, 96]` tensors (~6.5 GB RAM).

## Model Architecture

**TinyCNN + CBAM (256-dim)**:

```
Input: (B, 1, 96, 96)
Conv2d(1, 32, 3, pad=1) -> BN -> ReLU -> MaxPool(2)    # 96 -> 48
Conv2d(32, 64, 3, pad=1) -> BN -> ReLU -> MaxPool(2)    # 48 -> 24
Conv2d(64, 128, 3, pad=1) -> BN -> ReLU -> MaxPool(2)   # 24 -> 12
Conv2d(128, 256, 3, pad=1) -> BN -> ReLU -> MaxPool(2)  # 12 -> 6
CBAM(256):                                                # 6x6 attention
  ChannelAttention: GAP+GMP -> MLP(256->16->256) -> sigmoid
  SpatialAttention: cat(avg,max) -> Conv2d(2,1,7) -> sigmoid
AdaptiveAvgPool2d(1)                                      # 6 -> 1
Flatten -> Dropout(0.5) -> Linear(256, 3755)
```

| Component | Parameters |
|-----------|-----------|
| Backbone (conv blocks) | 388,320 |
| CBAM | 8,290 |
| Classifier (Linear 256->3755) | 963,580 |
| **Total** | **1,360,190** |

## Training Parameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR schedule | CosineAnnealingLR (T_max=30, eta_min=1e-6) |
| Batch size | 256 |
| Gradient accumulation | 2 steps |
| Weight decay | 1e-4 |
| Dropout | 0.5 |
| Label smoothing | 0.1 |
| Image size | **96x96** |
| CBAM | **Enabled** |
| Augmentation | RandomRotation(10 deg) |
| Normalization | mean=0.5, std=0.5 |
| AMP | Enabled |
| Seed | 42 |

## Results (killed at epoch 14)

| Epoch | Train Loss | Val Loss | Val Acc | LR |
|-------|-----------|----------|---------|-----|
| 1 | 6.4417 | 5.1588 | 0.1705 | 0.001000 |
| 2 | 4.7762 | 4.1391 | 0.3593 | 0.000997 |
| 3 | 4.1629 | 3.8407 | 0.4019 | 0.000989 |
| 4 | 3.8445 | 3.3653 | 0.5291 | 0.000976 |
| 5 | 3.6504 | 3.2164 | 0.5578 | 0.000957 |
| 6 | 3.5093 | 3.1415 | 0.5679 | 0.000933 |
| 7 | 3.4051 | 3.0420 | 0.5950 | 0.000905 |
| 8 | 3.3255 | 2.9174 | 0.6251 | 0.000872 |
| 9 | 3.2572 | 2.9443 | 0.6047 | 0.000835 |
| 10 | 3.1973 | 2.8405 | 0.6385 | 0.000794 |
| 11 | 3.1503 | 2.7775 | 0.6591 | 0.000750 |
| 12 | 3.1019 | 2.7735 | 0.6533 | 0.000704 |
| 13 | 3.0629 | 2.7408 | 0.6657 | 0.000655 |
| 14 | 3.0234 | 2.6873 | 0.6802 | 0.000604 |

## Comparison with Experiment 01 (64x64)

| Epoch | Exp 01 (64x64) | Exp 02 (96x96+CBAM) | Gap |
|-------|---------------|---------------------|-----|
| 1 | 43.8% | 17.1% | -26.7 |
| 4 | 79.6% | 52.9% | -26.7 |
| 6 | 82.0% | 56.8% | -25.2 |
| 10 | 83.9% | 63.9% | -20.0 |
| 13 | 85.0% | 66.6% | -18.4 |
| 14 | — | 68.0% | — |

## Timing

- ~7 minutes per epoch (vs ~3.5 for 64x64)
- Total runtime: ~98 minutes (14 epochs)

## Analysis

1. **256-dim bottleneck is the primary issue.** The model extracts 2.25x more spatial info
   from 96x96 input but crushes it into the same 256 dimensions. The richer input actually
   hurts because the bottleneck can't represent it.

2. **CBAM adds overhead without sufficient capacity.** CBAM helps select what to attend to,
   but it can't expand the information capacity of the 256-dim embedding. Early on, the
   random attention gating actively hurts convergence.

3. **Two variables changed simultaneously.** Cannot isolate whether the regression is from
   96x96 input, CBAM, or their interaction with the 256-dim bottleneck.

## Decision

Abandoned in favor of Experiment 03: 96x96 + 512-dim backbone (no CBAM).
CBAM may be revisited later with a wider backbone where the bottleneck isn't the
limiting factor.
