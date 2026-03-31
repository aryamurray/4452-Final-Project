# Experiment 01: TinyCNN Baseline (64x64, No CBAM)

## Overview

First baseline run of the TinyCNN character classification model on CASIA-HWDB1.1.
Trained for 13 epochs before manual termination (plateauing at ~85% val accuracy).

## Hardware

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce GTX 1060 6GB |
| CUDA | 12.6 (driver), PyTorch 2.10.0+cu126 |
| OS | Windows (MINGW32) |

## Dataset

| Split | Samples | Source |
|-------|---------|--------|
| Train | ~750,000 | CASIA-HWDB1.1 train (.gnt files) |
| Val | ~188,000 | 20% split from train (by .gnt filename, seed=42) |
| Classes | 3,755 (GB2312-L1) | |

Data loading: All samples preloaded as uint8 tensors `[N, 1, 64, 64]` into RAM (~3 GB),
normalized on-the-fly in `__getitem__` to float32 `[-1, 1]`.

## Model Architecture

**TinyCNN** (classification-only baseline):

```
Input: (B, 1, 64, 64)
Conv2d(1, 32, 3, pad=1) -> BN -> ReLU -> MaxPool(2)   # 64 -> 32
Conv2d(32, 64, 3, pad=1) -> BN -> ReLU -> MaxPool(2)   # 32 -> 16
Conv2d(64, 128, 3, pad=1) -> BN -> ReLU -> MaxPool(2)  # 16 -> 8
Conv2d(128, 256, 3, pad=1) -> BN -> ReLU -> MaxPool(2) # 8  -> 4
AdaptiveAvgPool2d(1)                                     # 4  -> 1
Flatten -> Dropout(0.5) -> Linear(256, 3755)
```

| Component | Parameters |
|-----------|-----------|
| Backbone | 388,320 |
| Classifier (Linear 256->3755) | 963,580 |
| **Total** | **1,351,900** |
| CBAM | Not used |

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
| Image size | 64x64 |
| Augmentation | RandomRotation(10 deg) |
| Normalization | mean=0.5, std=0.5 (maps to [-1, 1]) |
| Early stopping patience | 5 epochs |
| AMP (mixed precision) | Enabled (fp16 forward/backward) |
| cuDNN benchmark | Enabled |
| num_workers | 0 (preloaded data) |
| Seed | 42 |

## Results

| Epoch | Train Loss | Val Loss | Val Acc | LR |
|-------|-----------|----------|---------|-----|
| 1 | 6.2990 | 4.2320 | 0.4383 | 0.001000 |
| 2 | 3.8782 | 3.0825 | 0.6941 | 0.000997 |
| 3 | 3.2890 | 2.7951 | 0.7577 | 0.000989 |
| 4 | 3.0642 | 2.5855 | 0.7960 | 0.000976 |
| 5 | 2.9415 | 2.5177 | 0.8103 | 0.000957 |
| 6 | 2.8628 | 2.4414 | 0.8197 | 0.000933 |
| 7 | 2.8092 | 2.4168 | 0.8297 | 0.000905 |
| 8 | 2.7675 | 2.3988 | 0.8256 | 0.000872 |
| 9 | 2.7361 | 2.3499 | 0.8382 | 0.000835 |
| 10 | 2.7089 | 2.3508 | 0.8392 | 0.000794 |
| 11 | 2.6867 | 2.3328 | 0.8429 | 0.000750 |
| 12 | 2.6675 | 2.3056 | 0.8472 | 0.000704 |
| 13 | 2.6515 | 2.2815 | 0.8502 | 0.000655 |

**Best checkpoint**: Epoch 13, Val Acc = 85.02%

Training manually stopped at epoch 14 (gains slowing to ~0.3%/epoch, projected plateau ~86-87%).

## Timing

- ~3.5 minutes per epoch
- Total runtime: ~45 minutes (13 epochs)

## Observations

1. Very fast initial convergence: 44% -> 69% -> 76% in first 3 epochs.
2. Gains slowed significantly after epoch 7 (~83%), dropping to ~0.3%/epoch.
3. Small dip at epoch 8 (82.56% vs 82.97% at epoch 7), recovered by epoch 9.
4. GPU utilization remained low (~1-6%) due to model being too small to saturate the GPU.
   The bottleneck was CPU-side operations, not GPU compute.
5. No structural/radical supervision — this is pure character classification as a baseline.
6. Loss values appear high (~2.3 at epoch 13) because cross-entropy over 3,755 classes
   has a theoretical minimum around ln(1) = 0 for perfect predictions, but label smoothing
   (0.1) prevents the loss from reaching zero.

## Checkpoint

```
outputs/tinycnn/best_model_classification.pt
```

**Note**: This checkpoint uses 64x64 input and no CBAM. It is NOT architecture-compatible
with the improved models (96x96 + CBAM). A new baseline must be trained with matching
architecture before phased joint training.

## Next Steps

- Retrain baseline at 96x96 with CBAM attention (Experiment 02)
- Use that checkpoint as pretrained backbone for TinyCNNJoint phased training (Experiment 03)
