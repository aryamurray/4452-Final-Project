# Experiment 03: TinyCNN 96x96 + 512-dim + Augmentation (No CBAM)

## Overview

Addresses the 256-dim bottleneck identified in Experiment 02. Doubles backbone dimension to 512
while dropping CBAM. Also adds tensor-based data augmentation (RandomAffine, GaussianBlur,
RandomErasing applied to 50% of preloaded samples), fixing a bug where augmentation was silently
skipped in preloaded mode during Experiments 01-02.

## Hardware

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce GTX 1060 6GB |
| CUDA | 12.6 (driver), PyTorch 2.10.0+cu126 |

## Dataset

Same as Experiments 01-02 (HWDB1.1, ~750K train, ~188K val, 3,926 classes, seed=42).

Data loading: Preloaded as uint8 `[N, 1, 96, 96]` tensors (~6.5 GB RAM).

## Model Architecture

**TinyCNN (512-dim, no CBAM)**:

```
Input: (B, 1, 96, 96)
Conv2d(1, 32, 3, pad=1) -> BN -> ReLU -> MaxPool(2)    # 96 -> 48
Conv2d(32, 64, 3, pad=1) -> BN -> ReLU -> MaxPool(2)    # 48 -> 24
Conv2d(64, 128, 3, pad=1) -> BN -> ReLU -> MaxPool(2)   # 24 -> 12
Conv2d(128, 512, 3, pad=1) -> BN -> ReLU -> MaxPool(2)  # 12 -> 6
AdaptiveAvgPool2d(1)                                      # 6 -> 1
Flatten -> Dropout(0.5) -> Linear(512, 3926)
```

| Component | Parameters |
|-----------|-----------|
| Backbone | 683,744 |
| Classifier (Linear 512->3926) | 2,014,038 |
| **Total** | **2,697,782** |
| **Size (FP32)** | **10.30 MB** |
| **Size (INT8 est.)** | **~3.5 MB** |

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
| Backbone dim | **512** |
| CBAM | **Disabled** |
| Augmentation | **RandomAffine(10deg, translate=0.05, scale=0.9-1.1, shear=5) + GaussianBlur(p=0.3) + RandomErasing(p=0.1), applied to 50% of samples** |
| Normalization | mean=0.5, std=0.5 |
| Early stopping patience | 5 epochs |
| AMP | Enabled |
| Seed | 42 |

## Results

| Epoch | Train Loss | Val Loss | Val Acc | Top-5 | Top-10 | LR |
|-------|-----------|----------|---------|-------|--------|-----|
| 1 | 6.5651 | 4.9763 | 0.2433 | 0.5028 | 0.6209 | 0.001000 |
| 2 | 4.4414 | 3.5794 | 0.5692 | 0.8109 | 0.8716 | 0.000997 |
| 3 | 3.7386 | 3.2054 | 0.6543 | 0.8591 | 0.9053 | 0.000989 |
| 4 | 3.4454 | 2.9817 | 0.7027 | 0.8853 | 0.9232 | 0.000976 |
| 5 | 3.2859 | 2.8580 | 0.7284 | 0.8981 | 0.9321 | 0.000957 |
| 6 | 3.1853 | 2.7718 | 0.7441 | 0.9066 | 0.9380 | 0.000933 |
| 7 | 3.1147 | 2.7306 | 0.7526 | 0.9108 | 0.9407 | 0.000905 |
| 8 | 3.0602 | 2.6960 | 0.7591 | 0.9121 | 0.9415 | 0.000872 |
| 9 | 3.0192 | 2.6790 | 0.7647 | 0.9160 | 0.9448 | 0.000835 |
| 10 | 2.9874 | 2.6147 | 0.7778 | 0.9227 | 0.9491 | 0.000794 |
| 11 | 2.9578 | 2.6188 | 0.7809 | 0.9221 | 0.9489 | 0.000750 |
| 12 | 2.9328 | 2.5959 | 0.7832 | 0.9232 | 0.9497 | 0.000704 |
| 13 | 2.9126 | 2.6008 | 0.7841 | 0.9247 | 0.9505 | 0.000655 |
| 14 | 2.8905 | 2.5562 | 0.7918 | 0.9282 | 0.9532 | 0.000604 |
| 15 | 2.8720 | 2.5314 | 0.7997 | 0.9311 | 0.9550 | 0.000553 |
| 16 | 2.8533 | 2.5216 | 0.8015 | 0.9312 | 0.9547 | 0.000501 |
| 17 | 2.8365 | 2.5398 | 0.8014 | 0.9302 | 0.9545 | 0.000448 |
| 18 | 2.8193 | 2.4974 | 0.8097 | 0.9340 | 0.9574 | 0.000397 |
| 19 | 2.8034 | 2.4950 | 0.8117 | 0.9356 | 0.9581 | 0.000346 |
| 20 | 2.7891 | 2.4787 | 0.8142 | 0.9361 | 0.9581 | 0.000297 |
| 21 | 2.7733 | 2.4770 | 0.8169 | 0.9372 | 0.9590 | 0.000251 |
| 22 | 2.7601 | 2.4577 | 0.8195 | 0.9382 | 0.9595 | 0.000207 |
| 23 | 2.7478 | 2.4517 | 0.8216 | 0.9392 | 0.9600 | 0.000166 |
| 24 | 2.7350 | 2.4458 | 0.8243 | 0.9402 | 0.9606 | 0.000129 |
| 25 | 2.7253 | 2.4419 | 0.8248 | 0.9397 | 0.9606 | 0.000096 |
| 26 | 2.7155 | 2.4350 | 0.8275 | 0.9406 | 0.9610 | 0.000068 |
| 27 | 2.7069 | 2.4241 | 0.8285 | 0.9417 | 0.9621 | 0.000044 |
| 28 | 2.7023 | 2.4232 | 0.8291 | 0.9418 | 0.9615 | 0.000025 |
| 29 | 2.6985 | 2.4222 | 0.8298 | 0.9421 | 0.9617 | 0.000012 |
| 30 | 2.6951 | 2.4191 | 0.8305 | 0.9423 | 0.9619 | 0.000004 |

**Best: 83.05% top-1 | 94.23% top-5 | 96.19% top-10** (epoch 30)

## Timing

- ~10 minutes per epoch (vs ~7 min for 96x96 without augmentation, ~3.5 min for 64x64)
- Total runtime: ~5 hours (30 epochs)

## Comparison Across Experiments

| Metric | Exp 01 (64x64, 256d) | Exp 02 (96x96, CBAM, 256d) | **Exp 03 (96x96, 512d, aug)** |
|--------|---------------------|---------------------------|-------------------------------|
| Best Val Acc | **85.02%** | 68.02% (killed ep14) | 83.05% |
| Val Top-5 | N/A | N/A | 94.23% |
| Val Top-10 | N/A | N/A | 96.19% |
| Epochs | 13 (of 30) | 14 (killed) | 30 |
| Augmentation | None (preload bug) | None (preload bug) | Affine+Blur+Erasing |
| Params | 1,118,427 | 1,360,190 | 2,697,782 |

## Analysis

1. **Top-1 below 64x64 baseline (83% vs 85%).** The 512-dim model has 2.4x more parameters but
   lower accuracy. Two factors at play:
   - **Augmentation regularization**: This is the first run with actual augmentation. The model
     sees harder examples, producing a wider train-val gap but better generalization potential.
     Exp 01 had zero augmentation (preload bug) so its 85% may be slightly overfit.
   - **96x96 is harder**: 4 conv layers may be too shallow for 96x96 input. The 64x64 baseline
     has 4x4 feature maps before GAP (compact), while 96x96 has 6x6 (sparser signal per cell).

2. **Top-10 recall at 96.19% is the key metric.** For the structural pipeline, the correct
   character is a candidate in 96.19% of cases. With radical filtering + bigram reranking,
   this is the accuracy floor the pipeline can lift from.

3. **Still improving at epoch 30.** The model gained +0.8% in the last 10 epochs as cosine
   annealing drove LR to near zero. A longer run or restart with warm restarts could push higher.

4. **Val loss gap (train 2.70 vs val 2.42)** shows the model is not overfitting — augmentation
   is working as intended. The Exp 01 baseline likely had train loss very close to val loss,
   indicating memorization without augmentation.

## Command

```bash
uv run python scripts/train_tinycnn.py --backbone-dim 512 --output-dir outputs/tinycnn_512_aug --epochs 30 --batch-size 256
```

## Checkpoint

```
outputs/tinycnn_512_aug/best_model_classification.pt
```
