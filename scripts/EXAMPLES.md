# HCCR Evaluation Scripts - Usage Examples

## Quick Start Examples

### Example 1: Evaluate CLIP Models (Zero-Shot)

**Scenario:** You want to test CLIP's zero-shot performance on Chinese characters without training any models.

```bash
# Basic usage
python scripts/eval_clip_zeroshot.py

# Custom paths
python scripts/eval_clip_zeroshot.py \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json \
    --batch-size 64 \
    --output-dir outputs/results
```

**Expected Output:**
```
=============================================================
Evaluating CLIP Zero-Shot (multilingual)
=============================================================
Loading CLIP model: xlm-roberta-base-ViT-B-32 (laion5b_s13b_b90k)
Building text features from scratch...
Processed 3755/3755 characters
Running inference on test set...
CLIP-multilingual: 100%|██████████| 3125/3125 [15:32<00:00, 3.35it/s]

Results for CLIP-multilingual:
  Top-1 Accuracy: 12.34%
  Top-5 Accuracy: 28.56%
  Top-10 Accuracy: 38.92%

=============================================================
Evaluating CLIP Zero-Shot (english)
=============================================================
...
```

**Result Files:**
- `outputs/results/clip_zeroshot_results.json`
- `outputs/cache/clip_text_features_multilingual_3755.pt` (cached)
- `outputs/cache/clip_text_features_english_3755.pt` (cached)

---

### Example 2: Test Structural Post-Processing

**Scenario:** You have trained models and want to see if structural constraints improve accuracy.

```bash
# Evaluate with alpha sweep
python scripts/eval_structural.py \
    --checkpoint-dir outputs/checkpoints \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json \
    --radical-table resources/radical_table.json \
    --bigram-table resources/bigram_table.json
```

**Expected Output:**
```
=============================================================
Alpha sweep for tinycnn_joint
=============================================================

Testing alpha = 0.3
Evaluating structural pipeline (mode=joint)...
Evaluating: 100%|██████████| 3125/3125 [08:45<00:00, 5.95it/s]

Results for tinycnn_joint (alpha=0.3):
  Accuracy before: 0.8523
  Accuracy after: 0.8698
  Improvement: 0.0175

Testing alpha = 0.5
...

Best alpha for tinycnn_joint: 0.7
  Best accuracy: 0.8756
```

**Result Files:**
- `outputs/results/structural_results.json`

**Result Structure:**
```json
{
  "tinycnn_joint": {
    "model": "tinycnn_joint",
    "results": [
      {
        "alpha": 0.3,
        "acc_before": 0.8523,
        "acc_after": 0.8698,
        "improvement": 0.0175
      },
      ...
    ],
    "best_alpha": 0.7,
    "best_acc": 0.8756
  }
}
```

---

### Example 3: Analyze Bigram Effectiveness

**Scenario:** You want to understand when bigram language modeling helps most.

```bash
# Test bigram in different scenarios
python scripts/eval_bigram_settings.py \
    --checkpoint outputs/checkpoints/tinycnn_joint_best.pt \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json \
    --bigram-table resources/bigram_table.json
```

**Expected Output:**
```
=============================================================
Setting 1: Random Character Pairs
=============================================================
Evaluated 1000 random pairs
Accuracy without bigram: 0.7234
Accuracy with bigram: 0.7456
Improvement: 0.0222

=============================================================
Setting 2: Real Word Pairs (High-frequency bigrams)
=============================================================
Testing 200 high-frequency bigrams
Evaluated 200 real word pairs
Accuracy without bigram: 0.7850
Accuracy with bigram: 0.8325
Improvement: 0.0475

=============================================================
Setting 3: Adversarial Pairs (Visually similar)
=============================================================
Finding confusable character pairs...
Found 347 confusable pairs
Evaluated 347 adversarial pairs
Accuracy without bigram: 0.5823
Accuracy with bigram: 0.6234
Improvement: 0.0411
```

**Key Insight:** Bigram helps most with real word pairs (+4.75%), showing language context is valuable.

---

### Example 4: Run Efficiency Benchmarks

**Scenario:** You need to compare model efficiency for deployment.

```bash
# Full benchmark suite
python scripts/run_benchmarks.py \
    --checkpoint-dir outputs/checkpoints \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json

# Skip quantization to save time
python scripts/run_benchmarks.py --skip-quantization
```

**Expected Output:**
```
Benchmarking TinyCNN...
Benchmarking TinyCNNRadical...
Benchmarking TinyCNNJoint...
Benchmarking MobileNetV3...

Benchmark Results:
         model  size_mb   params  latency_mean_ms  latency_std_ms  top1_acc
      TinyCNN     0.78   204532             2.34            0.12     82.34
TinyCNNRadical     1.02   267890             2.56            0.15     83.12
 TinyCNNJoint     1.15   301456             2.89            0.18     85.67
  MobileNetV3     8.52  2543678            12.45            0.67     87.23

=============================================================
Quantization benchmark: TinyCNNJoint
=============================================================
Original size: 1.15 MB
Quantized size: 0.35 MB
Compression ratio: 3.29x
Original accuracy: 85.67%
Quantized accuracy: 84.92%
Accuracy drop: 0.75%

Best accuracy: MobileNetV3 (87.23%)
Smallest model: TinyCNN (0.78 MB)
Fastest model: TinyCNN (2.34 ms)
Most efficient: TinyCNNJoint (score: 74.50)
```

**Result Files:**
- `outputs/results/benchmark_results.json`
- `outputs/results/benchmark_table.csv`
- `outputs/figures/efficiency_pareto.png`

---

### Example 5: Full Pipeline Execution

**Scenario:** You want to run everything from scratch.

```bash
# Complete pipeline (takes 8-12 hours)
python scripts/run_all.py \
    --data-dir data \
    --output-dir outputs

# Only evaluation (models already trained)
python scripts/run_all.py \
    --skip-build \
    --skip-training \
    --data-dir data \
    --output-dir outputs
```

**Expected Output:**
```
============================================================
HCCR Full Pipeline Orchestration
============================================================
Data directory: data
Output directory: outputs

############################################################
STEP 1: Building Resources
############################################################

============================================================
Building label map
============================================================
Command: python scripts/build_label_map.py ...
Completed in 45.3s

============================================================
Building radical table
============================================================
Command: python scripts/build_radical_table.py ...
Completed in 12.8s

############################################################
STEP 2: Training Models
############################################################

============================================================
Training TinyCNN
============================================================
Command: python scripts/train_tinycnn.py ...
Completed in 7234.5s

...

############################################################
STEP 3: Running Evaluations
############################################################

============================================================
Evaluating CLIP zero-shot
============================================================
Command: python scripts/eval_clip_zeroshot.py ...
Completed in 1834.2s

...

############################################################
STEP 4: Generating Summary Report
############################################################

============================================================
FINAL SUMMARY REPORT
============================================================

1. CLIP Zero-Shot Performance:
   Multilingual (Chinese prompts):
     Top-1: 12.34%
     Top-5: 28.56%
   English prompts:
     Top-1: 8.92%
     Top-5: 21.45%

2. Structural Post-Processing (Best Alpha):
   tinycnn_joint:
     Best alpha: 0.7
     Best accuracy: 0.8756

3. Bigram Re-ranking Effectiveness:
   random_pairs:
     Improvement: 0.0222
     Accuracy: 0.7234 -> 0.7456
   real_word_pairs:
     Improvement: 0.0475
     Accuracy: 0.7850 -> 0.8325
   adversarial_pairs:
     Improvement: 0.0411
     Accuracy: 0.5823 -> 0.6234

4. Model Efficiency Benchmark:
   TinyCNN:
     Size: 0.78 MB
     Parameters: 204,532
     Latency: 2.34 ms
     Accuracy: 82.34%
   TinyCNNJoint:
     Size: 1.15 MB
     Parameters: 301,456
     Latency: 2.89 ms
     Accuracy: 85.67%
   MobileNetV3:
     Size: 8.52 MB
     Parameters: 2,543,678
     Latency: 12.45 ms
     Accuracy: 87.23%

5. Quantization (TinyCNNJoint):
   Original size: 1.15 MB
   Quantized size: 0.35 MB
   Compression: 3.29x
   Accuracy drop: 0.75%

============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================
Total time: 234.5 minutes
Results saved to: outputs
```

---

## Advanced Usage

### Parallel Evaluation (Multiple GPUs)

If you have multiple GPUs, run evaluations in parallel:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_clip_zeroshot.py &

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python scripts/eval_structural.py &

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python scripts/eval_bigram_settings.py &

# Terminal 4 (GPU 3)
CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmarks.py &

# Wait for all
wait
```

### Batch Processing with Custom Settings

```bash
# Evaluate multiple alpha values
for alpha in 0.3 0.5 0.7 0.9; do
    python scripts/eval_structural.py \
        --alpha $alpha \
        --output-dir "outputs/alpha_$alpha"
done

# Evaluate different batch sizes
for bs in 16 32 64 128; do
    python scripts/run_benchmarks.py \
        --batch-size $bs \
        --output-dir "outputs/bs_$bs"
done
```

### Debugging Single Model

```bash
# Test only TinyCNNJoint
python -c "
from pathlib import Path
import torch
from hccr.models import TinyCNNJoint
from hccr.utils import load_checkpoint, get_device

device = get_device()
model = TinyCNNJoint(num_classes=3755, num_radicals=500, num_structures=13).to(device)
load_checkpoint(model, Path('outputs/checkpoints/tinycnn_joint_best.pt'), device)
print('Model loaded successfully!')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

### Export Results for Paper

```bash
# Generate publication-ready tables
python scripts/run_benchmarks.py

# Convert to LaTeX
python -c "
import pandas as pd
df = pd.read_csv('outputs/results/benchmark_table.csv')
latex = df.to_latex(index=False, float_format='%.2f')
with open('outputs/results/benchmark_table.tex', 'w') as f:
    f.write(latex)
print('LaTeX table saved!')
"
```

---

## Common Workflows

### Workflow 1: Quick Model Comparison

```bash
# Train 2 models for quick comparison
python scripts/train_tinycnn.py --epochs 10
python scripts/train_tinycnn_joint.py --epochs 10

# Benchmark them
python scripts/run_benchmarks.py
```

### Workflow 2: Hyperparameter Tuning

```bash
# Train with different settings
python scripts/train_tinycnn_joint.py --lr 1e-3 --epochs 30
python scripts/train_tinycnn_joint.py --lr 5e-4 --epochs 30

# Evaluate both
python scripts/eval_structural.py
```

### Workflow 3: Ablation Study

```bash
# Baseline (no structural processing)
python scripts/train_tinycnn.py

# + Radical auxiliary task
python scripts/train_tinycnn_radical.py

# + Full multi-task
python scripts/train_tinycnn_joint.py

# Compare all
python scripts/run_benchmarks.py
```

---

## Troubleshooting Examples

### Issue: Out of Memory

```bash
# Reduce batch size
python scripts/eval_clip_zeroshot.py --batch-size 8

# Use CPU
CUDA_VISIBLE_DEVICES=-1 python scripts/eval_structural.py
```

### Issue: Slow CLIP Evaluation

```bash
# Use cache from previous run (instant)
python scripts/eval_clip_zeroshot.py
# Text features loaded from cache in <1 second
```

### Issue: Missing Checkpoints

```bash
# Check what's available
ls outputs/checkpoints/

# Train missing model
python scripts/train_tinycnn_joint.py
```

---

## Expected Runtimes (GPU: RTX 3090)

| Script | First Run | Cached | Memory |
|--------|-----------|--------|--------|
| `eval_clip_zeroshot.py` | 45 min | 12 min | 6 GB |
| `eval_structural.py` | 25 min | 25 min | 4 GB |
| `eval_bigram_settings.py` | 20 min | 20 min | 3 GB |
| `run_benchmarks.py` | 15 min | 15 min | 4 GB |
| `run_all.py` (full) | 10 hrs | 90 min | 8 GB |

---

## Tips and Tricks

1. **Cache Everything:** First run builds caches (indices, embeddings). Keep `outputs/cache/` directory.

2. **Use `screen` or `tmux`:** For long runs:
   ```bash
   screen -S hccr
   python scripts/run_all.py
   # Ctrl+A, D to detach
   ```

3. **Monitor GPU:**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Checkpoints:** Save intermediate results:
   ```bash
   python scripts/run_all.py --skip-training 2>&1 | tee eval.log
   ```

5. **Incremental Development:** Test on small subset first:
   ```python
   # Modify dataset loading to use subset
   test_dataset = Subset(test_dataset, range(1000))
   ```
