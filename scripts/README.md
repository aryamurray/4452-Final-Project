# HCCR Evaluation and Orchestration Scripts

This directory contains evaluation and orchestration scripts for the HCCR (Handwritten Chinese Character Recognition) project.

## Scripts Overview

### Evaluation Scripts

#### 1. `eval_clip_zeroshot.py`
Evaluates CLIP zero-shot classification on HWDB test set.

**Features:**
- Tests both multilingual (Chinese prompts) and English CLIP models
- Computes top-1, top-5, and top-10 accuracies
- Caches text embeddings for faster re-runs
- Saves results to JSON

**Usage:**
```bash
python scripts/eval_clip_zeroshot.py \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json \
    --cache-dir outputs/cache \
    --batch-size 32 \
    --output-dir outputs/results
```

**Output:**
- `outputs/results/clip_zeroshot_results.json`

---

#### 2. `eval_structural.py`
Evaluates structural post-processing on trained models.

**Features:**
- Tests radical filtering and structure constraints
- Sweeps alpha values [0.3, 0.5, 0.7, 0.9] to find optimal configuration
- Evaluates multiple models (TinyCNN, TinyCNNJoint, MobileNetV3)
- Reports accuracy improvement from structural processing

**Usage:**
```bash
python scripts/eval_structural.py \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json \
    --radical-table resources/radical_table.json \
    --bigram-table resources/bigram_table.json \
    --checkpoint-dir outputs/checkpoints \
    --output-dir outputs/results
```

**Output:**
- `outputs/results/structural_results.json`

---

#### 3. `eval_bigram_settings.py`
Evaluates bigram language model in three test settings.

**Features:**
- **Setting 1:** Random character pairs from test set
- **Setting 2:** Real word pairs from high-frequency bigrams
- **Setting 3:** Adversarial pairs with visually similar characters
- Compares accuracy with/without bigram re-ranking

**Usage:**
```bash
python scripts/eval_bigram_settings.py \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json \
    --bigram-table resources/bigram_table.json \
    --checkpoint outputs/checkpoints/tinycnn_joint_best.pt \
    --output-dir outputs/results
```

**Output:**
- `outputs/results/bigram_settings_results.json`

---

#### 4. `run_benchmarks.py`
Comprehensive efficiency benchmarking for all models.

**Features:**
- Measures model size, parameter count, inference latency
- Tests quantization on TinyCNNJoint (int8)
- Generates efficiency Pareto plot (size vs accuracy)
- Saves benchmark table as CSV and JSON

**Usage:**
```bash
python scripts/run_benchmarks.py \
    --checkpoint-dir outputs/checkpoints \
    --test-dir data/HWDB1.1/test \
    --label-map resources/label_map.json \
    --output-dir outputs/results \
    --batch-size 64
```

**Output:**
- `outputs/results/benchmark_results.json`
- `outputs/results/benchmark_table.csv`
- `outputs/figures/efficiency_pareto.png`

---

### Orchestration Script

#### 5. `run_all.py`
Full pipeline orchestration script.

**Features:**
- Runs complete HCCR pipeline from start to finish
- Builds resources (label map, radical table, bigram table)
- Trains all models (TinyCNN, TinyCNNRadical, TinyCNNJoint, MobileNetV3)
- Runs all evaluation scripts
- Generates comprehensive summary report

**Usage:**

**Full pipeline:**
```bash
python scripts/run_all.py \
    --data-dir data \
    --output-dir outputs
```

**Skip training (if already trained):**
```bash
python scripts/run_all.py \
    --skip-training \
    --data-dir data \
    --output-dir outputs
```

**Skip building resources (if already built):**
```bash
python scripts/run_all.py \
    --skip-build \
    --data-dir data \
    --output-dir outputs
```

**Only generate summary (skip everything else):**
```bash
python scripts/run_all.py \
    --skip-build \
    --skip-training \
    --skip-evaluation \
    --output-dir outputs
```

**Output:**
- All evaluation results
- `outputs/results/summary_report.txt`

---

## Directory Structure

After running all scripts, the output directory will have:

```
outputs/
├── cache/
│   ├── test_index.pkl
│   └── clip_text_features_*.pt
├── checkpoints/
│   ├── tinycnn_best.pt
│   ├── tinycnn_radical_best.pt
│   ├── tinycnn_joint_best.pt
│   └── mobilenetv3_best.pt
├── results/
│   ├── clip_zeroshot_results.json
│   ├── structural_results.json
│   ├── bigram_settings_results.json
│   ├── benchmark_results.json
│   ├── benchmark_table.csv
│   └── summary_report.txt
└── figures/
    └── efficiency_pareto.png
```

---

## Requirements

All scripts require the HCCR package to be installed:

```bash
pip install -e .
```

Key dependencies:
- PyTorch
- open_clip_torch (for CLIP evaluation)
- pandas (for benchmark tables)
- matplotlib (for plots)
- scikit-learn (for metrics)

---

## Notes

1. **CLIP models** require internet connection on first run to download pretrained weights
2. **Text embeddings** are cached to speed up repeated CLIP evaluations
3. **Alpha sweep** in structural evaluation tests multiple hyperparameter values
4. **Quantization benchmark** may take longer as it tests both original and quantized models
5. **Orchestration script** can take several hours to run complete pipeline

---

## Typical Workflow

### Quick evaluation (models already trained):
```bash
# Evaluate CLIP
python scripts/eval_clip_zeroshot.py

# Evaluate structural processing
python scripts/eval_structural.py

# Evaluate bigram settings
python scripts/eval_bigram_settings.py

# Run benchmarks
python scripts/run_benchmarks.py
```

### Full experiment from scratch:
```bash
# Run everything
python scripts/run_all.py
```

### Re-run evaluations after training:
```bash
# Skip build and training
python scripts/run_all.py --skip-build --skip-training
```

---

## Citation

If you use these scripts, please cite:

```
HCCR: Handwritten Chinese Character Recognition
https://github.com/your-repo/hccr
```
