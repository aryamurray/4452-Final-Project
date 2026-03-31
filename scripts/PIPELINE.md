# HCCR Evaluation Pipeline Architecture

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_all.py (Orchestrator)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Step 1:      │   │  Step 2:      │   │  Step 3:      │
│  Build        │   │  Train        │   │  Evaluate     │
│  Resources    │   │  Models       │   │  All          │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ label_map     │   │ tinycnn       │   │ eval_clip     │
│ radical_table │   │ tinycnn_radical│  │ eval_struct   │
│ bigram_table  │   │ tinycnn_joint │   │ eval_bigram   │
│               │   │ mobilenetv3   │   │ run_benchmark │
└───────────────┘   └───────────────┘   └───────────────┘
                                                  │
                                                  ▼
                                        ┌───────────────┐
                                        │  Step 4:      │
                                        │  Generate     │
                                        │  Summary      │
                                        └───────────────┘
```

## Script Dependencies

### Resource Building Scripts
```
build_label_map.py
    └── Output: resources/label_map.json

build_radical_table.py
    ├── Input: resources/label_map.json
    └── Output: resources/radical_table.json

build_bigram_table.py
    ├── Input: resources/label_map.json
    └── Output: resources/bigram_table.json
```

### Training Scripts
```
train_tinycnn.py
    ├── Input: resources/label_map.json
    └── Output: checkpoints/tinycnn_best.pt

train_tinycnn_radical.py
    ├── Input: resources/label_map.json
    │          resources/radical_table.json
    └── Output: checkpoints/tinycnn_radical_best.pt

train_tinycnn_joint.py
    ├── Input: resources/label_map.json
    │          resources/radical_table.json
    └── Output: checkpoints/tinycnn_joint_best.pt

train_mobilenetv3.py
    ├── Input: resources/label_map.json
    └── Output: checkpoints/mobilenetv3_best.pt
```

### Evaluation Scripts
```
eval_clip_zeroshot.py
    ├── Input: resources/label_map.json
    │          data/HWDB1.1/test/
    └── Output: results/clip_zeroshot_results.json
               cache/clip_text_features_*.pt

eval_structural.py
    ├── Input: resources/label_map.json
    │          resources/radical_table.json
    │          resources/bigram_table.json
    │          checkpoints/*.pt
    │          data/HWDB1.1/test/
    └── Output: results/structural_results.json

eval_bigram_settings.py
    ├── Input: resources/label_map.json
    │          resources/bigram_table.json
    │          checkpoints/tinycnn_joint_best.pt
    │          data/HWDB1.1/test/
    └── Output: results/bigram_settings_results.json

run_benchmarks.py
    ├── Input: resources/label_map.json
    │          checkpoints/*.pt
    │          data/HWDB1.1/test/
    └── Output: results/benchmark_results.json
               results/benchmark_table.csv
               figures/efficiency_pareto.png
```

## Key Features by Script

### eval_clip_zeroshot.py
- ✓ Multilingual CLIP (Chinese prompts)
- ✓ English CLIP (English prompts)
- ✓ Top-1, Top-5, Top-10 accuracy
- ✓ Text embedding caching
- ✓ Batch processing

### eval_structural.py
- ✓ Radical filtering evaluation
- ✓ Structure constraint evaluation
- ✓ Alpha hyperparameter sweep
- ✓ Multi-model comparison
- ✓ Before/after accuracy reporting

### eval_bigram_settings.py
- ✓ Random pair testing
- ✓ Real word pair testing
- ✓ Adversarial pair testing
- ✓ Bigram re-ranking impact
- ✓ Confusion pattern analysis

### run_benchmarks.py
- ✓ Model size measurement
- ✓ Inference latency profiling
- ✓ Parameter counting
- ✓ Quantization testing (int8)
- ✓ Pareto frontier plotting
- ✓ Efficiency scoring

### run_all.py
- ✓ Complete pipeline orchestration
- ✓ Conditional execution (skip flags)
- ✓ Progress logging
- ✓ Error handling
- ✓ Summary report generation
- ✓ Time tracking

## Execution Order

### Minimal (Evaluation Only)
Assumes models are already trained.

```bash
# 1. CLIP zero-shot
python scripts/eval_clip_zeroshot.py

# 2. Structural post-processing
python scripts/eval_structural.py

# 3. Bigram settings
python scripts/eval_bigram_settings.py

# 4. Efficiency benchmarks
python scripts/run_benchmarks.py
```

### Full Pipeline
Complete end-to-end execution.

```bash
python scripts/run_all.py
```

### Staged Execution
Run specific stages only.

```bash
# Stage 1: Resources only
python scripts/run_all.py --skip-training --skip-evaluation

# Stage 2: Training only
python scripts/run_all.py --skip-build --skip-evaluation

# Stage 3: Evaluation only
python scripts/run_all.py --skip-build --skip-training
```

## Output File Manifest

```
outputs/
├── cache/
│   ├── test_index.pkl                  # Dataset index cache
│   ├── clip_text_features_multilingual_3755.pt
│   └── clip_text_features_english_3755.pt
│
├── checkpoints/
│   ├── tinycnn_best.pt                 # 0.8 MB, ~200K params
│   ├── tinycnn_radical_best.pt         # 1.0 MB, ~250K params
│   ├── tinycnn_joint_best.pt           # 1.1 MB, ~280K params
│   └── mobilenetv3_best.pt             # 8.5 MB, ~2.5M params
│
├── results/
│   ├── clip_zeroshot_results.json      # CLIP evaluation
│   ├── structural_results.json          # Structural post-processing
│   ├── bigram_settings_results.json     # Bigram evaluation
│   ├── benchmark_results.json           # Efficiency metrics
│   ├── benchmark_table.csv              # Tabular benchmarks
│   └── summary_report.txt               # Final summary
│
└── figures/
    └── efficiency_pareto.png            # Size vs accuracy plot
```

## Performance Expectations

### eval_clip_zeroshot.py
- Runtime: ~30-60 minutes (first run with text embedding generation)
- Runtime: ~10-15 minutes (subsequent runs with cache)
- Memory: ~4-8 GB GPU

### eval_structural.py
- Runtime: ~20-30 minutes (alpha sweep × 3 models)
- Memory: ~2-4 GB GPU

### eval_bigram_settings.py
- Runtime: ~15-25 minutes (3 settings × 500-1000 pairs each)
- Memory: ~2-4 GB GPU

### run_benchmarks.py
- Runtime: ~10-20 minutes (4 models + quantization)
- Memory: ~2-4 GB GPU

### run_all.py (Full Pipeline)
- Runtime: ~8-12 hours (including training)
- Runtime: ~1-2 hours (skip training)
- Memory: ~8 GB GPU recommended

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size: `--batch-size 32` → `--batch-size 16`
   - Use CPU for evaluation: Set `CUDA_VISIBLE_DEVICES=-1`

2. **Missing Checkpoints**
   - Use `--skip-training` only if models are trained
   - Check `outputs/checkpoints/` directory

3. **CLIP Download Issues**
   - Ensure internet connection for first run
   - Check firewall settings
   - Set `HF_HOME` environment variable

4. **Slow Evaluation**
   - Reduce `num_samples` in bigram settings
   - Use cached indices: Will auto-cache after first run
   - Skip quantization: `--skip-quantization` flag

## Best Practices

1. **Always run resource building first**
   ```bash
   python scripts/run_all.py --skip-training --skip-evaluation
   ```

2. **Train models separately for debugging**
   ```bash
   python scripts/train_tinycnn.py
   python scripts/train_tinycnn_joint.py
   ```

3. **Run evaluations in parallel** (if multiple GPUs)
   ```bash
   # Terminal 1
   CUDA_VISIBLE_DEVICES=0 python scripts/eval_clip_zeroshot.py

   # Terminal 2
   CUDA_VISIBLE_DEVICES=1 python scripts/eval_structural.py
   ```

4. **Monitor progress with logs**
   ```bash
   python scripts/run_all.py 2>&1 | tee run_all.log
   ```

5. **Verify outputs after each stage**
   ```bash
   ls outputs/results/
   cat outputs/results/summary_report.txt
   ```
