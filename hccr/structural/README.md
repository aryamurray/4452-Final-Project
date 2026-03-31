# Structural Post-Processing Module

This module provides structural post-processing components for HCCR (Handwritten Chinese Character Recognition) to improve recognition accuracy through linguistic and structural constraints.

## Components

### 1. RadicalTable (`radical_table.py`)

Parses CJKVI IDS (Ideographic Description Sequence) data to build radical decomposition tables.

**Key Features:**
- Extracts 13 structural composition patterns (IDC types: ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻)
- Recursively decomposes characters to atomic radicals
- Estimates stroke counts from radical components
- Provides multi-hot radical vectors for each character

**Usage:**
```python
from hccr.structural import RadicalTable

# Build from IDS file
table = RadicalTable.build_from_ids_file(
    ids_path="resources/ids.txt",
    char_set=set(label_map.values())  # 3755 characters
)

# Get radical information
radicals = table.char_to_radicals["信"]  # ['亻', '言']
structure = table.get_structure("信")     # 0 (left-right ⿰)
strokes = table.get_strokes("信")         # 9
vector = table.get_radical_vector("信")   # [0,0,1,...,1,0] (multi-hot)

# Save/load
table.save("resources/radical_table.json")
table = RadicalTable.load("resources/radical_table.json")
```

### 2. RadicalFilter (`radical_filter.py`)

Reweights classifier candidates using radical predictions.

**Key Features:**
- Combines character scores with radical match scores
- Applies structure and stroke count constraints
- Configurable weighting (alpha parameter)

**Usage:**
```python
from hccr.structural import RadicalFilter

filter = RadicalFilter(radical_table, alpha=0.7)

# Basic reweighting (character + radical)
candidates = filter.reweight(
    char_probs=char_probs,        # (num_classes,)
    radical_probs=radical_probs,  # (num_radicals,)
    label_map=label_map,
    top_k=10
)

# Full reweighting (+ structure + strokes)
candidates = filter.reweight_with_structure(
    char_probs=char_probs,
    radical_probs=radical_probs,
    structure_probs=structure_probs,  # (13,)
    stroke_pred=8.5,
    label_map=label_map,
    top_k=10
)
# Returns: [(class_idx, score), ...]
```

### 3. BigramModel (`bigram.py`)

Character bigram language model for sequence re-ranking.

**Key Features:**
- Builds from word frequency data
- Laplace smoothing for unseen bigrams
- Log-probability based scoring
- Perplexity computation

**Usage:**
```python
from hccr.structural import BigramModel

# Build from frequency file
model = BigramModel.build_from_word_freq(
    freq_path="resources/word_freq.txt",
    min_freq=1
)

# Get probabilities
log_p = model.log_prob("前", "景")  # log P(景|前)

# Re-rank candidates with context
reranked = model.rerank(
    candidates=candidates,
    prev_char="前",
    label_map=label_map,
    beta=0.3
)

# Evaluate sequence
perplexity = model.perplexity("前景广阔")

# Save/load
model.save("resources/bigram_table.json")
model = BigramModel.load("resources/bigram_table.json")
```

### 4. StructuralPipeline (`combined.py`)

Full pipeline combining all components with beam search.

**Key Features:**
- Single-character prediction with all constraints
- Beam search for multi-character sequences
- Batch evaluation on test sets
- Error analysis tools

**Usage:**
```python
from hccr.structural import StructuralPipeline
from hccr.config import StructuralConfig

config = StructuralConfig(
    alpha=0.7,      # radical filter weight
    beta=0.3,       # bigram weight
    top_k=10,       # candidate count
    beam_width=5    # beam search width
)

pipeline = StructuralPipeline(
    radical_filter=radical_filter,
    bigram_model=bigram_model,
    label_map=label_map,
    config=config
)

# Single character prediction
candidates = pipeline.predict_single(
    char_probs=char_probs,
    radical_probs=radical_probs,
    structure_probs=structure_probs,
    stroke_pred=8.5,
    prev_char="前"  # optional context
)

# Sequence prediction with beam search
sequence = pipeline.predict_sequence_beam(
    char_probs_seq=[...],      # list of arrays
    radical_probs_seq=[...],
    structure_probs_seq=[...],
    stroke_pred_seq=[...],
    beam_width=5
)

# Evaluate on test set
results = pipeline.evaluate_on_loader(
    model=model,
    test_loader=test_loader,
    device=device,
    mode="joint"  # or "simple"
)
# Returns: {acc_before, acc_after, top5_before, top5_after, improvement}

# Error analysis
stats = pipeline.analyze_errors(
    model=model,
    test_loader=test_loader,
    device=device,
    num_samples=100
)
```

## IDS Format

The IDS (Ideographic Description Sequence) format in `ids.txt`:

```
U+4FE1	信	⿰亻言
U+524D	前	⿱丷⿱⿰丷丷刀
U+666F	景	⿱日京
```

- **Format:** `codepoint\tchar\tIDS_sequence`
- **IDC markers:** ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻ (U+2FF0-U+2FFB)
- **13 structure types** map to indices 0-12

## Configuration

Structural configuration in `config.py`:

```python
@dataclass
class StructuralConfig:
    alpha: float = 0.7      # classifier vs radical weight
    beta: float = 0.3       # bigram re-ranking weight
    top_k: int = 10         # candidates from classifier
    beam_width: int = 5     # beam search width
```

## Integration Example

```python
from pathlib import Path
from hccr.structural import (
    RadicalTable, RadicalFilter, BigramModel, StructuralPipeline
)
from hccr.config import StructuralConfig
from hccr.utils import load_json

# Load label map
label_map = load_json("resources/label_map.json")
label_map = {int(k): v for k, v in label_map.items()}

# Build/load radical table
if Path("resources/radical_table.json").exists():
    radical_table = RadicalTable.load("resources/radical_table.json")
else:
    char_set = set(label_map.values())
    radical_table = RadicalTable.build_from_ids_file(
        "resources/ids.txt", char_set
    )
    radical_table.save("resources/radical_table.json")

# Build/load bigram model
if Path("resources/bigram_table.json").exists():
    bigram_model = BigramModel.load("resources/bigram_table.json")
else:
    bigram_model = BigramModel.build_from_word_freq(
        "resources/word_freq.txt"
    )
    bigram_model.save("resources/bigram_table.json")

# Create pipeline
config = StructuralConfig()
radical_filter = RadicalFilter(radical_table, alpha=config.alpha)
pipeline = StructuralPipeline(
    radical_filter, bigram_model, label_map, config
)

# Use in evaluation
results = pipeline.evaluate_on_loader(model, test_loader, device)
print(f"Accuracy improvement: {results['improvement']:.4f}")
```

## Performance Considerations

1. **Radical Table Building:** One-time preprocessing, save to JSON
2. **Bigram Model:** Load once at startup, cache in memory
3. **Inference:** Minimal overhead (~10-20ms per sample)
4. **Beam Search:** O(beam_width * top_k * seq_len) complexity

## Requirements

- numpy
- torch
- tqdm (for progress bars)
- Standard library: json, logging, math, pathlib
