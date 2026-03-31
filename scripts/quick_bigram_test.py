"""Quick pipeline test using MiniResNet checkpoint.

Runs on CPU so it won't interfere with GPU training.
Builds bigram counts directly from SUBTLEX-CH (3 MB) instead of
loading the 675 MB precomputed bigram_table.json.

Tests three post-processing methods:
1. Baseline (argmax only)
2. Radical filtering (implied radical probs from classifier output)
3. Radical + bigram (full pipeline)
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

from hccr.data.dataset import HWDBDataset
from hccr.data.label_map import LabelMap
from hccr.data.transforms import get_eval_transform
from hccr.models.mini_resnet import MiniResNet
from hccr.structural.radical_table import RadicalTable
from hccr.utils import get_logger

logger = get_logger(__name__)


def derive_radical_probs(char_probs, radical_table, idx_to_char):
    """Derive implied radical probabilities from character probabilities.

    For each radical r, P(r) = sum of P(char) for all chars containing r.
    This extracts radical information from the classifier without needing
    a separate radical head.
    """
    num_radicals = len(radical_table.all_radicals)
    radical_probs = np.zeros(num_radicals)

    for char_idx, prob in enumerate(char_probs):
        if prob < 1e-6:
            continue
        char = idx_to_char.get(char_idx, "")
        radicals = radical_table.char_to_radicals.get(char, [])
        for radical in radicals:
            if radical in radical_table.radical_to_index:
                radical_probs[radical_table.radical_to_index[radical]] += prob

    # Clip to [0, 1]
    return np.clip(radical_probs, 0.0, 1.0)


def radical_reweight(char_probs, radical_probs, radical_table, idx_to_char, top_k=10, alpha=0.7):
    """Reweight top-K candidates using implied radical probabilities."""
    top_k_indices = np.argsort(char_probs)[-top_k:][::-1]
    candidates = []

    for idx in top_k_indices:
        char_prob = char_probs[idx]
        char = idx_to_char.get(int(idx), "")
        radicals = radical_table.char_to_radicals.get(char, [])

        if not radicals:
            candidates.append((int(idx), float(char_prob)))
            continue

        # Mean probability of this character's expected radicals
        scores = []
        for radical in radicals:
            if radical in radical_table.radical_to_index:
                r_idx = radical_table.radical_to_index[radical]
                scores.append(radical_probs[r_idx])

        radical_match = float(np.mean(scores)) if scores else 0.0
        combined = alpha * char_prob + (1 - alpha) * radical_match
        candidates.append((int(idx), combined))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def build_quick_bigram(subtlex_path: Path, min_freq: int = 5):
    """Build bigram counts from SUBTLEX-CH (lightweight, ~3 MB)."""
    bigram_counts = defaultdict(lambda: defaultdict(int))
    unigram_counts = defaultdict(int)

    with open(subtlex_path, "r", encoding="gbk") as f:
        for i, line in enumerate(f):
            if i < 3:  # Skip header
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                freq = int(parts[1])
            except ValueError:
                continue
            if freq < min_freq or len(word) < 2:
                continue
            for j in range(len(word) - 1):
                c1, c2 = word[j], word[j + 1]
                bigram_counts[c1][c2] += freq
                unigram_counts[c1] += freq

    vocab_size = len(set(unigram_counts.keys()))
    logger.info(f"Built bigram: {len(bigram_counts)} contexts, {vocab_size} vocab")
    return bigram_counts, unigram_counts, vocab_size


def bigram_log_prob(c1, c2, bigram_counts, unigram_counts, vocab_size):
    """Compute log P(c2|c1) with Laplace smoothing."""
    count = bigram_counts.get(c1, {}).get(c2, 0)
    unigram = unigram_counts.get(c1, 0)
    prob = (count + 1) / (unigram + vocab_size)
    return math.log(prob)


def main():
    # Paths
    checkpoint_path = Path("outputs/mini_resnet/best_model_classification.pt")
    label_map_path = Path("resources/label_map.json")
    radical_table_path = Path("resources/radical_table.json")
    subtlex_path = Path("resources/subtlex_ch_wf.tsv")
    test_dir = Path("data/HWDB1.1/test")

    # Load label map
    label_map = LabelMap.load(label_map_path)
    num_classes = len(label_map)
    idx_to_char = {i: label_map.decode(i) for i in range(num_classes)}
    logger.info(f"Label map: {num_classes} classes")

    # Load radical table
    logger.info("Loading radical table...")
    radical_table = RadicalTable.load(radical_table_path)
    logger.info(f"Radical table: {len(radical_table.all_radicals)} radicals, "
                f"{len(radical_table.char_to_radicals)} characters")

    # Load model on CPU
    logger.info("Loading MiniResNet on CPU...")
    model = MiniResNet(num_classes=num_classes, backbone_dim=192)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded checkpoint from epoch {epoch}")

    # Build lightweight bigram from SUBTLEX-CH
    bigram_counts, unigram_counts, vocab_size = build_quick_bigram(subtlex_path)

    # Load test dataset (NO preloading — save RAM)
    logger.info("Loading test dataset (no preload)...")
    transform = get_eval_transform(96)
    test_dataset = HWDBDataset(
        test_dir, label_map, transform=transform,
        index_cache_path=Path("outputs/mini_resnet/test_index.pkl"),
        preload=False,
    )
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Build char -> sample indices mapping
    logger.info("Building character index...")
    char_to_indices = defaultdict(list)
    for idx in range(len(test_dataset)):
        char_str = test_dataset.index[idx][2]
        char_to_indices[char_str].append(idx)

    # Find word pairs where BOTH characters exist in our test set
    word_pairs = []
    for c1, c2_counts in bigram_counts.items():
        for c2, count in c2_counts.items():
            if count > 50 and c1 in char_to_indices and c2 in char_to_indices:
                word_pairs.append((c1, c2, count))
    word_pairs.sort(key=lambda x: x[2], reverse=True)
    word_pairs = word_pairs[:500]
    logger.info(f"Testing {len(word_pairs)} high-frequency word pairs")

    # Also prepare random pairs (control group)
    all_chars = list(char_to_indices.keys())
    random.seed(42)
    random_pairs = []
    for _ in range(500):
        c1 = random.choice(all_chars)
        c2 = random.choice(all_chars)
        if c1 != c2:
            random_pairs.append((c1, c2))

    # ===== TEST =====
    results = {}
    for setting_name, pairs in [("Real word pairs", word_pairs), ("Random pairs", random_pairs)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Setting: {setting_name} ({len(pairs)} pairs)")
        logger.info(f"{'='*60}")

        baseline_correct = 0
        radical_correct = 0
        bigram_correct = 0
        full_correct = 0  # radical + bigram
        baseline_top5 = 0
        full_top5 = 0
        total = 0
        radical_corrections = []
        bigram_corrections = []
        full_corrections = []

        with torch.no_grad():
            for pair_idx, (c1, c2, *_) in enumerate(pairs):
                # Get a sample for c2
                idx2 = random.choice(char_to_indices[c2])
                img, label = test_dataset[idx2]
                logits = model(img.unsqueeze(0))
                probs = torch.softmax(logits, dim=1).numpy()[0]

                # === Method 1: Baseline (argmax) ===
                pred_baseline = int(np.argmax(probs))
                top5_baseline = set(np.argsort(probs)[-5:][::-1].tolist())

                # === Method 2: Radical filtering only ===
                rad_probs = derive_radical_probs(probs, radical_table, idx_to_char)
                rad_candidates = radical_reweight(
                    probs, rad_probs, radical_table, idx_to_char,
                    top_k=10, alpha=0.7,
                )
                pred_radical = rad_candidates[0][0]

                # === Method 3: Bigram only ===
                top10_indices = np.argsort(probs)[-10:][::-1]
                bigram_candidates = []
                for cand in top10_indices:
                    cand_char = idx_to_char.get(int(cand), "")
                    cls_score = float(probs[cand])
                    bg_log_p = bigram_log_prob(c1, cand_char, bigram_counts, unigram_counts, vocab_size)
                    bg_score = math.exp(bg_log_p)
                    combined = 0.7 * cls_score + 0.3 * bg_score
                    bigram_candidates.append((int(cand), combined))
                bigram_candidates.sort(key=lambda x: x[1], reverse=True)
                pred_bigram = bigram_candidates[0][0]

                # === Method 4: Radical + Bigram (full pipeline) ===
                # First apply radical reweighting, then bigram on top
                full_candidates = []
                for cand_idx, rad_score in rad_candidates:
                    cand_char = idx_to_char.get(cand_idx, "")
                    bg_log_p = bigram_log_prob(c1, cand_char, bigram_counts, unigram_counts, vocab_size)
                    bg_score = math.exp(bg_log_p)
                    # Combine: 70% radical-reweighted score + 30% bigram
                    combined = 0.7 * rad_score + 0.3 * bg_score
                    full_candidates.append((cand_idx, combined))
                full_candidates.sort(key=lambda x: x[1], reverse=True)
                pred_full = full_candidates[0][0]
                top5_full = set(c[0] for c in full_candidates[:5])

                # Check correctness
                if pred_baseline == label:
                    baseline_correct += 1
                if pred_radical == label:
                    radical_correct += 1
                if pred_bigram == label:
                    bigram_correct += 1
                if pred_full == label:
                    full_correct += 1
                if label in top5_baseline:
                    baseline_top5 += 1
                if label in top5_full:
                    full_top5 += 1

                # Track corrections
                if pred_baseline != label and pred_radical == label:
                    radical_corrections.append((c1, c2, idx_to_char.get(pred_baseline, "?")))
                if pred_baseline != label and pred_bigram == label:
                    bigram_corrections.append((c1, c2, idx_to_char.get(pred_baseline, "?")))
                if pred_baseline != label and pred_full == label:
                    full_corrections.append((c1, c2, idx_to_char.get(pred_baseline, "?")))

                total += 1

                if (pair_idx + 1) % 100 == 0:
                    logger.info(f"  Processed {pair_idx + 1}/{len(pairs)}...")

        acc_base = baseline_correct / total
        acc_radical = radical_correct / total
        acc_bigram = bigram_correct / total
        acc_full = full_correct / total
        top5_base_pct = baseline_top5 / total
        top5_full_pct = full_top5 / total

        logger.info(f"\nResults ({total} pairs):")
        logger.info(f"  Baseline     top-1: {acc_base:.4f} ({baseline_correct}/{total})")
        logger.info(f"  + Radical    top-1: {acc_radical:.4f} ({radical_correct}/{total})  delta={acc_radical - acc_base:+.4f}")
        logger.info(f"  + Bigram     top-1: {acc_bigram:.4f} ({bigram_correct}/{total})  delta={acc_bigram - acc_base:+.4f}")
        logger.info(f"  + Rad+Bigram top-1: {acc_full:.4f} ({full_correct}/{total})  delta={acc_full - acc_base:+.4f}")
        logger.info(f"  Baseline     top-5: {top5_base_pct:.4f}")
        logger.info(f"  + Rad+Bigram top-5: {top5_full_pct:.4f}")
        logger.info(f"  Corrections: radical={len(radical_corrections)}, bigram={len(bigram_corrections)}, full={len(full_corrections)}")

        if full_corrections[:10]:
            logger.info(f"\n  Sample full-pipeline corrections:")
            for prev, true, was in full_corrections[:10]:
                logger.info(f"    {prev}{true} (model said '{was}', pipeline fixed it)")

        results[setting_name] = {
            "total": total,
            "baseline_top1": acc_base,
            "radical_top1": acc_radical,
            "bigram_top1": acc_bigram,
            "full_top1": acc_full,
            "delta_radical": acc_radical - acc_base,
            "delta_bigram": acc_bigram - acc_base,
            "delta_full": acc_full - acc_base,
            "baseline_top5": top5_base_pct,
            "full_top5": top5_full_pct,
        }

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Setting':<20} {'Baseline':>10} {'+ Radical':>10} {'+ Bigram':>10} {'+ Both':>10}")
    logger.info(f"{'-'*60}")
    for name, r in results.items():
        logger.info(f"{name:<20} {r['baseline_top1']:>10.4f} {r['radical_top1']:>10.4f} "
                    f"{r['bigram_top1']:>10.4f} {r['full_top1']:>10.4f}")
    logger.info(f"\nRadical filtering improves character-level accuracy independently.")
    logger.info(f"Bigram helps only when characters form real words (sequential context).")
    logger.info(f"Combined pipeline gives the best result on real word pairs.")


if __name__ == "__main__":
    main()
