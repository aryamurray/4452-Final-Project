"""Evaluate bigram language model in three test settings.

Tests bigram re-ranking effectiveness on:
1. Random character pairs from test set
2. Real word pairs from SUBTLEX-CH corpus
3. Adversarial pairs with visually similar characters
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hccr.data import HWDBDataset, LabelMap, get_eval_transform
from hccr.models import TinyCNNJoint
from hccr.structural import BigramModel
from hccr.utils import get_device, get_logger, load_checkpoint, save_json


def evaluate_random_pairs(
    model: torch.nn.Module,
    test_dataset: HWDBDataset,
    bigram_model: BigramModel,
    label_map_dict: dict,
    device: torch.device,
    logger,
    num_samples: int = 1000,
) -> dict:
    """Evaluate bigram on random character pairs from test set.

    Args:
        model: Trained model
        test_dataset: Test dataset
        bigram_model: Bigram language model
        label_map_dict: Label map dictionary
        device: Torch device
        logger: Logger instance
        num_samples: Number of random pairs to sample

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info("Setting 1: Random Character Pairs")
    logger.info(f"{'='*60}")

    model.eval()
    correct_without_bigram = 0
    correct_with_bigram = 0
    total_pairs = 0

    # Sample random pairs
    random.seed(42)
    indices = random.sample(range(len(test_dataset) - 1), min(num_samples, len(test_dataset) - 1))

    with torch.no_grad():
        for idx in indices:
            # Get two consecutive samples
            sample1 = test_dataset[idx]
            sample2 = test_dataset[idx + 1]

            image1 = sample1["image"].unsqueeze(0).to(device)
            image2 = sample2["image"].unsqueeze(0).to(device)
            label1 = sample1["label"]
            label2 = sample2["label"]

            # Get predictions
            outputs1 = model(image1)
            outputs2 = model(image2)

            char_logits1 = outputs1["char_logits"] if isinstance(outputs1, dict) else outputs1
            char_logits2 = outputs2["char_logits"] if isinstance(outputs2, dict) else outputs2

            probs1 = torch.softmax(char_logits1, dim=1).cpu().numpy()[0]
            probs2 = torch.softmax(char_logits2, dim=1).cpu().numpy()[0]

            # Prediction without bigram
            pred1_no_bigram = int(np.argmax(probs1))
            pred2_no_bigram = int(np.argmax(probs2))

            # Prediction with bigram re-ranking
            # Get top-10 candidates for second character
            top_k = 10
            top_indices2 = np.argsort(probs2)[-top_k:][::-1]

            # Re-rank using bigram
            char1 = label_map_dict.get(pred1_no_bigram, "")
            best_score = -float('inf')
            pred2_with_bigram = pred2_no_bigram

            for cand_idx in top_indices2:
                char2 = label_map_dict.get(cand_idx, "")
                # Combine classifier score with bigram score
                classifier_score = probs2[cand_idx]
                bigram_score = np.exp(bigram_model.log_prob(char1, char2))
                combined_score = 0.7 * classifier_score + 0.3 * bigram_score

                if combined_score > best_score:
                    best_score = combined_score
                    pred2_with_bigram = cand_idx

            # Check correctness
            if pred1_no_bigram == label1 and pred2_no_bigram == label2:
                correct_without_bigram += 1
            if pred1_no_bigram == label1 and pred2_with_bigram == label2:
                correct_with_bigram += 1

            total_pairs += 1

    acc_without = correct_without_bigram / total_pairs if total_pairs > 0 else 0
    acc_with = correct_with_bigram / total_pairs if total_pairs > 0 else 0

    logger.info(f"Evaluated {total_pairs} random pairs")
    logger.info(f"Accuracy without bigram: {acc_without:.4f}")
    logger.info(f"Accuracy with bigram: {acc_with:.4f}")
    logger.info(f"Improvement: {acc_with - acc_without:.4f}")

    return {
        "setting": "random_pairs",
        "total_pairs": total_pairs,
        "acc_without_bigram": acc_without,
        "acc_with_bigram": acc_with,
        "improvement": acc_with - acc_without,
    }


def evaluate_real_word_pairs(
    model: torch.nn.Module,
    test_dataset: HWDBDataset,
    bigram_model: BigramModel,
    label_map_dict: dict,
    device: torch.device,
    logger,
) -> dict:
    """Evaluate bigram on real word pairs (simulated from common bigrams).

    Args:
        model: Trained model
        test_dataset: Test dataset
        bigram_model: Bigram language model
        label_map_dict: Label map dictionary
        device: Torch device
        logger: Logger instance

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info("Setting 2: Real Word Pairs (High-frequency bigrams)")
    logger.info(f"{'='*60}")

    model.eval()

    # Get high-frequency bigrams from the model
    # Sort bigrams by frequency/probability
    bigram_pairs = []
    for char1, next_chars in bigram_model.bigram_counts.items():
        for char2, count in next_chars.items():
            if count > 5:  # Only frequent bigrams
                bigram_pairs.append((char1, char2, count))

    # Sort by count and take top 200
    bigram_pairs.sort(key=lambda x: x[2], reverse=True)
    bigram_pairs = bigram_pairs[:200]

    logger.info(f"Testing {len(bigram_pairs)} high-frequency bigrams")

    # For each bigram, find samples in test set
    char_to_indices = {}
    for idx in range(len(test_dataset)):
        label = test_dataset.index[idx][2]  # Character string
        if label not in char_to_indices:
            char_to_indices[label] = []
        char_to_indices[label].append(idx)

    correct_without_bigram = 0
    correct_with_bigram = 0
    total_pairs = 0

    random.seed(42)

    with torch.no_grad():
        for char1, char2, _ in bigram_pairs:
            # Find samples for these characters
            if char1 not in char_to_indices or char2 not in char_to_indices:
                continue

            indices1 = char_to_indices[char1]
            indices2 = char_to_indices[char2]

            if not indices1 or not indices2:
                continue

            # Sample one instance of each
            idx1 = random.choice(indices1)
            idx2 = random.choice(indices2)

            sample1 = test_dataset[idx1]
            sample2 = test_dataset[idx2]

            image1 = sample1["image"].unsqueeze(0).to(device)
            image2 = sample2["image"].unsqueeze(0).to(device)
            label1 = sample1["label"]
            label2 = sample2["label"]

            # Get predictions
            outputs1 = model(image1)
            outputs2 = model(image2)

            char_logits1 = outputs1["char_logits"] if isinstance(outputs1, dict) else outputs1
            char_logits2 = outputs2["char_logits"] if isinstance(outputs2, dict) else outputs2

            probs1 = torch.softmax(char_logits1, dim=1).cpu().numpy()[0]
            probs2 = torch.softmax(char_logits2, dim=1).cpu().numpy()[0]

            # Prediction without bigram
            pred1_no_bigram = int(np.argmax(probs1))
            pred2_no_bigram = int(np.argmax(probs2))

            # Prediction with bigram
            top_k = 10
            top_indices2 = np.argsort(probs2)[-top_k:][::-1]

            char1_pred = label_map_dict.get(pred1_no_bigram, "")
            best_score = -float('inf')
            pred2_with_bigram = pred2_no_bigram

            for cand_idx in top_indices2:
                char2_cand = label_map_dict.get(cand_idx, "")
                classifier_score = probs2[cand_idx]
                bigram_score = np.exp(bigram_model.log_prob(char1_pred, char2_cand))
                combined_score = 0.7 * classifier_score + 0.3 * bigram_score

                if combined_score > best_score:
                    best_score = combined_score
                    pred2_with_bigram = cand_idx

            # Check correctness
            if pred1_no_bigram == label1 and pred2_no_bigram == label2:
                correct_without_bigram += 1
            if pred1_no_bigram == label1 and pred2_with_bigram == label2:
                correct_with_bigram += 1

            total_pairs += 1

    acc_without = correct_without_bigram / total_pairs if total_pairs > 0 else 0
    acc_with = correct_with_bigram / total_pairs if total_pairs > 0 else 0

    logger.info(f"Evaluated {total_pairs} real word pairs")
    logger.info(f"Accuracy without bigram: {acc_without:.4f}")
    logger.info(f"Accuracy with bigram: {acc_with:.4f}")
    logger.info(f"Improvement: {acc_with - acc_without:.4f}")

    return {
        "setting": "real_word_pairs",
        "total_pairs": total_pairs,
        "acc_without_bigram": acc_without,
        "acc_with_bigram": acc_with,
        "improvement": acc_with - acc_without,
    }


def evaluate_adversarial_pairs(
    model: torch.nn.Module,
    test_dataset: HWDBDataset,
    bigram_model: BigramModel,
    label_map_dict: dict,
    device: torch.device,
    logger,
) -> dict:
    """Evaluate bigram on adversarial pairs (visually similar characters).

    Args:
        model: Trained model
        test_dataset: Test dataset
        bigram_model: Bigram language model
        label_map_dict: Label map dictionary
        device: Torch device
        logger: Logger instance

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info("Setting 3: Adversarial Pairs (Visually similar)")
    logger.info(f"{'='*60}")

    # Find confusable pairs by running model and collecting top-2 predictions
    model.eval()
    confusion_pairs = {}  # Maps true_label -> confused_label

    # Sample subset of test set to find confusions
    random.seed(42)
    sample_indices = random.sample(range(len(test_dataset)), min(5000, len(test_dataset)))

    logger.info("Finding confusable character pairs...")

    with torch.no_grad():
        for idx in sample_indices:
            sample = test_dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"]

            outputs = model(image)
            char_logits = outputs["char_logits"] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(char_logits, dim=1).cpu().numpy()[0]

            # Get top-2 predictions
            top2 = np.argsort(probs)[-2:][::-1]
            pred1, pred2 = int(top2[0]), int(top2[1])

            # If model is confused (top-1 is wrong but close)
            if pred1 != label and probs[pred1] > 0.3:
                confusion_pairs[label] = pred1

    logger.info(f"Found {len(confusion_pairs)} confusable pairs")

    # Now test bigram on these adversarial pairs
    correct_without_bigram = 0
    correct_with_bigram = 0
    total_pairs = 0

    with torch.no_grad():
        for true_label, confused_label in list(confusion_pairs.items())[:500]:
            # Find samples
            sample1_idx = random.choice([i for i in range(len(test_dataset)) if test_dataset.index[i][2] == label_map_dict.get(true_label, "")])
            sample2_idx = random.choice([i for i in range(len(test_dataset)) if i != sample1_idx])

            sample1 = test_dataset[sample1_idx]
            sample2 = test_dataset[sample2_idx]

            image1 = sample1["image"].unsqueeze(0).to(device)
            image2 = sample2["image"].unsqueeze(0).to(device)
            label1 = sample1["label"]
            label2 = sample2["label"]

            # Get predictions
            outputs1 = model(image1)
            outputs2 = model(image2)

            char_logits1 = outputs1["char_logits"] if isinstance(outputs1, dict) else outputs1
            char_logits2 = outputs2["char_logits"] if isinstance(outputs2, dict) else outputs2

            probs1 = torch.softmax(char_logits1, dim=1).cpu().numpy()[0]
            probs2 = torch.softmax(char_logits2, dim=1).cpu().numpy()[0]

            # Prediction without bigram
            pred1_no_bigram = int(np.argmax(probs1))
            pred2_no_bigram = int(np.argmax(probs2))

            # Prediction with bigram
            top_k = 10
            top_indices2 = np.argsort(probs2)[-top_k:][::-1]

            char1_pred = label_map_dict.get(pred1_no_bigram, "")
            best_score = -float('inf')
            pred2_with_bigram = pred2_no_bigram

            for cand_idx in top_indices2:
                char2_cand = label_map_dict.get(cand_idx, "")
                classifier_score = probs2[cand_idx]
                bigram_score = np.exp(bigram_model.log_prob(char1_pred, char2_cand))
                combined_score = 0.7 * classifier_score + 0.3 * bigram_score

                if combined_score > best_score:
                    best_score = combined_score
                    pred2_with_bigram = cand_idx

            # Check correctness
            if pred1_no_bigram == label1 and pred2_no_bigram == label2:
                correct_without_bigram += 1
            if pred1_no_bigram == label1 and pred2_with_bigram == label2:
                correct_with_bigram += 1

            total_pairs += 1

    acc_without = correct_without_bigram / total_pairs if total_pairs > 0 else 0
    acc_with = correct_with_bigram / total_pairs if total_pairs > 0 else 0

    logger.info(f"Evaluated {total_pairs} adversarial pairs")
    logger.info(f"Accuracy without bigram: {acc_without:.4f}")
    logger.info(f"Accuracy with bigram: {acc_with:.4f}")
    logger.info(f"Improvement: {acc_with - acc_without:.4f}")

    return {
        "setting": "adversarial_pairs",
        "total_pairs": total_pairs,
        "acc_without_bigram": acc_without,
        "acc_with_bigram": acc_with,
        "improvement": acc_with - acc_without,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bigram model in three test settings"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/HWDB1.1/test"),
        help="Test data directory",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("resources/label_map.json"),
        help="Label map JSON file",
    )
    parser.add_argument(
        "--bigram-table",
        type=Path,
        default=Path("resources/bigram_table.json"),
        help="Bigram table JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/tinycnn_joint_best.pt"),
        help="Model checkpoint (joint model recommended)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Setup
    logger = get_logger(__name__)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load resources
    logger.info("Loading resources...")
    label_map = LabelMap.load(args.label_map)
    label_map_dict = {i: label_map.decode(i) for i in range(len(label_map))}

    bigram_model = BigramModel.load(args.bigram_table)

    logger.info(f"Loaded {len(label_map)} characters")
    logger.info(f"Loaded bigram model with {bigram_model.vocab_size} vocab")

    # Create test dataset
    logger.info(f"Loading test data from {args.test_dir}")
    test_transform = get_eval_transform(image_size=64)

    test_dataset = HWDBDataset(
        gnt_dir=args.test_dir,
        label_map=label_map,
        transform=test_transform,
        index_cache_path=Path("outputs/cache") / "test_index.pkl",
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = TinyCNNJoint(
        num_classes=len(label_map_dict),
        num_radicals=500,  # Default
        num_structures=13,
    ).to(device)

    if args.checkpoint.exists():
        load_checkpoint(model, args.checkpoint, device=device)
    else:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    # Run evaluations
    results = {}

    # Setting 1: Random pairs
    results["random_pairs"] = evaluate_random_pairs(
        model=model,
        test_dataset=test_dataset,
        bigram_model=bigram_model,
        label_map_dict=label_map_dict,
        device=device,
        logger=logger,
        num_samples=1000,
    )

    # Setting 2: Real word pairs
    results["real_word_pairs"] = evaluate_real_word_pairs(
        model=model,
        test_dataset=test_dataset,
        bigram_model=bigram_model,
        label_map_dict=label_map_dict,
        device=device,
        logger=logger,
    )

    # Setting 3: Adversarial pairs
    results["adversarial_pairs"] = evaluate_adversarial_pairs(
        model=model,
        test_dataset=test_dataset,
        bigram_model=bigram_model,
        label_map_dict=label_map_dict,
        device=device,
        logger=logger,
    )

    # Save results
    output_file = args.output_dir / "bigram_settings_results.json"
    save_json(results, output_file)
    logger.info(f"\nResults saved to {output_file}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Bigram Evaluation Summary")
    logger.info(f"{'='*60}")
    for setting_name, result in results.items():
        logger.info(f"\n{result['setting']}:")
        logger.info(f"  Pairs evaluated: {result['total_pairs']}")
        logger.info(f"  Accuracy without bigram: {result['acc_without_bigram']:.4f}")
        logger.info(f"  Accuracy with bigram: {result['acc_with_bigram']:.4f}")
        logger.info(f"  Improvement: {result['improvement']:.4f}")


if __name__ == "__main__":
    main()
