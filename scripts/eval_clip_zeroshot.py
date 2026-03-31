"""Evaluate CLIP zero-shot classification on HWDB test set.

Runs both multilingual (Chinese prompts) and English CLIP models for zero-shot
Chinese character recognition.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hccr.data import HWDBDataset, LabelMap, get_clip_transform
from hccr.evaluation import top_k_accuracy
from hccr.models import CLIPZeroShot
from hccr.utils import get_device, get_logger, save_json


def evaluate_clip_mode(
    mode: str,
    test_loader: DataLoader,
    label_map_dict: dict,
    device: torch.device,
    cache_dir: Path,
    logger,
) -> dict:
    """Evaluate CLIP in specified mode (multilingual or english).

    Args:
        mode: "multilingual" or "english"
        test_loader: Test data loader with CLIP transforms
        label_map_dict: Dictionary mapping indices to characters
        device: Torch device
        cache_dir: Cache directory for text embeddings
        logger: Logger instance

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating CLIP Zero-Shot ({mode})")
    logger.info(f"{'='*60}")

    # Create CLIP model
    clip_model = CLIPZeroShot(
        mode=mode,
        label_map=label_map_dict,
        device=device,
        cache_dir=cache_dir,
    )

    all_predictions = []
    all_targets = []

    logger.info(f"Running inference on test set...")

    # Iterate over test set
    for batch in tqdm(test_loader, desc=f"CLIP-{mode}"):
        images = batch["image"]
        labels = batch["label"]

        # Predict batch
        pred_indices, pred_scores = clip_model.predict_batch(
            images=images,
            top_k=10,
            batch_size=32
        )

        # Store predictions (top-1 only for now, but keep top-10 for metrics)
        all_predictions.append(torch.from_numpy(pred_indices))
        all_targets.append(labels)

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)  # (N, 10)
    all_targets = torch.cat(all_targets, dim=0)  # (N,)

    # Compute top-k accuracies
    # Convert predictions to scores (just use rank-based scores)
    # Create dummy logits where top predictions have highest scores
    num_classes = len(label_map_dict)
    logits = torch.zeros(len(all_predictions), num_classes)
    for i in range(len(all_predictions)):
        for rank, pred_idx in enumerate(all_predictions[i]):
            logits[i, pred_idx] = 10 - rank  # Higher score for better rank

    accuracies = top_k_accuracy(logits, all_targets, k_values=[1, 5, 10])

    logger.info(f"\nResults for CLIP-{mode}:")
    logger.info(f"  Top-1 Accuracy: {accuracies[1]:.2f}%")
    logger.info(f"  Top-5 Accuracy: {accuracies[5]:.2f}%")
    logger.info(f"  Top-10 Accuracy: {accuracies[10]:.2f}%")

    return {
        "mode": mode,
        "top1_acc": accuracies[1],
        "top5_acc": accuracies[5],
        "top10_acc": accuracies[10],
        "total_samples": len(all_targets),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP zero-shot on HWDB test set"
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
        "--cache-dir",
        type=Path,
        default=Path("outputs/cache"),
        help="Cache directory for CLIP text embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
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

    # Load label map
    logger.info(f"Loading label map from {args.label_map}")
    label_map = LabelMap.load(args.label_map)
    label_map_dict = {i: label_map.decode(i) for i in range(len(label_map))}

    logger.info(f"Loaded {len(label_map)} characters")

    # Create test dataset with CLIP transforms
    logger.info(f"Loading test data from {args.test_dir}")
    clip_transform = get_clip_transform(image_size=224)

    test_dataset = HWDBDataset(
        gnt_dir=args.test_dir,
        label_map=label_map,
        transform=clip_transform,
        index_cache_path=args.cache_dir / "test_index.pkl",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Evaluate both modes
    results = {}

    # Mode 1: Multilingual (Chinese prompts)
    results["multilingual"] = evaluate_clip_mode(
        mode="multilingual",
        test_loader=test_loader,
        label_map_dict=label_map_dict,
        device=device,
        cache_dir=args.cache_dir,
        logger=logger,
    )

    # Mode 2: English prompts
    results["english"] = evaluate_clip_mode(
        mode="english",
        test_loader=test_loader,
        label_map_dict=label_map_dict,
        device=device,
        cache_dir=args.cache_dir,
        logger=logger,
    )

    # Save results
    output_file = args.output_dir / "clip_zeroshot_results.json"
    save_json(results, output_file)
    logger.info(f"\nResults saved to {output_file}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("CLIP Zero-Shot Evaluation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Multilingual (Chinese prompts):")
    logger.info(f"  Top-1: {results['multilingual']['top1_acc']:.2f}%")
    logger.info(f"  Top-5: {results['multilingual']['top5_acc']:.2f}%")
    logger.info(f"\nEnglish prompts:")
    logger.info(f"  Top-1: {results['english']['top1_acc']:.2f}%")
    logger.info(f"  Top-5: {results['english']['top5_acc']:.2f}%")


if __name__ == "__main__":
    main()
