"""Evaluate structural post-processing on trained models.

Tests the impact of radical filtering, structure constraints, and bigram
re-ranking on model accuracy across different model architectures.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hccr.config import StructuralConfig
from hccr.data import HWDBDataset, LabelMap, get_eval_transform
from hccr.models import TinyCNN, TinyCNNJoint, MobileNetV3Wrapper
from hccr.structural import BigramModel, RadicalFilter, RadicalTable, StructuralPipeline
from hccr.utils import get_device, get_logger, load_checkpoint, save_json


def evaluate_model_with_structural(
    model_name: str,
    model: torch.nn.Module,
    checkpoint_path: Path,
    pipeline: StructuralPipeline,
    test_loader: DataLoader,
    device: torch.device,
    logger,
    alpha: float = 0.7,
) -> dict:
    """Evaluate a single model with structural post-processing.

    Args:
        model_name: Name of the model
        model: Model instance
        checkpoint_path: Path to model checkpoint
        pipeline: Structural pipeline instance
        test_loader: Test data loader
        device: Torch device
        logger: Logger instance
        alpha: Weight for classifier vs radical score

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name}")
    logger.info(f"{'='*60}")

    # Load checkpoint
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, device=device)
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return {
            "model": model_name,
            "error": "checkpoint_not_found",
        }

    # Update pipeline alpha
    pipeline.config.alpha = alpha

    # Determine evaluation mode based on model architecture
    mode = "joint" if hasattr(model, "radical_head") else "simple"

    # Evaluate with structural pipeline
    results = pipeline.evaluate_on_loader(
        model=model,
        test_loader=test_loader,
        device=device,
        mode=mode,
    )

    # Add metadata
    results["model"] = model_name
    results["alpha"] = alpha
    results["mode"] = mode

    logger.info(f"\nResults for {model_name} (alpha={alpha}):")
    logger.info(f"  Accuracy before: {results['acc_before']:.4f}")
    logger.info(f"  Accuracy after: {results['acc_after']:.4f}")
    logger.info(f"  Improvement: {results['improvement']:.4f}")

    return results


def sweep_alpha_values(
    model_name: str,
    model: torch.nn.Module,
    checkpoint_path: Path,
    radical_filter: RadicalFilter,
    bigram_model: BigramModel,
    label_map_dict: dict,
    test_loader: DataLoader,
    device: torch.device,
    logger,
    alpha_values: list,
) -> dict:
    """Sweep alpha values to find optimal configuration.

    Args:
        model_name: Model name
        model: Model instance
        checkpoint_path: Checkpoint path
        radical_filter: Radical filter
        bigram_model: Bigram model
        label_map_dict: Label map dictionary
        test_loader: Test data loader
        device: Device
        logger: Logger
        alpha_values: List of alpha values to try

    Returns:
        Dictionary with results for each alpha value
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Alpha sweep for {model_name}")
    logger.info(f"{'='*60}")

    results = []

    for alpha in alpha_values:
        logger.info(f"\nTesting alpha = {alpha}")

        # Create pipeline with current alpha
        config = StructuralConfig(alpha=alpha)
        pipeline = StructuralPipeline(
            radical_filter=radical_filter,
            bigram_model=bigram_model,
            label_map=label_map_dict,
            config=config,
        )

        # Evaluate
        result = evaluate_model_with_structural(
            model_name=model_name,
            model=model,
            checkpoint_path=checkpoint_path,
            pipeline=pipeline,
            test_loader=test_loader,
            device=device,
            logger=logger,
            alpha=alpha,
        )

        results.append(result)

    # Find best alpha
    best_result = max(results, key=lambda x: x.get("acc_after", 0))
    logger.info(f"\nBest alpha for {model_name}: {best_result['alpha']}")
    logger.info(f"  Best accuracy: {best_result['acc_after']:.4f}")

    return {
        "model": model_name,
        "results": results,
        "best_alpha": best_result["alpha"],
        "best_acc": best_result["acc_after"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate structural post-processing on trained models"
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
        "--radical-table",
        type=Path,
        default=Path("resources/radical_table.json"),
        help="Radical table JSON file",
    )
    parser.add_argument(
        "--bigram-table",
        type=Path,
        default=Path("resources/bigram_table.json"),
        help="Bigram table JSON file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
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

    radical_table = RadicalTable.load(args.radical_table)

    bigram_model = BigramModel.load(args.bigram_table)

    # Create radical filter
    radical_filter = RadicalFilter(radical_table)

    logger.info(f"Loaded {len(label_map)} characters")
    logger.info(f"Loaded {len(radical_table.all_radicals)} radicals")
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Define models to evaluate
    num_classes = len(label_map_dict)

    models_to_evaluate = {
        "tinycnn": {
            "model": TinyCNN(num_classes=num_classes),
            "checkpoint": args.checkpoint_dir / "tinycnn_best.pt",
        },
        "tinycnn_joint": {
            "model": TinyCNNJoint(
                num_classes=num_classes,
                num_radicals=len(radical_table.all_radicals),
                num_structures=13,
            ),
            "checkpoint": args.checkpoint_dir / "tinycnn_joint_best.pt",
        },
        "mobilenetv3": {
            "model": MobileNetV3Wrapper(num_classes=num_classes),
            "checkpoint": args.checkpoint_dir / "mobilenetv3_best.pt",
        },
    }

    # Alpha values to sweep
    alpha_values = [0.3, 0.5, 0.7, 0.9]

    all_results = {}

    # Evaluate each model with alpha sweep
    for model_name, model_info in models_to_evaluate.items():
        sweep_results = sweep_alpha_values(
            model_name=model_name,
            model=model_info["model"].to(device),
            checkpoint_path=model_info["checkpoint"],
            radical_filter=radical_filter,
            bigram_model=bigram_model,
            label_map_dict=label_map_dict,
            test_loader=test_loader,
            device=device,
            logger=logger,
            alpha_values=alpha_values,
        )

        all_results[model_name] = sweep_results

    # Save results
    output_file = args.output_dir / "structural_results.json"
    save_json(all_results, output_file)
    logger.info(f"\nResults saved to {output_file}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Structural Post-Processing Summary")
    logger.info(f"{'='*60}")
    for model_name, results in all_results.items():
        if "best_alpha" in results:
            logger.info(f"\n{model_name}:")
            logger.info(f"  Best alpha: {results['best_alpha']}")
            logger.info(f"  Best accuracy: {results['best_acc']:.4f}")


if __name__ == "__main__":
    main()
