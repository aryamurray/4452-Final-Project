"""Comprehensive efficiency benchmarking for HCCR models.

Measures model size, inference latency, parameter count, and quantization
effects for all model architectures.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hccr.data import HWDBDataset, LabelMap, get_eval_transform
from hccr.evaluation import (
    measure_inference_latency,
    measure_model_size,
    quantize_and_measure,
    run_full_benchmark,
    top_k_accuracy,
)
from hccr.models import (
    MobileNetV3Wrapper,
    TinyCNN,
    TinyCNNJoint,
    TinyCNNRadical,
)
from hccr.utils import (
    count_parameters,
    get_device,
    get_logger,
    load_checkpoint,
    save_json,
)


def load_model_with_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    logger,
) -> torch.nn.Module:
    """Load model checkpoint if it exists.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint
        device: Torch device
        logger: Logger instance

    Returns:
        Loaded model (or original if checkpoint not found)
    """
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, device=device)
        return model
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, using random weights")
        return model


def evaluate_model_accuracy(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model top-1 accuracy.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Torch device

    Returns:
        Top-1 accuracy percentage
    """
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            # Handle multi-output models
            if isinstance(outputs, dict):
                if "char_logits" not in outputs:
                    return -1.0  # Model has no char head (e.g., TinyCNNRadical)
                char_logits = outputs["char_logits"]
            else:
                char_logits = outputs

            all_outputs.append(char_logits.cpu())
            all_targets.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    acc = top_k_accuracy(all_outputs, all_targets, k_values=[1])
    return acc[1]


def benchmark_quantization(
    model: torch.nn.Module,
    model_name: str,
    test_loader: DataLoader,
    device: torch.device,
    logger,
) -> dict:
    """Benchmark quantization for a model.

    Args:
        model: Model to quantize
        model_name: Name of the model
        test_loader: Test data loader
        device: Torch device
        logger: Logger instance

    Returns:
        Dictionary with quantization results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Quantization benchmark: {model_name}")
    logger.info(f"{'='*60}")

    results = quantize_and_measure(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    logger.info(f"Original size: {results['original_size_mb']:.2f} MB")
    logger.info(f"Quantized size: {results['quantized_size_mb']:.2f} MB")
    logger.info(f"Compression ratio: {results['compression_ratio']:.2f}x")
    logger.info(f"Original accuracy: {results['original_acc']:.2f}%")
    logger.info(f"Quantized accuracy: {results['quantized_acc']:.2f}%")
    logger.info(f"Accuracy drop: {results['original_acc'] - results['quantized_acc']:.2f}%")

    return results


def plot_efficiency_pareto(
    benchmark_df: pd.DataFrame,
    output_path: Path,
    logger,
) -> None:
    """Generate Pareto plot of model size vs accuracy.

    Args:
        benchmark_df: Benchmark results dataframe
        output_path: Path to save plot
        logger: Logger instance
    """
    logger.info(f"\nGenerating efficiency Pareto plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for _, row in benchmark_df.iterrows():
        ax.scatter(
            row["size_mb"],
            row.get("top1_acc", 0),
            s=100,
            alpha=0.7,
            label=row["model"],
        )
        ax.annotate(
            row["model"],
            (row["size_mb"], row.get("top1_acc", 0)),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Model Size (MB)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Efficiency Pareto Frontier: Size vs Accuracy", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Pareto plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive efficiency benchmarks"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Checkpoint directory",
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
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip quantization benchmark",
    )

    args = parser.parse_args()

    # Setup
    logger = get_logger(__name__)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load label map
    logger.info("Loading resources...")
    label_map = LabelMap.load(args.label_map)
    label_map_dict = {i: label_map.decode(i) for i in range(len(label_map))}
    num_classes = len(label_map)

    logger.info(f"Loaded {num_classes} characters")

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

    # Define models
    logger.info("\nInitializing models...")
    models_dict = {
        "TinyCNN": TinyCNN(num_classes=num_classes),
        "TinyCNNRadical": TinyCNNRadical(
            num_radicals=500,
            num_structures=13,
        ),
        "TinyCNNJoint": TinyCNNJoint(
            num_classes=num_classes,
            num_radicals=500,
            num_structures=13,
        ),
        "MobileNetV3": MobileNetV3Wrapper(num_classes=num_classes),
    }

    # Load checkpoints
    checkpoint_mapping = {
        "TinyCNN": args.checkpoint_dir / "tinycnn_best.pt",
        "TinyCNNRadical": args.checkpoint_dir / "tinycnn_radical_best.pt",
        "TinyCNNJoint": args.checkpoint_dir / "tinycnn_joint_best.pt",
        "MobileNetV3": args.checkpoint_dir / "mobilenetv3_best.pt",
    }

    for model_name, model in models_dict.items():
        checkpoint_path = checkpoint_mapping[model_name]
        models_dict[model_name] = load_model_with_checkpoint(
            model=model.to(device),
            checkpoint_path=checkpoint_path,
            device=device,
            logger=logger,
        )

    # Run full benchmark
    logger.info(f"\n{'='*60}")
    logger.info("Running comprehensive benchmark")
    logger.info(f"{'='*60}")

    benchmark_df = run_full_benchmark(
        models=models_dict,
        device=device,
        test_loader=test_loader,
    )

    # Save benchmark table
    csv_path = args.output_dir / "benchmark_table.csv"
    benchmark_df.to_csv(csv_path, index=False)
    logger.info(f"\nBenchmark table saved to {csv_path}")

    # Display table
    logger.info("\nBenchmark Results:")
    logger.info("\n" + benchmark_df.to_string(index=False))

    # Quantization benchmark (only for TinyCNNJoint)
    quantization_results = None
    if not args.skip_quantization:
        quantization_results = benchmark_quantization(
            model=models_dict["TinyCNNJoint"],
            model_name="TinyCNNJoint",
            test_loader=test_loader,
            device=device,
            logger=logger,
        )

    # Compile all results
    all_results = {
        "benchmark_table": benchmark_df.to_dict(orient="records"),
        "quantization": quantization_results,
    }

    # Save JSON results
    json_path = args.output_dir / "benchmark_results.json"
    save_json(all_results, json_path)
    logger.info(f"\nBenchmark results saved to {json_path}")

    # Generate Pareto plot
    if "top1_acc" in benchmark_df.columns:
        pareto_plot_path = figures_dir / "efficiency_pareto.png"
        plot_efficiency_pareto(
            benchmark_df=benchmark_df,
            output_path=pareto_plot_path,
            logger=logger,
        )

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Benchmark Summary")
    logger.info(f"{'='*60}")

    # Find best model by different criteria
    if "top1_acc" in benchmark_df.columns:
        best_acc = benchmark_df.loc[benchmark_df["top1_acc"].idxmax()]
        logger.info(f"\nBest accuracy: {best_acc['model']} ({best_acc['top1_acc']:.2f}%)")

    smallest = benchmark_df.loc[benchmark_df["size_mb"].idxmin()]
    logger.info(f"Smallest model: {smallest['model']} ({smallest['size_mb']:.2f} MB)")

    fastest = benchmark_df.loc[benchmark_df["latency_mean_ms"].idxmin()]
    logger.info(f"Fastest model: {fastest['model']} ({fastest['latency_mean_ms']:.2f} ms)")

    # Compute efficiency scores (accuracy / size)
    if "top1_acc" in benchmark_df.columns:
        benchmark_df["efficiency_score"] = benchmark_df["top1_acc"] / benchmark_df["size_mb"]
        most_efficient = benchmark_df.loc[benchmark_df["efficiency_score"].idxmax()]
        logger.info(
            f"Most efficient: {most_efficient['model']} "
            f"(score: {most_efficient['efficiency_score']:.2f})"
        )


if __name__ == "__main__":
    main()
