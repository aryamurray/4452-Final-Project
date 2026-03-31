"""Full orchestration script for HCCR project.

Runs the complete pipeline:
1. Build resources (label_map, radical_table, bigram_table)
2. Train all models (tinycnn, tinycnn_radical, tinycnn_joint, mobilenetv3)
3. Evaluate CLIP zero-shot
4. Evaluate structural post-processing
5. Evaluate bigram settings
6. Run benchmarks
7. Generate summary report
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hccr.utils import get_logger, load_json


def run_command(
    cmd: list,
    description: str,
    logger,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command and log output.

    Args:
        cmd: Command and arguments as list
        description: Human-readable description
        logger: Logger instance
        check: Whether to raise on non-zero exit

    Returns:
        CompletedProcess instance
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{description}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {' '.join(str(c) for c in cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=False,  # Show output in real-time
            text=True,
        )
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.1f}s")
        return result

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed after {elapsed:.1f}s with exit code {e.returncode}")
        raise


def build_resources(data_dir: Path, output_dir: Path, logger) -> None:
    """Build all required resources.

    Args:
        data_dir: Data directory
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info(f"\n{'#'*60}")
    logger.info("STEP 1: Building Resources")
    logger.info(f"{'#'*60}")

    scripts_dir = Path(__file__).parent

    # Build label map
    run_command(
        cmd=[
            sys.executable,
            str(scripts_dir / "build_label_map.py"),
            "--train-dir", str(data_dir / "HWDB1.1/train"),
            "--test-dir", str(data_dir / "HWDB1.1/test"),
            "--output", str(output_dir / "resources/label_map.json"),
        ],
        description="Building label map",
        logger=logger,
        check=False,  # May already exist
    )

    # Build radical table
    run_command(
        cmd=[
            sys.executable,
            str(scripts_dir / "build_radical_table.py"),
            "--ids-file", str(output_dir / "resources/ids.txt"),
            "--label-map", str(output_dir / "resources/label_map.json"),
            "--output", str(output_dir / "resources/radical_table.json"),
        ],
        description="Building radical table",
        logger=logger,
        check=False,
    )

    # Build bigram table
    run_command(
        cmd=[
            sys.executable,
            str(scripts_dir / "build_bigram_table.py"),
            "--freq-file", str(data_dir / "SUBTLEX-CH/SUBTLEX_CH_131210_CE.utf8"),
            "--label-map", str(output_dir / "resources/label_map.json"),
            "--output", str(output_dir / "resources/bigram_table.json"),
        ],
        description="Building bigram table",
        logger=logger,
        check=False,
    )


def train_models(data_dir: Path, output_dir: Path, logger) -> None:
    """Train all models.

    Args:
        data_dir: Data directory
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info(f"\n{'#'*60}")
    logger.info("STEP 2: Training Models")
    logger.info(f"{'#'*60}")

    scripts_dir = Path(__file__).parent

    models = [
        ("TinyCNN", "train_tinycnn.py"),
        ("TinyCNN Radical", "train_tinycnn_radical.py"),
        ("TinyCNN Joint", "train_tinycnn_joint.py"),
        ("MobileNetV3", "train_mobilenetv3.py"),
    ]

    for model_name, script_name in models:
        run_command(
            cmd=[
                sys.executable,
                str(scripts_dir / script_name),
                "--data-dir", str(data_dir),
                "--output-dir", str(output_dir),
                "--epochs", "30",
            ],
            description=f"Training {model_name}",
            logger=logger,
        )


def evaluate_all(data_dir: Path, output_dir: Path, logger) -> None:
    """Run all evaluation scripts.

    Args:
        data_dir: Data directory
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info(f"\n{'#'*60}")
    logger.info("STEP 3: Running Evaluations")
    logger.info(f"{'#'*60}")

    scripts_dir = Path(__file__).parent

    # CLIP zero-shot evaluation
    run_command(
        cmd=[
            sys.executable,
            str(scripts_dir / "eval_clip_zeroshot.py"),
            "--test-dir", str(data_dir / "HWDB1.1/test"),
            "--label-map", str(output_dir / "resources/label_map.json"),
            "--cache-dir", str(output_dir / "cache"),
            "--output-dir", str(output_dir / "results"),
        ],
        description="Evaluating CLIP zero-shot",
        logger=logger,
    )

    # Structural post-processing evaluation
    run_command(
        cmd=[
            sys.executable,
            str(scripts_dir / "eval_structural.py"),
            "--test-dir", str(data_dir / "HWDB1.1/test"),
            "--label-map", str(output_dir / "resources/label_map.json"),
            "--radical-table", str(output_dir / "resources/radical_table.json"),
            "--bigram-table", str(output_dir / "resources/bigram_table.json"),
            "--checkpoint-dir", str(output_dir / "checkpoints"),
            "--output-dir", str(output_dir / "results"),
        ],
        description="Evaluating structural post-processing",
        logger=logger,
    )

    # Bigram settings evaluation
    run_command(
        cmd=[
            sys.executable,
            str(scripts_dir / "eval_bigram_settings.py"),
            "--test-dir", str(data_dir / "HWDB1.1/test"),
            "--label-map", str(output_dir / "resources/label_map.json"),
            "--bigram-table", str(output_dir / "resources/bigram_table.json"),
            "--checkpoint", str(output_dir / "checkpoints/tinycnn_joint_best.pt"),
            "--output-dir", str(output_dir / "results"),
        ],
        description="Evaluating bigram settings",
        logger=logger,
    )

    # Benchmarks
    run_command(
        cmd=[
            sys.executable,
            str(scripts_dir / "run_benchmarks.py"),
            "--checkpoint-dir", str(output_dir / "checkpoints"),
            "--test-dir", str(data_dir / "HWDB1.1/test"),
            "--label-map", str(output_dir / "resources/label_map.json"),
            "--output-dir", str(output_dir / "results"),
        ],
        description="Running efficiency benchmarks",
        logger=logger,
    )


def generate_summary_report(output_dir: Path, logger) -> None:
    """Generate final summary report.

    Args:
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info(f"\n{'#'*60}")
    logger.info("STEP 4: Generating Summary Report")
    logger.info(f"{'#'*60}")

    results_dir = output_dir / "results"

    # Load all results
    try:
        clip_results = load_json(results_dir / "clip_zeroshot_results.json")
        structural_results = load_json(results_dir / "structural_results.json")
        bigram_results = load_json(results_dir / "bigram_settings_results.json")
        benchmark_results = load_json(results_dir / "benchmark_results.json")
    except FileNotFoundError as e:
        logger.warning(f"Could not load all results: {e}")
        return

    # Print comprehensive summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY REPORT")
    logger.info(f"{'='*60}")

    # CLIP Results
    logger.info("\n1. CLIP Zero-Shot Performance:")
    logger.info(f"   Multilingual (Chinese prompts):")
    logger.info(f"     Top-1: {clip_results['multilingual']['top1_acc']:.2f}%")
    logger.info(f"     Top-5: {clip_results['multilingual']['top5_acc']:.2f}%")
    logger.info(f"   English prompts:")
    logger.info(f"     Top-1: {clip_results['english']['top1_acc']:.2f}%")
    logger.info(f"     Top-5: {clip_results['english']['top5_acc']:.2f}%")

    # Structural Results
    logger.info("\n2. Structural Post-Processing (Best Alpha):")
    for model_name, results in structural_results.items():
        if "best_alpha" in results:
            logger.info(f"   {model_name}:")
            logger.info(f"     Best alpha: {results['best_alpha']}")
            logger.info(f"     Best accuracy: {results['best_acc']:.4f}")

    # Bigram Results
    logger.info("\n3. Bigram Re-ranking Effectiveness:")
    for setting_name, result in bigram_results.items():
        logger.info(f"   {result['setting']}:")
        logger.info(f"     Improvement: {result['improvement']:.4f}")
        logger.info(f"     Accuracy: {result['acc_without_bigram']:.4f} -> {result['acc_with_bigram']:.4f}")

    # Benchmark Results
    logger.info("\n4. Model Efficiency Benchmark:")
    if "benchmark_table" in benchmark_results:
        for model in benchmark_results["benchmark_table"]:
            logger.info(f"   {model['model']}:")
            logger.info(f"     Size: {model['size_mb']:.2f} MB")
            logger.info(f"     Parameters: {model['params']:,}")
            logger.info(f"     Latency: {model['latency_mean_ms']:.2f} ms")
            if "top1_acc" in model:
                logger.info(f"     Accuracy: {model['top1_acc']:.2f}%")

    # Quantization Results
    if benchmark_results.get("quantization"):
        quant = benchmark_results["quantization"]
        logger.info("\n5. Quantization (TinyCNNJoint):")
        logger.info(f"   Original size: {quant['original_size_mb']:.2f} MB")
        logger.info(f"   Quantized size: {quant['quantized_size_mb']:.2f} MB")
        logger.info(f"   Compression: {quant['compression_ratio']:.2f}x")
        logger.info(f"   Accuracy drop: {quant['original_acc'] - quant['quantized_acc']:.2f}%")

    # Save summary to text file
    summary_file = output_dir / "results" / "summary_report.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("HCCR Project Summary Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        f.write("1. CLIP Zero-Shot Performance\n")
        f.write(f"   Multilingual: Top-1={clip_results['multilingual']['top1_acc']:.2f}%, "
                f"Top-5={clip_results['multilingual']['top5_acc']:.2f}%\n")
        f.write(f"   English: Top-1={clip_results['english']['top1_acc']:.2f}%, "
                f"Top-5={clip_results['english']['top5_acc']:.2f}%\n\n")

        f.write("2. Structural Post-Processing\n")
        for model_name, results in structural_results.items():
            if "best_alpha" in results:
                f.write(f"   {model_name}: alpha={results['best_alpha']}, "
                        f"acc={results['best_acc']:.4f}\n")

        f.write("\n3. Bigram Re-ranking\n")
        for setting_name, result in bigram_results.items():
            f.write(f"   {result['setting']}: improvement={result['improvement']:.4f}\n")

        f.write("\n4. Model Efficiency\n")
        if "benchmark_table" in benchmark_results:
            for model in benchmark_results["benchmark_table"]:
                f.write(f"   {model['model']}: {model['size_mb']:.2f} MB, "
                        f"{model['params']:,} params, {model['latency_mean_ms']:.2f} ms\n")

    logger.info(f"\nSummary report saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete HCCR pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip resource building (if already done)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (if already done)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation (only generate summary)",
    )

    args = parser.parse_args()

    # Setup
    logger = get_logger(__name__)
    start_time = time.time()

    logger.info("="*60)
    logger.info("HCCR Full Pipeline Orchestration")
    logger.info("="*60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resources").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache").mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Build resources
        if not args.skip_build:
            build_resources(args.data_dir, args.output_dir, logger)
        else:
            logger.info("\nSkipping resource building (--skip-build)")

        # Step 2: Train models
        if not args.skip_training:
            train_models(args.data_dir, args.output_dir, logger)
        else:
            logger.info("\nSkipping model training (--skip-training)")

        # Step 3: Run evaluations
        if not args.skip_evaluation:
            evaluate_all(args.data_dir, args.output_dir, logger)
        else:
            logger.info("\nSkipping evaluation (--skip-evaluation)")

        # Step 4: Generate summary
        generate_summary_report(args.output_dir, logger)

        # Final report
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"\n{'='*60}")
        logger.error("PIPELINE FAILED")
        logger.error(f"{'='*60}")
        logger.error(f"Error: {e}")
        logger.error(f"Time elapsed: {elapsed/60:.1f} minutes")
        raise


if __name__ == "__main__":
    main()
