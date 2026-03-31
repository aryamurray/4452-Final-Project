"""Main CLI dispatcher for HCCR project.

Provides subcommands for all major pipeline operations:
- Data preparation: build-labels, build-radicals, build-bigrams
- Model training: train-tinycnn, train-radical, train-joint, train-mobilenet
- Evaluation: eval-clip, eval-structural, eval-bigram, run-benchmarks
- Full pipeline: run-all
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Handwritten Chinese Character Recognition (HCCR) Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Data preparation commands
    subparsers.add_parser(
        "build-labels",
        help="Build character label map from training data",
    )

    subparsers.add_parser(
        "build-radicals",
        help="Build radical decomposition table from IDS file",
    )

    subparsers.add_parser(
        "build-bigrams",
        help="Build character bigram language model from word frequency data",
    )

    # Training commands
    subparsers.add_parser(
        "train-tinycnn",
        help="Train TinyCNN baseline model (classification only)",
    )

    subparsers.add_parser(
        "train-radical",
        help="Train TinyCNNRadical model (multi-task radical supervision)",
    )

    subparsers.add_parser(
        "train-joint",
        help="Train TinyCNNJoint model (flagship: classification + radicals)",
    )

    subparsers.add_parser(
        "train-mobilenet",
        help="Train MobileNetV3 transfer learning model",
    )

    subparsers.add_parser(
        "train-symbolic",
        help="Train with differentiable symbolic constraint layer",
    )

    subparsers.add_parser(
        "train-neurosymbolic",
        help="Train MiniResNetJoint with 3-phase neurosymbolic pipeline",
    )

    # Data preparation: constraint tensors
    subparsers.add_parser(
        "build-constraints",
        help="Build master_table.pt and radical_mask.pt for symbolic layer",
    )

    subparsers.add_parser(
        "collision-analysis",
        help="Analyze symbolic signature collisions (theoretical ceiling)",
    )

    # Evaluation commands
    subparsers.add_parser(
        "eval-clip",
        help="Evaluate CLIP zero-shot performance",
    )

    subparsers.add_parser(
        "eval-structural",
        help="Evaluate structural re-ranking pipeline",
    )

    subparsers.add_parser(
        "eval-bigram",
        help="Evaluate bigram language model re-ranking",
    )

    subparsers.add_parser(
        "run-benchmarks",
        help="Run comprehensive benchmarks on all models",
    )

    subparsers.add_parser(
        "eval-ablation",
        help="Run component ablation study on symbolic layer",
    )

    subparsers.add_parser(
        "eval-zeroshot",
        help="Evaluate zero-shot character recognition via symbolic layer",
    )

    subparsers.add_parser(
        "run-all",
        help="Execute full pipeline: data prep + training + evaluation",
    )

    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Strip the subcommand from sys.argv so sub-scripts' argparse works
    sys.argv = [sys.argv[0]] + remaining

    # Dispatch to appropriate script
    if args.command == "build-labels":
        from scripts.build_label_map import main as build_labels_main
        build_labels_main()

    elif args.command == "build-radicals":
        from scripts.build_radical_table import main as build_radicals_main
        build_radicals_main()

    elif args.command == "build-bigrams":
        from scripts.build_bigram_table import main as build_bigrams_main
        build_bigrams_main()

    elif args.command == "train-tinycnn":
        from scripts.train_tinycnn import main as train_tinycnn_main
        train_tinycnn_main()

    elif args.command == "train-radical":
        from scripts.train_tinycnn_radical import main as train_radical_main
        train_radical_main()

    elif args.command == "train-joint":
        from scripts.train_tinycnn_joint import main as train_joint_main
        train_joint_main()

    elif args.command == "train-mobilenet":
        from scripts.train_mobilenetv3 import main as train_mobilenet_main
        train_mobilenet_main()

    elif args.command == "train-symbolic":
        from scripts.train_symbolic import main as train_symbolic_main
        train_symbolic_main()

    elif args.command == "train-neurosymbolic":
        from scripts.train_neurosymbolic import main as train_neurosymbolic_main
        train_neurosymbolic_main()

    elif args.command == "build-constraints":
        from scripts.build_constraint_tensors import main as build_constraints_main
        build_constraints_main()

    elif args.command == "collision-analysis":
        from scripts.build_collision_analysis import main as collision_main
        collision_main()

    elif args.command == "eval-clip":
        from scripts.eval_clip_zeroshot import main as eval_clip_main
        eval_clip_main()

    elif args.command == "eval-structural":
        from scripts.eval_structural import main as eval_structural_main
        eval_structural_main()

    elif args.command == "eval-bigram":
        from scripts.eval_bigram_settings import main as eval_bigram_main
        eval_bigram_main()

    elif args.command == "run-benchmarks":
        from scripts.run_benchmarks import main as run_benchmarks_main
        run_benchmarks_main()

    elif args.command == "eval-ablation":
        from scripts.eval_ablation import main as eval_ablation_main
        eval_ablation_main()

    elif args.command == "eval-zeroshot":
        from scripts.eval_zero_shot import main as eval_zeroshot_main
        eval_zeroshot_main()

    elif args.command == "run-all":
        from scripts.run_all import main as run_all_main
        run_all_main()

    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
