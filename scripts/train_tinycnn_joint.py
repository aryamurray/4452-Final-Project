"""Train TinyCNNJoint model with phased training on HWDB1.1.

Flagship model with three training phases:
  Phase 1: Load pretrained TinyCNN backbone + char_head (from baseline checkpoint)
  Phase 2: Freeze backbone + char_head, train radical/structure/stroke heads (10 epochs)
  Phase 3: Unfreeze all, joint fine-tune at low LR (5 epochs)
"""

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hccr.config import TrainConfig
from hccr.data.dataset import HWDBRadicalDataset
from hccr.data.label_map import LabelMap
from hccr.data.transforms import get_train_transform, get_eval_transform, get_train_tensor_augment
from hccr.models.tiny_cnn_joint import TinyCNNJoint
from hccr.structural.radical_table import RadicalTable
from hccr.training.trainer import Trainer
from hccr.utils import get_device, get_logger, plot_training_curves, set_seed

logger = get_logger(__name__)


def prepare_radical_table_dict(radical_table: RadicalTable) -> dict:
    """Convert RadicalTable to dictionary format expected by HWDBRadicalDataset."""
    result = {}
    for char in radical_table.char_to_radicals:
        radicals = radical_table.char_to_radicals.get(char, [])
        radical_indices = [
            radical_table.radical_to_index[r]
            for r in radicals
            if r in radical_table.radical_to_index
        ]
        result[char] = {
            "radicals": radical_indices,
            "structure": radical_table.char_to_structure.get(char, 0),
            "strokes": radical_table.char_to_strokes.get(char, 0),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyCNNJoint (flagship, phased)")
    parser.add_argument("--train-dir", type=Path, default=Path("data/HWDB1.1/train"))
    parser.add_argument("--label-map", type=Path, default=Path("resources/label_map.json"))
    parser.add_argument("--radical-table", type=Path, default=Path("resources/radical_table.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tinycnn_joint"))
    parser.add_argument(
        "--pretrained-checkpoint", type=Path, default=Path("outputs/tinycnn/best_model_classification.pt"),
        help="Path to pretrained TinyCNN baseline checkpoint (Phase 1)",
    )
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--phase3-epochs", type=int, default=5)
    parser.add_argument("--phase2-lr", type=float, default=1e-3)
    parser.add_argument("--phase3-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cbam", action="store_true", default=False)
    parser.add_argument("--no-cbam", dest="use_cbam", action="store_false")
    parser.add_argument("--num-strokes", type=int, default=30)
    parser.add_argument(
        "--skip-to-phase3", type=Path, default=None,
        help="Skip phases 1-2, load this joint checkpoint and go straight to phase 3",
    )
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Training TinyCNNJoint (Phased Training - Flagship)")
    logger.info("=" * 60)

    # Load label map and radical table
    label_map = LabelMap.load(args.label_map)
    num_classes = len(label_map)

    radical_table = RadicalTable.load(args.radical_table)
    num_radicals = len(radical_table.all_radicals)
    logger.info(f"Classes: {num_classes}, Radicals: {num_radicals}")

    radical_dict = prepare_radical_table_dict(radical_table)

    # Split GNT files
    all_files = list(args.train_dir.glob("*.gnt"))
    num_val = int(len(all_files) * args.val_ratio)
    random.seed(args.seed)
    shuffled = all_files.copy()
    random.shuffle(shuffled)
    train_files = shuffled[num_val:]
    val_files = shuffled[:num_val]

    # Create temp directories with symlinks
    temp_dir = Path(tempfile.mkdtemp())
    train_temp = temp_dir / "train"
    val_temp = temp_dir / "val"
    train_temp.mkdir()
    val_temp.mkdir()

    for f in train_files:
        dst = train_temp / f.name
        try:
            dst.symlink_to(f.resolve())
        except OSError:
            shutil.copy(f, dst)
    for f in val_files:
        dst = val_temp / f.name
        try:
            dst.symlink_to(f.resolve())
        except OSError:
            shutil.copy(f, dst)

    # Create datasets with preloading
    train_transform = get_train_transform(args.image_size)
    val_transform = get_eval_transform(args.image_size)
    tensor_augment = get_train_tensor_augment()

    train_dataset = HWDBRadicalDataset(
        train_temp, label_map, radical_dict,
        num_radicals=num_radicals, num_structures=13,
        transform=train_transform,
        tensor_augment=tensor_augment,
        index_cache_path=args.output_dir / "train_index.pkl",
        preload=True, image_size=args.image_size,
        stroke_as_class=True, max_strokes=args.num_strokes,
    )
    val_dataset = HWDBRadicalDataset(
        val_temp, label_map, radical_dict,
        num_radicals=num_radicals, num_structures=13,
        transform=val_transform,
        index_cache_path=args.output_dir / "val_index.pkl",
        preload=True, image_size=args.image_size,
        stroke_as_class=True, max_strokes=args.num_strokes,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = TinyCNNJoint(
        num_classes=num_classes,
        num_radicals=num_radicals,
        num_structures=13,
        num_strokes=args.num_strokes,
        use_cbam=args.use_cbam,
    )
    total, _ = model.get_num_params()
    logger.info(f"Model parameters: {total:,} (CBAM={'on' if args.use_cbam else 'off'})")

    if args.skip_to_phase3:
        # Skip phases 1-2, load joint checkpoint directly
        logger.info(f"\n{'='*60}")
        logger.info(f"Skipping to Phase 3: Loading checkpoint {args.skip_to_phase3}")
        logger.info(f"{'='*60}")
        model.load_joint_checkpoint(args.skip_to_phase3)
        history_p2 = {}
    else:
        # ── Phase 1: Load pretrained backbone ──────────────────────────
        if args.pretrained_checkpoint.exists():
            logger.info(f"\n{'='*60}")
            logger.info("Phase 1: Loading pretrained backbone from TinyCNN baseline")
            logger.info(f"{'='*60}")
            model.load_pretrained_backbone(str(args.pretrained_checkpoint))
            logger.info(f"Loaded weights from {args.pretrained_checkpoint}")
        else:
            logger.warning(
                f"No pretrained checkpoint at {args.pretrained_checkpoint}. "
                "Training aux heads from scratch (Phase 1 skipped)."
            )

        # ── Phase 2: Freeze backbone + char_head, train aux heads ──────
        logger.info(f"\n{'='*60}")
        logger.info("Phase 2: Training auxiliary heads (backbone + char_head frozen)")
        logger.info(f"{'='*60}")

        model.freeze_backbone_and_char_head()
        _, trainable = model.get_num_params()
        logger.info(f"Trainable parameters: {trainable:,}")

        config_p2 = TrainConfig(
            epochs=args.phase2_epochs,
            batch_size=args.batch_size,
            lr=args.phase2_lr,
            image_size=args.image_size,
            num_strokes=args.num_strokes,
            lambda_radical=1.0,
            lambda_structure=0.5,
            lambda_stroke=0.1,
        )

        trainer_p2 = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            config=config_p2, mode="joint", device=device,
            log_dir=args.output_dir / "phase2",
        )
        history_p2 = trainer_p2.train()

    # ── Phase 3: Unfreeze all, joint fine-tune at low LR ───────────
    logger.info(f"\n{'='*60}")
    logger.info("Phase 3: Joint fine-tuning (all params, low LR)")
    logger.info(f"{'='*60}")

    model.unfreeze_all()
    _, trainable = model.get_num_params()
    logger.info(f"Trainable parameters: {trainable:,}")

    config_p3 = TrainConfig(
        epochs=args.phase3_epochs,
        batch_size=args.batch_size,
        lr=args.phase3_lr,
        image_size=args.image_size,
        num_strokes=args.num_strokes,
        lambda_radical=0.3,
        lambda_structure=0.1,
        lambda_stroke=0.1,
    )

    trainer_p3 = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=config_p3, mode="joint", device=device,
        log_dir=args.output_dir / "phase3",
    )
    history_p3 = trainer_p3.train()

    # Plot combined training curves
    combined = {k: history_p2.get(k, []) + history_p3.get(k, [])
                for k in set(list(history_p2.keys()) + list(history_p3.keys()))}
    plot_training_curves(
        combined.get("train_loss", []),
        combined.get("val_loss", []),
        combined.get("val_acc", []),
        save_path=args.output_dir / "training_curves.png",
    )

    # Cleanup
    shutil.rmtree(temp_dir)

    logger.info(f"\nPhased training complete!")
    if not args.skip_to_phase3:
        logger.info(f"Phase 2 best val_acc: {trainer_p2.best_metric:.4f}")
    logger.info(f"Phase 3 best val_acc: {trainer_p3.best_metric:.4f}")
    logger.info(f"Final checkpoint: {args.output_dir / 'phase3' / 'best_model_joint.pt'}")


if __name__ == "__main__":
    main()
