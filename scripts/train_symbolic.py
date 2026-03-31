"""Train with differentiable symbolic layer.

Loads a pretrained TinyCNNJoint or MiniResNetJoint checkpoint, attaches
constraint tensors (master_table + radical_mask), and fine-tunes with
SymbolicLoss so that gradients from the symbolic layer flow back into
all four prediction heads.

Usage:
    python scripts/train_symbolic.py \
        --pretrained-checkpoint outputs/tinycnn_joint/phase3/best_model_joint.pt \
        --master-table resources/master_table.pt \
        --radical-mask resources/radical_mask.pt \
        --epochs 20 --lr 1e-4
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
from hccr.data.transforms import get_eval_transform, get_train_tensor_augment, get_train_transform
from hccr.models.tiny_cnn_joint import TinyCNNJoint
from hccr.models.mini_resnet import MiniResNetJoint
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
    parser = argparse.ArgumentParser(description="Train with differentiable symbolic layer")
    parser.add_argument("--train-dir", type=Path, default=Path("data/HWDB1.1/train"))
    parser.add_argument("--label-map", type=Path, default=Path("resources/label_map.json"))
    parser.add_argument("--radical-table", type=Path, default=Path("resources/radical_table.json"))
    parser.add_argument("--master-table", type=Path, default=Path("resources/master_table.pt"))
    parser.add_argument("--radical-mask", type=Path, default=Path("resources/radical_mask.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/symbolic"))
    parser.add_argument(
        "--pretrained-checkpoint", type=Path,
        default=Path("outputs/tinycnn_joint/phase3/best_model_joint.pt"),
        help="Path to pretrained joint checkpoint",
    )
    parser.add_argument("--model", choices=["tinycnn", "miniresnet"], default="tinycnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbolic-temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-combined", type=float, default=1.0)
    parser.add_argument("--lambda-char", type=float, default=0.5)
    parser.add_argument("--lambda-radical", type=float, default=0.5)
    parser.add_argument("--lambda-structure", type=float, default=0.3)
    parser.add_argument("--lambda-stroke", type=float, default=0.1)
    parser.add_argument("--use-cbam", action="store_true", default=False)
    parser.add_argument("--no-cbam", dest="use_cbam", action="store_false")
    parser.add_argument("--num-strokes", type=int, default=30)
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Symbolic Fine-Tuning (Differentiable Constraint Layer)")
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

    # Create datasets with stroke_as_class=True for CE loss
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
    if args.model == "tinycnn":
        model = TinyCNNJoint(
            num_classes=num_classes,
            num_radicals=num_radicals,
            num_structures=13,
            num_strokes=args.num_strokes,
            use_cbam=args.use_cbam,
            symbolic_temperature=args.symbolic_temperature,
        )
    else:
        model = MiniResNetJoint(
            num_classes=num_classes,
            num_radicals=num_radicals,
            num_structures=13,
            num_strokes=args.num_strokes,
            symbolic_temperature=args.symbolic_temperature,
        )

    total, _ = model.get_num_params()
    logger.info(f"Model: {args.model}, Parameters: {total:,}")

    # Phase A: Load pretrained checkpoint
    if args.pretrained_checkpoint.exists():
        logger.info(f"\nLoading pretrained checkpoint: {args.pretrained_checkpoint}")
        model.load_joint_checkpoint(args.pretrained_checkpoint)
        logger.info("Checkpoint loaded (stroke head reinitialized if shape mismatch)")
    else:
        logger.warning(
            f"No checkpoint at {args.pretrained_checkpoint}. "
            "Training symbolic layer from scratch."
        )

    # Load constraint tensors
    logger.info(f"Loading constraint tensors...")
    model.load_constraint_tensors(args.master_table, args.radical_mask)
    logger.info(
        f"  master_table: {tuple(model.master_table.shape)}, "
        f"radical_mask: {tuple(model.radical_mask.shape)}"
    )

    # Phase B: Symbolic fine-tuning
    logger.info(f"\n{'='*60}")
    logger.info("Symbolic Fine-Tuning")
    logger.info(f"  Temperature: {args.symbolic_temperature}")
    logger.info(f"  Grad clip norm: {args.grad_clip_norm}")
    logger.info(f"  LR: {args.lr}")
    logger.info(f"{'='*60}")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        num_strokes=args.num_strokes,
        lambda_combined=args.lambda_combined,
        lambda_char=args.lambda_char,
        lambda_radical=args.lambda_radical,
        lambda_structure=args.lambda_structure,
        lambda_stroke=args.lambda_stroke,
        symbolic_temperature=args.symbolic_temperature,
        grad_clip_norm=args.grad_clip_norm,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        mode="symbolic",
        device=device,
        log_dir=args.output_dir,
    )

    history = trainer.train()

    # Plot training curves
    plot_training_curves(
        history.get("train_loss", []),
        history.get("val_loss", []),
        history.get("val_acc", []),
        save_path=args.output_dir / "training_curves.png",
    )

    # Cleanup
    shutil.rmtree(temp_dir)

    logger.info(f"\nSymbolic training complete!")
    logger.info(f"Best val_acc: {trainer.best_metric:.4f}")
    logger.info(f"Checkpoint: {args.output_dir / 'best_model_symbolic.pt'}")


if __name__ == "__main__":
    main()
