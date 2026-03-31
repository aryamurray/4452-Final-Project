"""Train MiniResNet on HWDB1.1 dataset.

Custom residual CNN with 13 conv layers for fine-grained character recognition.
"""

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hccr.config import TrainConfig
from hccr.data.dataset import HWDBDataset
from hccr.data.label_map import LabelMap
from hccr.data.transforms import get_train_transform, get_eval_transform, get_train_tensor_augment
from hccr.models.mini_resnet import MiniResNet
from hccr.training.trainer import Trainer
from hccr.utils import get_device, get_logger, plot_training_curves, set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniResNet")
    parser.add_argument("--train-dir", type=Path, default=Path("data/HWDB1.1/train"))
    parser.add_argument("--label-map", type=Path, default=Path("resources/label_map.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mini_resnet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--backbone-dim", type=int, default=192)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume training from checkpoint path")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Training MiniResNet (Classification Mode)")
    logger.info("=" * 60)

    # Load label map
    label_map = LabelMap.load(args.label_map)
    num_classes = len(label_map)
    logger.info(f"Number of classes: {num_classes}")

    # Split GNT files into train/val
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

    train_dataset = HWDBDataset(
        train_temp, label_map,
        transform=train_transform,
        tensor_augment=tensor_augment,
        index_cache_path=args.output_dir / "train_index.pkl",
        preload=True, image_size=args.image_size,
    )
    val_dataset = HWDBDataset(
        val_temp, label_map,
        transform=val_transform,
        index_cache_path=args.output_dir / "val_index.pkl",
        preload=True, image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create model
    model = MiniResNet(num_classes=num_classes, backbone_dim=args.backbone_dim)
    total, _ = model.get_num_params()
    logger.info(f"Model parameters: {total:,} (dim={args.backbone_dim})")

    # Load checkpoint weights if resuming
    if args.resume:
        logger.info(f"Loading model weights from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Create trainer
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        mode="classification",
        device=device,
        log_dir=args.output_dir,
        resume_from=args.resume,
    )

    # Train
    history = trainer.train()

    # Plot training curves
    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        history.get("val_acc", []),
        save_path=args.output_dir / "training_curves.png",
    )
    logger.info(f"Training curves saved to {args.output_dir / 'training_curves.png'}")

    # Cleanup
    shutil.rmtree(temp_dir)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
