"""Train MobileNetV3 transfer learning model on HWDB1.1.

Uses pretrained MobileNetV3-Small with 224x224 input size.
"""

import argparse
from pathlib import Path
import shutil
import tempfile
import random

import torch
from torch.utils.data import DataLoader

from hccr.config import TrainConfig
from hccr.data.dataset import HWDBDataset
from hccr.data.label_map import LabelMap
from hccr.data.transforms import get_train_transform, get_eval_transform
from hccr.models.mobilenetv3 import MobileNetV3Wrapper
from hccr.training.trainer import Trainer
from hccr.utils import get_device, get_logger, plot_training_curves, set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 transfer learning model")
    parser.add_argument("--train-dir", type=Path, default=Path("data/HWDB1.1/train"))
    parser.add_argument("--label-map", type=Path, default=Path("resources/label_map.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mobilenetv3"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Training MobileNetV3 (Transfer Learning)")
    logger.info("=" * 60)

    # Load label map
    logger.info(f"Loading label map from {args.label_map}")
    label_map = LabelMap.load(args.label_map)
    num_classes = len(label_map)
    logger.info(f"Number of classes: {num_classes}")

    # MobileNetV3 requires 224x224 input
    image_size = 224
    logger.info(f"Using image size: {image_size}x{image_size} (MobileNet requirement)")

    # Create datasets with 224x224 transforms
    logger.info(f"Loading training data from {args.train_dir}")
    train_transform = get_train_transform(image_size)
    val_transform = get_eval_transform(image_size)

    all_files = list(args.train_dir.glob("*.gnt"))
    num_val = int(len(all_files) * args.val_ratio)
    random.seed(args.seed)
    shuffled = all_files.copy()
    random.shuffle(shuffled)

    train_files = shuffled[num_val:]
    val_files = shuffled[:num_val]

    temp_dir = Path(tempfile.mkdtemp())
    train_temp = temp_dir / "train"
    val_temp = temp_dir / "val"
    train_temp.mkdir()
    val_temp.mkdir()

    for f in train_files:
        shutil.copy(f, train_temp / f.name)
    for f in val_files:
        shutil.copy(f, val_temp / f.name)

    train_dataset = HWDBDataset(
        train_temp,
        label_map,
        transform=train_transform,
        index_cache_path=args.output_dir / "train_index.pkl",
    )
    val_dataset = HWDBDataset(
        val_temp,
        label_map,
        transform=val_transform,
        index_cache_path=args.output_dir / "val_index.pkl",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create model
    model = MobileNetV3Wrapper(num_classes=num_classes, pretrained=args.pretrained)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Pretrained: {args.pretrained}")

    # Create trainer
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=image_size,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        mode="classification",
        device=device,
        log_dir=args.output_dir,
    )

    # Train
    history = trainer.train()

    # Plot training curves
    plot_training_curves(
        history,
        save_path=args.output_dir / "training_curves.png",
    )
    logger.info(f"Training curves saved to {args.output_dir / 'training_curves.png'}")

    # Cleanup
    shutil.rmtree(temp_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
