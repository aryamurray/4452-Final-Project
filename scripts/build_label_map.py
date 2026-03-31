"""Build label map from HWDB training data.

Scans all .gnt files in the training directory and creates a bidirectional
character-to-index mapping, sorted by Unicode codepoint.
"""

import argparse
from pathlib import Path

from hccr.data.label_map import LabelMap
from hccr.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build character label map from HWDB training data"
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/HWDB1.1/train"),
        help="Directory containing training .gnt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/label_map.json"),
        help="Output path for label map JSON file",
    )
    parser.add_argument(
        "--chinese-only",
        action="store_true",
        default=True,
        help="Only include CJK Unified Ideographs (default: True)",
    )
    parser.add_argument(
        "--all-chars",
        dest="chinese_only",
        action="store_false",
        help="Include all characters (ASCII, punctuation, etc.)",
    )
    args = parser.parse_args()

    logger.info(f"Building label map from {args.train_dir}")
    if args.chinese_only:
        logger.info("Filtering to CJK Unified Ideographs only (U+4E00–U+9FFF)")

    if not args.train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {args.train_dir}")

    # Build label map
    label_map = LabelMap.build_from_gnt_dir(args.train_dir, chinese_only=args.chinese_only)

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_map.save(args.output)

    logger.info(f"Label map saved to {args.output}")
    logger.info(f"Total classes: {len(label_map)}")
    logger.info(f"Sample characters: {label_map.chars[:10]}")


if __name__ == "__main__":
    main()
