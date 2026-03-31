"""Build character bigram language model from word frequency data.

Reads a word frequency corpus (e.g., SUBTLEX-CH) and constructs bigram
probabilities for character-level sequence modeling.
"""

import argparse
from pathlib import Path

from hccr.structural.bigram import BigramModel
from hccr.data.label_map import LabelMap
from hccr.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build character bigram model from word frequency data"
    )
    parser.add_argument(
        "--freq-file",
        type=Path,
        required=True,
        help="Path to word frequency file (CSV/TSV with word and frequency columns)",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("resources/label_map.json"),
        help="Path to label map JSON (optional, for statistics)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/bigram_table.json"),
        help="Output path for bigram table JSON file",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum word frequency threshold",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding for frequency file",
    )
    args = parser.parse_args()

    logger.info(f"Building bigram model from {args.freq_file}")

    if not args.freq_file.exists():
        raise FileNotFoundError(f"Frequency file not found: {args.freq_file}")

    # Build bigram model
    bigram_model = BigramModel.build_from_word_freq(
        args.freq_file,
        min_freq=args.min_freq,
        encoding=args.encoding,
    )

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bigram_model.save(args.output)

    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("Bigram Model Statistics")
    logger.info("=" * 60)
    logger.info(f"Vocabulary size: {bigram_model.vocab_size}")
    logger.info(f"Unique bigram contexts: {len(bigram_model.bigram_counts)}")

    # Coverage analysis if label map provided
    if args.label_map.exists():
        label_map = LabelMap.load(args.label_map)
        overlap = len(set(label_map.chars) & bigram_model.vocab)
        logger.info(f"\nCharacter set overlap: {overlap}/{len(label_map)} "
                   f"({overlap/len(label_map)*100:.1f}%)")

    # Example continuations
    example_chars = ["我", "中", "学", "人", "大"]
    logger.info("\nExample top continuations:")
    for char in example_chars:
        if char in bigram_model.vocab:
            top = bigram_model.get_top_continuations(char, k=5)
            if top:
                top_str = ", ".join([f"{c}({p:.2f})" for c, p in top])
                logger.info(f"  {char} → {top_str}")

    logger.info(f"\nBigram model saved to {args.output}")


if __name__ == "__main__":
    main()
