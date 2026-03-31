"""Character bigram language model for sequence re-ranking.

Builds bigram probabilities from word frequency data and uses them
to improve candidate selection in multi-character recognition.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from hccr.utils import load_json, save_json

logger = logging.getLogger(__name__)


class BigramModel:
    """Character bigram language model with Laplace smoothing.

    Stores log probabilities P(c2 | c1) for character pairs.
    Used to re-rank candidates based on context.
    """

    def __init__(self) -> None:
        """Initialize empty bigram model."""
        self.bigram_counts: dict[str, dict[str, int]] = {}
        self.unigram_counts: dict[str, int] = {}
        self.vocab: set[str] = set()
        self.log_probs: dict[str, dict[str, float]] = {}
        self.vocab_size: int = 0
        self.total_bigrams: int = 0

    @classmethod
    def build_from_word_freq(
        cls,
        freq_path: Path,
        min_freq: int = 1,
        encoding: str = "utf-8",
    ) -> BigramModel:
        """Build bigram model from word frequency file.

        Args:
            freq_path: Path to frequency file (CSV/TSV with word and freq columns)
            min_freq: Minimum frequency threshold to include word
            encoding: File encoding

        Returns:
            Trained BigramModel instance
        """
        model = cls()
        freq_path = Path(freq_path)

        if not freq_path.exists():
            logger.warning(f"Frequency file not found: {freq_path}")
            return model

        logger.info(f"Building bigram model from {freq_path}")

        # Read frequency file
        words_with_freq: list[tuple[str, int]] = []
        with open(freq_path, "r", encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Try tab-separated, then comma-separated
                parts = line.split("\t") if "\t" in line else line.split(",")

                if len(parts) < 2:
                    continue

                word = parts[0].strip()
                try:
                    freq = int(parts[1].strip())
                except ValueError:
                    logger.debug(f"Line {line_num}: invalid frequency '{parts[1]}'")
                    continue

                if freq >= min_freq and len(word) >= 2:
                    words_with_freq.append((word, freq))

        logger.info(f"Loaded {len(words_with_freq)} words with freq >= {min_freq}")

        # Count bigrams
        for word, freq in words_with_freq:
            # Extract character bigrams
            for i in range(len(word) - 1):
                c1 = word[i]
                c2 = word[i + 1]

                # Update counts (weighted by word frequency)
                if c1 not in model.bigram_counts:
                    model.bigram_counts[c1] = {}

                model.bigram_counts[c1][c2] = model.bigram_counts[c1].get(c2, 0) + freq
                model.unigram_counts[c1] = model.unigram_counts.get(c1, 0) + freq
                model.unigram_counts[c2] = model.unigram_counts.get(c2, 0) + freq

                model.vocab.add(c1)
                model.vocab.add(c2)

        model.vocab_size = len(model.vocab)
        logger.info(f"Vocabulary size: {model.vocab_size}")

        # Compute log probabilities with Laplace smoothing
        model._compute_log_probs()

        logger.info(f"Built bigram model with {len(model.bigram_counts)} contexts")

        return model

    def _compute_log_probs(self) -> None:
        """Compute log probabilities with add-1 (Laplace) smoothing.

        P(c2 | c1) = (count(c1, c2) + 1) / (count(c1) + V)
        where V = vocabulary size
        """
        self.log_probs = {}

        for c1 in self.bigram_counts:
            self.log_probs[c1] = {}

            unigram_count = self.unigram_counts.get(c1, 0)

            for c2 in self.vocab:
                bigram_count = self.bigram_counts[c1].get(c2, 0)

                # Laplace smoothing
                prob = (bigram_count + 1) / (unigram_count + self.vocab_size)

                # Store log probability
                self.log_probs[c1][c2] = math.log(prob)

    def log_prob(self, prev_char: str, next_char: str) -> float:
        """Get log probability P(next_char | prev_char).

        Args:
            prev_char: Previous character (context)
            next_char: Next character

        Returns:
            Log probability (negative value, higher is better)
        """
        if prev_char in self.log_probs and next_char in self.log_probs[prev_char]:
            return self.log_probs[prev_char][next_char]

        # Unseen bigram: use uniform smoothing
        # P(c2 | c1) = 1 / V
        if self.vocab_size > 0:
            return math.log(1.0 / self.vocab_size)

        return -10.0  # Very low probability for empty model

    def rerank(
        self,
        candidates: list[tuple[int, float]],
        prev_char: str | None,
        label_map: dict[int, str],
        beta: float = 0.3,
    ) -> list[tuple[int, float]]:
        """Re-rank candidates using bigram language model.

        Args:
            candidates: List of (class_index, score) tuples
            prev_char: Previous character for context (None for first char)
            label_map: Maps class index to character
            beta: Weight for bigram score
                  final_score = (1-beta) * score + beta * normalized_bigram_score

        Returns:
            Re-ranked list of (class_index, score) tuples
        """
        if prev_char is None or prev_char not in self.log_probs:
            # No context or unknown context, return candidates unchanged
            return candidates

        if not candidates:
            return candidates

        # Compute bigram scores for all candidates
        bigram_scores: list[float] = []

        for idx, _ in candidates:
            char = label_map.get(idx)
            if char is None:
                # Unknown character, use minimum score
                bigram_scores.append(-10.0)
            else:
                bigram_scores.append(self.log_prob(prev_char, char))

        # Normalize bigram scores to [0, 1] range
        min_score = min(bigram_scores)
        max_score = max(bigram_scores)

        if max_score > min_score:
            normalized_bigram = [
                (score - min_score) / (max_score - min_score)
                for score in bigram_scores
            ]
        else:
            # All scores equal, use uniform
            normalized_bigram = [0.5] * len(bigram_scores)

        # Combine with original scores
        reranked: list[tuple[int, float]] = []

        for i, (idx, orig_score) in enumerate(candidates):
            new_score = (1 - beta) * orig_score + beta * normalized_bigram[i]
            reranked.append((idx, new_score))

        # Sort by new score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked

    def perplexity(self, text: str) -> float:
        """Compute perplexity of a text sequence.

        Lower perplexity indicates better fit to the model.

        Args:
            text: Text sequence to evaluate

        Returns:
            Perplexity score
        """
        if len(text) < 2:
            return float("inf")

        log_prob_sum = 0.0
        num_bigrams = 0

        for i in range(len(text) - 1):
            c1 = text[i]
            c2 = text[i + 1]
            log_prob_sum += self.log_prob(c1, c2)
            num_bigrams += 1

        if num_bigrams == 0:
            return float("inf")

        # Perplexity = exp(-mean_log_prob)
        mean_log_prob = log_prob_sum / num_bigrams
        return math.exp(-mean_log_prob)

    def save(self, path: Path) -> None:
        """Save bigram model to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "bigram_counts": {
                c1: dict(c2_counts) for c1, c2_counts in self.bigram_counts.items()
            },
            "unigram_counts": self.unigram_counts,
            "vocab": sorted(self.vocab),
            "log_probs": {
                c1: dict(c2_probs) for c1, c2_probs in self.log_probs.items()
            },
            "vocab_size": self.vocab_size,
        }
        save_json(data, path)
        logger.info(f"Saved bigram model to {path}")

    @classmethod
    def load(cls, path: Path) -> BigramModel:
        """Load bigram model from JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded BigramModel instance
        """
        data = load_json(path)
        model = cls()

        model.bigram_counts = {
            c1: dict(c2_counts) for c1, c2_counts in data["bigram_counts"].items()
        }
        model.unigram_counts = data["unigram_counts"]
        model.vocab = set(data["vocab"])
        model.log_probs = {
            c1: dict(c2_probs) for c1, c2_probs in data["log_probs"].items()
        }
        model.vocab_size = data["vocab_size"]

        logger.info(f"Loaded bigram model from {path}")
        return model

    def get_top_continuations(
        self,
        prev_char: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Get top-K most likely next characters given context.

        Useful for autocomplete and debugging.

        Args:
            prev_char: Previous character (context)
            k: Number of top candidates to return

        Returns:
            List of (character, log_prob) tuples, sorted by probability
        """
        if prev_char not in self.log_probs:
            return []

        # Get all continuations with their log probs
        continuations = [
            (c2, log_p) for c2, log_p in self.log_probs[prev_char].items()
        ]

        # Sort by log prob descending
        continuations.sort(key=lambda x: x[1], reverse=True)

        return continuations[:k]
