"""CLIP-based zero-shot Chinese character recognition.

Uses pretrained CLIP models (OpenAI/OpenCLIP) for zero-shot classification
via text-image similarity with Chinese character prompts.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import open_clip


class CLIPZeroShot:
    """CLIP zero-shot classifier for Chinese character recognition.

    Two modes:
    - multilingual: XLM-RoBERTa backbone, Chinese prompts
    - english: ViT-B/32 backbone, English prompts

    Precomputes text embeddings for all characters using prompt templates,
    then classifies images via cosine similarity.

    Args:
        mode: Either "multilingual" or "english"
        label_map: Dictionary mapping class indices to characters
        device: torch device (cuda/cpu)
        cache_dir: Directory to cache precomputed text embeddings

    Attributes:
        mode: Model mode (multilingual/english)
        label_map: Index to character mapping
        device: Computation device
        cache_dir: Cache directory for embeddings
        model: CLIP image encoder
        preprocess: CLIP image preprocessing
        tokenizer: CLIP text tokenizer
        text_features: Cached text embeddings (num_classes, embed_dim)
    """

    # Model configurations
    CONFIGS = {
        "multilingual": {
            "model_name": "xlm-roberta-base-ViT-B-32",
            "pretrained": "laion5b_s13b_b90k",
            "prompts": [
                "手写汉字{char}",
                "中文字符{char}",
            ],
        },
        "english": {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "prompts": [
                "a handwritten Chinese character {char}",
                "the Chinese character {char}",
            ],
        },
    }

    def __init__(
        self,
        mode: str,
        label_map: Dict[int, str],
        device: torch.device,
        cache_dir: Path,
    ) -> None:
        """Initialize CLIP zero-shot classifier.

        Args:
            mode: "multilingual" or "english"
            label_map: Dictionary mapping class indices to character strings
            device: torch device for computation
            cache_dir: Directory to save/load cached text embeddings

        Raises:
            ValueError: If mode is not recognized
        """
        if mode not in self.CONFIGS:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'multilingual' or 'english'."
            )

        self.mode = mode
        self.label_map = label_map
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model configuration
        config = self.CONFIGS[mode]
        self.model_name = config["model_name"]
        self.pretrained = config["pretrained"]
        self.prompt_templates = config["prompts"]

        # Load CLIP model
        print(f"Loading CLIP model: {self.model_name} ({self.pretrained})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Build or load cached text features
        self.text_features = self._load_or_build_text_features()

    def _get_cache_path(self) -> Path:
        """Get cache file path for text embeddings.

        Returns:
            Path to cache file
        """
        cache_filename = f"clip_text_features_{self.mode}_{len(self.label_map)}.pt"
        return self.cache_dir / cache_filename

    def _load_or_build_text_features(self) -> torch.Tensor:
        """Load cached text features or build from scratch.

        Returns:
            Text feature tensor of shape (num_classes, embed_dim)
        """
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"Loading cached text features from {cache_path}")
            text_features = torch.load(cache_path, map_location=self.device)
            print(f"Loaded text features: {text_features.shape}")
            return text_features

        print("Building text features from scratch...")
        text_features = self._build_text_features()

        # Cache for future use
        print(f"Caching text features to {cache_path}")
        torch.save(text_features, cache_path)

        return text_features

    def _build_text_features(self) -> torch.Tensor:
        """Build text features for all characters using prompt templates.

        For each character:
        1. Apply all prompt templates
        2. Tokenize and encode with CLIP text encoder
        3. Average embeddings across prompts
        4. L2 normalize

        Returns:
            Text feature tensor of shape (num_classes, embed_dim)
        """
        num_classes = len(self.label_map)
        all_features = []

        with torch.no_grad():
            for idx in range(num_classes):
                char = self.label_map[idx]

                # Generate prompts for this character
                prompts = [
                    template.format(char=char)
                    for template in self.prompt_templates
                ]

                # Tokenize prompts
                text_tokens = self.tokenizer(prompts).to(self.device)

                # Encode text
                text_embeddings = self.model.encode_text(text_tokens)

                # Average across prompt templates
                char_feature = text_embeddings.mean(dim=0)

                # L2 normalize
                char_feature = F.normalize(char_feature, dim=0)

                all_features.append(char_feature)

                if (idx + 1) % 500 == 0:
                    print(f"Processed {idx + 1}/{num_classes} characters")

        # Stack into single tensor
        text_features = torch.stack(all_features)  # (num_classes, embed_dim)
        print(f"Built text features: {text_features.shape}")

        return text_features

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict top-k characters for input images.

        Args:
            images: Preprocessed image tensor of shape (B, 3, 224, 224)
                   Must be RGB and normalized for CLIP
            top_k: Number of top predictions to return

        Returns:
            Tuple of (indices, scores):
                - indices: Top-k class indices, shape (B, top_k)
                - scores: Cosine similarity scores, shape (B, top_k)
        """
        self.model.eval()

        # Encode images
        image_features = self.model.encode_image(images.to(self.device))

        # L2 normalize
        image_features = F.normalize(image_features, dim=-1)

        # Compute cosine similarity with all text features
        # (B, embed_dim) @ (embed_dim, num_classes) -> (B, num_classes)
        similarity = image_features @ self.text_features.T

        # Get top-k predictions
        scores, indices = similarity.topk(k=top_k, dim=-1)

        return indices.cpu().numpy(), scores.cpu().numpy()

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        top_k: int = 10,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict top-k characters for large batches of images.

        Processes images in smaller batches to avoid OOM errors.

        Args:
            images: Preprocessed image tensor of shape (N, 3, 224, 224)
            top_k: Number of top predictions to return
            batch_size: Batch size for processing

        Returns:
            Tuple of (indices, scores):
                - indices: Top-k class indices, shape (N, top_k)
                - scores: Cosine similarity scores, shape (N, top_k)
        """
        self.model.eval()

        all_indices = []
        all_scores = []

        num_images = len(images)
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(start_idx + batch_size, num_images)
            batch = images[start_idx:end_idx]

            indices, scores = self.predict(batch, top_k=top_k)
            all_indices.append(indices)
            all_scores.append(scores)

        # Concatenate results
        all_indices = np.concatenate(all_indices, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)

        return all_indices, all_scores

    def get_char_from_index(self, idx: int) -> str:
        """Get character string from class index.

        Args:
            idx: Class index

        Returns:
            Character string
        """
        return self.label_map[idx]

    def decode_predictions(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
    ) -> List[List[Tuple[str, float]]]:
        """Decode prediction indices to character-score pairs.

        Args:
            indices: Prediction indices of shape (B, top_k)
            scores: Prediction scores of shape (B, top_k)

        Returns:
            List of lists containing (character, score) tuples for each image
        """
        results = []
        for idx_row, score_row in zip(indices, scores):
            predictions = [
                (self.label_map[idx], float(score))
                for idx, score in zip(idx_row, score_row)
            ]
            results.append(predictions)
        return results
