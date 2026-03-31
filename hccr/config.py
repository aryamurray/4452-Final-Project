"""Configuration dataclasses for HCCR pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.5
    image_size: int = 64
    early_stopping_patience: int = 5
    seed: int = 42
    grad_accum_steps: int = 2
    num_workers: int = 2
    label_smoothing: float = 0.1
    # Multi-task loss weights
    lambda_radical: float = 1.0
    lambda_structure: float = 0.5
    lambda_stroke: float = 0.1
    # Symbolic training
    num_strokes: int = 30
    lambda_combined: float = 1.0
    lambda_char: float = 0.5
    symbolic_temperature: float = 1.0
    grad_clip_norm: float = 1.0


@dataclass
class DataConfig:
    train_dir: Path = Path("data/HWDB1.1/train")
    test_dir: Path = Path("data/HWDB1.1/test")
    label_map_path: Path = Path("resources/label_map.json")
    radical_table_path: Path = Path("resources/radical_table.json")
    bigram_table_path: Path = Path("resources/bigram_table.json")
    ids_path: Path = Path("resources/ids.txt")
    index_cache_dir: Path = Path("outputs/cache")
    val_ratio: float = 0.2


@dataclass
class StructuralConfig:
    alpha: float = 0.7  # weight for classifier score vs radical score
    beta: float = 0.3  # weight for bigram re-ranking
    top_k: int = 10  # candidates from classifier
    beam_width: int = 5


@dataclass
class ModelConfig:
    num_classes: int = 3755
    num_radicals: int = 500
    num_structures: int = 13
    num_strokes: int = 30
    backbone_dim: int = 256


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: Path = Path("outputs")

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"
