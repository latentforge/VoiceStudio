"""
Synthesis configuration classes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    name: str
    sample_rate: int = 22050
    max_duration: Optional[float] = None  # seconds
    min_duration: Optional[float] = 1.0   # seconds


@dataclass
class ModelConfig:
    """Configuration for TTS models."""
    name: str
    device: str = "cuda"
    model_path: Optional[str] = None
    language: str = "en"
    temperature: float = 0.1
    top_k: int = 10
    top_p: float = 0.7


@dataclass
class GenerationConfig:
    """Configuration for generation methods."""
    method1_samples: int = 100
    method2_ref_samples: int = 10
    method2_syn_per_ref: int = 10
    output_dir: Path = Path("results")


class SynthesisConfig:
    """Main synthesis configuration manager."""

    def __init__(self):
        # Dataset configurations
        self.datasets = {
            "vctk": DatasetConfig(name="vctk", sample_rate=48000),
            "ljspeech": DatasetConfig(name="ljspeech", sample_rate=22050),
            "libritts": DatasetConfig(name="libritts", sample_rate=24000),
        }

        # Model configurations
        self.models = {
            "xtts_v2": ModelConfig(name="xtts_v2"),
            "parler_tts_mini_v1": ModelConfig(name="parler_tts_mini_v1"),
            "higgs_v2": ModelConfig(name="higgs_v2"),
        }

        # Generation configuration
        self.generation = GenerationConfig()

    def get_dataset_config(self, name: str) -> DatasetConfig:
        """Get dataset configuration by name."""
        if name not in self.datasets:
            raise ValueError(f"Unknown dataset: {name}")
        return self.datasets[name]

    def get_model_config(self, name: str) -> ModelConfig:
        """Get model configuration by name."""
        if name not in self.models:
            raise ValueError(f"Unknown model: {name}")
        return self.models[name]