"""
Base dataset loader for synthesis.
"""

import random
from abc import ABC, abstractmethod
from pathlib import Path


class BaseSynthesisDataset(ABC):
    """Base class for synthesis datasets."""

    def __init__(self, config, download: bool = True):
        self.config = config
        self.download = download
        self.dataset = None

    @abstractmethod
    def load_dataset(self) -> None:
        """Load the dataset."""
        pass

    @abstractmethod
    def get_sample(self, index: int) -> tuple[str, Path, str | None, str | None]:
        """Get a sample by index.

        Returns:
            Tuple of (transcript, audio_path, style_prompt, speaker_id)
        """
        pass

    @abstractmethod
    def get_total_samples(self) -> int:
        """Get total number of samples."""
        pass

    def select_samples(self, num_samples: int, seed: int = 42) -> list[int]:
        """Select random samples for synthesis."""
        random.seed(seed)
        total = self.get_total_samples()
        if num_samples > total:
            raise ValueError(f"Requested {num_samples} samples but only {total} available")

        return random.sample(range(total), num_samples)

    def filter_by_duration(self, indices: list[int]) -> list[int]:
        """Filter samples by duration constraints."""
        if not self.config.max_duration and not self.config.min_duration:
            return indices

        # For now, return all indices. Can add duration filtering later if needed.
        return indices
