"""
Base metric calculator with error handling and resource management.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any, Dict
from dataclasses import dataclass
import torch
import numpy as np
from tqdm import tqdm


@dataclass
class ModelConfig:
    """Model configuration for metric calculators."""
    name: str
    model_path: Optional[Path] = None
    batch_size: int = 8
    device: str = "cuda"
    additional_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class MetricCalculationError(Exception):
    """Custom exception for metric calculation errors."""
    pass


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass


class BaseMetricCalculator(ABC):
    """
    Abstract base class for metric calculators with improved error handling,
    resource management, and progress tracking.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def load_model(self) -> None:
        """Load the required model for metric calculation."""
        try:
            self._load_model_impl()
            self._is_initialized = True
            self.logger.info(f"Successfully loaded {self.get_name()} model")
        except Exception as e:
            self.logger.error(f"Failed to load {self.get_name()} model: {e}")
            raise ModelLoadError(f"Failed to load {self.get_name()} model: {e}")

    @abstractmethod
    def _load_model_impl(self) -> None:
        """Actual model loading implementation."""
        pass

    def calculate_pair(self, ref_path: Path, syn_path: Path) -> float:
        """Calculate metric for a single reference-synthesis pair."""
        if not self._is_initialized:
            raise MetricCalculationError("Model not initialized. Call load_model() first.")

        try:
            return self._calculate_pair_impl(ref_path, syn_path)
        except Exception as e:
            self.logger.error(f"Error calculating {self.get_name()} for pair ({ref_path}, {syn_path}): {e}")
            raise MetricCalculationError(f"Error calculating {self.get_name()}: {e}")

    @abstractmethod
    def _calculate_pair_impl(self, ref_path: Path, syn_path: Path) -> float:
        """Actual pair calculation implementation."""
        pass

    def calculate_batch(
        self,
        pairs: List[Tuple[Path, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[float]:
        """
        Calculate metric for multiple pairs with progress tracking.

        Args:
            pairs: List of (reference, synthesis) path pairs
            progress_callback: Optional callback function for progress updates

        Returns:
            List of metric scores
        """
        if not self._is_initialized:
            raise MetricCalculationError("Model not initialized. Call load_model() first.")

        results = []
        total_pairs = len(pairs)

        # Use tqdm for progress bar by default
        iterator = tqdm(pairs, desc=f"Calculating {self.get_name()}", disable=False)

        for i, (ref_path, syn_path) in enumerate(iterator):
            try:
                score = self._calculate_pair_impl(ref_path, syn_path)
                results.append(score)

                if progress_callback:
                    progress_callback(i + 1, total_pairs)

            except Exception as e:
                self.logger.warning(f"Skipping pair ({ref_path}, {syn_path}) due to error: {e}")
                results.append(np.nan)  # Use NaN for failed calculations

        return results

    def calculate_batch_optimized(self, pairs: List[Tuple[Path, Path]]) -> List[float]:
        """
        Optimized batch calculation (can be overridden by subclasses for true batch processing).
        Default implementation falls back to individual calculations.
        """
        return self.calculate_batch(pairs)

    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass

    def validate_audio_files(self, pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
        """
        Validate that all audio files exist and are readable.

        Args:
            pairs: List of (reference, synthesis) path pairs

        Returns:
            List of valid pairs
        """
        valid_pairs = []

        for ref_path, syn_path in pairs:
            if not ref_path.exists():
                self.logger.warning(f"Reference file not found: {ref_path}")
                continue

            if not syn_path.exists():
                self.logger.warning(f"Synthesis file not found: {syn_path}")
                continue

            # Additional validation can be added here (file format, duration, etc.)
            valid_pairs.append((ref_path, syn_path))

        self.logger.info(f"Validated {len(valid_pairs)}/{len(pairs)} audio pairs")
        return valid_pairs

    def get_device(self) -> torch.device:
        """Get the appropriate device for computation."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def cleanup(self) -> None:
        """Clean up resources if needed."""
        pass
