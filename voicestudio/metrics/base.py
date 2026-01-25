"""
Base metric calculator with error handling and resource management.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from ..utils.loader import AudioLoader


@dataclass
class ModelConfig:
    """Model configuration for metric calculators."""

    name: str
    model_path: Path | None = None
    batch_size: int = 8
    device: str = "cuda"
    additional_params: dict[str, Any] | None = None

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

    Note:
        This class is not thread-safe. Use separate instances for concurrent processing.
    """

    DEFAULT_SAMPLE_RATE = 16000

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
        self._audio_loaders: dict[int, AudioLoader] = {}  # Cache by sample rate

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

    def forward(
        self,
        synthesis: torch.Tensor | str | Path,
        reference: torch.Tensor | str | Path | None = None,
        **kwargs,
    ) -> float | torch.Tensor:
        """
        Main entry point for calculating the metric.

        Args:
            synthesis: Synthesis input
                - torch.Tensor: Audio waveform
                - Path: Path to audio file (will be loaded as tensor)
                - str: Transcription text (for WER)
            reference: Reference input
                - torch.Tensor: Reference audio waveform
                - Path: Path to reference audio file
                - str: Ground truth transcription (for WER)
                - None: Not used (for UTMOS)
            **kwargs: Metric-specific parameters (e.g., sample_rate)

        Returns:
            Metric score
        """
        if not self._is_initialized:
            raise MetricCalculationError("Model not initialized")

        try:
            sample_rate = kwargs.get("sample_rate")

            # Prepare reference input: Path or Tensor as audio, str as text
            if reference is not None and isinstance(reference, (Path, torch.Tensor)):
                if isinstance(reference, Path):
                    kwargs["orig_reference"] = reference
                reference = self._prepare_audio_input(reference, sample_rate)

            # Prepare synthesis input: Path or Tensor as audio, str as text
            if isinstance(synthesis, (Path, torch.Tensor)):
                if isinstance(synthesis, Path):
                    kwargs["orig_synthesis"] = synthesis
                synthesis = self._prepare_audio_input(synthesis, sample_rate)

            return self._forward_impl(synthesis, reference, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {self.get_name()} forward pass: {e}")
            raise MetricCalculationError(
                f"Error in {self.get_name()} forward pass: {e}"
            )

    @abstractmethod
    def _forward_impl(
        self,
        synthesis: torch.Tensor | str,
        reference: torch.Tensor | str | None = None,
        **kwargs,
    ) -> float | torch.Tensor:
        """
        Actual metric calculation implementation to be overridden by subclasses.

        Args:
            synthesis: Synthesis input
                - torch.Tensor: Audio waveform for audio-based metrics
                - str: Transcription text for WER metric
            reference: Reference input (metric-dependent)
                - torch.Tensor: Reference audio for similarity metrics (SIM, MCD, FFE)
                - str: Ground truth transcription for WER metric
                - None: Not used for no-reference metrics (UTMOS)
            **kwargs: Metric-specific parameters

        Returns:
            Metric score (float or tensor)
        """
        pass

    def _prepare_audio_input(
        self, audio: torch.Tensor | str | Path, sample_rate: int | None = None
    ) -> torch.Tensor:
        """
        Prepare audio input for processing.

        Args:
            audio: Audio tensor or file path
            sample_rate: Target sample rate for file inputs

        Returns:
            Processed audio tensor
        """
        if sample_rate is None:
            sample_rate = self.config.additional_params.get(
                "sample_rate", self.DEFAULT_SAMPLE_RATE
            )

        # 1. Load waveform
        if isinstance(audio, torch.Tensor):
            waveform = audio
        elif isinstance(audio, (str, Path)):
            # Cache loaders by sample rate to preserve resampler caches
            if sample_rate not in self._audio_loaders:
                self._audio_loaders[sample_rate] = AudioLoader(
                    sr=sample_rate, cache=False
                )
            waveform = self._audio_loaders[sample_rate].load(audio)
        else:
            raise TypeError(
                f"Unsupported audio type: {type(audio).__name__}. "
                f"Expected torch.Tensor, pathlib.Path, or str."
            )

        # 2. Basic normalization / dimensionality check
        if waveform.dim() > 1:
            waveform = waveform.mean(0)

        return waveform

    def __call__(
        self,
        synthesis: torch.Tensor | str | Path,
        reference: torch.Tensor | str | Path | None = None,
        **kwargs,
    ) -> float | torch.Tensor:
        """Callable interface."""
        return self.forward(synthesis, reference, **kwargs)

    def calculate_batch(
        self,
        pairs: list[tuple[Path, Path]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        """
        Calculate metric for multiple pairs with progress tracking.

        Args:
            pairs: List of (reference, synthesis) path pairs
            progress_callback: Optional callback function for progress updates

        Returns:
            List of metric scores
        """
        if not self._is_initialized:
            raise MetricCalculationError(
                "Model not initialized. Call load_model() first."
            )

        results = []
        total_pairs = len(pairs)
        iterator = tqdm(pairs, desc=f"Calculating {self.get_name()}", disable=False)

        for i, (ref_path, syn_path) in enumerate(iterator):
            try:
                # Use callable interface (forward internally) for uniform processing
                score = self(synthesis=syn_path, reference=ref_path)
                results.append(float(score))

                if progress_callback:
                    progress_callback(i + 1, total_pairs)

            except Exception as e:
                self.logger.warning(
                    f"Skipping pair ({ref_path}, {syn_path}) due to error: {e}"
                )
                results.append(np.nan)

        return results

    def calculate_batch_optimized(self, pairs: list[tuple[Path, Path]]) -> list[float]:
        """
        Optimized batch calculation (can be overridden by subclasses).
        Default implementation falls back to individual calculations via forward.
        """
        return self.calculate_batch(pairs)

    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass

    def validate_audio_files(
        self, pairs: list[tuple[Path, Path]]
    ) -> list[tuple[Path, Path]]:
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
            valid_pairs.append((ref_path, syn_path))
        
        self.logger.info(f"Validated {len(valid_pairs)}/{len(pairs)} audio pairs")
        return valid_pairs

    def get_device(self) -> torch.device:
        """Get the appropriate device for computation."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def cleanup(self) -> None:
        """Clean up resources."""
        self._audio_loaders.clear()
