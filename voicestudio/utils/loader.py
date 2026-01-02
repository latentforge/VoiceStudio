"""
Audio loading utilities for metric calculation.
Separates file I/O from metric computation logic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
import warnings


class AudioLoader:
    """
    Audio file loader with caching and preprocessing.

    Handles file I/O, resampling, and caching to separate concerns
    from metric calculation logic.
    """

    DEFAULT_SAMPLE_RATE = 16000

    def __init__(
        self,
        sr: Optional[int] = None,
        mono: bool = True,
        cache: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize AudioLoader.

        Args:
            sr: Sample rate for resampling (default: 16000 for speech)
            mono: Convert to mono if True (uses channel averaging)
            cache: Enable caching of loaded audio
            device: Device to load tensors to (None for CPU)
        """
        self.sr = sr if sr is not None else self.DEFAULT_SAMPLE_RATE
        self.mono = mono
        self.device = torch.device(device) if device else torch.device("cpu")
        self._cache: Dict[Path, torch.Tensor] = {} if cache else None
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def load(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load audio file and convert to tensor.

        Args:
            path: Path to audio file

        Returns:
            Audio tensor of shape [samples] if mono, [channels, samples] otherwise

        Note:
            Mono conversion uses channel averaging (same as librosa.to_mono).
        """
        # Normalize path to absolute
        path = Path(path).resolve()

        if self._cache is not None and path in self._cache:
            return self._cache[path]

        try:
            waveform, sample_rate = torchaudio.load(str(path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {path}")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load audio {path}: {e}")

        original_dtype = waveform.dtype

        # Validate audio quality
        if torch.isnan(waveform).any():
            raise ValueError(f"Audio contains NaN values: {path}")
        if torch.isinf(waveform).any():
            raise ValueError(f"Audio contains Inf values: {path}")
        if waveform.abs().max() < 1e-6:
            warnings.warn(
                f"Near-silent audio detected (max amplitude < 1e-6): {path}",
                UserWarning,
            )

        # Convert to mono if requested
        if self.mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary (with dtype preservation)
        if sample_rate != self.sr:
            if sample_rate not in self._resamplers:
                # Create resampler with matching dtype for precision
                self._resamplers[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sr,
                    dtype=original_dtype,  # Preserve precision
                )
            waveform = self._resamplers[sample_rate](waveform)

        waveform = waveform.to(self.device)

        # Squeeze if mono
        if self.mono:
            waveform = waveform.squeeze(0)

        if self._cache is not None:
            self._cache[path] = waveform

        return waveform

    def load_batch(
        self, paths: List[Union[str, Path]], unique_only: bool = True
    ) -> Dict[Path, torch.Tensor]:
        """
        Load multiple audio files.

        Args:
            paths: List of paths to audio files
            unique_only: Only load unique paths to avoid duplicates

        Returns:
            Dictionary mapping paths to audio tensors
        """
        paths = [Path(p).resolve() for p in paths]

        if unique_only:
            paths = list(set(paths))

        return {path: self.load(path) for path in paths}

    def clear_cache(self):
        """Clear the audio cache."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached audio files."""
        return len(self._cache) if self._cache is not None else 0

    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._cache is not None


if __name__ == "__main__":
    loader = AudioLoader(sr=16000, cache=True)

    audio = loader.load("example.wav")
    print(f"Loaded audio shape: {audio.shape}")

    # Load batch
    paths = ["audio1.wav", "audio2.wav", "audio1.wav"]  # audio1 appears twice
    batch = loader.load_batch(paths, unique_only=True)
    print(f"Loaded {len(batch)} unique files")
