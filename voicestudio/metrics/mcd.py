"""
MCD (Mel Cepstral Distortion) calculator for spectral quality evaluation.
"""

import math
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pysptk
import pyworld
from librosa.sequence import dtw
from tqdm import tqdm

from .base import BaseMetricCalculator, MetricCalculationError, ModelConfig


class MCDCalculator(BaseMetricCalculator):
    """Mel Cepstral Distortion calculator."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.target_sr = self.config.additional_params.get("sample_rate", 16000)
        self.frame_period = self.config.additional_params.get("frame_period", 5.0)
        self.alpha = self.config.additional_params.get("alpha", 0.65)
        self.fft_size = self.config.additional_params.get("fft_size", 512)
        self.mcep_size = self.config.additional_params.get("mcep_size", 25)

    def _load_model_impl(self) -> None:
        """Initialize MCD calculation components."""
        try:
            self.logger.info(f"Initialized MCD calculator with parameters:")
            self.logger.info(f"  Sample rate: {self.target_sr}")
            self.logger.info(f"  Frame period: {self.frame_period}")
            self.logger.info(f"  Alpha: {self.alpha}")
            self.logger.info(f"  FFT size: {self.fft_size}")
            self.logger.info(f"  MCEP size: {self.mcep_size}")

        except ImportError as e:
            raise MetricCalculationError(f"Required libraries not installed: {e}")
        except Exception as e:
            raise MetricCalculationError(f"Failed to initialize MCD calculator: {e}")

    def _load_and_preprocess_audio(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio for MCEP extraction."""
        try:
            # Load audio using librosa (consistent with original code)
            wav, _ = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
            return wav.astype(np.double)

        except Exception as e:
            raise MetricCalculationError(f"Failed to load audio {audio_path}: {e}")

    def extract_mcep(self, audio_path: Path) -> np.ndarray:
        """
        Extract MCEP features from audio.

        Args:
            audio_path: Path to audio file

        Returns:
            MCEP feature matrix
        """
        try:
            # Load audio
            loaded_wav = self._load_and_preprocess_audio(audio_path)

            # Use WORLD vocoder to extract spectral envelope
            _, sp, _ = pyworld.wav2world(
                loaded_wav.astype(np.double),
                fs=self.target_sr,
                frame_period=self.frame_period,
                fft_size=self.fft_size,
            )

            # Extract MCEP features
            mgc = pysptk.sp2mc(sp, order=self.mcep_size, alpha=self.alpha)

            return mgc

        except Exception as e:
            self.logger.error(f"Failed to extract MCEP from {audio_path}: {e}")
            raise MetricCalculationError(f"MCEP extraction failed: {e}")

    @staticmethod
    def log_spec_dB_dist(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate log spectral dB distance."""
        log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
        diff = x - y
        return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

    def calculate_mcd_dtw(self, ref_mcep: np.ndarray, syn_mcep: np.ndarray) -> float:
        """
        Calculate MCD between MCEP features using Dynamic Time Warping.

        Args:
            ref_mcep: Reference MCEP features
            syn_mcep: Synthesis MCEP features

        Returns:
            MCD score
        """
        try:
            # Use DTW to align sequences (skip 0th coefficient)
            cost_matrix, warping_path = dtw(
                ref_mcep[:, 1:].T, syn_mcep[:, 1:].T, metric=self.log_spec_dB_dist
            )

            return cost_matrix[-1, -1] / len(warping_path)

        except Exception as e:
            raise MetricCalculationError(f"MCD DTW calculation failed: {e}")

    def calculate_mcd_frame_by_frame(
        self, ref_mcep: np.ndarray, syn_mcep: np.ndarray
    ) -> float:
        """
        Calculate frame-by-frame MCD (without DTW alignment).

        Args:
            ref_mcep: Reference MCEP features
            syn_mcep: Synthesis MCEP features

        Returns:
            MCD score
        """
        try:
            # Align sequences by taking minimum length
            min_len = min(len(ref_mcep), len(syn_mcep))
            ref_aligned = ref_mcep[:min_len, 1:]  # Skip 0th coefficient
            syn_aligned = syn_mcep[:min_len, 1:]  # Skip 0th coefficient

            # Calculate frame-wise distances
            distances = []
            for i in range(min_len):
                dist = self.log_spec_dB_dist(ref_aligned[i], syn_aligned[i])
                distances.append(dist)

            return np.mean(distances)

        except Exception as e:
            raise MetricCalculationError(f"Frame-by-frame MCD calculation failed: {e}")

    def calculate_mcd(
        self, ref_mcep: np.ndarray, syn_mcep: np.ndarray, use_dtw: bool = True
    ) -> float:
        """
        Calculate MCD between MCEP features.

        Args:
            ref_mcep: Reference MCEP features
            syn_mcep: Synthesis MCEP features
            use_dtw: Whether to use DTW alignment

        Returns:
            MCD score in dB
        """
        try:
            if use_dtw:
                return self.calculate_mcd_dtw(ref_mcep, syn_mcep)
            else:
                return self.calculate_mcd_frame_by_frame(ref_mcep, syn_mcep)

        except Exception as e:
            raise MetricCalculationError(f"MCD calculation failed: {e}")

    def _calculate_pair_impl(self, ref_path: Path, syn_path: Path) -> float:
        """Calculate MCD for a reference-synthesis pair."""
        try:
            # Extract MCEP features from both audio files
            ref_mcep = self.extract_mcep(ref_path)
            syn_mcep = self.extract_mcep(syn_path)

            # Calculate MCD (use DTW by default for better alignment)
            use_dtw = self.config.additional_params.get("use_dtw", True)
            mcd_score = self.calculate_mcd(ref_mcep, syn_mcep, use_dtw=use_dtw)

            self.logger.debug(f"MCD score: {mcd_score:.4f} dB")

            return mcd_score

        except Exception as e:
            raise MetricCalculationError(f"Failed to calculate MCD for pair: {e}")

    def calculate_batch_optimized(self, pairs: List[Tuple[Path, Path]]) -> List[float]:
        """
        Optimized batch calculation for MCD.
        Extracts all MCEP features first, then calculates MCD scores.
        """
        try:
            # Extract all unique audio files
            all_paths = set()
            for ref_path, syn_path in pairs:
                all_paths.add(ref_path)
                all_paths.add(syn_path)

            # Extract MCEP features for all files
            mcep_features = {}

            self.logger.info(
                f"Extracting MCEP features for {len(all_paths)} unique audio files"
            )

            for audio_path in tqdm(all_paths, desc="Extracting MCEP features"):
                try:
                    mcep_features[audio_path] = self.extract_mcep(audio_path)
                except Exception as e:
                    self.logger.warning(f"Failed to extract MCEP for {audio_path}: {e}")
                    mcep_features[audio_path] = None

            # Calculate MCD for all pairs
            results = []
            use_dtw = self.config.additional_params.get("use_dtw", True)

            for ref_path, syn_path in tqdm(pairs, desc="Calculating MCD scores"):
                try:
                    ref_mcep = mcep_features.get(ref_path)
                    syn_mcep = mcep_features.get(syn_path)

                    if ref_mcep is not None and syn_mcep is not None:
                        mcd_score = self.calculate_mcd(
                            ref_mcep, syn_mcep, use_dtw=use_dtw
                        )
                        results.append(mcd_score)
                    else:
                        results.append(np.nan)

                except Exception as e:
                    self.logger.warning(f"Failed MCD calculation for pair: {e}")
                    results.append(np.nan)

            return results

        except Exception as e:
            self.logger.warning(
                f"Batch processing failed, falling back to individual: {e}"
            )
            return super().calculate_batch_optimized(pairs)

    def get_name(self) -> str:
        return "MCD"


if __name__ == "__main__":
    ref_path = Path("data/test/ref.wav")
    syn_path = Path("data/test/syn.wav")

    config = ModelConfig(
        name="mcd",
        batch_size=8,
        device="cpu",
        additional_params={
            "sample_rate": 16000,
            "frame_period": 5.0,
            "alpha": 0.65,
            "fft_size": 512,
            "mcep_size": 25,
            "use_dtw": True,
        },
    )

    try:
        with MCDCalculator(config) as calculator:
            print(f"Testing {calculator.get_name()} calculator...")
            score = calculator.calculate_pair(ref_path, syn_path)
            print(f"MCD Score: {score:.4f} dB")
    except Exception as e:
        print(f"Test failed: {e}")
