"""
FFE (F0 Frame Error) calculator for fundamental frequency evaluation.
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .base import BaseMetricCalculator, MetricCalculationError, ModelConfig


class PitchExtractor:
    """Pitch extraction using YIN algorithm (similar to the provided code)."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def compute_yin(
        self,
        sig: np.ndarray,
        w_len: int = 512,
        w_step: int = 256,
        f0_min: int = 100,
        f0_max: int = 500,
        harmo_thresh: float = 0.1,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Compute the Yin Algorithm for F0 estimation.

        Args:
            sig: Audio signal
            w_len: Analysis window size
            w_step: Step size between windows
            f0_min: Minimum F0 frequency
            f0_max: Maximum F0 frequency
            harmo_thresh: Harmonicity threshold

        Returns:
            Tuple of (pitches, harmonic_rates, argmins, times)
        """
        tau_min = int(self.sr / f0_max)
        tau_max = int(self.sr / f0_min)

        time_scale = range(0, len(sig) - w_len, w_step)
        times = [t / float(self.sr) for t in time_scale]
        frames = [sig[t : t + w_len] for t in time_scale]

        pitches = [0.0] * len(time_scale)
        harmonic_rates = [0.0] * len(time_scale)
        argmins = [0.0] * len(time_scale)

        for i, frame in enumerate(frames):
            # Compute YIN
            df = self._difference_function(frame, w_len, tau_max)
            cm_df = self._cumulative_mean_normalized_difference_function(df, tau_max)
            p = self._get_pitch(cm_df, tau_min, tau_max, harmo_thresh)

            # Get results
            if np.argmin(cm_df) > tau_min:
                argmins[i] = float(self.sr / np.argmin(cm_df))
            if p != 0:  # A pitch was found
                pitches[i] = float(self.sr / p)
                harmonic_rates[i] = cm_df[p]
            else:  # No pitch, but we compute a value of the harmonic rate
                harmonic_rates[i] = min(cm_df)

        return pitches, harmonic_rates, argmins, times

    @staticmethod
    def _difference_function(x: np.ndarray, n: int, tau_max: int) -> np.ndarray:
        """Compute difference function using FFT."""
        x = np.array(x, np.float64)
        w = x.size
        tau_max = min(tau_max, w)
        x_cumsum = np.concatenate((np.array([0.0]), (x * x).cumsum()))
        size = w + tau_max
        p2 = (size // 32).bit_length()
        nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
        size_pad = min(x * 2**p2 for x in nice_numbers if x * 2**p2 >= size)
        fc = np.fft.rfft(x, size_pad)
        conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
        return (
            x_cumsum[w : w - tau_max : -1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv
        )

    @staticmethod
    def _cumulative_mean_normalized_difference_function(
        df: np.ndarray, n: int
    ) -> np.ndarray:
        """Compute cumulative mean normalized difference function (CMND)."""
        cmn_df = df[1:] * range(1, n) / np.cumsum(df[1:]).astype(float)
        return np.insert(cmn_df, 0, 1)

    @staticmethod
    def _get_pitch(
        cmdf: np.ndarray, tau_min: int, tau_max: int, harmo_th: float = 0.1
    ) -> int:
        """Return fundamental period based on CMND function."""
        tau = tau_min
        while tau < tau_max:
            if cmdf[tau] < harmo_th:
                while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                    tau += 1
                return tau
            tau += 1
        return 0  # if unvoiced


class FFECalculator(BaseMetricCalculator):
    """F0 Frame Error calculator for pitch accuracy evaluation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.target_sr = self.config.additional_params.get("sample_rate", 16000)
        self.pitch_extractor = None

    def _load_model_impl(self) -> None:
        """Initialize F0 extraction components."""
        try:
            self.pitch_extractor = PitchExtractor(sr=self.target_sr)
            self.logger.info(
                f"Initialized FFE calculator with sample rate: {self.target_sr}"
            )

        except Exception as e:
            raise MetricCalculationError(f"Failed to initialize FFE calculator: {e}")

    def extract_f0(
        self, audio_path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract F0 from audio using YIN algorithm."""
        waveform = self._prepare_audio_input(audio_path)
        audio = waveform.numpy().astype(np.double)
        return self._extract_f0_from_wav(audio)

    def _extract_f0_from_wav(
        self, wav: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Internal helper for YIN-based F0 extraction. Returns (times, pitches, argmins, harmonic_rates)."""
        pitches, harmonic_rates, argmins, times = self.pitch_extractor.compute_yin(wav)
        return (
            np.array(times),
            np.array(pitches),
            np.array(argmins),
            np.array(harmonic_rates),
        )

    def calculate_ffe(self, ref_f0_data: tuple, syn_f0_data: tuple) -> float:
        """Calculate F0 Frame Error between reference and synthesis F0."""
        try:
            ref_times, ref_f0, _, _ = ref_f0_data
            syn_times, syn_f0, _, _ = syn_f0_data

            ref_f0 = np.array(ref_f0)
            syn_f0 = np.array(syn_f0)

            # Align F0 sequences (simple alignment by taking minimum length)
            min_len = min(len(ref_f0), len(syn_f0))
            ref_f0_aligned = ref_f0[:min_len]
            syn_f0_aligned = syn_f0[:min_len]

            # Calculate frame errors
            gross_pitch_error_frames = self._gross_pitch_error_frames(
                ref_f0_aligned, syn_f0_aligned
            )
            voicing_decision_error_frames = self._voicing_decision_error_frames(
                ref_f0_aligned, syn_f0_aligned
            )

            total_errors = (
                np.sum(gross_pitch_error_frames)
                + np.sum(voicing_decision_error_frames)
            )
            total_frames = len(ref_f0_aligned)

            if total_frames == 0:
                return 0.0

            ffe_score = total_errors / total_frames
            return float(ffe_score)

        except Exception as e:
            raise MetricCalculationError(f"FFE calculation failed: {e}")

    @staticmethod
    def _voicing_decision_error_frames(true_f0: np.ndarray, est_f0: np.ndarray) -> bool:
        """Calculate voicing decision error frames."""
        return (est_f0 != 0) != (true_f0 != 0)

    @staticmethod
    def _true_voiced_frames(true_f0: np.ndarray, est_f0: np.ndarray) -> bool:
        """Find frames where both reference and estimate are voiced."""
        return (est_f0 != 0) & (true_f0 != 0)

    def _gross_pitch_error_frames(
        self, true_f0: np.ndarray, est_f0: np.ndarray, eps: float = 1e-8
    ) -> np.ndarray:
        """Calculate gross pitch error frames."""
        voiced_frames = self._true_voiced_frames(true_f0, est_f0)
        true_f0_eps = true_f0 + eps
        pitch_error_frames = np.abs(est_f0 / true_f0_eps - 1) > 0.2
        return voiced_frames & pitch_error_frames

    def _forward_impl(
        self,
        synthesis: torch.Tensor,
        reference: torch.Tensor | None = None,
        **kwargs,
    ) -> float:
        """Calculate FFE between synthesis and reference tensors."""
        if reference is None:
            raise MetricCalculationError("FFE requires a reference audio tensor")

        try:
            # WORLD/SPTK work with numpy doubles
            syn_audio = synthesis.cpu().numpy().astype(np.double)
            ref_audio = reference.cpu().numpy().astype(np.double)

            # Extract F0 using YIN helper
            syn_f0_data = self._extract_f0_from_wav(syn_audio)
            ref_f0_data = self._extract_f0_from_wav(ref_audio)

            return self.calculate_ffe(ref_f0_data, syn_f0_data)

        except Exception as e:
            raise MetricCalculationError(f"FFE forward pass failed: {e}")

    def calculate_batch_optimized(self, pairs: list[tuple[Path, Path]]) -> list[float]:
        """
        Optimized batch calculation for FFE.
        Extracts all F0 features first, then calculates FFE scores.
        """
        if not pairs:
            return []

        try:
            # 1. Extract unique paths
            all_paths = list(set([p for pair in pairs for p in pair]))
            f0_features = {}

            self.logger.info(f"Extracting F0 features for {len(all_paths)} unique files")
            for audio_path in tqdm(all_paths, desc="Extracting F0 features"):
                try:
                    f0_features[audio_path] = self.extract_f0(audio_path)
                except Exception as e:
                    self.logger.warning(f"Failed F0 extraction for {audio_path}: {e}")
                    f0_features[audio_path] = None

            # 2. Calculate FFE scores
            self.logger.info(f"Calculating FFE scores for {len(pairs)} pairs")
            results = []
            for ref_path, syn_path in tqdm(pairs, desc="Calculating FFE scores"):
                try:
                    ref_data = f0_features.get(ref_path)
                    syn_data = f0_features.get(syn_path)

                    if ref_data is not None and syn_data is not None:
                        results.append(self.calculate_ffe(ref_data, syn_data))
                    else:
                        results.append(np.nan)
                except Exception as e:
                    self.logger.warning(f"Failed calculation for pair: {e}")
                    results.append(np.nan)

            return results

        except Exception as e:
            self.logger.error(f"FFE batch optimization failed: {e}. Falling back to individual.")
            return super().calculate_batch_optimized(pairs)

    def get_name(self) -> str:
        return "FFE"


if __name__ == "__main__":
    from pathlib import Path

    ref_path = Path("data/test/ref.wav")
    syn_path = Path("data/test/syn.wav")

    config = ModelConfig(
        name="ffe", batch_size=8, device="cpu", additional_params={"sample_rate": 16000}
    )

    try:
        with FFECalculator(config) as calculator:
            print(f"Testing {calculator.get_name()} calculator...")
            score = calculator(reference=ref_path, synthesis=syn_path)
            print(f"FFE Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
