"""
FFE (F0 Frame Error) calculator for fundamental frequency evaluation.
"""
from abc import ABC
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torchaudio

from .base import BaseMetricCalculator, ModelConfig, MetricCalculationError


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
        harmo_thresh: float = 0.1
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
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
        frames = [sig[t:t + w_len] for t in time_scale]

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
        x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
        size = w + tau_max
        p2 = (size // 32).bit_length()
        nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
        size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
        fc = np.fft.rfft(x, size_pad)
        conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
        return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv

    @staticmethod
    def _cumulative_mean_normalized_difference_function(df: np.ndarray, n: int) -> np.ndarray:
        """Compute cumulative mean normalized difference function (CMND)."""
        cmn_df = df[1:] * range(1, n) / np.cumsum(df[1:]).astype(float)
        return np.insert(cmn_df, 0, 1)

    @staticmethod
    def _get_pitch(cmdf: np.ndarray, tau_min: int, tau_max: int, harmo_th: float = 0.1) -> int:
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
        self.target_sr = self.config.additional_params.get('sample_rate', 16000)
        self.pitch_extractor = None

    def _load_model_impl(self) -> None:
        """Initialize F0 extraction components."""
        try:
            self.pitch_extractor = PitchExtractor(sr=self.target_sr)
            self.logger.info(f"Initialized FFE calculator with sample rate: {self.target_sr}")

        except Exception as e:
            raise MetricCalculationError(f"Failed to initialize FFE calculator: {e}")

    def _load_and_preprocess_audio(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio for F0 extraction."""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sr
                )
                waveform = resampler(waveform)

            # Convert to numpy and normalize
            audio_np = waveform.squeeze().numpy().astype(np.double)

            return audio_np

        except Exception as e:
            raise MetricCalculationError(f"Failed to load audio {audio_path}: {e}")

    def extract_f0(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract F0 from audio using YIN algorithm."""
        try:
            audio = self._load_and_preprocess_audio(audio_path)

            # Extract F0 using YIN
            pitches, harmonic_rates, argmins, times = self.pitch_extractor.compute_yin(audio, self.target_sr)

            return (
                np.array(pitches),
                np.array(harmonic_rates),
                np.array(argmins),
                np.array(times)
            )

        except Exception as e:
            self.logger.error(f"Failed to extract F0 from {audio_path}: {e}")
            raise MetricCalculationError(f"F0 extraction failed: {e}")

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
            gross_pitch_error_frames = self._gross_pitch_error_frames(ref_f0_aligned, syn_f0_aligned)
            voicing_decision_error_frames = self._voicing_decision_error_frames(ref_f0_aligned, syn_f0_aligned)

            total_errors = np.sum(gross_pitch_error_frames) + np.sum(voicing_decision_error_frames)
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

    def _gross_pitch_error_frames(self, true_f0: np.ndarray, est_f0: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Calculate gross pitch error frames."""
        voiced_frames = self._true_voiced_frames(true_f0, est_f0)
        true_f0_eps = true_f0 + eps  # Avoid division by zero
        pitch_error_frames = np.abs(est_f0 / true_f0_eps - 1) > 0.2
        return voiced_frames & pitch_error_frames

    def _calculate_pair_impl(self, ref_path: Path, syn_path: Path) -> float:
        """Calculate FFE for a reference-synthesis pair."""
        try:
            # Extract F0 from both audio files
            ref_f0_data = self.extract_f0(ref_path)
            syn_f0_data = self.extract_f0(syn_path)

            # Calculate FFE
            ffe_score = self.calculate_ffe(ref_f0_data, syn_f0_data)

            self.logger.debug(f"FFE score: {ffe_score:.4f}")

            return ffe_score

        except Exception as e:
            raise MetricCalculationError(f"Failed to calculate FFE for pair: {e}")

    def get_name(self) -> str:
        return "FFE"


if __name__ == '__main__':
    from pathlib import Path

    ref_path = Path("data/test/ref.wav")
    syn_path = Path("data/test/syn.wav")

    config = ModelConfig(
        name="ffe",
        batch_size=8,
        device="cpu",
        additional_params={'sample_rate': 16000}
    )

    try:
        with FFECalculator(config) as calculator:
            print(f"Testing {calculator.get_name()} calculator...")
            score = calculator.calculate_pair(ref_path, syn_path)
            print(f"FFE Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
