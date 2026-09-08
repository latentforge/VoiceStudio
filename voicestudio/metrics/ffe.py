"""F0 frame error between a reference recording and a generated one."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .base import Metric, MetricConfig
from ..utils.audio_utils import load_batch





def yin(frames: torch.Tensor, sampling_rate: int, tau_min: int, tau_max: int, threshold: float):
    """Estimates F0 for many frames at once with YIN.

    Args:
        frames (`torch.Tensor`):
            Analysis frames, shaped `(frames, frame_length)`.
        sampling_rate (`int`):
            Rate the frames were sampled at.
        tau_min (`int`):
            Shortest lag considered, which bounds F0 from above.
        tau_max (`int`):
            Longest lag considered, which bounds F0 from below.
        threshold (`float`):
            Harmonicity below which a lag counts as a pitch candidate.

    Returns:
        `torch.Tensor`: F0 in Hz per frame, `0` where the frame is unvoiced.
    """
    count, length = frames.shape
    frames = frames.double()

    # Difference function by autocorrelation, following YIN's own FFT formulation.
    power = torch.cat([frames.new_zeros(count, 1), (frames * frames).cumsum(1)], dim=1)
    size = 1 << (length + tau_max - 1).bit_length()
    spectrum = torch.fft.rfft(frames, size)
    correlation = torch.fft.irfft(spectrum * spectrum.conj(), size)[:, :tau_max]
    difference = (
        power[:, length - tau_max + 1 : length + 1].flip(1)
        + power[:, length : length + 1]
        - power[:, :tau_max]
        - 2 * correlation
    )

    # Cumulative mean normalized difference, which is defined as 1 at lag zero.
    tail = difference[:, 1:]
    lags = torch.arange(1, tau_max, device=frames.device, dtype=frames.dtype)
    running = tail.cumsum(1)
    # A silent frame has no difference to normalize by. Holding it at 1 keeps it above any threshold,
    # so it reads as unvoiced rather than as a perfect match at the shortest lag.
    normalized = torch.where(running > 0, tail * lags / running, frames.new_ones(()))
    normalized = torch.cat([frames.new_ones(count, 1), normalized], dim=1)

    # First lag under the threshold, then downhill to the local minimum, as YIN specifies.
    below = normalized[:, tau_min:tau_max] < threshold
    voiced = below.any(1)
    tau = below.to(torch.int64).argmax(1) + tau_min
    for _ in range(tau_max - tau_min):
        following = (tau + 1).clamp(max=tau_max - 1)
        descending = normalized.gather(1, following[:, None]) < normalized.gather(1, tau[:, None])
        step = descending.squeeze(1) & (tau + 1 < tau_max)
        if not step.any():
            break
        tau = tau + step.to(tau.dtype)

    return torch.where(voiced, sampling_rate / tau.to(frames.dtype), frames.new_zeros(()))


class Ffe(Metric):
    def __init__(
        self,
        config: MetricConfig,
        sampling_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 256,
        f0_min: int = 100,
        f0_max: int = 500,
        harmonic_threshold: float = 0.1,
        pitch_tolerance: float = 0.2,
        batch_size: int = 16,
        device: str | None = None,
        **kwargs,
    ):
        """
        Args:
            config (`MetricConfig`):
                What the metric is called, and where its model runs.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Rate both sides are resampled to before analysis.
            frame_length (`int`, *optional*, defaults to 512):
                Samples per analysis frame.
            hop_length (`int`, *optional*, defaults to 256):
                Samples between consecutive frames.
            f0_min (`int`, *optional*, defaults to 100):
                Lowest F0 searched for, in Hz.
            f0_max (`int`, *optional*, defaults to 500):
                Highest F0 searched for, in Hz.
            harmonic_threshold (`float`, *optional*, defaults to 0.1):
                YIN's absolute threshold on the normalized difference.
            pitch_tolerance (`float`, *optional*, defaults to 0.2):
                Relative F0 departure above which a voiced frame counts as a gross pitch error.
            batch_size (`int`, *optional*, defaults to 16):
                Clips whose frames are analysed in one pass.
            device (`str`, *optional*):
                Device the analysis runs on. Defaults to CUDA where it is available.
        """
        super().__init__(config)
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.harmonic_threshold = harmonic_threshold
        self.pitch_tolerance = pitch_tolerance
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def estimate_f0(self, audio_paths: list[str]) -> dict[str, torch.Tensor]:
        """Estimates an F0 contour per clip.

        Args:
            audio_paths (`list[str]`):
                Paths to the audio to analyse. A repeated path is analysed once.

        Returns:
            `dict[str, torch.Tensor]`: The F0 contour of each distinct path, in Hz, zero where unvoiced.
        """
        tau_min = int(self.sampling_rate / self.f0_max)
        tau_max = int(self.sampling_rate / self.f0_min)
        unique_paths = list(dict.fromkeys(audio_paths))
        contours = {}

        for start in range(0, len(unique_paths), self.batch_size):
            batch = unique_paths[start : start + self.batch_size]
            waveforms, lengths = load_batch(batch, self.sampling_rate)
            waveforms = waveforms.to(self.device)
            if waveforms.shape[1] <= self.frame_length:
                contours.update((path, torch.zeros(0)) for path in batch)
                continue

            frames = waveforms.unfold(1, self.frame_length, self.hop_length)
            pitch = yin(
                frames.reshape(-1, self.frame_length),
                self.sampling_rate,
                tau_min,
                tau_max,
                self.harmonic_threshold,
            ).reshape(frames.shape[0], frames.shape[1])

            # Frames that would run past the clip's own samples belong to the padding, not the clip.
            for index, path in enumerate(batch):
                usable = -(-(int(lengths[index]) - self.frame_length) // self.hop_length)
                contours[path] = pitch[index, : max(usable, 0)].cpu()

        return contours

    def score(self, artifacts: Sequence[Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Compares each pair's pitch contour frame by frame.

        Args:
            artifacts (`Sequence[Any]`):
                Artifacts carrying `audio` and `reference_audio`, both paths.
            **kwargs:
                Unused.

        Returns:
            `list[dict[str, Any]]`: One mapping per artifact, holding the frames compared and the
            frames each kind of error was found on.
        """
        generated = [artifact["audio"] for artifact in artifacts]
        reference = [artifact["reference_audio"] for artifact in artifacts]
        contours = self.estimate_f0(generated + reference)

        statistics = []
        for artifact, source, target in zip(artifacts, generated, reference):
            estimated, truth = contours[source], contours[target]
            frames = min(estimated.shape[0], truth.shape[0])
            estimated, truth = estimated[:frames], truth[:frames]

            voicing_error = (estimated != 0) != (truth != 0)
            both_voiced = (estimated != 0) & (truth != 0)
            departure = (estimated / (truth + torch.finfo(truth.dtype).eps) - 1).abs()
            gross_error = both_voiced & (departure > self.pitch_tolerance)
            statistics.append(
                {
                    "id": artifact.get("id"),
                    "frames": int(frames),
                    "gross_pitch_error": int(gross_error.sum()),
                    "voicing_decision_error": int(voicing_error.sum()),
                    "audio": source,
                    "reference_audio": target,
                }
            )
        return statistics

    def pool(self, statistics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Divides the summed error frames by the summed frames.

        Args:
            statistics (`Sequence[Mapping[str, Any]]`):
                Everything [`Ffe.score`] returned.

        Returns:
            `dict[str, float]`: The rate under this metric's name, its two terms read separately,
            and the frames they were divided by. A pitch error and a voicing error are different
            failures and a total that hides which one moved says little.
        """
        totals = {key: 0 for key in ("frames", "gross_pitch_error", "voicing_decision_error")}
        for statistic in statistics:
            for key in totals:
                totals[key] += int(statistic.get(key, 0))
        frames = totals["frames"]
        errors = totals["gross_pitch_error"] + totals["voicing_decision_error"]
        return {
            self.name: errors / frames if frames else float("inf"),
            "gross_pitch_error": totals["gross_pitch_error"] / frames if frames else float("inf"),
            "voicing_decision_error": (
                totals["voicing_decision_error"] / frames if frames else float("inf")
            ),
            "frames": float(frames),
        }

__all__ = ["Ffe"]
