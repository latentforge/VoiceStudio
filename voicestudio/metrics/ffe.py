"""F0 frame error between a reference recording and a generated one."""

import datasets
import evaluate
import torch

from .base import load_batch


_DESCRIPTION = """
F0 frame error between a reference recording and a generated one, the fraction of frames on which the
two disagree about pitch. A frame counts as an error when the two disagree about voicing, or when both
are voiced and the generated F0 departs from the reference by more than `pitch_tolerance`. Pitch comes
from YIN, run over every frame of every clip at once.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list[str]`): Paths to the generated audio.
    references (`list[str]`): Paths to the reference audio, one per generation.

Returns:
    ffe (`float`): Error frames over compared frames, pooled across every pair.
    gross_pitch_error (`float`): Frames where both are voiced but the pitches disagree, over compared
        frames.
    voicing_decision_error (`float`): Frames where the two disagree about voicing, over compared frames.
    frames (`int`): Frames compared, the denominator of the three rates.
    utterances (`list[float]`): Per-pair F0 frame error, in input order.
"""


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


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Ffe(evaluate.Metric):
    def __init__(
        self,
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
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.harmonic_threshold = harmonic_threshold
        self.pitch_tolerance = pitch_tolerance
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
        )

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

    def _compute(self, predictions: list[str], references: list[str]) -> dict:
        contours = self.estimate_f0(list(predictions) + list(references))

        utterances = []
        total_frames = 0
        total_gross = 0
        total_voicing = 0
        for prediction, reference in zip(predictions, references):
            estimated = contours[prediction]
            truth = contours[reference]
            frames = min(estimated.shape[0], truth.shape[0])
            estimated, truth = estimated[:frames], truth[:frames]

            voicing_error = (estimated != 0) != (truth != 0)
            both_voiced = (estimated != 0) & (truth != 0)
            departure = (estimated / (truth + torch.finfo(truth.dtype).eps) - 1).abs()
            gross_error = both_voiced & (departure > self.pitch_tolerance)

            errors = int(gross_error.sum()) + int(voicing_error.sum())
            utterances.append(errors / frames if frames else float("inf"))
            total_frames += frames
            total_gross += int(gross_error.sum())
            total_voicing += int(voicing_error.sum())

        return {
            "ffe": (total_gross + total_voicing) / total_frames if total_frames else float("inf"),
            "gross_pitch_error": total_gross / total_frames if total_frames else float("inf"),
            "voicing_decision_error": total_voicing / total_frames if total_frames else float("inf"),
            "frames": total_frames,
            "utterances": utterances,
        }


__all__ = ["Ffe"]
