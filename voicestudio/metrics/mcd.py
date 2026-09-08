"""Mel cepstral distortion between a reference recording and a generated one."""

import math
from concurrent.futures import ThreadPoolExecutor

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .base import Metric, MetricConfig
from ..utils.audio_utils import CheapTrickEnvelope, WorldF0Estimator, load_audio




# Converts a Euclidean cepstral distance into the dB figure MCD is quoted in.
_DECIBEL_CONSTANT = 10.0 / math.log(10.0) * math.sqrt(2.0)


def warping_matrix(source_order: int, target_order: int, alpha: float) -> torch.Tensor:
    """Builds the linear map from a real cepstrum to a mel cepstrum.

    The frequency transform behind mel cepstral analysis warps the frequency axis through an all-pass
    filter of coefficient `alpha`, by a recursion over the source coefficients. That recursion is
    linear in its input, so running it once against a basis collapses it into a matrix that later
    frames reuse.

    Args:
        source_order (`int`):
            Highest real cepstral coefficient fed in.
        target_order (`int`):
            Highest mel cepstral coefficient produced.
        alpha (`float`):
            All-pass coefficient, which sets how strongly the frequency axis is warped.

    Returns:
        `torch.Tensor`: The map, shaped `(source_order + 1, target_order + 1)`.
    """
    basis = torch.eye(source_order + 1, dtype=torch.float64)
    warped = torch.zeros(source_order + 1, target_order + 1, dtype=torch.float64)
    for index in range(source_order, -1, -1):
        previous = warped.clone()
        warped[:, 0] = basis[:, index] + alpha * previous[:, 0]
        if target_order >= 1:
            warped[:, 1] = (1.0 - alpha * alpha) * previous[:, 0] + alpha * previous[:, 1]
        for order in range(2, target_order + 1):
            warped[:, order] = previous[:, order - 1] + alpha * (previous[:, order] - warped[:, order - 1])
    return warped


def real_cepstrum(power_spectrum: torch.Tensor, n_fft: int, order: int) -> torch.Tensor:
    """Takes the real cepstrum of a power spectrum.

    Args:
        power_spectrum (`torch.Tensor`):
            Power spectrum, shaped `(..., n_fft // 2 + 1)`.
        n_fft (`int`):
            Window the spectrum was taken over.
        order (`int`):
            Highest coefficient kept.

    Returns:
        `torch.Tensor`: The cepstrum, shaped `(..., order + 1)`.
    """
    log_spectrum = power_spectrum.clamp_min(torch.finfo(power_spectrum.dtype).tiny).log()
    cepstrum = torch.fft.irfft(log_spectrum.to(torch.complex128), n=n_fft).real
    # The spectrum is real and even, so its own reflection doubles the two coefficients that have no
    # partner to be reflected onto.
    cepstrum[..., 0] = cepstrum[..., 0] / 2.0
    cepstrum[..., n_fft // 2] = cepstrum[..., n_fft // 2] / 2.0
    return cepstrum[..., : order + 1]


def warp(costs: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs dynamic time warping over a batch of frame-to-frame cost matrices.

    Args:
        costs (`torch.Tensor`):
            Local costs between every reference and generated frame, shaped
            `(pairs, reference, generated)`, right-padded where a pair is shorter than the batch.
        rows (`torch.Tensor`):
            Reference frames in each pair.
        columns (`torch.Tensor`):
            Generated frames in each pair.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The accumulated cost of the cheapest monotonic alignment
        per pair, and the number of steps along it.
    """
    pairs, height, width = costs.shape
    accumulated = costs.new_full((pairs, height + 1, width + 1), float("inf"))
    steps = costs.new_zeros((pairs, height + 1, width + 1))
    accumulated[:, 0, 0] = 0.0

    # One anti-diagonal at a time, across every pair at once: a cell reads only from earlier
    # diagonals, so the whole batch advances in one vectorized step per diagonal. Counting steps
    # alongside the cost removes the per-pair backtrace that would otherwise follow.
    for total in range(2, height + width + 1):
        row = torch.arange(max(1, total - width), min(height, total - 1) + 1, device=costs.device)
        column = total - row
        candidates = torch.stack(
            [
                accumulated[:, row - 1, column - 1],
                accumulated[:, row - 1, column],
                accumulated[:, row, column - 1],
            ]
        )
        taken = torch.stack(
            [
                steps[:, row - 1, column - 1],
                steps[:, row - 1, column],
                steps[:, row, column - 1],
            ]
        )
        best, chosen = candidates.min(dim=0)
        accumulated[:, row, column] = costs[:, row - 1, column - 1] + best
        steps[:, row, column] = taken.gather(0, chosen[None])[0] + 1

    index = torch.arange(pairs, device=costs.device)
    # Padding beyond a pair's own frames never feeds the cell its result is read from.
    return accumulated[index, rows, columns], steps[index, rows, columns]


def warp_budget(device: str) -> int:
    """Reports the bytes a warp may hold in its accumulators.

    Args:
        device (`str`):
            Device the alignment runs on.

    Returns:
        `int`: The budget in bytes, a quarter of what CUDA currently has free, or a fixed allowance
        on any other device where free memory is not knowable from here.
    """
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        return int(free // 4)
    return 2 * 1024**3


def plan_chunks(rows: list[int], columns: list[int], budget: int, element_size: int) -> list[list[int]]:
    """Groups pairs into warps that fit the budget.

    A warp allocates one accumulated cost and one step count per pair per cell, over a grid squared
    off to the largest pair in the group, so a group's cost is set by its longest member rather than
    its average.

    Args:
        rows (`list[int]`):
            Reference frames per pair.
        columns (`list[int]`):
            Generated frames per pair.
        budget (`int`):
            Bytes the accumulators may occupy.
        element_size (`int`):
            Bytes per accumulator cell.

    Returns:
        `list[list[int]]`: Indices into `rows` and `columns`, grouped per warp.
    """
    # Grouping similar lengths keeps a long pair from squaring off the grid for short ones.
    order = sorted(range(len(rows)), key=lambda index: (rows[index], columns[index]))
    chunks: list[list[int]] = []
    current: list[int] = []
    tallest = widest = 0
    for index in order:
        height = max(tallest, rows[index] + 1)
        width = max(widest, columns[index] + 1)
        # Two accumulators, one for cost and one for the step count.
        if current and (len(current) + 1) * height * width * element_size * 2 > budget:
            chunks.append(current)
            current, tallest, widest = [], 0, 0
            height, width = rows[index] + 1, columns[index] + 1
        current.append(index)
        tallest, widest = height, width
    if current:
        chunks.append(current)
    return chunks


def pad_stack(sequences: list[torch.Tensor]) -> torch.Tensor:
    """Right-pads cepstral sequences of differing length into one batch.

    Args:
        sequences (`list[torch.Tensor]`):
            Sequences shaped `(frames, coefficients)`.

    Returns:
        `torch.Tensor`: The batch, shaped `(sequences, longest, coefficients)`.
    """
    longest = max(sequence.shape[0] for sequence in sequences)
    stacked = sequences[0].new_zeros(len(sequences), longest, sequences[0].shape[1])
    for index, sequence in enumerate(sequences):
        stacked[index, : sequence.shape[0]] = sequence
    return stacked


class Mcd(Metric):
    def __init__(
        self,
        config: MetricConfig,
        sampling_rate: int = 16000,
        frame_period: float = 5.0,
        n_fft: int = 512,
        order: int = 25,
        alpha: float = 0.65,
        use_dtw: bool = True,
        pairs_per_warp: int | None = None,
        workers: int = 4,
        device: str | None = None,
        **kwargs,
    ):
        """
        Args:
            config (`MetricConfig`):
                What the metric is called, and where its model runs.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Rate both sides are resampled to before analysis.
            frame_period (`float`, *optional*, defaults to 5.0):
                Milliseconds between consecutive frames.
            n_fft (`int`, *optional*, defaults to 512):
                Window the spectrum is taken over.
            order (`int`, *optional*, defaults to 25):
                Highest mel cepstral coefficient kept. The zeroth is dropped from the distance, so
                the distance runs over `order` coefficients.
            alpha (`float`, *optional*, defaults to 0.65):
                All-pass coefficient of the frequency warping, which sets the mel scale it
                approximates for the sampling rate in use.
            use_dtw (`bool`, *optional*, defaults to `True`):
                Align the two with dynamic time warping. Set `False` to compare frame by frame after
                truncating to the shorter of the two, which only makes sense for durations that match.
            workers (`int`, *optional*, defaults to 4):
                Clips analysed at once. The f0 search is a clip at a time, so this is where the
                analysis parallelises.
            pairs_per_warp (`int`, *optional*):
                Pairs warped together. Left unset, each warp is filled to a quarter of the free CUDA
                memory, or to 2 GiB elsewhere. Warping more pairs at once costs almost nothing once
                the grid is wide, because the number of steps along it does not depend on how many
                pairs share it.
            device (`str`, *optional*):
                Device the analysis runs on. Defaults to CUDA where it is available.
        """
        super().__init__(config)
        self.sampling_rate = sampling_rate
        self.frame_period = frame_period
        self.n_fft = n_fft
        self.order = order
        self.alpha = alpha
        self.use_dtw = use_dtw
        self.workers = workers
        self.pairs_per_warp = pairs_per_warp
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._estimator = None
        self._envelope = None
        self._warping = None

    @property
    def hop_length(self) -> int:
        return int(self.sampling_rate * self.frame_period / 1000)

    def extract_cepstra(self, audio_paths: list[str]) -> dict[str, torch.Tensor]:
        """Computes mel cepstra per clip.

        Args:
            audio_paths (`list[str]`):
                Paths to the audio to analyse. A repeated path is analysed once.

        Returns:
            `dict[str, torch.Tensor]`: Cepstra of each distinct path, shaped `(frames, order)`
            with the zeroth coefficient already dropped.
        """
        if self._estimator is None:
            self._estimator = WorldF0Estimator(sampling_rate=self.sampling_rate, device=self.device)
            self._envelope = CheapTrickEnvelope(sampling_rate=self.sampling_rate, fft_size=self.n_fft)
            # `freqt` consumes the whole cepstrum, so the source runs to the transform length.
            self._warping = warping_matrix(self.n_fft - 1, self.order, self.alpha).to(self.device)

        # The f0 search is a clip at a time, so the clips are loaded and searched together and the
        # refinement, which is the expensive half, runs over every clip's frames in one pass.
        unique_paths = list(dict.fromkeys(audio_paths))
        with ThreadPoolExecutor(self.workers) as pool:
            samples = list(pool.map(lambda path: load_audio(path, self.sampling_rate).double().numpy(), unique_paths))
            searched = list(
                pool.map(lambda clip: self._estimator.dio(clip, frame_period=self.frame_period), samples)
            )
        positions = [found[1] for found in searched]
        contours = self._estimator.stonemask_batch(samples, positions, [found[0] for found in searched])

        cepstra = {}
        for path, clip, contour, frames in zip(unique_paths, samples, contours, positions):
            envelope = self._envelope.envelope(
                torch.from_numpy(clip).to(self.device),
                torch.from_numpy(contour).to(self.device),
                torch.from_numpy(frames).to(self.device),
            )
            cepstra[path] = (real_cepstrum(envelope, self.n_fft, self.n_fft - 1) @ self._warping)[:, 1:].cpu()

        return cepstra

    def distort(self, reference: list[torch.Tensor], generated: list[torch.Tensor]) -> list[float]:
        """Scores cepstral sequences pair by pair.

        Args:
            reference (`list[torch.Tensor]`):
                Reference cepstra per pair, each shaped `(frames, coefficients)`.
            generated (`list[torch.Tensor]`):
                Generated cepstra per pair, each shaped `(frames, coefficients)`.

        Returns:
            `list[float]`: The distortion in dB per pair, `inf` where either side has no frames.
        """
        scores = [float("inf")] * len(reference)
        scored = [
            index
            for index, (left, right) in enumerate(zip(reference, generated))
            if left.shape[0] and right.shape[0]
        ]
        if not scored:
            return scores

        if not self.use_dtw:
            for index in scored:
                frames = min(reference[index].shape[0], generated[index].shape[0])
                distance = (reference[index][:frames] - generated[index][:frames]).norm(dim=1)
                scores[index] = float(_DECIBEL_CONSTANT * distance.mean())
            return scores

        # Every pair walks the same anti-diagonals, so they are warped together.
        heights = [reference[index].shape[0] for index in scored]
        widths = [generated[index].shape[0] for index in scored]
        if self.pairs_per_warp is None:
            groups = plan_chunks(heights, widths, warp_budget(self.device), torch.finfo(torch.float64).bits // 8)
        else:
            groups = [
                list(range(start, min(start + self.pairs_per_warp, len(scored))))
                for start in range(0, len(scored), self.pairs_per_warp)
            ]

        for group in groups:
            chunk = [scored[position] for position in group]
            left = pad_stack([reference[index].double() for index in chunk]).to(self.device)
            right = pad_stack([generated[index].double() for index in chunk]).to(self.device)
            rows = torch.tensor([reference[index].shape[0] for index in chunk], device=self.device)
            columns = torch.tensor([generated[index].shape[0] for index in chunk], device=self.device)
            total, steps = warp(torch.cdist(left, right), rows, columns)
            for offset, index in enumerate(chunk):
                scores[index] = float(_DECIBEL_CONSTANT * total[offset] / steps[offset])
        return scores

    def score(self, artifacts: Sequence[Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Aligns each pair's cepstra and reports the distortion between them.

        Args:
            artifacts (`Sequence[Any]`):
                Artifacts carrying `audio` and `reference_audio`, both paths.
            **kwargs:
                Unused.

        Returns:
            `list[dict[str, Any]]`: One mapping per artifact, holding the distortion in decibels
            and the pair it was measured over.
        """
        generated = [artifact["audio"] for artifact in artifacts]
        reference = [artifact["reference_audio"] for artifact in artifacts]
        cepstra = self.extract_cepstra(generated + reference)
        distortions = self.distort(
            [cepstra[path] for path in reference], [cepstra[path] for path in generated]
        )
        return [
            {
                "id": artifact.get("id"),
                "mcd": distortion,
                "audio": source,
                "reference_audio": target,
            }
            for artifact, source, target, distortion in zip(
                artifacts, generated, reference, distortions
            )
        ]

    def pool(self, statistics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Averages the distortions over the pairs an alignment could be found for.

        A pair whose alignment failed reports `inf` rather than a number, and averaging that in
        would take the corpus with it, so those are counted and left out.

        Args:
            statistics (`Sequence[Mapping[str, Any]]`):
                Everything [`Mcd.score`] returned.

        Returns:
            `dict[str, float]`: The mean distortion under this metric's name, the pairs it covers,
            and the pairs that could not be aligned. Read the mean beside `unaligned`, since a mean
            over a subset says nothing about the rest.
        """
        values = [float(statistic["mcd"]) for statistic in statistics]
        finite = [value for value in values if value != float("inf")]
        return {
            self.name: sum(finite) / len(finite) if finite else float("inf"),
            "pairs": float(len(values)),
            "unaligned": float(len(values) - len(finite)),
        }

__all__ = ["Mcd"]
