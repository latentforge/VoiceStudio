"""Predicted mean opinion score of generated speech, without a reference recording."""

import numpy as np
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..models.utmos_v2 import UTMOSv2FeatureExtractor, UTMOSv2ForAudioClassification
from .base import Metric, MetricConfig
from ..utils.audio_utils import load_audio


class Utmos(Metric):
    def __init__(
        self,
        config: MetricConfig,
        model_id: str = "sarulab-speech/UTMOSv2",
        domain: str = "sarulab",
        draws: int = 64,
        seed: int | None = None,
        device: str | None = None,
        dtype: str | torch.dtype = torch.float32,
        batch_size: int = 4,
        **kwargs,
    ):
        """
        Args:
            model_id (`str`, *optional*, defaults to `"sarulab-speech/UTMOSv2"`):
                Repository id of the scorer.
            domain (`str`, *optional*, defaults to `"sarulab"`):
                Listening test the score should imitate, a name in
                [`~models.utmos_v2.DOMAINS`]. The scale differs between corpora, so scores carrying
                different domains are not comparable.
            draws (`int`, *optional*, defaults to 64):
                Feature draws averaged per clip. Sixteen leaves a spread wider than the differences a
                score is usually asked to resolve, so the default follows the count upstream's own
                comparison needed.
            seed (`int`, *optional*):
                Seed of the draw positions. Given one, a run repeats exactly.
            device (`str`, *optional*):
                Device the scorer runs on. Defaults to CUDA where it is available.
            dtype (`str` or `torch.dtype`, *optional*, defaults to `torch.float32`):
                Precision the scorer is loaded in.
            batch_size (`int`, *optional*, defaults to 4):
                Clips per forward pass. Each clip carries eight 512 by 512 three channel images, and
                the scorer is an ensemble of five predictors, so this is bounded by memory rather
                than by throughput.
        """
        super().__init__(config)
        self.model_id = model_id
        self.domain = domain
        self.draws = draws
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.batch_size = batch_size
        self._model = None
        self._extractor = None

    def load(self) -> None:
        """Loads the UTMOSv2 checkpoint named by `model_id`, once."""
        if self._model is not None:
            return
        self._extractor = UTMOSv2FeatureExtractor.from_pretrained(self.model_id)
        self._model = UTMOSv2ForAudioClassification.from_pretrained(self.model_id, dtype=self.dtype)
        self._model.to(self.device).eval()

    def release(self) -> None:
        """Drops the loaded scorer and frees the memory it held."""
        self._model = None
        self._extractor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, audio_paths: list[str]) -> torch.Tensor:
        """Scores each clip once per draw.

        Args:
            audio_paths (`list[str]`):
                Paths to the audio to score.

        Returns:
            `torch.Tensor`: Every score, shaped `(clips, draws)`, `nan` for a clip holding no speech.
        """
        self.load()
        sampling_rate = self._extractor.sampling_rate
        loaded = [load_audio(path, sampling_rate).numpy() for path in audio_paths]

        # The scorer reads an excerpt of what is left once the quiet stretches are dropped, and a clip
        # that is quiet throughout leaves nothing to excerpt. Asking the same question the extractor
        # asks keeps the two from disagreeing about what counts as silence.
        speaking = [self._extractor.remove_silent_sections(clip).shape[0] > 0 for clip in loaded]
        waveforms = [clip for clip, heard in zip(loaded, speaking) if heard]
        scores = torch.full((len(loaded), self.draws), float("nan"))
        if not waveforms:
            return scores
        rows = [index for index, heard in enumerate(speaking) if heard]

        drawn = torch.empty(len(waveforms), self.draws)
        for draw in range(self.draws):
            # A seeded run advances the source per draw, so the draws differ from one another while
            # the run as a whole repeats.
            generator = np.random.default_rng(self.seed + draw) if self.seed is not None else None
            for start in range(0, len(waveforms), self.batch_size):
                batch = waveforms[start : start + self.batch_size]
                inputs = self._extractor(
                    batch, sampling_rate=sampling_rate, domain=self.domain, generator=generator
                ).to(self.device)
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                drawn[start : start + len(batch), draw] = logits.reshape(-1).float().cpu()
        scores[rows] = drawn
        return scores

    def score(self, artifacts: Sequence[Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Predicts the opinion score of each clip, over several feature draws.

        Args:
            artifacts (`Sequence[Any]`):
                Artifacts carrying `audio`, a path. No reference is read: the score is predicted
                from the clip alone.
            **kwargs:
                Unused.

        Returns:
            `list[dict[str, Any]]`: One mapping per artifact, holding the mean over the draws and
            the spread across them. A mean quoted without the spread cannot be told from a
            neighbouring one.
        """
        paths = [artifact["audio"] for artifact in artifacts]
        # A path repeated across the batch is scored once. The scorer dominates the runtime here.
        unique_paths = list(dict.fromkeys(paths))
        by_path = dict(zip(unique_paths, self.predict(unique_paths)))
        return [
            {
                "id": artifact.get("id"),
                "utmos": float(by_path[path].mean()),
                "deviation": float(by_path[path].std()),
                "audio": path,
                "draws": self.draws,
                "domain": self.domain,
            }
            for artifact, path in zip(artifacts, paths)
        ]

    def pool(self, statistics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Averages the clips that could be scored.

        Args:
            statistics (`Sequence[Mapping[str, Any]]`):
                Everything [`Utmos.score`] returned.

        Returns:
            `dict[str, float]`: The mean score under this metric's name, on the five point opinion
            scale, and the clips behind it. Read the mean beside `unscorable`, since a mean over a
            subset says nothing about the rest.
        """
        values = [float(statistic["utmos"]) for statistic in statistics]
        scored = [value for value in values if value == value and abs(value) != float("inf")]
        return {
            self.name: sum(scored) / len(scored) if scored else float("nan"),
            "scored": float(len(scored)),
            "unscorable": float(len(values) - len(scored)),
            "draws": float(self.draws),
        }

__all__ = ["Utmos"]
