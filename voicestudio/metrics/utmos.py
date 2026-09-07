"""Predicted mean opinion score of generated speech, without a reference recording."""

import datasets
import evaluate
import numpy as np
import torch

from ..models.utmos_v2 import UTMOSv2FeatureExtractor, UTMOSv2ForAudioClassification
from .base import load_audio


_DESCRIPTION = """
Mean opinion score a listening panel would give a clip, predicted by UTMOSv2 and needing no reference
recording. The features are drawn at random from the clip, so one call is one sample of the score
rather than the score; each clip is drawn `draws` times and the spread across those draws is reported
alongside the mean, because a mean quoted without it cannot be told from a neighbouring one.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list[str]`): Paths to the audio to score.

Returns:
    utmos (`float`): Mean predicted score across every clip, on the five point opinion scale.
    utterances (`list[float]`): Per-clip mean over its draws, in input order.
    deviation (`list[float]`): Per-clip standard deviation over its draws, in input order.
    draws (`int`): Draws averaged per clip.
    domain (`str`): Listening test the prediction imitates.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Utmos(evaluate.Metric):
    def __init__(
        self,
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
        super().__init__(**kwargs)
        self.model_id = model_id
        self.domain = domain
        self.draws = draws
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.batch_size = batch_size
        self._model = None
        self._extractor = None

    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({"predictions": datasets.Value("string")}),
            codebase_urls=["https://github.com/sarulab-speech/UTMOSv2"],
        )

    def load_scorer(self) -> None:
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

    def score(self, audio_paths: list[str]) -> torch.Tensor:
        """Scores each clip once per draw.

        Args:
            audio_paths (`list[str]`):
                Paths to the audio to score.

        Returns:
            `torch.Tensor`: Every score, shaped `(clips, draws)`.
        """
        self.load_scorer()
        sampling_rate = self._extractor.sampling_rate
        waveforms = [load_audio(path, sampling_rate).numpy() for path in audio_paths]

        scores = torch.empty(len(waveforms), self.draws)
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
                scores[start : start + len(batch), draw] = logits.reshape(-1).float().cpu()
        return scores

    def _compute(self, predictions: list[str]) -> dict:
        # A path repeated across the batch is scored once. The scorer dominates the runtime here.
        unique_paths = list(dict.fromkeys(predictions))
        scores = self.score(unique_paths)
        by_path = dict(zip(unique_paths, scores))

        drawn = torch.stack([by_path[path] for path in predictions])
        means = drawn.mean(1)
        return {
            "utmos": float(means.mean()),
            "utterances": means.tolist(),
            "deviation": drawn.std(1).tolist(),
            "draws": self.draws,
            "domain": self.domain,
        }


__all__ = ["Utmos"]
