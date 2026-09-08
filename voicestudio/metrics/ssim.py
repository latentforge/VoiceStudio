"""Speaker similarity between a reference recording and a generated one."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..models.ecapa_tdnn import EcapaTdnnFeatureExtractor, EcapaTdnnForXVector
from .base import Metric, MetricConfig
from ..utils.audio_utils import load_audio





class Ssim(Metric):
    def __init__(
        self,
        config: MetricConfig,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str | None = None,
        dtype: str | torch.dtype = torch.float32,
        batch_size: int = 8,
        **kwargs,
    ):
        """
        Args:
            model_id (`str`, *optional*, defaults to `"speechbrain/spkrec-ecapa-voxceleb"`):
                Repository id of the speaker encoder.
            device (`str`, *optional*):
                Device the encoder runs on. Defaults to CUDA where it is available.
            dtype (`str` or `torch.dtype`, *optional*, defaults to `torch.float32`):
                Precision the encoder is loaded in.
            batch_size (`int`, *optional*, defaults to 8):
                Clips per forward pass.
        """
        super().__init__(config)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.batch_size = batch_size
        self._model = None
        self._extractor = None

    def load(self) -> None:
        """Loads the speaker encoder named by `model_id`, once."""
        if self._model is not None:
            return
        self._extractor = EcapaTdnnFeatureExtractor.from_pretrained(self.model_id)
        self._model = EcapaTdnnForXVector.from_pretrained(self.model_id, dtype=self.dtype)
        self._model.to(self.device).eval()

    def release(self) -> None:
        """Drops the loaded encoder and frees the memory it held."""
        self._model = None
        self._extractor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed(self, audio_paths: list[str]) -> dict[str, torch.Tensor]:
        """Embeds each clip as a unit-length speaker vector.

        Args:
            audio_paths (`list[str]`):
                Paths to the audio to embed. A repeated path is embedded once.

        Returns:
            `dict[str, torch.Tensor]`: The embedding of each distinct path.
        """
        self.load()
        sampling_rate = self._extractor.sampling_rate

        # Sorting by length keeps a batch's padding to what its own members need, which costs nothing
        # once the padded frames are masked out but keeps the batches cheap.
        unique_paths = list(dict.fromkeys(audio_paths))
        waveforms = {path: load_audio(path, sampling_rate) for path in unique_paths}
        ordered = sorted(unique_paths, key=lambda path: waveforms[path].shape[0])

        embeddings = {}
        for start in range(0, len(ordered), self.batch_size):
            batch = ordered[start : start + self.batch_size]
            features = self._extractor(
                [waveforms[path].numpy() for path in batch], sampling_rate=sampling_rate
            )
            with torch.no_grad():
                # The attention mask is what keeps the padding out of the pooled statistics, and so
                # keeps a clip's embedding independent of whatever else shared its batch.
                output = self._model(
                    input_features=features["input_features"].to(self.device, self.dtype),
                    attention_mask=features["attention_mask"].to(self.device),
                )
            vectors = torch.nn.functional.normalize(output.embeddings.flatten(1).float(), dim=1)
            embeddings.update(zip(batch, vectors.cpu()))
        return embeddings

    def score(self, artifacts: Sequence[Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Embeds each pair and reports the cosine between them.

        Args:
            artifacts (`Sequence[Any]`):
                Artifacts carrying `audio` and `reference_audio`, both paths.
            **kwargs:
                Unused.

        Returns:
            `list[dict[str, Any]]`: One mapping per artifact, holding the similarity, the pair it
            was measured over and the encoder that produced the embeddings.
        """
        generated = [artifact["audio"] for artifact in artifacts]
        reference = [artifact["reference_audio"] for artifact in artifacts]
        embeddings = self.embed(generated + reference)
        return [
            {
                "id": artifact.get("id"),
                "similarity": float(embeddings[target] @ embeddings[source]),
                "audio": source,
                "reference_audio": target,
                "encoder": self.model_id,
            }
            for artifact, source, target in zip(artifacts, generated, reference)
        ]

    def pool(self, statistics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Averages the similarities over every pair.

        Args:
            statistics (`Sequence[Mapping[str, Any]]`):
                Everything [`Ssim.score`] returned.

        Returns:
            `dict[str, float]`: The mean cosine under this metric's name, and the pairs behind it.
        """
        values = [float(statistic["similarity"]) for statistic in statistics]
        return {
            self.name: sum(values) / len(values) if values else float("nan"),
            "pairs": float(len(values)),
        }

__all__ = ["Ssim"]
