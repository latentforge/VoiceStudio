"""Speaker similarity between a reference recording and a generated one."""

import datasets
import evaluate
import torch

from ..models.ecapa_tdnn import EcapaTdnnFeatureExtractor, EcapaTdnnForXVector
from .base import load_audio


_DESCRIPTION = """
Cosine similarity of the ECAPA-TDNN speaker embeddings of a reference recording and a generated one,
which is what a speaker verification system compares. The embedding pools statistics over the whole
utterance, so the padding that squares a batch off has to be masked out of that pooling; leaving it in
moves an embedding by as much as 0.13 cosine and makes a clip's score depend on what shared its batch.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list[str]`): Paths to the generated audio.
    references (`list[str]`): Paths to the reference audio, one per generation.

Returns:
    similarity (`float`): Mean cosine similarity across every pair.
    utterances (`list[float]`): Per-pair cosine similarity, in input order.
    pairs (`int`): Pairs compared.

Examples:
    >>> ssim = Ssim()
    >>> ssim.add_batch(predictions=["generated.wav"], references=["target.wav"])
    >>> ssim.compute()["similarity"]
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Ssim(evaluate.Metric):
    def __init__(
        self,
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
        super().__init__(**kwargs)
        self.model_id = model_id
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
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            codebase_urls=["https://github.com/speechbrain/speechbrain"],
        )

    def load_encoder(self) -> None:
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
        self.load_encoder()
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

    def _compute(self, predictions: list[str], references: list[str]) -> dict:
        embeddings = self.embed(list(predictions) + list(references))
        similarities = [
            float(embeddings[reference] @ embeddings[prediction])
            for prediction, reference in zip(predictions, references)
        ]
        return {
            "similarity": sum(similarities) / len(similarities) if similarities else float("nan"),
            "utterances": similarities,
            "pairs": len(similarities),
        }


__all__ = ["Ssim"]
