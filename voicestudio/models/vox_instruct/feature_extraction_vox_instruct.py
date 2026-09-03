"""Feature extractor class for VoxInstruct."""

import numpy as np
import torch
import torchaudio
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


class VoxInstructFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a VoxInstruct feature extractor. A speech prompt is read twice, at the sampling rate of the acoustic
    tokenizer and at the sampling rate of the semantic tokenizer, so this returns both views of the same clip.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            Number of audio channels.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate the acoustic tokenizer consumes.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to pad shorter clips.
        semantic_sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate the semantic tokenizer consumes.
        semantic_frame_multiple (`int`, *optional*, defaults to 320):
            The semantic view is truncated to a multiple of this many samples, matching the stride of the semantic
            tokenizer feature encoder.
    """

    model_input_names = ["input_values", "padding_mask", "semantic_input_values"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        semantic_sampling_rate: int = 16000,
        semantic_frame_multiple: int = 320,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.semantic_sampling_rate = semantic_sampling_rate
        self.semantic_frame_multiple = semantic_frame_multiple

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        """Averages a multi channel clip down to one channel."""
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        return waveform.reshape(-1, waveform.shape[-1]).mean(dim=0, keepdim=True)

    def __call__(
        self,
        raw_audio,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        r"""
        Args:
            raw_audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                One speech prompt, or a list of them. Mono or multi channel, in which case the channels are averaged.
            sampling_rate (`int`, *optional*):
                Sampling rate of `raw_audio`. Required whenever it differs from `self.sampling_rate`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                Only `"pt"` is supported.

        Returns:
            [`~feature_extraction_utils.BatchFeature`] with `input_values` of shape
            `(batch_size, 1, num_samples)` at `self.sampling_rate`, `padding_mask` of the same shape, and
            `semantic_input_values` of shape `(batch_size, num_semantic_samples)` at `self.semantic_sampling_rate`.

        Raises:
            ValueError: If `return_tensors` is not `"pt"`, or if more than one clip of differing length is passed.
        """
        if return_tensors not in ("pt", TensorType.PYTORCH):
            raise ValueError("VoxInstruct feature extraction only supports `return_tensors='pt'`.")

        if isinstance(raw_audio, (np.ndarray, torch.Tensor)) and raw_audio.ndim <= 2:
            clips = [raw_audio]
        else:
            clips = list(raw_audio)

        source_rate = sampling_rate if sampling_rate is not None else self.sampling_rate
        acoustic, semantic = [], []
        for clip in clips:
            waveform = clip if torch.is_tensor(clip) else torch.as_tensor(np.asarray(clip))
            waveform = self._to_mono(waveform.to(torch.float32))
            acoustic.append(torchaudio.functional.resample(waveform, source_rate, self.sampling_rate))
            resampled = torchaudio.functional.resample(waveform, source_rate, self.semantic_sampling_rate)
            length = resampled.shape[-1] // self.semantic_frame_multiple * self.semantic_frame_multiple
            semantic.append(resampled[..., :length])

        if len({clip.shape[-1] for clip in acoustic}) != 1:
            raise ValueError("VoxInstruct takes one speech prompt at a time, or several of the same length.")

        input_values = torch.stack(acoustic)
        return BatchFeature(
            data={
                "input_values": input_values,
                "padding_mask": torch.ones_like(input_values, dtype=torch.bool),
                "semantic_input_values": torch.stack(semantic).squeeze(1),
            }
        )


__all__ = ["VoxInstructFeatureExtractor"]
