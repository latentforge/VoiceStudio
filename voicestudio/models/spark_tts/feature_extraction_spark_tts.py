"""Feature extractor class for Spark-TTS."""

import copy
from typing import Any

import numpy as np

from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from transformers.utils import PaddingStrategy, TensorType, logging
from transformers.utils.import_utils import is_torch_available, is_torchaudio_available, requires


if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio


logger = logging.get_logger(__name__)


@requires(backends=("torchaudio",))
class SparkTTSFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Spark-TTS feature extractor. It prepares the two views of an audio clip that
    [`SparkTTSBiCodecModel`] consumes: the waveform the self-supervised semantic model runs on, and the mel
    spectrogram of a fixed-length reference excerpt the speaker encoder runs on.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            Feature dimension of `input_values`, which is a raw waveform.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sample rate, in hertz, the audio inputs are expected to be digitalized at.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value `input_values` is padded with.
        volume_normalize (`bool`, *optional*, defaults to `True`):
            Whether to rescale each clip so that its loud percentile sits at `volume_normalize_target` before any
            other processing.
        volume_normalize_target (`float`, *optional*, defaults to 0.2):
            Amplitude the mean of the 90th-to-99th percentile of the clip is normalized to.
        ref_segment_duration (`float`, *optional*, defaults to 6.0):
            Duration, in seconds, of the reference excerpt the mel spectrogram is computed from. Shorter clips are
            tiled and longer clips are truncated.
        hop_length (`int`, *optional*, defaults to 320):
            Hop size, in samples, of the mel spectrogram. Also the granularity the reference excerpt length is
            rounded down to.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform of the mel spectrogram.
        win_length (`int`, *optional*, defaults to 640):
            Window size, in samples, of the mel spectrogram.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel filters.
        mel_fmin (`float`, *optional*, defaults to 10):
            Lowest frequency, in hertz, covered by the mel filters.
        mel_fmax (`float`, *optional*):
            Highest frequency, in hertz, covered by the mel filters. Defaults to half the sampling rate.
    """

    model_input_names = ["input_values", "attention_mask", "reference_input_features"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        volume_normalize=True,
        volume_normalize_target=0.2,
        ref_segment_duration=6.0,
        hop_length=320,
        n_fft=1024,
        win_length=640,
        num_mel_bins=128,
        mel_fmin=10,
        mel_fmax=None,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.volume_normalize = volume_normalize
        self.volume_normalize_target = volume_normalize_target
        self.ref_segment_duration = ref_segment_duration
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.num_mel_bins = num_mel_bins
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        # The semantic model is a Wav2Vec2 checkpoint, so `input_values` must carry the exact per-clip zero-mean
        # unit-variance normalization and padding that Wav2Vec2 was trained with.
        self.semantic_feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=mel_fmin,
            f_max=mel_fmax,
            n_mels=num_mel_bins,
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    @property
    def ref_segment_length(self) -> int:
        """Length, in samples, of the reference excerpt, rounded down to a whole number of mel frames."""
        return int(self.sampling_rate * self.ref_segment_duration) // self.hop_length * self.hop_length

    def volume_normalized(self, audio: np.ndarray) -> np.ndarray:
        """
        Rescale a clip so that the mean amplitude of its 90th-to-99th percentile equals
        `volume_normalize_target`, leaving near-silent clips and clips with too few audible samples alone.

        Args:
            audio (`np.ndarray` of shape `(sequence_length,)`):
                Waveform with samples in `[-1, 1]`.

        Returns:
            `np.ndarray` of shape `(sequence_length,)`: The rescaled waveform.
        """
        sorted_magnitudes = np.sort(np.abs(audio))

        if sorted_magnitudes[-1] < 0.1:
            audio = audio / max(sorted_magnitudes[-1], 1e-3) * 0.1

        audible = sorted_magnitudes[sorted_magnitudes > 0.01]
        if audible.shape[0] <= 10:
            return audio

        volume = np.mean(audible[int(0.9 * audible.shape[0]) : int(0.99 * audible.shape[0])])
        audio = audio * np.clip(self.volume_normalize_target / volume, a_min=0.1, a_max=10)

        peak = np.max(np.abs(audio))
        if peak > 1:
            audio = audio / peak
        return audio

    def reference_clip(self, audio: np.ndarray) -> np.ndarray:
        """
        Cut a `ref_segment_length` excerpt out of a clip, tiling it first when it is too short.

        Args:
            audio (`np.ndarray` of shape `(sequence_length,)`):
                Waveform to excerpt.

        Returns:
            `np.ndarray` of shape `(ref_segment_length,)`: The excerpt.
        """
        target_length = self.ref_segment_length
        if target_length > audio.shape[0]:
            audio = np.tile(audio, target_length // audio.shape[0] + 1)
        return audio[:target_length]

    def __call__(
        self,
        audio: AudioInput,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | TensorType | None = "pt",
        sampling_rate: int | None = None,
        return_labels: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Mono waveform, or a list of them for a batch of inputs.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Padding strategy applied to `input_values`.
            max_length (`int`, *optional*):
                Maximum length `input_values` is padded or truncated to.
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to cut clips longer than `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                Framework of the returned tensors.
            sampling_rate (`int`, *optional*):
                Sample rate the `audio` input was digitalized at. Strongly recommended, so that a mismatch with
                `self.sampling_rate` raises instead of silently degrading the codes.
            return_labels (`bool`, *optional*, defaults to `False`):
                Whether to also return the padded waveform without the zero-mean unit-variance normalization
                `input_values` carries, which is the target [`SparkTTSBiCodecModel`] measures its reconstruction
                against.

        Returns:
            [`BatchFeature`]: With `input_values` and `attention_mask` for the semantic model, and
            `reference_input_features` for the speaker encoder, plus `labels` when `return_labels` is set.

        Raises:
            ValueError: If `sampling_rate` disagrees with `self.sampling_rate`, or if a clip is not mono.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using"
                    f" a sampling rate of {self.sampling_rate}. Please make sure that the provided `audio` input was"
                    f" sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        waveforms = []
        for example in make_list_of_audio(audio):
            example = np.asarray(example, dtype=np.float32)
            if example.ndim > 1:
                if example.shape[0] != 1:
                    raise ValueError(f"Expected mono audio of shape (1, length) or (length,) but got {example.shape}")
                example = example[0]
            if self.volume_normalize:
                example = self.volume_normalized(example)
            waveforms.append(example.astype(np.float32))

        semantic_inputs = self.semantic_feature_extractor(
            waveforms,
            sampling_rate=self.sampling_rate,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=True,
            return_tensors="pt",
        )

        reference_clips = torch.from_numpy(np.stack([self.reference_clip(wave) for wave in waveforms]))
        reference_input_features = self.mel_transform(reference_clips).transpose(1, 2)

        data = {
            "input_values": semantic_inputs["input_values"],
            "attention_mask": semantic_inputs["attention_mask"],
            "reference_input_features": reference_input_features,
        }

        if return_labels:
            targets = self.pad(
                BatchFeature({"input_values": waveforms}),
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                return_attention_mask=False,
                return_tensors="pt",
            )
            data["labels"] = targets["input_values"]

        return BatchFeature(data, tensor_type=return_tensors)

    def to_dict(self) -> dict[str, Any]:
        """
        Returns:
            `dict[str, Any]`: The serializable attributes of this feature extractor, without the sub-extractor and
            the mel transform, which [`~SparkTTSFeatureExtractor.__init__`] rebuilds from the other fields.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        output.pop("semantic_feature_extractor", None)
        output.pop("mel_transform", None)
        return output


__all__ = ["SparkTTSFeatureExtractor"]
