# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# Copyright 2024 SpeechBrain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature extractor class for ECAPA-TDNN."""

import math

import numpy as np
import torch

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)

_MEL_FILTER_CACHE = {}
_WINDOW_CACHE = {}


class EcapaTdnnFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an ECAPA-TDNN feature extractor. It turns a waveform into the log filterbank features
    [`EcapaTdnnModel`] reads, then subtracts each utterance's own mean so that a recording's channel is not part
    of what the speaker embedding sees.

    The filterbank is SpeechBrain's rather than the usual one. Its triangles are symmetric in hertz, each half as
    wide as the gap to the next centre frequency, where a torchaudio or librosa filter spans the gap on either
    side of its centre and is therefore asymmetric. The two are not interchangeable.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            Number of filterbank channels.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate the waveforms are expected at, in hertz.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the Fourier transform.
        win_length_ms (`float`, *optional*, defaults to 25.0):
            Length of the analysis window, in milliseconds.
        hop_length_ms (`float`, *optional*, defaults to 10.0):
            Distance between neighbouring frames, in milliseconds.
        f_min (`float`, *optional*, defaults to 0.0):
            Lowest centre frequency of the filterbank, in hertz.
        f_max (`float`, *optional*):
            Highest centre frequency of the filterbank, in hertz. Defaults to half the sampling rate.
        amin (`float`, *optional*, defaults to 1e-10):
            Smallest value a filterbank channel is clipped to before its logarithm is taken.
        top_db (`float`, *optional*, defaults to 80.0):
            Decibel floor below the loudest bin of an utterance.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value the waveforms of a batch are padded with.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the frame level mask that marks the unpadded part of each utterance.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        win_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        f_min: float = 0.0,
        f_max: float | None = None,
        amin: float = 1e-10,
        top_db: float = 80.0,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.win_length_ms = win_length_ms
        self.hop_length_ms = hop_length_ms
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sampling_rate / 2.0
        self.amin = amin
        self.top_db = top_db

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Loads the extractor of an ECAPA-TDNN checkpoint, from a published SpeechBrain repository as it stands or
        from a directory [`~weight_conversion.convert`] wrote.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                `"speechbrain/spkrec-ecapa-voxceleb"`, or any repository id or directory holding one of the two
                layouts.
            kwargs (`dict`, *optional*):
                Keyword arguments of [`~FeatureExtractionMixin.from_pretrained`].

        Returns:
            [`EcapaTdnnFeatureExtractor`]: The extractor.
        """
        from .weight_conversion import build_feature_extractor, is_published_layout

        if pretrained_model_name_or_path is not None and is_published_layout(pretrained_model_name_or_path):
            return build_feature_extractor(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @property
    def win_length(self) -> int:
        r"""
        Returns:
            `int`: Length of the analysis window in waveform samples.
        """
        return int(round(self.sampling_rate / 1000.0 * self.win_length_ms))

    @property
    def hop_length(self) -> int:
        r"""
        Returns:
            `int`: Distance between neighbouring frames in waveform samples.
        """
        return int(round(self.sampling_rate / 1000.0 * self.hop_length_ms))

    @staticmethod
    def _to_mel(hertz: float) -> float:
        return 2595.0 * math.log10(1.0 + hertz / 700.0)

    @staticmethod
    def _to_hertz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _get_filters(self, device, dtype) -> torch.Tensor:
        key = (self.n_fft, self.feature_size, self.sampling_rate, self.f_min, self.f_max, device, dtype)
        if key not in _MEL_FILTER_CACHE:
            num_bins = self.n_fft // 2 + 1
            mel = torch.linspace(self._to_mel(self.f_min), self._to_mel(self.f_max), self.feature_size + 2)
            hertz = self._to_hertz(mel)
            centres = hertz[1:-1].unsqueeze(1)
            bands = (hertz[1:] - hertz[:-1])[:-1].unsqueeze(1)
            frequencies = torch.linspace(0, self.sampling_rate // 2, num_bins).repeat(self.feature_size, 1)
            slope = (frequencies - centres) / bands
            filters = torch.max(torch.zeros(1), torch.min(slope + 1.0, 1.0 - slope)).transpose(0, 1)
            _MEL_FILTER_CACHE[key] = filters.to(device=device, dtype=dtype)
        return _MEL_FILTER_CACHE[key]

    def _get_window(self, device, dtype) -> torch.Tensor:
        key = (self.win_length, device, dtype)
        if key not in _WINDOW_CACHE:
            _WINDOW_CACHE[key] = torch.hamming_window(self.win_length, device=device, dtype=dtype)
        return _WINDOW_CACHE[key]

    def filterbank(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (`torch.Tensor`):
                Waveform of shape `(batch_size, num_samples)` at `sampling_rate`.

        Returns:
            `torch.Tensor`: Log filterbank features of shape `(batch_size, num_frames, feature_size)`, in decibels
            and floored at `top_db` below each utterance's loudest bin.
        """
        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._get_window(waveform.device, waveform.dtype),
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrogram.abs().pow(2).transpose(1, 2)
        features = torch.matmul(power, self._get_filters(waveform.device, waveform.dtype))

        decibels = 10.0 * torch.log10(features.clamp(min=self.amin))
        decibels = decibels - 10.0 * math.log10(max(self.amin, 1.0))
        floor = decibels.amax(dim=(-2, -1)) - self.top_db
        return torch.max(decibels, floor.view(-1, 1, 1))

    def __call__(
        self,
        raw_speech,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        r"""
        Turns one or more waveforms into the features [`EcapaTdnnModel`] reads.

        Args:
            raw_speech (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]` or `list[list[float]]`):
                One waveform or a batch of them, mono and at `sampling_rate`.
            sampling_rate (`int`, *optional*):
                Sampling rate of `raw_speech`, to be checked against `sampling_rate`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                Framework of the returned tensors. Only `"pt"` is supported.

        Returns:
            [`~feature_extraction_utils.BatchFeature`]: A dictionary with `input_features` of shape
            `(batch_size, num_frames, feature_size)` and, where `return_attention_mask` is set, `attention_mask`
            of shape `(batch_size, num_frames)`.

        Raises:
            ValueError: If `sampling_rate` disagrees with the one this extractor was configured with, or if
                `return_tensors` is not `"pt"`.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"This feature extractor expects {self.sampling_rate} Hz audio, got {sampling_rate} Hz. Resample"
                " the waveform first."
            )
        if return_tensors not in ("pt", TensorType.PYTORCH):
            raise ValueError(f"`return_tensors` must be 'pt', got {return_tensors}.")

        if isinstance(raw_speech, torch.Tensor):
            raw_speech = raw_speech.cpu().numpy()
        if np.ndim(raw_speech[0]) == 0:
            waveforms = [torch.as_tensor(np.asarray(raw_speech), dtype=torch.float32)]
        else:
            waveforms = [
                torch.as_tensor(
                    waveform.cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform),
                    dtype=torch.float32,
                )
                for waveform in raw_speech
            ]

        num_samples = max(waveform.shape[0] for waveform in waveforms)
        padded = torch.full((len(waveforms), num_samples), float(self.padding_value))
        for index, waveform in enumerate(waveforms):
            padded[index, : waveform.shape[0]] = waveform

        features = self.filterbank(padded)
        num_frames = features.shape[1]
        lengths = torch.tensor(
            [min(waveform.shape[0] // self.hop_length + 1, num_frames) for waveform in waveforms]
        )
        mask = torch.arange(num_frames)[None, :] < lengths[:, None]

        # Each utterance is normalized on its own, over its unpadded frames alone. The mean comes off every
        # frame including the padded ones, which the convolutions read either way and which zeroing here would
        # give a value the network was never trained behind.
        counts = mask.sum(dim=1, keepdim=True).unsqueeze(2)
        mean = (features * mask.unsqueeze(2)).sum(dim=1, keepdim=True) / counts
        features = features - mean

        data = {"input_features": features}
        if self.return_attention_mask:
            data["attention_mask"] = mask.long()
        return BatchFeature(data)


__all__ = ["EcapaTdnnFeatureExtractor"]
