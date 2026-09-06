# coding=utf-8
# Copyright 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du) and the HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for CosyVoice v1."""

from typing import Optional, Union

import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature


class CosyVoiceV1FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CosyVoice v1 feature extractor, which turns a waveform into the log mel spectrogram
    the flow matching model is conditioned on and trained against.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            Number of mel bins.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Rate the incoming waveform is resampled to before the mel spectrogram is taken.
        mel_sampling_rate (`int`, *optional*, defaults to 22050):
            Rate the mel filter bank is built for.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 256):
            Hop between two consecutive frames.
        win_length (`int`, *optional*, defaults to 1024):
            Size of the analysis window.
        fmin (`float`, *optional*, defaults to 0.0):
            Lowest frequency of the mel filter bank.
        fmax (`float`, *optional*, defaults to 8000.0):
            Highest frequency of the mel filter bank.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to pad batches of spectrograms.
    """

    model_input_names = ["speech_feat"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 24000,
        mel_sampling_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs
        )
        self.mel_sampling_rate = mel_sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.mel_filters = librosa_mel(
            sr=mel_sampling_rate, n_fft=n_fft, n_mels=feature_size, fmin=fmin, fmax=fmax
        )

    def _mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (`torch.Tensor` of shape `(batch_size, num_samples)`):
                Waveform sampled at `sampling_rate`.

        Returns:
            `torch.Tensor` of shape `(batch_size, feature_size, num_frames)`: the log mel spectrogram.
        """
        padding = int((self.n_fft - self.hop_length) / 2)
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=waveform.device),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spectrogram = torch.view_as_real(spectrogram)
        spectrogram = torch.sqrt(spectrogram.pow(2).sum(-1) + 1e-9)
        mel_filters = torch.from_numpy(self.mel_filters).to(spectrogram)
        return torch.log(torch.clamp(torch.matmul(mel_filters, spectrogram), min=1e-5))

    def __call__(
        self,
        raw_speech: Union[np.ndarray, torch.Tensor, list],
        sampling_rate: Optional[int] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            raw_speech (`np.ndarray`, `torch.Tensor` or `list`):
                Mono waveform of shape `(num_samples,)` or `(1, num_samples)`.
            sampling_rate (`int`, *optional*):
                Rate of `raw_speech`. It is resampled to `self.sampling_rate` when the two differ.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Only `"pt"` is supported.

        Returns:
            [`BatchFeature`]: `speech_feat` of shape `(1, num_frames, feature_size)` and
            `speech_feat_lengths` of shape `(1,)`.

        Raises:
            ValueError: If `return_tensors` is not `"pt"`.
        """
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports return_tensors='pt'")
        waveform = raw_speech if isinstance(raw_speech, torch.Tensor) else torch.as_tensor(np.asarray(raw_speech))
        waveform = waveform.reshape(1, -1).float()
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sampling_rate, self.sampling_rate)
        speech_feat = self._mel_spectrogram(waveform).transpose(1, 2)
        lengths = torch.tensor([speech_feat.shape[1]], dtype=torch.int32)
        return BatchFeature({"speech_feat": speech_feat, "speech_feat_lengths": lengths})


__all__ = ["CosyVoiceV1FeatureExtractor"]
