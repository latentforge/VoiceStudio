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
"""Feature extractor class for CosyVoice v2."""

from ..cosyvoice_v1.feature_extraction_cosyvoice_v1 import CosyVoiceV1FeatureExtractor


class CosyVoiceV2FeatureExtractor(CosyVoiceV1FeatureExtractor):
    r"""
    Constructs a CosyVoice v2 feature extractor, which turns a waveform into the 24 kHz log mel
    spectrogram the flow matching model is conditioned on and trained against.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            Number of mel bins.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Rate the incoming waveform is resampled to before the mel spectrogram is taken.
        mel_sampling_rate (`int`, *optional*, defaults to 24000):
            Rate the mel filter bank is built for.
        n_fft (`int`, *optional*, defaults to 1920):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 480):
            Hop between two consecutive frames.
        win_length (`int`, *optional*, defaults to 1920):
            Size of the analysis window.
        fmin (`float`, *optional*, defaults to 0.0):
            Lowest frequency of the mel filter bank.
        fmax (`float`, *optional*, defaults to 8000.0):
            Highest frequency of the mel filter bank.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to pad batches of spectrograms.
    """

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 24000,
        mel_sampling_rate: int = 24000,
        n_fft: int = 1920,
        hop_length: int = 480,
        win_length: int = 1920,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            mel_sampling_rate=mel_sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            padding_value=padding_value,
            **kwargs,
        )


__all__ = ["CosyVoiceV2FeatureExtractor"]
