# MIT License
#
# Copyright (c) 2024 Yushen CHEN
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Feature extractor class for F5-TTS."""

import numpy as np
import torch
import torchaudio

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)

_MEL_FILTER_CACHE = {}
_WINDOW_CACHE = {}


class F5TTSFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an F5-TTS feature extractor. It turns a waveform into the log mel spectrogram the conditional flow is
    defined over, with either the `"vocos"` front end, a magnitude spectrogram on centered frames mapped through an
    HTK mel scale, or the `"bigvgan"` front end, a magnitude spectrogram on reflection padded uncentered frames
    mapped through a Slaney mel scale.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 100):
            Number of mel filterbank channels.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate, in Hz, the waveform is expected at.
        hop_length (`int`, *optional*, defaults to 256):
            Distance in waveform samples between neighbouring frames.
        win_length (`int`, *optional*, defaults to 1024):
            Width in waveform samples of one analysis window.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        mel_spec_type (`str`, *optional*, defaults to `"vocos"`):
            Which front end to use, `"vocos"` or `"bigvgan"`. It has to match the vocoder the checkpoint was
            trained against.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value the spectrograms of a batch are padded with.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the frame level mask that marks the unpadded part of each spectrogram.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 100,
        sampling_rate: int = 24000,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        mel_spec_type: str = "vocos",
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        if mel_spec_type not in ("vocos", "bigvgan"):
            raise ValueError(f"`mel_spec_type` must be one of 'vocos' or 'bigvgan', got {mel_spec_type}.")
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.mel_spec_type = mel_spec_type

    def _get_filters(self, device, dtype) -> torch.Tensor:
        norm, mel_scale = (None, "htk") if self.mel_spec_type == "vocos" else ("slaney", "slaney")
        key = (self.n_fft, self.feature_size, self.sampling_rate, norm, mel_scale, device, dtype)
        if key not in _MEL_FILTER_CACHE:
            _MEL_FILTER_CACHE[key] = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=0.0,
                f_max=self.sampling_rate / 2.0,
                n_mels=self.feature_size,
                sample_rate=self.sampling_rate,
                norm=norm,
                mel_scale=mel_scale,
            ).to(device=device, dtype=dtype)
        return _MEL_FILTER_CACHE[key]

    def _get_window(self, device, dtype) -> torch.Tensor:
        key = (self.win_length, device, dtype)
        if key not in _WINDOW_CACHE:
            _WINDOW_CACHE[key] = torch.hann_window(self.win_length, device=device, dtype=dtype)
        return _WINDOW_CACHE[key]

    def mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (`torch.Tensor`):
                Waveform of shape `(batch_size, num_samples)` at `sampling_rate`.

        Returns:
            `torch.Tensor`: Log mel spectrogram of shape `(batch_size, num_frames, feature_size)`.
        """
        window = self._get_window(waveform.device, waveform.dtype)
        filters = self._get_filters(waveform.device, waveform.dtype)

        if self.mel_spec_type == "bigvgan":
            padding = (self.n_fft - self.hop_length) // 2
            waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
            center = False
        else:
            center = True

        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        if self.mel_spec_type == "bigvgan":
            magnitude = torch.sqrt(torch.view_as_real(spectrogram).pow(2).sum(-1) + 1e-9)
        else:
            magnitude = spectrogram.abs()

        mel = torch.matmul(filters.transpose(-1, -2), magnitude)
        return torch.log(torch.clamp(mel, min=1e-5)).transpose(-1, -2)

    def __call__(
        self,
        raw_speech,
        sampling_rate: int | None = None,
        padding: bool = True,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        r"""
        Args:
            raw_speech (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Mono waveform, or batch of mono waveforms, at `sampling_rate`.
            sampling_rate (`int`, *optional*):
                Sampling rate of `raw_speech`. Passing it lets the extractor verify it matches its own.
            padding (`bool`, *optional*, defaults to `True`):
                Whether the spectrograms of a batch are padded to the longest one.
            return_attention_mask (`bool`, *optional*):
                Whether to return the frame level mask. Defaults to the value set on the extractor.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                Framework of the returned tensors. Only `"pt"` is supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with an `input_features` entry of shape
            `(batch_size, num_frames, feature_size)` and, when asked for, an `attention_mask` entry of shape
            `(batch_size, num_frames)`.

        Raises:
            ValueError: If `sampling_rate` does not match the extractor's, or if `return_tensors` is not `"pt"`.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{self.__class__.__name__} was instantiated for {self.sampling_rate} Hz audio but got a "
                f"{sampling_rate} Hz waveform. Resample it first."
            )
        if return_tensors is not None and return_tensors != TensorType.PYTORCH and return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only returns PyTorch tensors, got {return_tensors}.")

        if isinstance(raw_speech, (np.ndarray, torch.Tensor)) and raw_speech.ndim == 1:
            waveforms = [raw_speech]
        elif isinstance(raw_speech, (np.ndarray, torch.Tensor)):
            waveforms = list(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and not isinstance(
            raw_speech[0], (list, tuple, np.ndarray, torch.Tensor)
        ):
            waveforms = [raw_speech]
        else:
            waveforms = list(raw_speech)

        waveforms = [torch.as_tensor(waveform, dtype=torch.float32).reshape(-1) for waveform in waveforms]
        spectrograms = [self.mel_spectrogram(waveform.unsqueeze(0))[0] for waveform in waveforms]

        lengths = torch.tensor([spectrogram.shape[0] for spectrogram in spectrograms], dtype=torch.long)
        max_length = int(lengths.amax()) if padding else None

        if padding:
            spectrograms = [
                torch.nn.functional.pad(
                    spectrogram, (0, 0, 0, max_length - spectrogram.shape[0]), value=self.padding_value
                )
                for spectrogram in spectrograms
            ]
            input_features = torch.stack(spectrograms)
        else:
            input_features = spectrograms

        data = {"input_features": input_features}

        return_attention_mask = (
            self.return_attention_mask if return_attention_mask is None else return_attention_mask
        )
        if return_attention_mask and padding:
            positions = torch.arange(max_length)
            data["attention_mask"] = positions[None, :] < lengths[:, None]

        return BatchFeature(data=data)


__all__ = ["F5TTSFeatureExtractor"]
