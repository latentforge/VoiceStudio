"""Feature extractor class for Vocos."""

import numpy as np
import torch
import torchaudio

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)

_MEL_FILTER_CACHE = {}
_WINDOW_CACHE = {}


class VocosFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Vocos feature extractor. It turns a waveform into the log mel spectrogram the `"mel"` front end of
    [`VocosModel`] is trained to invert, a magnitude spectrogram mapped through an unnormalized HTK mel scale and
    clipped before its logarithm is taken.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 100):
            Number of mel filterbank channels.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate, in Hz, the waveform is expected at.
        hop_length (`int`, *optional*, defaults to 256):
            Distance in waveform samples between neighbouring frames.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform. It also sets the width of one analysis window.
        padding (`str`, *optional*, defaults to `"center"`):
            How the frames are laid over the waveform, `"center"` for the centered frames of
            `torch.stft(center=True)` or `"same"` for reflection padding the waveform by
            `(n_fft - hop_length) // 2` samples on each side and framing it uncentered.
        clip_value (`float`, *optional*, defaults to 1e-07):
            Smallest value the mel spectrogram is clipped to before its logarithm is taken.
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
        n_fft: int = 1024,
        padding: str = "center",
        clip_value: float = 1e-7,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        if padding not in ("center", "same"):
            raise ValueError(f"`padding` must be one of 'center' or 'same', got {padding}.")
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.padding = padding
        self.clip_value = clip_value

    def _get_filters(self, device, dtype) -> torch.Tensor:
        key = (self.n_fft, self.feature_size, self.sampling_rate, device, dtype)
        if key not in _MEL_FILTER_CACHE:
            _MEL_FILTER_CACHE[key] = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=0.0,
                f_max=self.sampling_rate / 2.0,
                n_mels=self.feature_size,
                sample_rate=self.sampling_rate,
                norm=None,
                mel_scale="htk",
            ).to(device=device, dtype=dtype)
        return _MEL_FILTER_CACHE[key]

    def _get_window(self, device, dtype) -> torch.Tensor:
        key = (self.n_fft, device, dtype)
        if key not in _WINDOW_CACHE:
            _WINDOW_CACHE[key] = torch.hann_window(self.n_fft, device=device, dtype=dtype)
        return _WINDOW_CACHE[key]

    def mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (`torch.Tensor`):
                Waveform of shape `(batch_size, num_samples)` at `sampling_rate`.

        Returns:
            `torch.Tensor`: Log mel spectrogram of shape `(batch_size, feature_size, num_frames)`.
        """
        window = self._get_window(waveform.device, waveform.dtype)
        filters = self._get_filters(waveform.device, waveform.dtype)

        if self.padding == "same":
            pad = (self.n_fft - self.hop_length) // 2
            waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=self.padding == "center",
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        mel = torch.matmul(filters.transpose(-1, -2), spectrogram.abs())
        return torch.log(torch.clip(mel, min=self.clip_value))

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
            `(batch_size, feature_size, num_frames)` and, when asked for, an `attention_mask` entry of shape
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

        lengths = torch.tensor([spectrogram.shape[-1] for spectrogram in spectrograms], dtype=torch.long)
        max_length = int(lengths.amax()) if padding else None

        if padding:
            spectrograms = [
                torch.nn.functional.pad(
                    spectrogram, (0, max_length - spectrogram.shape[-1]), value=self.padding_value
                )
                for spectrogram in spectrograms
            ]
            input_features = torch.stack(spectrograms)
        else:
            input_features = spectrograms

        data = {"input_features": input_features}

        return_attention_mask = self.return_attention_mask if return_attention_mask is None else return_attention_mask
        if return_attention_mask and padding:
            positions = torch.arange(max_length)
            data["attention_mask"] = positions[None, :] < lengths[:, None]

        return BatchFeature(data=data)


__all__ = ["VocosFeatureExtractor"]
