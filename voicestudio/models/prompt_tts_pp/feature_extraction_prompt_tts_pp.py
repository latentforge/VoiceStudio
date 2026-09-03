"""Feature extractor class for PromptTTS++."""

import numpy as np
import torch
import torchaudio

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


class PromptTTSPPFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a PromptTTS++ feature extractor. It turns a waveform into the log mel spectrogram the model reads a
    style embedding from and predicts, a magnitude spectrogram on centered frames mapped through a Slaney mel scale
    and standardized by the corpus level mean and standard deviation of the training set.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            Number of mel filterbank channels.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate, in Hz, the waveform is expected at.
        hop_length (`int`, *optional*, defaults to 240):
            Distance in waveform samples between neighbouring frames.
        win_length (`int`, *optional*, defaults to 480):
            Width in waveform samples of one analysis window.
        n_fft (`int`, *optional*, defaults to 512):
            Size of the Fourier transform.
        f_min (`float`, *optional*, defaults to 63.0):
            Lowest frequency, in Hz, of the mel filterbank.
        f_max (`float`, *optional*, defaults to 12000.0):
            Highest frequency, in Hz, of the mel filterbank.
        mel_floor (`float`, *optional*, defaults to 1e-05):
            Value the mel spectrogram is clamped to before the logarithm.
        mel_mean (`float`, *optional*, defaults to -6.708349227905273):
            Mean the log mel spectrogram is shifted by. Defaults to the statistic of the LibriTTS-R training set
            the released checkpoint was trained on.
        mel_std (`float`, *optional*, defaults to 2.529783010482788):
            Standard deviation the log mel spectrogram is scaled by. Defaults to the statistic of the LibriTTS-R
            training set the released checkpoint was trained on.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to standardize the log mel spectrogram by `mel_mean` and `mel_std`.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value the spectrograms of a batch are padded with.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the frame level mask that marks the unpadded part of each spectrogram.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 24000,
        hop_length: int = 240,
        win_length: int = 480,
        n_fft: int = 512,
        f_min: float = 63.0,
        f_max: float = 12000.0,
        mel_floor: float = 1e-5,
        mel_mean: float = -6.708349227905273,
        mel_std: float = 2.529783010482788,
        do_normalize: bool = True,
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
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.mel_floor = mel_floor
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.do_normalize = do_normalize
        self._filters = None
        self._window = None

    def _get_filters(self, device, dtype) -> torch.Tensor:
        if self._filters is None or self._filters.device != device or self._filters.dtype != dtype:
            self._filters = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.feature_size,
                sample_rate=self.sampling_rate,
                norm="slaney",
                mel_scale="slaney",
            ).to(device=device, dtype=dtype)
        return self._filters

    def _get_window(self, device, dtype) -> torch.Tensor:
        if self._window is None or self._window.device != device or self._window.dtype != dtype:
            self._window = torch.hann_window(self.win_length, device=device, dtype=dtype)
        return self._window

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

        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = spectrogram.abs()
        mel = torch.matmul(filters.transpose(-1, -2), magnitude)
        mel = torch.log(torch.clamp(mel, min=self.mel_floor))
        if self.do_normalize:
            mel = self.normalize(mel)
        return mel

    def normalize(self, spectrogram: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            spectrogram (`torch.Tensor`):
                Log mel spectrogram.

        Returns:
            `torch.Tensor`: The spectrogram standardized by `mel_mean` and `mel_std`.
        """
        return (spectrogram - self.mel_mean) / self.mel_std

    def denormalize(self, spectrogram: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            spectrogram (`torch.Tensor`):
                Standardized log mel spectrogram, as the model predicts it.

        Returns:
            `torch.Tensor`: The log mel spectrogram on the scale the vocoder expects.
        """
        return spectrogram * self.mel_std + self.mel_mean

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

        return_attention_mask = (
            self.return_attention_mask if return_attention_mask is None else return_attention_mask
        )
        if return_attention_mask and padding:
            positions = torch.arange(max_length)
            data["attention_mask"] = positions[None, :] < lengths[:, None]

        return BatchFeature(data=data)


__all__ = ["PromptTTSPPFeatureExtractor"]
