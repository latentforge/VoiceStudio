# MIT License
#
# Copyright (c) 2024 sarulab-speech
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
"""Feature extractor class for UTMOSv2."""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)

_MEL_FILTER_CACHE = {}
_WINDOW_CACHE = {}

# The listening-test corpora the published checkpoint was trained on, in the order its domain vector indexes them.
DOMAINS = (
    "bvcc",
    "sarulab",
    "blizzard2008",
    "blizzard2009",
    "blizzard2010-EH1",
    "blizzard2010-EH2",
    "blizzard2010-ES1",
    "blizzard2010-ES3",
    "blizzard2011",
    "somos",
)


class UTMOSv2FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a UTMOSv2 feature extractor. It turns a waveform into the two views [`UTMOSv2Model`] reads: a fixed
    length excerpt for the self-supervised branch, and a stack of mel spectrograms of shorter excerpts rendered as
    square images for the spectrogram branch.

    Both views are drawn at random positions, and each spectrogram is mixed with a second draw, so repeated calls
    on one waveform give different features. That is what the model was trained and evaluated under, and averaging
    a handful of calls is how upstream reports a score.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 512):
            Number of mel filterbank channels.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate the waveforms are expected at, in hertz.
        n_fft (`int`, *optional*, defaults to 4096):
            Size of the Fourier transform behind every spectrogram.
        hop_length (`int`, *optional*, defaults to 32):
            Distance in waveform samples between neighbouring spectrogram frames.
        win_lengths (`list[int]`, *optional*, defaults to `[4096, 2048, 1024, 512]`):
            Window length of each of the spectrogram resolutions, one per encoder of the spectrogram branch.
        image_size (`int`, *optional*, defaults to 512):
            Side of the square each spectrogram is resampled to.
        top_db (`float`, *optional*, defaults to 80.0):
            Decibel floor below the loudest bin. It also normalizes the result, which is divided by it.
        frame_seconds (`float`, *optional*, defaults to 1.4):
            Duration of each excerpt the spectrogram branch reads.
        num_frames (`int`, *optional*, defaults to 2):
            Number of such excerpts.
        mixup_alpha (`float`, *optional*, defaults to 0.4):
            Both parameters of the beta distribution the mixing ratio of the two draws behind one spectrogram is
            sampled from.
        ssl_seconds (`float`, *optional*, defaults to 3.0):
            Duration of the excerpt the self-supervised branch reads.
        remove_silence (`bool`, *optional*, defaults to `True`):
            Whether to drop the quiet stretches of a waveform before excerpting it.
        silence_threshold (`float`, *optional*, defaults to 0.1):
            Amplitude below which a sample counts as quiet.
        min_silence_samples (`int`, *optional*, defaults to 4800):
            Shortest run of quiet samples that is dropped. Shorter runs are kept.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value the waveforms of a batch are padded with.
    """

    model_input_names = ["input_values", "input_features", "domain_ids"]

    def __init__(
        self,
        feature_size: int = 512,
        sampling_rate: int = 16000,
        n_fft: int = 4096,
        hop_length: int = 32,
        win_lengths: list[int] | None = None,
        image_size: int = 512,
        top_db: float = 80.0,
        frame_seconds: float = 1.4,
        num_frames: int = 2,
        mixup_alpha: float = 0.4,
        ssl_seconds: float = 3.0,
        remove_silence: bool = True,
        silence_threshold: float = 0.1,
        min_silence_samples: int = 4800,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_lengths = list(win_lengths) if win_lengths is not None else [4096, 2048, 1024, 512]
        self.image_size = image_size
        self.top_db = top_db
        self.frame_seconds = frame_seconds
        self.num_frames = num_frames
        self.mixup_alpha = mixup_alpha
        self.ssl_seconds = ssl_seconds
        self.remove_silence = remove_silence
        self.silence_threshold = silence_threshold
        self.min_silence_samples = min_silence_samples

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Loads the extractor of a UTMOSv2 checkpoint, from the published repository as it stands or from a
        directory [`~weight_conversion.convert`] wrote.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                `"sarulab-speech/UTMOSv2"`, or any repository id or directory holding one of the two layouts.
            kwargs (`dict`, *optional*):
                Keyword arguments of [`~FeatureExtractionMixin.from_pretrained`].

        Returns:
            [`UTMOSv2FeatureExtractor`]: The extractor.
        """
        from .weight_conversion import is_published_layout

        if pretrained_model_name_or_path is not None and is_published_layout(pretrained_model_name_or_path):
            return cls()
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def _get_filters(self, device, dtype) -> torch.Tensor:
        key = (self.n_fft, self.feature_size, self.sampling_rate, device, dtype)
        if key not in _MEL_FILTER_CACHE:
            _MEL_FILTER_CACHE[key] = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=0.0,
                f_max=self.sampling_rate / 2.0,
                n_mels=self.feature_size,
                sample_rate=self.sampling_rate,
                norm="slaney",
                mel_scale="slaney",
            ).to(device=device, dtype=dtype)
        return _MEL_FILTER_CACHE[key]

    def _get_window(self, win_length: int, device, dtype) -> torch.Tensor:
        key = (win_length, device, dtype)
        if key not in _WINDOW_CACHE:
            _WINDOW_CACHE[key] = torch.hann_window(win_length, device=device, dtype=dtype)
        return _WINDOW_CACHE[key]

    def remove_silent_sections(self, waveform: np.ndarray) -> np.ndarray:
        r"""
        Drops every run of at least `min_silence_samples` samples that stays below `silence_threshold`.

        Args:
            waveform (`np.ndarray`):
                Waveform of shape `(num_samples,)`.

        Returns:
            `np.ndarray`: The waveform with its quiet stretches removed.
        """
        quiet = waveform < self.silence_threshold
        edges = np.pad(quiet, (1, 0)) ^ np.pad(quiet, (0, 1))
        indices = np.where(edges)[0]
        runs = indices[1::2] - indices[::2]
        indices = indices[np.repeat(runs > self.min_silence_samples, 2)]
        marks = np.zeros(waveform.shape[0] + 1, dtype=int)
        marks[indices] = np.where(np.arange(indices.shape[0]) % 2, -1, 1)
        return waveform[~np.cumsum(marks).astype(bool)[:-1]]

    def mel_spectrogram(self, waveform: torch.Tensor, win_length: int) -> torch.Tensor:
        r"""
        Args:
            waveform (`torch.Tensor`):
                Waveform of shape `(batch_size, num_samples)` at `sampling_rate`.
            win_length (`int`):
                Window length of this resolution. Windows shorter than `n_fft` are centred in a zero padded frame.

        Returns:
            `torch.Tensor`: Mel spectrogram of shape `(batch_size, feature_size, num_frames)`, in decibels below
            the loudest bin, floored at `-top_db` and divided by `top_db` to land in `[0, 1]`.
        """
        window = self._get_window(win_length, waveform.device, waveform.dtype)
        filters = self._get_filters(waveform.device, waveform.dtype)

        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        mel = torch.matmul(filters.transpose(-1, -2), spectrogram.abs().pow(2))
        decibels = 10.0 * torch.log10(mel.clamp(min=1e-10))
        decibels = decibels - decibels.amax(dim=(-2, -1), keepdim=True)
        return decibels.clamp(min=-self.top_db) / self.top_db + 1.0

    def _excerpt(self, waveform: np.ndarray, length: int, generator: np.random.Generator) -> np.ndarray:
        if waveform.shape[0] <= length:
            waveform = np.tile(waveform, length // waveform.shape[0] + 1)
        start = generator.integers(0, waveform.shape[0] - length)
        return waveform[start : start + length]

    def _spectrogram_stack(self, waveform: np.ndarray, generator: np.random.Generator) -> torch.Tensor:
        length = int(self.frame_seconds * self.sampling_rate)
        images = []
        for _ in range(self.num_frames):
            excerpt = torch.from_numpy(self._excerpt(waveform, length, generator)).float()
            for win_length in self.win_lengths:
                partner = torch.from_numpy(self._excerpt(waveform, length, generator)).float()
                ratio = generator.beta(self.mixup_alpha, self.mixup_alpha)
                spectrogram = ratio * self.mel_spectrogram(excerpt, win_length) + (
                    1.0 - ratio
                ) * self.mel_spectrogram(partner, win_length)
                image = spectrogram.expand(3, -1, -1).unsqueeze(0)
                image = F.interpolate(
                    image,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                images.append(image.squeeze(0))
        return torch.stack(images)

    def __call__(
        self,
        raw_speech,
        sampling_rate: int | None = None,
        domain: str | int | list[str | int] = "sarulab",
        generator: np.random.Generator | None = None,
        return_tensors: str | TensorType | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        r"""
        Turns one or more waveforms into the features [`UTMOSv2Model`] reads.

        Args:
            raw_speech (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]` or `list[list[float]]`):
                One waveform or a batch of them, mono and at `sampling_rate`.
            sampling_rate (`int`, *optional*):
                Sampling rate of `raw_speech`, to be checked against `sampling_rate`.
            domain (`str`, `int` or `list`, *optional*, defaults to `"sarulab"`):
                Listening-test corpus each prediction should imitate, a name in `DOMAINS` or its index. Pass one
                value for the whole batch or one per waveform.
            generator (`np.random.Generator`, *optional*):
                Source of the excerpt positions and the mixing ratios. Pass a seeded one to make a call
                reproducible.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                Framework of the returned tensors. Only `"pt"` is supported.

        Returns:
            [`~feature_extraction_utils.BatchFeature`]: A dictionary with `input_values` of shape
            `(batch_size, ssl_seconds * sampling_rate)`, `input_features` of shape
            `(batch_size, num_frames * len(win_lengths), 3, image_size, image_size)`, and `domain_ids` of shape
            `(batch_size,)`.

        Raises:
            ValueError: If `sampling_rate` disagrees with the one this extractor was configured with, if
                `return_tensors` is not `"pt"`, or if a domain is not one of `DOMAINS`.
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
        waveforms = [np.asarray(raw_speech, dtype=np.float64)] if np.ndim(raw_speech[0]) == 0 else [
            np.asarray(waveform.cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform, dtype=np.float64)
            for waveform in raw_speech
        ]
        domains = domain if isinstance(domain, list) else [domain] * len(waveforms)
        if len(domains) != len(waveforms):
            raise ValueError(f"Got {len(domains)} domains for {len(waveforms)} waveforms.")
        domain_ids = []
        for name in domains:
            if isinstance(name, str):
                if name not in DOMAINS:
                    raise ValueError(f"Unknown domain {name!r}. Must be one of {list(DOMAINS)}.")
                name = DOMAINS.index(name)
            domain_ids.append(name)

        generator = generator if generator is not None else np.random.default_rng()
        ssl_length = int(self.ssl_seconds * self.sampling_rate)
        excerpts, stacks = [], []
        for waveform in waveforms:
            if self.remove_silence:
                waveform = self.remove_silent_sections(waveform)
            excerpt = torch.from_numpy(self._excerpt(waveform, ssl_length, generator)).float()
            excerpts.append((excerpt - excerpt.mean()) / torch.sqrt(excerpt.var(correction=0) + 1e-7))
            stacks.append(self._spectrogram_stack(waveform, generator))

        return BatchFeature(
            {
                "input_values": torch.stack(excerpts),
                "input_features": torch.stack(stacks),
                "domain_ids": torch.tensor(domain_ids, dtype=torch.long),
            }
        )


__all__ = ["UTMOSv2FeatureExtractor"]
