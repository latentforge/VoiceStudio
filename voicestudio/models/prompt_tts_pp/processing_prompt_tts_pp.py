"""Processor class for PromptTTS++."""

import numpy as np
import torch
import torchaudio

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging
from transformers.utils.import_utils import requires


logger = logging.get_logger(__name__)


def butterworth_lowpass(order: int, normalized_cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Designs a digital Butterworth lowpass filter by bilinear transform of the analog prototype.

    Args:
        order (`int`):
            Order of the filter.
        normalized_cutoff (`float`):
            Cutoff frequency, as a fraction of the Nyquist frequency.

    Returns:
        `tuple[np.ndarray, np.ndarray]`: The numerator and denominator coefficients, highest power first.
    """
    # The prototype is designed against a sampling rate of two, so the Nyquist frequency is one.
    warped = 4 * np.tan(np.pi * normalized_cutoff / 2)
    poles = -np.exp(1j * np.pi * np.arange(-order + 1, order, 2) / (2 * order)) * warped
    gain = warped**order

    discrete_poles = (4.0 + poles) / (4.0 - poles)
    discrete_gain = gain * np.real(1.0 / np.prod(4.0 - poles))

    numerator = discrete_gain * np.real(np.poly(-np.ones(order)))
    denominator = np.real(np.poly(discrete_poles))
    return numerator, denominator


def lowpass_filter(waveform: torch.Tensor, sampling_rate: int, cutoff: int, order: int = 5) -> torch.Tensor:
    """
    Zero phase lowpass filtering with a Butterworth filter.

    Args:
        waveform (`torch.Tensor`):
            Signal to filter along its last dimension.
        sampling_rate (`int`):
            Sampling rate of `waveform`, in Hz.
        cutoff (`int`):
            Cutoff frequency, in Hz.
        order (`int`, *optional*, defaults to 5):
            Order of the filter.

    Returns:
        `torch.Tensor`: The filtered signal, unchanged when it is too short for the filter.
    """
    numerator, denominator = butterworth_lowpass(order, cutoff / (sampling_rate // 2))
    if waveform.shape[-1] <= max(len(numerator), len(denominator)) * (order // 2 + 1):
        return waveform

    numerator = torch.as_tensor(numerator, dtype=torch.float32, device=waveform.device)
    denominator = torch.as_tensor(denominator, dtype=torch.float32, device=waveform.device)
    return torchaudio.functional.filtfilt(waveform, denominator, numerator, clamp=False)


@requires(backends=("torch",))
class PromptTTSPPProcessor(ProcessorMixin):
    r"""
    Constructs a PromptTTS++ processor which wraps a [`PromptTTSPPFeatureExtractor`], a
    [`PromptTTSPPTokenizer`] and the tokenizer of the prompt encoder into a single processor.

    PromptTTS++ reads its content from a phoneme sequence and its speaker from either a natural language style
    prompt or a reference waveform, so the processor tokenizes the two texts with two different tokenizers and
    turns the reference waveform into the normalized mel spectrogram the style encoder expects.

    Args:
        feature_extractor ([`PromptTTSPPFeatureExtractor`], *optional*):
            Feature extractor turning a reference waveform into a normalized log mel spectrogram.
        tokenizer ([`PromptTTSPPTokenizer`], *optional*):
            Phoneme tokenizer of the model.
        prompt_tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            Tokenizer of the BERT prompt encoder.
        chat_template (`str`, *optional*):
            Template string used by [`~ProcessorMixin.apply_chat_template`].
    """

    def __init__(self, feature_extractor=None, tokenizer=None, prompt_tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, prompt_tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text=None,
        style_prompt=None,
        audio=None,
        sampling_rate: int | None = None,
        padding: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        r"""
        Args:
            text (`str` or `list[str]`, *optional*):
                Text to speak, phonemized by the phoneme tokenizer.
            style_prompt (`str` or `list[str]`, *optional*):
                Natural language description of the speaker and the speaking style.
            audio (`np.ndarray`, `torch.Tensor` or `list`, *optional*):
                Reference waveform whose speaker the style encoder reads, as an alternative to `style_prompt`.
            sampling_rate (`int`, *optional*):
                Sampling rate of `audio`.
            padding (`bool`, *optional*, defaults to `True`):
                Whether sequences of a batch are padded to the longest one.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Framework of the returned tensors. Only `"pt"` is supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] holding `input_ids` and `attention_mask` for the phonemes,
            `prompt_input_ids` and `prompt_attention_mask` for the style prompt, and `reference_spectrogram`
            and `reference_spectrogram_lengths` for the reference waveform, whichever of the three were given.

        Raises:
            ValueError: If neither `style_prompt` nor `audio` is given, or if `return_tensors` is not `"pt"`.
        """
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only returns PyTorch tensors, got {return_tensors}.")
        if style_prompt is None and audio is None:
            raise ValueError("One of `style_prompt` or `audio` must be given to condition the speaker on.")

        data = {}
        if text is not None:
            encoding = self.tokenizer(text, padding=padding, return_tensors=return_tensors, **kwargs)
            data["input_ids"] = encoding["input_ids"]
            data["attention_mask"] = encoding["attention_mask"]

        if style_prompt is not None:
            prompt_encoding = self.prompt_tokenizer(style_prompt, padding=padding, return_tensors=return_tensors)
            data["prompt_input_ids"] = prompt_encoding["input_ids"]
            data["prompt_attention_mask"] = prompt_encoding["attention_mask"]

        if audio is not None:
            features = self.feature_extractor(
                audio, sampling_rate=sampling_rate, padding=padding, return_tensors=return_tensors
            )
            data["reference_spectrogram"] = features["input_features"]
            if "attention_mask" in features:
                data["reference_spectrogram_lengths"] = features["attention_mask"].sum(dim=-1)

        return BatchFeature(data=data)

    def postprocess(self, outputs, frame_rate: int = 100, cutoff: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Turns the model's output into the two inputs of the f0 aware vocoder.

        Args:
            outputs ([`PromptTTSPPOutput`]):
                Output of [`PromptTTSPPForConditionalGeneration`], holding the generated spectrogram, the
                predicted log continuous f0 and the predicted voicing.
            frame_rate (`int`, *optional*, defaults to 100):
                Number of spectrogram frames per second, which the f0 contour is filtered at.
            cutoff (`int`, *optional*, defaults to 20):
                Cutoff frequency, in Hz, of the lowpass filter smoothing the f0 contour.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: The spectrogram on the scale the vocoder was trained on, of
            shape `(batch_size, num_mel_bins, num_frames)`, and the fundamental frequency in Hz of each frame,
            of shape `(batch_size, 1, num_frames)`, zero on the unvoiced frames.
        """
        log_f0 = lowpass_filter(outputs.log_f0, frame_rate, cutoff)
        f0 = log_f0.exp()
        f0 = torch.where(outputs.vuv < 0.5, torch.zeros_like(f0), f0)
        spectrogram = self.feature_extractor.denormalize(outputs.spectrogram).transpose(1, 2)
        return spectrogram, f0


__all__ = ["PromptTTSPPProcessor"]
