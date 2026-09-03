"""Processor class for CosyVoice v2."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.models.auto.tokenization_auto import AutoTokenizer

from ..cosyvoice_v1.processing_cosyvoice_v1 import CosyVoiceV1FeatureExtractor, CosyVoiceV1Processor
from ..cosyvoice_v1.weight_conversion import resolve_checkpoint
from .configuration_cosyvoice_v2 import CosyVoiceV2Config
from .modeling_cosyvoice_v2 import CosyVoiceV2SpeechTokenizer
from .weight_conversion import SPEECH_TOKENIZER_FILE, TEXT_MODEL_SUBDIR


# The tokens upstream's `CosyVoice2Tokenizer` adds to the Qwen2 tokenizer, in the order it adds them.
SPECIAL_TOKENS = [
    "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
    "[breath]", "<strong>", "</strong>", "[noise]",
    "[laughter]", "[cough]", "[clucking]", "[accent]",
    "[quick_breath]",
    "<laughter>", "</laughter>",
    "[hissing]", "[sigh]", "[vocalized-noise]",
    "[lipsmack]", "[mn]",
]


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


class CosyVoiceV2Processor(CosyVoiceV1Processor):
    r"""
    Constructs a CosyVoice v2 processor, which wraps the Qwen2 text tokenizer, the 24 kHz mel
    spectrogram extractor of the flow matching model, the supervised semantic speech tokenizer and the
    speaker encoder into a single object.

    Its speech tokenizer is [`CosyVoiceV2SpeechTokenizer`], whose weights are read out of the
    `speech_tokenizer_v2.onnx` graph the released directory ships, and whose two strided convolutions
    put it at half v1's token rate.

    Args:
        feature_extractor ([`CosyVoiceV2FeatureExtractor`]):
            Mel spectrogram extractor of the flow matching model.
        tokenizer ([`Qwen2TokenizerFast`]):
            Text tokenizer, loaded from the `CosyVoice-BlankEN` directory of the released checkpoint.
        speech_token_model_path (`str`, *optional*):
            Path of the `speech_tokenizer_v2.onnx` graph the speech tokenizer is built from.
        speaker_encoder_model_path (`str`, *optional*):
            Path of the CAM++ weights the speaker encoder is built from.
        speaker_info_path (`str`, *optional*):
            Path of a `spk2info.pt`. The released v2 directory ships none.
        speech_token_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel bins of the log mel spectrogram the speech tokenizer consumes.
        token_mel_ratio (`int`, *optional*, defaults to 2):
            Number of mel frames per speech token. A prompt is truncated so that its mel spectrogram
            and its speech tokens keep exactly this ratio.
    """

    feature_extractor_type = CosyVoiceV2FeatureExtractor
    speech_tokenizer_type = CosyVoiceV2SpeechTokenizer
    model_config_type = CosyVoiceV2Config
    speech_tokenizer_file = SPEECH_TOKENIZER_FILE

    @classmethod
    def _released_processor(cls, directory: "str | Path") -> "CosyVoiceV2Processor":
        r"""
        Builds the processor of a released CosyVoice v2 directory.

        The directory carries the speech tokenizer, the speaker encoder and the Qwen2 directory the
        text tokenizer is read from, to which upstream's own special tokens are added.

        Args:
            directory (`str` or `os.PathLike`):
                Local directory of the released checkpoint.

        Returns:
            [`CosyVoiceV2Processor`]: The processor.
        """
        directory = Path(directory)
        tokenizer = AutoTokenizer.from_pretrained(str(directory / TEXT_MODEL_SUBDIR))
        cls.add_special_tokens(tokenizer)
        return cls(
            feature_extractor=cls.feature_extractor_type(),
            tokenizer=tokenizer,
            speech_token_model_path=str(directory / cls.speech_tokenizer_file),
        )

    @classmethod
    def _resolve_released_checkpoint(cls, source, **kwargs) -> "Path | None":
        r"""
        Fetches the files a released CosyVoice v2 directory holds for the processor.

        Args:
            source (`str` or `os.PathLike`, *optional*):
                Repository id or local directory.
            kwargs (`dict`, *optional*):
                Fields of `weight_conversion.DOWNLOAD_KWARGS` selecting a revision and a cache.

        Returns:
            `Path` or `None`: The local directory, or `None` when `source` holds no released
            checkpoint.
        """
        return resolve_checkpoint(source, (cls.speech_tokenizer_file,), (f"{TEXT_MODEL_SUBDIR}/*",), **kwargs)

    @staticmethod
    def add_special_tokens(tokenizer, tokens: Optional[list[str]] = None) -> int:
        r"""
        Adds upstream's v2 special tokens to a tokenizer, in upstream's order.

        Args:
            tokenizer (`PreTrainedTokenizerBase`):
                Tokenizer to extend.
            tokens (`list[str]`, *optional*):
                Tokens to add. Defaults to [`SPECIAL_TOKENS`].

        Returns:
            `int`: The number of tokens the tokenizer did not already carry.
        """
        return tokenizer.add_special_tokens(
            {
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": SPECIAL_TOKENS if tokens is None else tokens,
            }
        )

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        speech_token_model_path: Optional[str] = None,
        speaker_encoder_model_path: Optional[str] = None,
        speaker_info_path: Optional[str] = None,
        speech_token_mel_bins: int = 128,
        token_mel_ratio: int = 2,
        **kwargs,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            speech_token_model_path=speech_token_model_path,
            speaker_encoder_model_path=speaker_encoder_model_path,
            speaker_info_path=speaker_info_path,
            speech_token_mel_bins=speech_token_mel_bins,
            **kwargs,
        )
        self.token_mel_ratio = token_mel_ratio

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sampling_rate: Optional[int] = None,
        prompt_text: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            text (`str` or `list[str]`, *optional*):
                Text to synthesize.
            audio (`np.ndarray` or `torch.Tensor`, *optional*):
                Mono waveform of the prompt utterance.
            sampling_rate (`int`, *optional*):
                Rate of `audio`.
            prompt_text (`str` or `list[str]`, *optional*):
                Transcript of the prompt utterance.

        Returns:
            [`BatchFeature`]: `input_ids` and `input_lengths` for the text, plus the prompt fields
            when `audio` is given, with the mel spectrogram and the speech tokens truncated to a whole
            number of `token_mel_ratio` sized groups.
        """
        data = super().__call__(
            text=text, audio=audio, sampling_rate=sampling_rate, prompt_text=prompt_text, **kwargs
        )
        if audio is None:
            return data
        token_length = min(
            int(data["speech_feat"].shape[1] // self.token_mel_ratio),
            int(data["prompt_speech_token_ids"].shape[1]),
        )
        data["speech_feat"] = data["speech_feat"][:, : self.token_mel_ratio * token_length]
        data["speech_feat_lengths"] = torch.tensor(
            [self.token_mel_ratio * token_length], dtype=torch.int32
        )
        data["prompt_speech_token_ids"] = data["prompt_speech_token_ids"][:, :token_length]
        data["prompt_speech_token_lengths"] = torch.tensor([token_length], dtype=torch.int32)
        return data


__all__ = ["SPECIAL_TOKENS", "CosyVoiceV2FeatureExtractor", "CosyVoiceV2Processor"]
