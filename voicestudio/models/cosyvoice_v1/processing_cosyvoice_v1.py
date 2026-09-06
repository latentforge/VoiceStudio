"""Processor class for CosyVoice v1."""

import re
import unicodedata
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from librosa.filters import mel as librosa_mel

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.processing_utils import ProcessorMixin

from .configuration_cosyvoice_v1 import CosyVoiceV1Config
from .modeling_cosyvoice_v1 import CosyVoiceV1SpeakerEncoder, CosyVoiceV1SpeechTokenizer
from .weight_conversion import (
    SPEAKER_ENCODER_REPO,
    SPEAKER_ENCODER_WEIGHTS,
    SPEAKER_INFO_FILE,
    SPEECH_TOKENIZER_FILE,
    TEXT_TOKENIZER_ID,
    convert_speech_tokenizer,
    resolve_checkpoint,
)


CHINESE_CHARACTERS = re.compile(r"[\u4e00-\u9fff]+")

CORNER_MARKS = {"\u00b2": "\u5e73\u65b9", "\u00b3": "\u7acb\u65b9"}

CHINESE_PUNCTUATION = ["\u3002", "\uff1f", "\uff01", "\uff1b", "\uff1a", "\u3001", ".", "?", "!", ";"]

ENGLISH_PUNCTUATION = [".", "?", "!", ";", ":"]

ENGLISH_UNITS = ("", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

ENGLISH_TEENS = (
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
)

ENGLISH_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

ENGLISH_SCALES = (
    "", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion", " sextillion",
    " septillion", " octillion", " nonillion", " decillion",
)


def contains_chinese(text: str) -> bool:
    """
    Args:
        text (`str`):
            Text to test.

    Returns:
        `bool`: Whether the text holds at least one Chinese character.
    """
    return bool(CHINESE_CHARACTERS.search(text))


def replace_corner_mark(text: str) -> str:
    """
    Spells the squared and cubed corner marks out in Chinese.

    Args:
        text (`str`):
            Text to rewrite.

    Returns:
        `str`: The rewritten text.
    """
    for mark, replacement in CORNER_MARKS.items():
        text = text.replace(mark, replacement)
    return text


def remove_bracket(text: str) -> str:
    """
    Drops the bracket and quote characters the Chinese front end treats as meaningless.

    Args:
        text (`str`):
            Text to rewrite.

    Returns:
        `str`: The rewritten text.
    """
    text = text.replace("\uff08", "").replace("\uff09", "")
    text = text.replace("\u3010", "").replace("\u3011", "")
    text = text.replace("`", "")
    text = text.replace("\u2014\u2014", " ")
    return text


def replace_blank(text: str) -> str:
    """
    Drops every space that does not sit between two ASCII characters, which is how a Chinese sentence
    keeps the spaces of an embedded English word and loses the rest.

    Args:
        text (`str`):
            Text to rewrite.

    Returns:
        `str`: The rewritten text.
    """
    characters = []
    for index, character in enumerate(text):
        if character != " ":
            characters.append(character)
        elif (
            text[index + 1].isascii()
            and text[index + 1] != " "
            and text[index - 1].isascii()
            and text[index - 1] != " "
        ):
            characters.append(character)
    return "".join(characters)


def spell_out_two_digits(tens: int, units: int) -> str:
    """
    Args:
        tens (`int`):
            Tens digit.
        units (`int`):
            Units digit.

    Returns:
        `str`: The two digits read out, hyphenated when both are nonzero and empty when both are
        zero.
    """
    if tens == 1:
        return ENGLISH_TEENS[units]
    return ENGLISH_TENS[tens] + ("-" if tens and units else "") + ENGLISH_UNITS[units]


def number_to_words(number: str) -> str:
    """
    Reads a run of decimal digits out in English: groups of three carrying a scale word and
    separated by commas, `and` after a hundreds digit and before a trailing group that is a single
    word, tens and units hyphenated, and leading zeros dropped.

    Args:
        number (`str`):
            Run of decimal digits, with no sign, decimal point or group separator. A run of zeros
            reads as `zero`.

    Returns:
        `str`: The reading.

    Raises:
        ValueError: If the run needs a scale word beyond `decillion`.
    """
    digits = number.lstrip("0")
    if digits == "":
        return "zero"
    if int(digits) == 1:
        return "one"
    groups = []
    while digits:
        groups.append(int(digits[-3:]))
        digits = digits[:-3]
    if len(groups) > len(ENGLISH_SCALES):
        raise ValueError(f"{number} is larger than the English scale words reach")
    spelled = []
    for index in range(len(groups) - 1, -1, -1):
        hundreds, remainder = divmod(groups[index], 100)
        if hundreds == 0 and remainder == 0:
            continue
        below_hundred = spell_out_two_digits(*divmod(remainder, 10))
        if hundreds:
            joiner = " and " if remainder else ""
            spelled.append(f"{ENGLISH_UNITS[hundreds]} hundred{joiner}{below_hundred}{ENGLISH_SCALES[index]}")
        else:
            spelled.append(f"{below_hundred}{ENGLISH_SCALES[index]}")
    words = ", ".join(spelled)
    head, separator, tail = words.rpartition(", ")
    if separator and " " not in tail:
        words = f"{head} and {tail}"
    return words


class CosyVoiceV1NumberSpeller:
    r"""
    Constructs the engine [`~CosyVoiceV1Processor.normalize_text`] reads an English digit run out
    with.
    """

    def number_to_words(self, number: str) -> str:
        """
        Args:
            number (`str`):
                Run of decimal digits.

        Returns:
            `str`: The reading, from [`number_to_words`].
        """
        return number_to_words(number)


def spell_out_number(text: str, number_speller) -> str:
    """
    Replaces every run of digits with its English reading.

    Args:
        text (`str`):
            Text to rewrite.
        number_speller ([`CosyVoiceV1NumberSpeller`]):
            Engine whose `number_to_words` reads a digit run out.

    Returns:
        `str`: The rewritten text.
    """
    spelled = []
    start = None
    for index, character in enumerate(text):
        if character.isdigit():
            if start is None:
                start = index
            continue
        if start is not None:
            spelled.append(number_speller.number_to_words(text[start:index]))
            start = None
        spelled.append(character)
    if start is not None and start < len(text):
        spelled.append(number_speller.number_to_words(text[start:]))
    return "".join(spelled)


def is_only_punctuation(text: str) -> bool:
    """
    Args:
        text (`str`):
            Text to test.

    Returns:
        `bool`: Whether the text is empty or made of punctuation and symbols alone.
    """
    return all(unicodedata.category(character)[0] in "PS" for character in text)


def split_paragraph(
    text: str,
    tokenize: Callable[[str], list[int]],
    lang: str = "zh",
    token_max_n: int = 80,
    token_min_n: int = 60,
    merge_len: int = 20,
    comma_split: bool = False,
) -> list[str]:
    """
    Splits a paragraph into sentences on punctuation, then greedily merges neighbouring sentences so
    that each piece stays under `token_max_n` and a trailing piece shorter than `merge_len` joins the
    one before it. A Chinese piece is measured in characters and any other in tokens.

    Args:
        text (`str`):
            Paragraph to split.
        tokenize (`Callable`):
            Callable returning the token ids of a string, used to measure a non Chinese piece.
        lang (`str`, *optional*, defaults to `"zh"`):
            `"zh"` to measure in characters and split on Chinese punctuation, anything else to measure
            in tokens and split on ASCII punctuation.
        token_max_n (`int`, *optional*, defaults to 80):
            Length above which a piece is closed, provided the piece is already longer than
            `token_min_n`.
        token_min_n (`int`, *optional*, defaults to 60):
            Length below which a piece keeps growing.
        merge_len (`int`, *optional*, defaults to 20):
            Length below which a trailing piece is merged into the one before it.
        comma_split (`bool`, *optional*, defaults to `False`):
            Whether a comma also ends a sentence.

    Returns:
        `list[str]`: The pieces.
    """

    def length(piece: str) -> int:
        return len(piece) if lang == "zh" else len(tokenize(piece))

    punctuation = list(CHINESE_PUNCTUATION if lang == "zh" else ENGLISH_PUNCTUATION)
    if comma_split:
        punctuation.extend(["\uff0c", ","])
    if text[-1] not in punctuation:
        text += "\u3002" if lang == "zh" else "."

    start, sentences = 0, []
    for index, character in enumerate(text):
        if character not in punctuation:
            continue
        if len(text[start:index]) > 0:
            sentences.append(text[start:index] + character)
        if index + 1 < len(text) and text[index + 1] in ['"', "\u201d"]:
            sentences.append(sentences.pop(-1) + text[index + 1])
            start = index + 2
        else:
            start = index + 1

    pieces, current = [], ""
    for sentence in sentences:
        if length(current + sentence) > token_max_n and length(current) > token_min_n:
            pieces.append(current)
            current = ""
        current = current + sentence
    if len(current) > 0:
        if length(current) < merge_len and len(pieces) != 0:
            pieces[-1] = pieces[-1] + current
        else:
            pieces.append(current)
    return pieces


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


class CosyVoiceV1Processor(ProcessorMixin):
    r"""
    Constructs a CosyVoice v1 processor, which wraps a Whisper tokenizer, the mel spectrogram feature
    extractor of the flow matching model, the supervised semantic speech tokenizer and the speaker
    encoder into a single object.

    The speech tokenizer is [`CosyVoiceV1SpeechTokenizer`], whose weights are read out of the
    `speech_tokenizer_v1.onnx` graph the released directory ships, which is the only form upstream
    publishes them in. The speaker encoder is [`CosyVoiceV1SpeakerEncoder`], built from the CAM++
    weights its authors published at [`SPEAKER_ENCODER_REPO`]. Both are built on first use, so text
    tokenization and the speaker table cost nothing.

    Args:
        feature_extractor ([`CosyVoiceV1FeatureExtractor`]):
            Mel spectrogram extractor of the flow matching model.
        tokenizer ([`WhisperTokenizer`]):
            Text tokenizer.
        speech_token_model_path (`str`, *optional*):
            Path of the `speech_tokenizer_v1.onnx` graph the speech tokenizer is built from.
        speaker_encoder_model_path (`str`, *optional*):
            Path of the CAM++ weights the speaker encoder is built from. Defaults to
            [`SPEAKER_ENCODER_WEIGHTS`] fetched from [`SPEAKER_ENCODER_REPO`].
        speaker_info_path (`str`, *optional*):
            Path of a `spk2info.pt`, the table of precomputed prompts the SFT and Instruct
            checkpoints ship. Read lazily by [`~CosyVoiceV1Processor.get_speaker`].
        speech_token_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel bins of the log mel spectrogram the speech tokenizer consumes.
    """

    speech_tokenizer_sampling_rate = 16000
    speaker_encoder_sampling_rate = 16000
    speaker_encoder_max_seconds = 10
    feature_extractor_type = CosyVoiceV1FeatureExtractor
    speech_tokenizer_type = CosyVoiceV1SpeechTokenizer
    model_config_type = CosyVoiceV1Config
    speech_tokenizer_file = SPEECH_TOKENIZER_FILE

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        speech_token_model_path: Optional[str] = None,
        speaker_encoder_model_path: Optional[str] = None,
        speaker_info_path: Optional[str] = None,
        speech_token_mel_bins: int = 128,
        **kwargs,
    ):
        super().__init__(feature_extractor, tokenizer, **kwargs)
        self.speech_token_model_path = speech_token_model_path
        self.speaker_encoder_model_path = speaker_encoder_model_path
        self.speaker_info_path = speaker_info_path
        self.speech_token_mel_bins = speech_token_mel_bins
        self._speech_token_features = None
        self._speech_tokenizer = None
        self._speaker_encoder = None
        self._speaker_info = None
        self._number_speller = None

    @classmethod
    def _released_processor(cls, directory: "str | Path") -> "CosyVoiceV1Processor":
        r"""
        Builds the processor of a released CosyVoice v1 directory.

        The directory carries the speech tokenizer, the speaker encoder and, on the SFT and Instruct
        releases, the speaker table. Its text tokenizer is not part of it: upstream builds one with
        `whisper.tokenizer.get_tokenizer`, whose vocabulary is the one [`TEXT_TOKENIZER_ID`] ships.

        Args:
            directory (`str` or `os.PathLike`):
                Local directory of the released checkpoint.

        Returns:
            [`CosyVoiceV1Processor`]: The processor.
        """
        directory = Path(directory)
        speaker_info = directory / SPEAKER_INFO_FILE
        return cls(
            feature_extractor=cls.feature_extractor_type(),
            tokenizer=WhisperTokenizer.from_pretrained(TEXT_TOKENIZER_ID, language="en", task="transcribe"),
            speech_token_model_path=str(directory / cls.speech_tokenizer_file),
            speaker_info_path=str(speaker_info) if speaker_info.is_file() else None,
        )

    @classmethod
    def _resolve_released_checkpoint(cls, source, **kwargs) -> "Path | None":
        r"""
        Fetches the files a released CosyVoice v1 directory holds for the processor.

        Args:
            source (`str` or `os.PathLike`, *optional*):
                Repository id or local directory.
            kwargs (`dict`, *optional*):
                Fields of `weight_conversion.DOWNLOAD_KWARGS` selecting a revision and a cache.

        Returns:
            `Path` or `None`: The local directory, or `None` when `source` holds no released
            checkpoint.
        """
        return resolve_checkpoint(source, (cls.speech_tokenizer_file,), (SPEAKER_INFO_FILE,), **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Repository id or local directory. The directories the CosyVoice authors published
                carry the speech tokenizer and the speaker encoder but no saved processor, and are
                read as they are.
            kwargs:
                Forwarded to [`~ProcessorMixin.from_pretrained`].

        Returns:
            [`CosyVoiceV1Processor`]: The processor.
        """
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            directory = cls._resolve_released_checkpoint(pretrained_model_name_or_path, **kwargs)
            if directory is None:
                raise
        return cls._released_processor(directory)

    @property
    def number_speller(self) -> CosyVoiceV1NumberSpeller:
        """
        Returns:
            [`CosyVoiceV1NumberSpeller`]: The engine that reads an English digit run out.
        """
        if self._number_speller is None:
            self._number_speller = CosyVoiceV1NumberSpeller()
        return self._number_speller

    @property
    def speech_token_feature_extractor(self):
        """
        Returns:
            [`WhisperFeatureExtractor`]: The log mel extractor feeding the speech tokenizer.
        """
        if self._speech_token_features is None:
            from transformers import WhisperFeatureExtractor

            self._speech_token_features = WhisperFeatureExtractor(
                feature_size=self.speech_token_mel_bins, sampling_rate=self.speech_tokenizer_sampling_rate
            )
        return self._speech_token_features

    @property
    def speech_tokenizer(self) -> CosyVoiceV1SpeechTokenizer:
        """
        Returns:
            [`CosyVoiceV1SpeechTokenizer`]: The speech tokenizer, in evaluation mode.
        """
        if self._speech_tokenizer is None:
            tokenizer = self.speech_tokenizer_type(self.model_config_type())
            tokenizer.load_state_dict(convert_speech_tokenizer(self.speech_token_model_path))
            self._speech_tokenizer = tokenizer.eval()
        return self._speech_tokenizer

    @property
    def speaker_encoder(self) -> CosyVoiceV1SpeakerEncoder:
        """
        Returns:
            [`CosyVoiceV1SpeakerEncoder`]: The speaker encoder, in evaluation mode.
        """
        if self._speaker_encoder is None:
            path = self.speaker_encoder_model_path
            if path is None:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(SPEAKER_ENCODER_REPO, SPEAKER_ENCODER_WEIGHTS)
            encoder = CosyVoiceV1SpeakerEncoder(self.model_config_type())
            encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self._speaker_encoder = encoder.eval()
        return self._speaker_encoder

    @property
    def speakers(self) -> list[str]:
        """
        Returns:
            `list[str]`: Names of the speakers `speaker_info_path` holds, empty when it is unset.
        """
        if self.speaker_info_path is None:
            return []
        if self._speaker_info is None:
            self._speaker_info = torch.load(self.speaker_info_path, map_location="cpu", weights_only=True)
        return list(self._speaker_info)

    def get_speaker(self, name: str) -> BatchFeature:
        """
        Reads one precomputed prompt out of `speaker_info_path`.

        Args:
            name (`str`):
                Name of the speaker, one of [`~CosyVoiceV1Processor.speakers`].

        Returns:
            [`BatchFeature`]: `speaker_embedding`, and `prompt_speech_token_ids` plus `speech_feat`
            when the table carries them.

        Raises:
            ValueError: If no speaker table is configured or `name` is not in it.
        """
        if not self.speakers:
            raise ValueError("this processor has no `speaker_info_path`, so it has no speaker table")
        if name not in self._speaker_info:
            raise ValueError(f"{name} is not one of {list(self._speaker_info)}")
        entry = self._speaker_info[name]
        data = {"speaker_embedding": entry["embedding"]}
        if "speech_token" in entry:
            data["prompt_speech_token_ids"] = entry["speech_token"].to(torch.int32)
            data["prompt_speech_token_lengths"] = torch.tensor(
                [entry["speech_token"].shape[1]], dtype=torch.int32
            )
        if "speech_feat" in entry:
            data["speech_feat"] = entry["speech_feat"]
            data["speech_feat_lengths"] = torch.tensor([entry["speech_feat"].shape[1]], dtype=torch.int32)
        return BatchFeature(data)

    def _resample(self, waveform: torch.Tensor, sampling_rate: Optional[int], target_rate: int) -> torch.Tensor:
        waveform = waveform.reshape(1, -1).float()
        if sampling_rate is not None and sampling_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sampling_rate, target_rate)
        return waveform

    def encode_speech_tokens(
        self, audio: Union[np.ndarray, torch.Tensor], sampling_rate: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Turns a waveform into supervised semantic speech tokens.

        Args:
            audio (`np.ndarray` or `torch.Tensor`):
                Mono waveform.
            sampling_rate (`int`, *optional*):
                Rate of `audio`.

        Returns:
            `tuple(torch.Tensor)`: the speech token ids of shape `(1, speech_length)` and their length.

        Raises:
            ValueError: If the waveform is longer than thirty seconds.
        """
        waveform = self._resample(
            audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio)),
            sampling_rate,
            self.speech_tokenizer_sampling_rate,
        )
        if waveform.shape[1] / self.speech_tokenizer_sampling_rate > 30:
            raise ValueError("the CosyVoice v1 speech tokenizer does not support audio longer than 30 seconds")
        features = self.speech_token_feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.speech_tokenizer_sampling_rate,
            padding=False,
            return_tensors="pt",
        ).input_features
        lengths = torch.tensor([features.shape[2]], dtype=torch.int32)
        with torch.no_grad():
            speech_token_ids = self.speech_tokenizer(features, lengths).to(torch.int32)
        return speech_token_ids, torch.tensor([speech_token_ids.shape[1]], dtype=torch.int32)

    def encode_speaker(
        self, audio: Union[np.ndarray, torch.Tensor], sampling_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Turns a waveform into an utterance level speaker embedding.

        Args:
            audio (`np.ndarray` or `torch.Tensor`):
                Mono waveform.
            sampling_rate (`int`, *optional*):
                Rate of `audio`.

        Returns:
            `torch.Tensor` of shape `(1, speaker_embedding_dim)`: the speaker embedding.
        """
        waveform = self._resample(
            audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio)),
            sampling_rate,
            self.speaker_encoder_sampling_rate,
        )
        features = kaldi.fbank(
            waveform, num_mel_bins=80, dither=0, sample_frequency=self.speaker_encoder_sampling_rate
        )
        features = features - features.mean(dim=0, keepdim=True)
        with torch.no_grad():
            return self.speaker_encoder(features.unsqueeze(dim=0))

    def normalize_text(self, text: str, split: bool = True) -> Union[str, list[str]]:
        """
        Rewrites a sentence the way upstream's text front end does, then optionally splits it into the
        pieces upstream synthesizes one at a time.

        A Chinese sentence loses the spaces that do not sit inside an embedded English word, has its
        corner marks spelled out, its brackets removed, its full stops and dashes replaced by their
        Chinese counterparts and a trailing run of commas turned into a full stop. Any other sentence
        has its digit runs read out by [`number_to_words`]. Text carrying a `<|` `|>` marker is returned
        untouched. Neither branch runs a text normalizer over numbers, dates or abbreviations, which is
        what upstream reaches `ttsfrd` or `wetext` for.

        Args:
            text (`str`):
                Text to rewrite.
            split (`bool`, *optional*, defaults to `True`):
                Whether the rewritten text is split into pieces.

        Returns:
            `str` or `list[str]`: The rewritten text, or its pieces when `split` is set.
        """
        if ("<|" in text and "|>" in text) or text == "":
            return [text] if split else text
        text = text.strip()

        def tokenize(piece: str) -> list[int]:
            return self.tokenizer.encode(piece, add_special_tokens=False)

        if contains_chinese(text):
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "\u3002")
            text = text.replace(" - ", "\uff0c")
            text = remove_bracket(text)
            text = re.sub(r"[\uff0c,\u3001]+$", "\u3002", text)
            pieces = split_paragraph(text, tokenize, "zh", token_max_n=80, token_min_n=60, merge_len=20)
        else:
            text = spell_out_number(text, self.number_speller)
            pieces = split_paragraph(text, tokenize, "en", token_max_n=80, token_min_n=60, merge_len=20)
        pieces = [piece for piece in pieces if not is_only_punctuation(piece)]
        return pieces if split else text

    def compute_f0(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sampling_rate: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extracts the f0 contour the vocoder objective regresses the f0 predictor onto, with the WORLD
        harvest estimator, falling back to dio when fewer than five frames come back voiced, refined
        by stonemask and interpolated onto the mel frame count.

        Args:
            audio (`np.ndarray` or `torch.Tensor`):
                Mono waveform.
            sampling_rate (`int`, *optional*):
                Rate of `audio`. It is resampled to the rate the mel filter bank is built for.
            num_frames (`int`, *optional*):
                Number of frames the contour is interpolated onto. Defaults to the number of mel frames
                of the resampled waveform.

        Returns:
            `torch.Tensor` of shape `(num_frames,)`: the f0 contour, in Hz.

        Raises:
            ImportError: If `pyworld` is not installed.
        """
        try:
            import pyworld
        except ImportError as error:
            raise ImportError(
                "the CosyVoice v1 vocoder objective regresses its f0 predictor onto the WORLD harvest "
                "contour, which upstream extracts with `pyworld`. This package does not depend on it. "
                "Pass a contour of your own to `CosyVoiceV1HiFTGenerator.compute_loss`, or install "
                "`pyworld` yourself."
            ) from error

        target_rate = self.feature_extractor.mel_sampling_rate
        hop_length = self.feature_extractor.hop_length
        waveform = self._resample(
            audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio)),
            sampling_rate,
            target_rate,
        )
        samples = waveform.squeeze(0).numpy().astype("double")
        frame_period = hop_length * 1000 / target_rate
        contour, time_axis = pyworld.harvest(samples, target_rate, frame_period=frame_period)
        if (contour != 0).sum() < 5:
            contour, time_axis = pyworld.dio(samples, target_rate, frame_period=frame_period)
        contour = pyworld.stonemask(samples, contour, time_axis, target_rate)
        if num_frames is None:
            num_frames = self.feature_extractor._mel_spectrogram(waveform).shape[-1]
        # The interpolation stays in the double precision the WORLD estimator returns: at a voicing
        # boundary a float32 source index can round onto the neighbouring frame.
        contour = torch.from_numpy(contour).view(1, 1, -1)
        contour = torch.nn.functional.interpolate(contour, size=num_frames, mode="linear")
        return contour.view(-1).float()

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sampling_rate: Optional[int] = None,
        prompt_text: Optional[Union[str, list[str]]] = None,
        normalize: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            text (`str` or `list[str]`, *optional*):
                Text to synthesize.
            audio (`np.ndarray` or `torch.Tensor`, *optional*):
                Mono waveform of the prompt utterance, turned into speech tokens, a mel spectrogram and
                a speaker embedding.
            sampling_rate (`int`, *optional*):
                Rate of `audio`.
            prompt_text (`str` or `list[str]`, *optional*):
                Transcript of the prompt utterance.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether every string is rewritten by [`~CosyVoiceV1Processor.normalize_text`] first.
                Splitting a long paragraph into the pieces upstream synthesizes one at a time is
                `normalize_text(text)` instead, since one call here encodes one sequence.

        Returns:
            [`BatchFeature`]: `input_ids` and `input_lengths` for the text, plus
            `prompt_input_ids`, `prompt_speech_token_ids`, `prompt_speech_token_lengths`,
            `speech_feat`, `speech_feat_lengths` and `speaker_embedding` when a prompt is given.
        """
        if normalize:
            if isinstance(text, str):
                text = self.normalize_text(text, split=False)
            elif text is not None:
                text = [self.normalize_text(item, split=False) for item in text]
            if isinstance(prompt_text, str):
                prompt_text = self.normalize_text(prompt_text, split=False)
            elif prompt_text is not None:
                prompt_text = [self.normalize_text(item, split=False) for item in prompt_text]
        data = {}
        if text is not None:
            encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt", **kwargs)
            data["input_ids"] = encoded["input_ids"].to(torch.int32)
            data["input_lengths"] = torch.tensor([data["input_ids"].shape[1]], dtype=torch.int32)
        if prompt_text is not None:
            encoded = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt", **kwargs)
            data["prompt_input_ids"] = encoded["input_ids"].to(torch.int32)
            data["prompt_input_lengths"] = torch.tensor([data["prompt_input_ids"].shape[1]], dtype=torch.int32)
        if audio is not None:
            speech_token_ids, speech_token_lengths = self.encode_speech_tokens(audio, sampling_rate)
            data["prompt_speech_token_ids"] = speech_token_ids
            data["prompt_speech_token_lengths"] = speech_token_lengths
            data.update(self.feature_extractor(audio, sampling_rate=sampling_rate))
            data["speaker_embedding"] = self.encode_speaker(audio, sampling_rate)
        return BatchFeature(data)


__all__ = [
    "CosyVoiceV1FeatureExtractor",
    "CosyVoiceV1NumberSpeller",
    "CosyVoiceV1Processor",
    "contains_chinese",
    "is_only_punctuation",
    "number_to_words",
    "remove_bracket",
    "replace_blank",
    "replace_corner_mark",
    "spell_out_number",
    "spell_out_two_digits",
    "split_paragraph",
]
