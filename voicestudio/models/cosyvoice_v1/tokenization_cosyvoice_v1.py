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
"""Tokenization class for CosyVoice v1."""

import base64
import re
from typing import Optional, Union

from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from transformers.convert_slow_tokenizer import TikTokenConverter, bytes_to_unicode
from transformers.tokenization_utils_tokenizers import TokenizersBackend
from transformers.utils import logging


logger = logging.get_logger(__name__)


CHINESE_CHARACTERS = re.compile(r"[\u4e00-\u9fff]+")


CORNER_MARKS = {"\u00b2": "\u5e73\u65b9", "\u00b3": "\u7acb\u65b9"}


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


def load_text_normalizer(**kwargs):
    """
    Builds the weighted finite state transducer normalizer upstream's text front end rewrites a date,
    a currency amount, a unit and an abbreviation with, from the optional `wetext` package.

    Args:
        kwargs:
            Forwarded to `wetext.Normalizer`.

    Returns:
        `wetext.Normalizer` or `None`: The normalizer, or `None` when `wetext` is not installed.
    """
    try:
        from wetext import Normalizer
    except ImportError:
        return None
    return Normalizer(**kwargs)


def warn_without_text_normalizer(text: str) -> None:
    """
    Reports that the text normalizer is unavailable, once per process, and only for text holding a
    digit, which is the text whose reading it changes.

    Args:
        text (`str`):
            Text the normalizer was going to be applied to.
    """
    if any(character.isdigit() for character in text):
        logger.warning_once(
            "`wetext` is not installed, so the text normalizer of the CosyVoice front end is skipped "
            "and this text reaches the model with its numbers written as they are. A date, a currency "
            "amount, a unit and a phone number are read wrongly without it; plain text is unaffected, "
            "and an English digit run is still read out. Install `wetext` to enable it."
        )


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


def normalize_english(text: str, english_normalizer, number_speller) -> str:
    """
    Rewrites an English sentence the way upstream's front end reads it out: the text normalizer
    first, then every digit run it leaves behind.

    Args:
        text (`str`):
            Text to rewrite. A string that is empty or whitespace only skips the normalizer, which
            asserts on one.
        english_normalizer (`wetext.Normalizer` or `None`):
            Normalizer rewriting a date, a currency amount, a unit and an abbreviation. `None` skips
            that step.
        number_speller ([`CosyVoiceV1NumberSpeller`]):
            Engine whose `number_to_words` reads a digit run out.

    Returns:
        `str`: The rewritten text.
    """
    if english_normalizer is None:
        warn_without_text_normalizer(text)
    elif text.strip():
        text = english_normalizer.normalize(text)
    return spell_out_number(text, number_speller)


VOCAB_FILES_NAMES = {
    "vocab_file": "multilingual_zh_ja_yue_char_del.tiktoken",
    "tokenizer_file": "tokenizer.json",
}

SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

NUM_TIMESTAMP_TOKENS = 1501

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
    "minnan": "minnan",
    "wuyu": "wuyu",
    "dialect": "dialect",
    "zh/en": "zh/en",
    "en/zh": "en/zh",
}

AUDIO_EVENTS = [
    "ASR",
    "AED",
    "SER",
    "Speech",
    "/Speech",
    "BGM",
    "/BGM",
    "Laughter",
    "/Laughter",
    "Applause",
    "/Applause",
]

EMOTIONS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"]

TTS_VOCAL_TOKENS = [
    "TTS/B",
    "TTS/O",
    "TTS/Q",
    "TTS/A",
    "TTS/CO",
    "TTS/CL",
    "TTS/H",
    *[f"TTS/SP{index:02d}" for index in range(1, 14)],
]


def build_special_tokens(num_languages: int = len(LANGUAGES)) -> list[str]:
    r"""
    Builds the special tokens of the CosyVoice v1 vocabulary, in the order their ids follow the
    mergeable ranks of the tiktoken file.

    Args:
        num_languages (`int`, *optional*, defaults to 105):
            Number of leading entries of `LANGUAGES` that get a language token. The released
            `CosyVoice-300M-25Hz` vocabulary of 60515 tokens covers all of them.

    Returns:
        `list[str]`: The special tokens, lowest id first.
    """
    return [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{language}|>" for language in list(LANGUAGES)[:num_languages]],
        *[f"<|{event}|>" for event in AUDIO_EVENTS],
        *[f"<|{emotion}|>" for emotion in EMOTIONS],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{index}|>" for index in range(1, 31)],
        *[f"<|{token}|>" for token in TTS_VOCAL_TOKENS],
        *[f"<|{index * 0.02:.2f}|>" for index in range(NUM_TIMESTAMP_TOKENS)],
    ]


class CosyVoiceV1TikTokenConverter(TikTokenConverter):
    r"""
    Converts a tiktoken rank file into the byte level BPE backend of [`CosyVoiceV1Tokenizer`].

    Args:
        vocab_file (`str`):
            Path of the `.tiktoken` file, which holds one `base64(token) rank` pair per line.
        pattern (`str`, *optional*):
            Regular expression the pre tokenizer isolates matches of.
        extra_special_tokens (`list[str]`, *optional*):
            Special tokens appended after the mergeable ranks, lowest id first.
    """

    def extract_vocab_merges_from_model(self, vocab_file: str) -> tuple[dict[str, int], list[tuple[str, str]]]:
        r"""
        Args:
            vocab_file (`str`):
                Path of the `.tiktoken` file.

        Returns:
            `tuple(dict, list)`: The vocabulary, mapping every token to its rank, and the merges that
            rebuild each multi byte token from the two ranked tokens it splits into.
        """
        with open(vocab_file, "rb") as ranks_file:
            bpe_ranks = {
                base64.b64decode(token): int(rank)
                for token, rank in (line.split() for line in ranks_file if line.strip())
            }
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(token: bytes) -> str:
            return "".join(byte_encoder[byte] for byte in token)

        vocab, merges = {}, []
        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            splits = []
            for index in range(1, len(token)):
                left, right = token[:index], token[index:]
                if left in bpe_ranks and right in bpe_ranks and (left + right) in bpe_ranks:
                    splits.append((left, right, rank))
            merges.extend(sorted(splits, key=lambda split: (bpe_ranks[split[0]], bpe_ranks[split[1]])))
        merges = sorted(merges, key=lambda split: split[2])
        return vocab, [(token_bytes_to_string(left), token_bytes_to_string(right)) for left, right, _ in merges]


class CosyVoiceV1Tokenizer(TokenizersBackend):
    r"""
    Constructs a CosyVoice v1 tokenizer, the byte level BPE the `CosyVoice-300M-25Hz` release encodes
    its text with. Its 58836 mergeable ranks are not the 51866 of `openai/whisper-large-v3`, and its
    special tokens carry audio event, emotion and TTS vocal markers Whisper has none of, so the two
    vocabularies are not interchangeable.

    This tokenizer inherits from [`PreTrainedTokenizerFast`], which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab (`dict[str, int]`, *optional*):
            Vocabulary mapping every byte level token to its id. Read back from a serialized backend;
            build it from `vocab_file` instead.
        merges (`list`, *optional*):
            Merges of the byte level BPE, read back from a serialized backend.
        vocab_file (`str`, *optional*):
            Path of the `.tiktoken` file holding the mergeable ranks. It is turned into `vocab` and
            `merges` by [`CosyVoiceV1TikTokenConverter`].
        num_languages (`int`, *optional*, defaults to 105):
            Number of language tokens, passed to [`build_special_tokens`]. The released vocabulary of
            60515 tokens covers every entry of `LANGUAGES`.
        bos_token (`str`, *optional*, defaults to `"<|startoftranscript|>"`):
            Beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            End of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            Padding token.
        unk_token (`str`, *optional*):
            Unknown token. A byte level vocabulary encodes every input, so there is none.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: Optional[dict[str, int]] = None,
        merges: Optional[list] = None,
        vocab_file: Optional[str] = None,
        num_languages: int = len(LANGUAGES),
        bos_token: Union[str, AddedToken] = "<|startoftranscript|>",
        eos_token: Union[str, AddedToken] = "<|endoftext|>",
        pad_token: Union[str, AddedToken] = "<|endoftext|>",
        unk_token: Optional[Union[str, AddedToken]] = None,
        **kwargs,
    ):
        if vocab is None and vocab_file is not None:
            vocab, merges = CosyVoiceV1TikTokenConverter(
                vocab_file=vocab_file, pattern=SPLIT_PATTERN
            ).extract_vocab_merges_from_model(vocab_file)

        self.num_languages = num_languages
        self._vocab = vocab if vocab is not None else {}
        self._merges = merges if merges is not None else []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )
        # A rank that is already a whole token is emitted as it stands rather than rebuilt from merges.
        self._tokenizer.model.ignore_merges = True
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        if self._vocab:
            self._tokenizer.add_special_tokens(
                [
                    AddedToken(token, normalized=False, special=True)
                    for token in build_special_tokens(num_languages)
                ]
            )

        super().__init__(
            vocab_file=vocab_file,
            num_languages=num_languages,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )


__all__ = [
    "CosyVoiceV1NumberSpeller",
    "CosyVoiceV1TikTokenConverter",
    "CosyVoiceV1Tokenizer",
    "build_special_tokens",
    "contains_chinese",
    "load_text_normalizer",
    "normalize_english",
    "number_to_words",
    "remove_bracket",
    "replace_blank",
    "replace_corner_mark",
    "spell_out_number",
    "spell_out_two_digits",
    "warn_without_text_normalizer",
]
