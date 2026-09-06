# coding=utf-8
# Copyright 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du) and the HuggingFace Inc. team. All rights reserved.
# Copyright 2010 M. Morise. All rights reserved.
#
# The f0 estimators in this file are derived from WORLD and are licensed under the
# BSD 3-Clause license:
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# The remainder is licensed under the Apache License, Version 2.0 (the "License");
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
"""Processor class for CosyVoice v1."""

import functools
import math
import re
import unicodedata
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging

from .configuration_cosyvoice_v1 import CosyVoiceV1Config
from .feature_extraction_cosyvoice_v1 import CosyVoiceV1FeatureExtractor
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


logger = logging.get_logger(__name__)


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


_LOG2 = 0.69314718055994529
_PI = 3.1415926535897932384
_SAFE_GUARD_MINIMUM = 0.000000000001
_FLOOR_F0_STONEMASK = 40.0
_CUT_OFF = 50.0
_MAXIMUM_VALUE = 100000.0

_DECIMATE_COEFFICIENTS = {
    2: ((0.041156734567757189, -0.42599112459189636, 0.041037215479961225), (0.16797464681802227, 0.50392394045406674)),
    3: ((0.95039378983237421, -0.67429146741526791, 0.15412211621346475), (0.071221945171178636, 0.21366583551353591)),
    4: ((1.4499664446880227, -0.98943497080950582, 0.24578252340690215), (0.036710750339322612, 0.11013225101796784)),
    5: ((1.7610939654280557, -1.2554914843859768, 0.3237186507788215), (0.021334858522387423, 0.06400457556716227)),
    6: ((1.9715352749512141, -1.4686795689225347, 0.3893908434965701), (0.013469181309343825, 0.040407543928031475)),
    7: ((2.1225239019534703, -1.6395144861046302, 0.44469707800587366), (0.0090366882681608418, 0.027110064804482525)),
    8: ((2.2357462340187593, -1.7780899984041358, 0.49152555365968692), (0.0063522763407111993, 0.019056829022133598)),
    9: ((2.3236003491759578, -1.8921545617463598, 0.53148928133729068), (0.0046331164041389372, 0.013899349212416812)),
    10: ((2.3936475118069387, -1.9873904075111861, 0.5658879979027055), (0.0034818622251927556, 0.010445586675578267)),
    11: ((2.450743295230728, -2.06794904601978, 0.59574774438332101), (0.0026822508007163792, 0.0080467524021491377)),
    12: ((2.4981398605924205, -2.1368928194784025, 0.62187513816221485), (0.0021097275904709001, 0.0063291827714127002)),
}

_SMOOTHING_A = (1.7347257688092754, -0.76600660094326412)
_SMOOTHING_B = (0.0078202080334971724, 0.015640416066994345)


@functools.cache
def _load_pyworld():
    """Returns the `pyworld` module when it is importable, and `None` otherwise."""
    try:
        import pyworld
    except ImportError:
        return None
    return pyworld


class CosyVoiceV1WorldEstimator:
    r"""
    Constructs the WORLD f0 estimator the vocoder objective's f0 term regresses onto, exposing
    [`~CosyVoiceV1WorldEstimator.harvest`], [`~CosyVoiceV1WorldEstimator.dio`] and
    [`~CosyVoiceV1WorldEstimator.stonemask`]. Each delegates to `pyworld` where that package is
    importable and runs the ported implementation otherwise; the two produce the same contour.

    Args:
        sampling_rate (`int`):
            Rate of the waveforms passed to the estimators.
        f0_floor (`float`, *optional*, defaults to 71.0):
            Lowest f0 the search considers, in Hz.
        f0_ceil (`float`, *optional*, defaults to 800.0):
            Highest f0 the search considers, in Hz.
        prefer_pyworld (`bool`, *optional*, defaults to `True`):
            Whether to delegate to `pyworld` when it is installed. Pass `False` to run the ported
            implementation regardless.
    """

    def __init__(
        self,
        sampling_rate: int,
        f0_floor: float = 71.0,
        f0_ceil: float = 800.0,
        prefer_pyworld: bool = True,
    ):
        self.sampling_rate = sampling_rate
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.pyworld = _load_pyworld() if prefer_pyworld else None

    def harvest(self, waveform: np.ndarray, frame_period: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the f0 contour with Harvest.

        Args:
            waveform (`np.ndarray`):
                Mono waveform.
            frame_period (`float`, *optional*, defaults to 5.0):
                Spacing between analysis frames, in milliseconds.

        Returns:
            `tuple[np.ndarray, np.ndarray]`: the f0 contour in Hz and the frame positions in seconds.
        """
        x = np.ascontiguousarray(waveform, dtype=np.float64)
        if self.pyworld is not None:
            return self.pyworld.harvest(
                x, self.sampling_rate, f0_floor=self.f0_floor, f0_ceil=self.f0_ceil, frame_period=frame_period
            )

        fs = self.sampling_rate
        channels_in_octave = 40.0
        dimension_ratio = self._matlab_round(fs / 8000.0)
        if frame_period == 1.0:
            positions, f0 = self._harvest_general_body(x, fs, 1, channels_in_octave, dimension_ratio)
            return f0, positions

        _, basic_f0 = self._harvest_general_body(x, fs, 1, channels_in_octave, dimension_ratio)
        f0_length = self._samples_for_harvest(fs, x.shape[0], frame_period)
        positions = np.arange(f0_length, dtype=np.float64) * frame_period / 1000.0
        index = np.minimum(basic_f0.shape[0] - 1, self._matlab_round_array(positions * 1000.0))
        return basic_f0[index], positions

    def dio(
        self,
        waveform: np.ndarray,
        frame_period: float = 5.0,
        channels_in_octave: float = 2.0,
        speed: int = 1,
        allowed_range: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the f0 contour with DIO.

        Args:
            waveform (`np.ndarray`):
                Mono waveform.
            frame_period (`float`, *optional*, defaults to 5.0):
                Spacing between analysis frames, in milliseconds.
            channels_in_octave (`float`, *optional*, defaults to 2.0):
                Number of band-pass channels per octave.
            speed (`int`, *optional*, defaults to 1):
                Decimation ratio applied before the analysis, from 1 to 12.
            allowed_range (`float`, *optional*, defaults to 0.1):
                Relative f0 jump the postprocessing tolerates between frames.

        Returns:
            `tuple[np.ndarray, np.ndarray]`: the f0 contour in Hz and the frame positions in seconds.
        """
        x = np.ascontiguousarray(waveform, dtype=np.float64)
        if self.pyworld is not None:
            return self.pyworld.dio(
                x,
                self.sampling_rate,
                f0_floor=self.f0_floor,
                f0_ceil=self.f0_ceil,
                channels_in_octave=channels_in_octave,
                frame_period=frame_period,
                speed=speed,
                allowed_range=allowed_range,
            )

        fs = self.sampling_rate
        x_length = x.shape[0]
        number_of_bands = 1 + int(math.log(self.f0_ceil / self.f0_floor) / _LOG2 * channels_in_octave)
        boundary_f0_list = self.f0_floor * np.power(
            2.0, np.arange(1, number_of_bands + 1, dtype=np.float64) / channels_in_octave
        )

        decimation_ratio = max(min(speed, 12), 1)
        y_length = 1 + int(x_length / decimation_ratio)
        actual_fs = fs / decimation_ratio
        fft_size = self._suitable_fft_size(
            y_length
            + self._matlab_round(actual_fs / _CUT_OFF) * 2
            + 1
            + 4 * int(1.0 + actual_fs / boundary_f0_list[0] / 2.0)
        )

        spectrum = self._spectrum_for_estimation(x, y_length, actual_fs, fft_size, decimation_ratio)
        f0_length = self._samples_for_harvest(fs, x_length, frame_period)
        temporal_positions = np.arange(f0_length, dtype=np.float64) * frame_period / 1000.0

        candidates, scores = self._dio_candidates_and_scores(
            boundary_f0_list, actual_fs, y_length, temporal_positions, spectrum, fft_size
        )
        best = self._dio_best_f0_contour(candidates, scores)
        f0 = self._dio_fix_f0_contour(frame_period, candidates, best, allowed_range)
        return f0, temporal_positions

    def stonemask(self, waveform: np.ndarray, temporal_positions: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """
        Refines an f0 contour by instantaneous frequency.

        Args:
            waveform (`np.ndarray`):
                Mono waveform.
            temporal_positions (`np.ndarray`):
                Frame positions of `f0`, in seconds.
            f0 (`np.ndarray`):
                Contour to refine, in Hz.

        Returns:
            `np.ndarray`: the refined contour, in Hz.
        """
        x = np.ascontiguousarray(waveform, dtype=np.float64)
        f0 = np.ascontiguousarray(f0, dtype=np.float64)
        positions = np.ascontiguousarray(temporal_positions, dtype=np.float64)
        if self.pyworld is not None:
            return self.pyworld.stonemask(x, f0, positions, self.sampling_rate)

        fs = self.sampling_rate
        refined = np.zeros_like(f0)
        frames = np.flatnonzero((f0 > _FLOOR_F0_STONEMASK) & (f0 <= fs / 12.0))
        if frames.shape[0] == 0:
            return refined

        f0s = f0[frames]
        half_window_lengths = (1.5 * fs / f0s + 1.0).astype(np.int64)
        for half_window_length in np.unique(half_window_lengths):
            selected = np.flatnonzero(half_window_lengths == half_window_length)
            width = int(half_window_length) * 2 + 1
            fft_size = int(pow(2.0, 2.0 + int(math.log(width) / _LOG2)))
            chunk = max(1, 2**25 // fft_size)
            for begin in range(0, selected.shape[0], chunk):
                rows = selected[begin : begin + chunk]
                refined[frames[rows]] = self._stonemask_batch(
                    x, fs, positions[frames[rows]], f0s[rows], int(half_window_length), fft_size
                )
        return refined

    def _matlab_round(self, x):
        return int(x + 0.5) if x > 0 else int(x - 0.5)

    def _matlab_round_array(self, x):
        return np.trunc(np.where(x > 0.0, x + 0.5, x - 0.5)).astype(np.int64)

    def _suitable_fft_size(self, sample):
        return int(pow(2.0, int(math.log(float(sample)) / _LOG2) + 1))

    def _nuttall_window(self, length):
        tmp = np.arange(length, dtype=np.float64) / (length - 1.0)
        return (
            0.355768
            - 0.487396 * np.cos(2.0 * _PI * tmp)
            + 0.144232 * np.cos(4.0 * _PI * tmp)
            - 0.012604 * np.cos(6.0 * _PI * tmp)
        )

    def _direct_form_2(self, x, a, b):
        """Runs a direct form II recursion whose state is `len(a)` samples wide."""
        order = len(a)
        y = np.empty(x.shape[0], dtype=np.float64)
        w = [0.0] * order
        source = x.tolist()
        for i, sample in enumerate(source):
            wt = sample
            for k in range(order):
                wt += a[k] * w[k]
            accumulator = b[0] * wt
            for k in range(order):
                accumulator += b[k + 1] * w[k]
            y[i] = accumulator
            for k in range(order - 1, 0, -1):
                w[k] = w[k - 1]
            w[0] = wt
        return y

    def _filter_for_decimate(self, x, ratio):
        a, b = _DECIMATE_COEFFICIENTS[ratio]
        return self._direct_form_2(x, a, (b[0], b[1], b[1], b[0]))

    def _decimate(self, x, ratio):
        n_fact = 9
        x_length = x.shape[0]
        padded = np.empty(x_length + n_fact * 2, dtype=np.float64)
        padded[:n_fact] = 2 * x[0] - x[n_fact:0:-1]
        padded[n_fact : n_fact + x_length] = x
        tail = np.arange(n_fact, dtype=np.int64)
        padded[n_fact + x_length :] = 2 * x[x_length - 1] - x[x_length - 2 - tail]

        filtered = self._filter_for_decimate(padded, ratio)
        filtered = self._filter_for_decimate(filtered[::-1].copy(), ratio)
        padded = filtered[::-1].copy()

        n_out = (x_length - 1) // ratio + 1
        n_beg = ratio - ratio * n_out + x_length
        positions = np.arange(n_beg, x_length + n_fact, ratio, dtype=np.int64)
        return padded[positions + n_fact - 1]

    def _interp1(self, x, y, xi):
        """Linear interpolation with the clamped-interval extrapolation `histc` gives WORLD."""
        index = np.searchsorted(x, xi, side="right")
        np.clip(index, 1, x.shape[0] - 1, out=index)
        lower = index - 1
        step = (xi - x[lower]) / (x[index] - x[lower])
        return y[lower] + step * (y[index] - y[lower])

    def _zero_crossing_engine(self, signal, length, fs):
        """Returns the reciprocal intervals between successive downward zero crossings."""
        head = signal[: length - 1]
        tail = signal[1:length]
        edges = np.flatnonzero((head > 0.0) & (tail <= 0.0)) + 1
        if edges.shape[0] < 2:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        before = signal[edges - 1]
        fine_edges = edges - before / (signal[edges] - before)
        intervals = fs / np.diff(fine_edges)
        locations = (fine_edges[:-1] + fine_edges[1:]) / 2.0 / fs
        return locations, intervals

    def _four_zero_crossing_intervals(self, signal, length, fs):
        negative = self._zero_crossing_engine(signal, length, fs)
        inverted = -signal
        positive = self._zero_crossing_engine(inverted, length, fs)
        differentiated = inverted[: length - 1] - inverted[1:length]
        peak = self._zero_crossing_engine(differentiated, length - 1, fs)
        dip = self._zero_crossing_engine(-differentiated, length - 1, fs)
        return negative, positive, peak, dip

    def _filtered_signal(self, boundary_f0, fft_size, fs, spectrum, y_length):
        """Convolves the analysed spectrum with a Nuttall-windowed band-pass filter."""
        filter_length_half = self._matlab_round(fs / boundary_f0 * 2.0)
        band_pass_filter = np.zeros(fft_size, dtype=np.float64)
        taps = self._nuttall_window(filter_length_half * 2 + 1)
        lags = np.arange(-filter_length_half, filter_length_half + 1, dtype=np.float64)
        band_pass_filter[: filter_length_half * 2 + 1] = taps * np.cos(2 * _PI * boundary_f0 * lags / fs)

        filter_spectrum = np.fft.rfft(band_pass_filter)
        signal = np.fft.irfft(spectrum * filter_spectrum, n=fft_size) * fft_size
        index_bias = filter_length_half + 1
        return signal[index_bias : index_bias + y_length]

    def _f0_candidate_contour(self, crossings, boundary_f0, temporal_positions):
        f0_length = temporal_positions.shape[0]
        if any(locations.shape[0] < 3 for locations, _ in crossings):
            return np.zeros(f0_length, dtype=np.float64)

        candidate = np.zeros(f0_length, dtype=np.float64)
        for locations, intervals in crossings:
            candidate += self._interp1(locations, intervals, temporal_positions)
        candidate /= 4.0

        rejected = (
            (candidate > boundary_f0 * 1.1)
            | (candidate < boundary_f0 * 0.9)
            | (candidate > self.f0_ceil)
            | (candidate < self.f0_floor)
        )
        candidate[rejected] = 0.0
        return candidate

    def _raw_f0_candidates(self, boundary_f0_list, actual_fs, y_length, temporal_positions, spectrum, fft_size):
        raw = np.empty((boundary_f0_list.shape[0], temporal_positions.shape[0]), dtype=np.float64)
        for band, boundary_f0 in enumerate(boundary_f0_list):
            signal = self._filtered_signal(boundary_f0, fft_size, actual_fs, spectrum, y_length)
            crossings = self._four_zero_crossing_intervals(signal, y_length, actual_fs)
            raw[band] = self._f0_candidate_contour(crossings, boundary_f0, temporal_positions)
        return raw

    def _detect_official_f0_candidates(self, raw, max_candidates):
        """Averages each band-contiguous voiced run of the per-channel candidates into one candidate."""
        number_of_channels, f0_length = raw.shape
        candidates = np.zeros((f0_length, max_candidates), dtype=np.float64)
        number_of_candidates = 0

        voiced = raw > 0
        voiced[0] = False
        voiced[number_of_channels - 1] = False
        transitions = voiced[1:].astype(np.int8) - voiced[:-1].astype(np.int8)

        for frame in range(f0_length):
            column = transitions[:, frame]
            starts = np.flatnonzero(column == 1) + 1
            ends = np.flatnonzero(column == -1) + 1
            sections = min(starts.shape[0], ends.shape[0])
            count = 0
            for section in range(sections):
                start, end = starts[section], ends[section]
                if end - start < 10:
                    continue
                candidates[frame, count] = raw[start:end, frame].mean()
                count += 1
            number_of_candidates = max(number_of_candidates, count)

        return candidates, number_of_candidates

    def _overlap_f0_candidates(self, candidates, number_of_candidates):
        """Spreads each frame's candidates onto the three frames either side of it."""
        f0_length = candidates.shape[0]
        n = 3
        for shift in range(1, n + 1):
            block = slice(number_of_candidates * shift, number_of_candidates * (shift + 1))
            candidates[shift:, block] = candidates[: f0_length - shift, :number_of_candidates]
            block = slice(number_of_candidates * (shift + n), number_of_candidates * (shift + n + 1))
            candidates[: f0_length - shift, block] = candidates[shift:, :number_of_candidates]
        return candidates

    def _refine_f0_batch(self, x, fs, positions, f0s, half_window_length, fft_size):
        """Refines one batch of candidates sharing a window length, by instantaneous frequency."""
        width = half_window_length * 2 + 1
        window_length_in_time = width / fs

        # One zero column either side lets the interior difference formula also produce the two
        # endpoint values the upstream loop special-cases.
        main_window = np.zeros((f0s.shape[0], width + 2), dtype=np.float64)
        offsets = np.arange(width, dtype=np.int64)

        basic_index = self._matlab_round_array((positions - half_window_length / fs) * fs + 0.001)
        base_index = basic_index[:, None] + offsets[None, :]
        elapsed = (base_index - 1.0) / fs - positions[:, None]
        window = main_window[:, 1 : width + 1]
        np.cos(2.0 * _PI * elapsed / window_length_in_time, out=window)
        window *= 0.5
        window += 0.42
        window += 0.08 * np.cos(4.0 * _PI * elapsed / window_length_in_time)
        diff_window = -(main_window[:, 2:] - main_window[:, :width]) / 2.0

        samples = x[np.clip(base_index - 1, 0, x.shape[0] - 1)]

        padded = np.zeros((f0s.shape[0], fft_size), dtype=np.float64)
        np.multiply(samples, window, out=padded[:, :width])
        main_spectrum = np.fft.rfft(padded, axis=-1)
        np.multiply(samples, diff_window, out=padded[:, :width])
        diff_spectrum = np.fft.rfft(padded, axis=-1)

        numerator_i = main_spectrum.real * diff_spectrum.imag - main_spectrum.imag * diff_spectrum.real
        power_spectrum = main_spectrum.real**2 + main_spectrum.imag**2

        harmonics = np.arange(1, 7, dtype=np.float64)
        counts = np.minimum((fs / 2.0 / f0s).astype(np.int64), 6)
        used = harmonics[None, :] <= counts[:, None]
        index = self._matlab_round_array(f0s[:, None] * fft_size / fs * harmonics[None, :])
        np.clip(index, 0, fft_size // 2, out=index)

        rows = np.arange(f0s.shape[0], dtype=np.int64)[:, None]
        power = power_spectrum[rows, index]
        silent = power == 0.0
        instantaneous = np.where(
            silent,
            0.0,
            index * fs / fft_size + numerator_i[rows, index] / np.where(silent, 1.0, power) * fs / 2.0 / _PI,
        )
        amplitude = np.sqrt(power)

        numerator = np.sum(np.where(used, amplitude * instantaneous, 0.0), axis=-1)
        denominator = np.sum(np.where(used, amplitude * harmonics[None, :], 0.0), axis=-1)
        deviation = np.sum(
            np.where(used, np.abs((instantaneous / harmonics[None, :] - f0s[:, None]) / f0s[:, None]), 0.0), axis=-1
        )

        refined = numerator / (denominator + _SAFE_GUARD_MINIMUM)
        score = 1.0 / (deviation / counts + _SAFE_GUARD_MINIMUM)
        rejected = (refined < self.f0_floor) | (refined > self.f0_ceil) | (score < 2.5)
        refined[rejected] = 0.0
        score[rejected] = 0.0
        return refined, score

    def _refine_f0_candidates(self, x, fs, temporal_positions, candidates):
        scores = np.zeros_like(candidates)
        frame, column = np.nonzero(candidates > 0.0)
        if frame.shape[0] == 0:
            return candidates, scores

        f0s = candidates[frame, column]
        positions = temporal_positions[frame]
        half_window_lengths = (1.5 * fs / f0s + 1.0).astype(np.int64)

        for half_window_length in np.unique(half_window_lengths):
            selected = np.flatnonzero(half_window_lengths == half_window_length)
            width = int(half_window_length) * 2 + 1
            fft_size = int(pow(2.0, 2.0 + int(math.log(width) / _LOG2)))
            chunk = max(1, 2**25 // fft_size)
            for start in range(0, selected.shape[0], chunk):
                rows = selected[start : start + chunk]
                refined, score = self._refine_f0_batch(
                    x, fs, positions[rows], f0s[rows], int(half_window_length), fft_size
                )
                candidates[frame[rows], column[rows]] = refined
                scores[frame[rows], column[rows]] = score

        return candidates, scores

    def _remove_unreliable_candidates(self, candidates, scores):
        """Zeroes a candidate that no neighbouring frame corroborates within five percent."""
        f0_length = candidates.shape[0]
        reference = candidates.copy()
        threshold = 0.05

        for start in range(1, f0_length - 1, 2048):
            stop = min(start + 2048, f0_length - 1)
            centre = reference[start:stop, :, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                forward = np.abs(centre - reference[start + 1 : stop + 1, None, :]) / centre
                backward = np.abs(centre - reference[start - 1 : stop - 1, None, :]) / centre
            error = np.minimum(
                np.minimum(forward.min(axis=-1), 1.0), np.minimum(backward.min(axis=-1), 1.0)
            )
            unreliable = (reference[start:stop] != 0.0) & (error > threshold)
            candidates[start:stop][unreliable] = 0.0
            scores[start:stop][unreliable] = 0.0
        return candidates, scores

    def _search_f0_base(self, candidates, scores):
        best = np.argmax(scores, axis=-1)
        rows = np.arange(candidates.shape[0], dtype=np.int64)
        contour = candidates[rows, best]
        contour[scores[rows, best] <= 0.0] = 0.0
        return contour

    def _fix_step_1(self, f0_base, allowed_range):
        f0_step1 = np.zeros_like(f0_base)
        body = f0_base[2:]
        with np.errstate(divide="ignore", invalid="ignore"):
            reference = f0_base[1:-1] * 2 - f0_base[:-2]
            jumped = np.abs((body - reference) / reference) > allowed_range
            stepped = np.abs(body - f0_base[1:-1]) / f0_base[1:-1] > allowed_range
        f0_step1[2:] = np.where(jumped & stepped, 0.0, body)
        return f0_step1

    def _boundary_list(self, f0):
        voiced = f0 > 0
        voiced[0] = False
        voiced[-1] = False
        edges = np.flatnonzero(voiced[1:] != voiced[:-1]) + 1
        return edges - np.arange(edges.shape[0], dtype=np.int64) % 2

    def _fix_step_2(self, f0_step1, voice_range_minimum):
        f0_step2 = f0_step1.copy()
        boundaries = self._boundary_list(f0_step1)
        for start, end in zip(boundaries[::2], boundaries[1::2]):
            if end - start < voice_range_minimum:
                f0_step2[start : end + 1] = 0.0
        return f0_step2

    def _multi_channel_f0(self, f0, boundaries):
        sections = boundaries.shape[0] // 2
        channels = np.zeros((sections, f0.shape[0]), dtype=np.float64)
        for section in range(sections):
            start, end = boundaries[section * 2], boundaries[section * 2 + 1]
            channels[section, start : end + 1] = f0[start : end + 1]
        return channels

    def _select_best_f0(self, reference_f0, candidates, allowed_range):
        best_f0 = 0.0
        best_error = allowed_range
        for candidate in candidates:
            error = abs(reference_f0 - candidate) / reference_f0
            if error > best_error:
                continue
            best_f0 = candidate
            best_error = error
        return best_f0, best_error

    def _extend_f0(self, origin, last_point, shift, candidates, allowed_range, extended_f0):
        threshold = 4
        tmp_f0 = extended_f0[origin]
        shifted_origin = origin
        count = 0
        for step in range(abs(last_point - origin) + 1):
            target = origin + shift * step + shift
            extended_f0[target], _ = self._select_best_f0(tmp_f0, candidates[target], allowed_range)
            if extended_f0[target] == 0.0:
                count += 1
            else:
                tmp_f0 = extended_f0[target]
                count = 0
                shifted_origin = target
            if count == threshold:
                break
        return shifted_origin

    def _extend(self, channels, f0_length, boundaries, candidates, allowed_range):
        """Grows each voiced section outwards along the candidates that continue its contour."""
        threshold = 100
        sections = channels.shape[0]
        for section in range(sections):
            boundaries[section * 2 + 1] = self._extend_f0(
                boundaries[section * 2 + 1],
                min(f0_length - 2, boundaries[section * 2 + 1] + threshold),
                1,
                candidates,
                allowed_range,
                channels[section],
            )
            boundaries[section * 2] = self._extend_f0(
                boundaries[section * 2],
                max(1, boundaries[section * 2] - threshold),
                -1,
                candidates,
                allowed_range,
                channels[section],
            )

        # The running mean is not reset between sections, matching the upstream accumulation.
        running_mean = 0.0
        count = 0
        for section in range(sections):
            start, end = boundaries[section * 2], boundaries[section * 2 + 1]
            running_mean += channels[section, start:end].sum()
            running_mean /= end - start
            if 2200.0 / running_mean < end - start:
                channels[[count, section]] = channels[[section, count]]
                boundaries[[count * 2, section * 2]] = boundaries[[section * 2, count * 2]]
                boundaries[[count * 2 + 1, section * 2 + 1]] = boundaries[[section * 2 + 1, count * 2 + 1]]
                count += 1
        return count

    def _make_sorted_order(self, boundaries, sections):
        """Reproduces the upstream ordering pass, which compares against the displaced element."""
        order = list(range(sections))
        for i in range(1, sections):
            for j in range(i - 1, -1, -1):
                if boundaries[order[j] * 2] > boundaries[order[i] * 2]:
                    order[i], order[j] = order[j], order[i]
                else:
                    break
        return order

    def _search_score(self, f0, candidates, scores):
        matched = candidates == f0
        if not matched.any():
            return 0.0
        return max(0.0, float(scores[matched].max()))

    def _merge_f0_sub(self, merged, start1, end1, channel, start2, end2, candidates, scores):
        if start1 <= start2 and end1 >= end2:
            return end1

        span = slice(start2, end1 + 1)
        score1 = sum(
            self._search_score(merged[i], candidates[i], scores[i]) for i in range(start2, end1 + 1)
        )
        score2 = sum(
            self._search_score(channel[i], candidates[i], scores[i]) for i in range(start2, end1 + 1)
        )
        del span
        if score1 > score2:
            merged[end1 : end2 + 1] = channel[end1 : end2 + 1]
        else:
            merged[start2 : end2 + 1] = channel[start2 : end2 + 1]
        return end2

    def _merge_f0(self, channels, boundaries, sections, candidates, scores):
        order = self._make_sorted_order(boundaries, sections)
        merged = channels[0].copy()

        for i in range(1, sections):
            current = order[i]
            if boundaries[current * 2] - boundaries[1] > 0:
                start, end = boundaries[current * 2], boundaries[current * 2 + 1]
                merged[start : end + 1] = channels[current, start : end + 1]
                boundaries[0] = start
                boundaries[1] = end
            else:
                boundaries[1] = self._merge_f0_sub(
                    merged,
                    boundaries[0],
                    boundaries[1],
                    channels[current],
                    boundaries[current * 2],
                    boundaries[current * 2 + 1],
                    candidates,
                    scores,
                )
        return merged

    def _fix_step_3(self, f0_step2, candidates, scores, allowed_range):
        f0_step3 = f0_step2.copy()
        f0_length = f0_step2.shape[0]
        boundaries = self._boundary_list(f0_step2)
        if boundaries.shape[0] == 0:
            return f0_step3

        channels = self._multi_channel_f0(f0_step2, boundaries)
        sections = self._extend(channels, f0_length, boundaries, candidates, allowed_range)
        if sections != 0:
            f0_step3 = self._merge_f0(channels, boundaries, sections, candidates, scores)
        return f0_step3

    def _fix_step_4(self, f0_step3, threshold):
        f0_step4 = f0_step3.copy()
        boundaries = self._boundary_list(f0_step3)
        for section in range(boundaries.shape[0] // 2 - 1):
            end = boundaries[section * 2 + 1]
            start = boundaries[(section + 1) * 2]
            distance = start - end - 1
            if distance >= threshold:
                continue
            head = f0_step3[end] + 1
            tail = f0_step3[start] - 1
            coefficient = (tail - head) / (distance + 1.0)
            steps = np.arange(1, start - end, dtype=np.float64)
            f0_step4[end + 1 : start] = head + coefficient * steps
        return f0_step4

    def _fix_f0_contour(self, candidates, scores):
        contour = self._search_f0_base(candidates, scores)
        contour = self._fix_step_1(contour, 0.008)
        contour = self._fix_step_2(contour, 6)
        contour = self._fix_step_3(contour, candidates, scores, 0.18)
        return self._fix_step_4(contour, 9)

    def _filtering_f0(self, x, start, end):
        x = x.copy()
        x[:start] = x[start]
        x[end + 1 :] = x[end]
        a = _SMOOTHING_A
        b = (_SMOOTHING_B[0], _SMOOTHING_B[1], _SMOOTHING_B[0])
        forward = self._direct_form_2(x, a, b)[::-1].copy()
        return self._direct_form_2(forward, a, b)[::-1].copy()

    def _smooth_f0_contour(self, f0):
        lag = 300
        f0_length = f0.shape[0]
        contour = np.zeros(f0_length + lag * 2, dtype=np.float64)
        contour[lag : lag + f0_length] = f0

        boundaries = self._boundary_list(contour)
        channels = self._multi_channel_f0(contour, boundaries)

        smoothed = np.zeros(f0_length, dtype=np.float64)
        for section in range(boundaries.shape[0] // 2):
            start, end = boundaries[section * 2], boundaries[section * 2 + 1]
            filtered = self._filtering_f0(channels[section], start, end)
            smoothed[start - lag : end + 1 - lag] = filtered[start : end + 1]
        return smoothed

    def _samples_for_harvest(self, fs, x_length, frame_period):
        return int(1000.0 * x_length / fs / frame_period) + 1

    def _waveform_and_spectrum(self, x, y_length, fft_size, decimation_ratio):
        y = np.zeros(fft_size, dtype=np.float64)
        if decimation_ratio == 1:
            y[: x.shape[0]] = x
        else:
            # The decimated waveform is noisy at both ends, so the input is extended first.
            lag = int(math.ceil(140.0 / decimation_ratio) * decimation_ratio)
            extended = np.empty(x.shape[0] + lag * 2, dtype=np.float64)
            extended[:lag] = x[0]
            extended[lag : lag + x.shape[0]] = x
            extended[lag + x.shape[0] :] = x[-1]
            decimated = self._decimate(extended, decimation_ratio)
            y[:y_length] = decimated[lag // decimation_ratio : lag // decimation_ratio + y_length]

        y[:y_length] -= y[:y_length].mean()
        y[y_length:] = 0.0
        return y, np.fft.rfft(y)

    def _harvest_general_body(self, x, fs, frame_period, channels_in_octave, speed):
        adjusted_f0_floor = self.f0_floor * 0.9
        adjusted_f0_ceil = self.f0_ceil * 1.1
        number_of_channels = 1 + int(math.log(adjusted_f0_ceil / adjusted_f0_floor) / _LOG2 * channels_in_octave)
        boundary_f0_list = adjusted_f0_floor * np.power(
            2.0, np.arange(1, number_of_channels + 1, dtype=np.float64) / channels_in_octave
        )

        x_length = x.shape[0]
        decimation_ratio = max(min(speed, 12), 1)
        y_length = int(math.ceil(x_length / decimation_ratio))
        actual_fs = fs / decimation_ratio
        fft_size = self._suitable_fft_size(y_length + 5 + 2 * int(2.0 * actual_fs / boundary_f0_list[0]))

        y, y_spectrum = self._waveform_and_spectrum(x, y_length, fft_size, decimation_ratio)

        f0_length = self._samples_for_harvest(fs, x_length, frame_period)
        temporal_positions = np.arange(f0_length, dtype=np.float64) * frame_period / 1000.0

        overlap_parameter = 7
        max_candidates = self._matlab_round(number_of_channels / 10.0) * overlap_parameter

        raw = self._raw_f0_candidates(
            boundary_f0_list, actual_fs, y_length, temporal_positions, y_spectrum, fft_size
        )
        candidates, detected = self._detect_official_f0_candidates(raw, max_candidates)
        candidates = self._overlap_f0_candidates(candidates, detected)
        number_of_candidates = detected * overlap_parameter

        candidates = candidates[:, :number_of_candidates]
        candidates, scores = self._refine_f0_candidates(
            y[:y_length], actual_fs, temporal_positions, candidates
        )
        candidates, scores = self._remove_unreliable_candidates(candidates, scores)

        contour = self._fix_f0_contour(candidates, scores)
        return temporal_positions, self._smooth_f0_contour(contour)

    def _fix_f0_stonemask(self, power_spectrum, numerator_i, fft_size, fs, f0s, number_of_harmonics):
        harmonics = np.arange(1, number_of_harmonics + 1, dtype=np.float64)
        index = self._matlab_round_array(f0s[:, None] * fft_size / fs * harmonics[None, :])
        np.clip(index, 0, fft_size // 2, out=index)

        rows = np.arange(f0s.shape[0], dtype=np.int64)[:, None]
        power = power_spectrum[rows, index]
        safe_power = np.where(power == 0.0, 1.0, power)
        instantaneous = np.where(
            power == 0.0,
            0.0,
            index * fs / fft_size + numerator_i[rows, index] / safe_power * fs / 2.0 / _PI,
        )
        amplitude = np.sqrt(power)
        numerator = np.sum(amplitude * instantaneous, axis=-1)
        denominator = np.sum(amplitude * harmonics[None, :], axis=-1)
        return numerator / (denominator + _SAFE_GUARD_MINIMUM)

    def _stonemask_batch(self, x, fs, positions, f0s, half_window_length, fft_size):
        width = half_window_length * 2 + 1
        window_length_in_time = width / fs

        offsets = np.arange(width, dtype=np.int64)
        base_time = (offsets[None, :] - half_window_length) / fs
        index_raw = self._matlab_round_array((positions[:, None] + base_time) * fs)

        main_window = np.zeros((f0s.shape[0], width + 2), dtype=np.float64)
        elapsed = (index_raw - 1.0) / fs - positions[:, None]
        window = main_window[:, 1 : width + 1]
        np.cos(2.0 * _PI * elapsed / window_length_in_time, out=window)
        window *= 0.5
        window += 0.42
        window += 0.08 * np.cos(4.0 * _PI * elapsed / window_length_in_time)
        diff_window = -(main_window[:, 2:] - main_window[:, :width]) / 2.0

        samples = x[np.clip(index_raw - 1, 0, x.shape[0] - 1)]
        padded = np.zeros((f0s.shape[0], fft_size), dtype=np.float64)
        np.multiply(samples, window, out=padded[:, :width])
        main_spectrum = np.fft.rfft(padded, axis=-1)
        np.multiply(samples, diff_window, out=padded[:, :width])
        diff_spectrum = np.fft.rfft(padded, axis=-1)

        numerator_i = main_spectrum.real * diff_spectrum.imag - main_spectrum.imag * diff_spectrum.real
        power_spectrum = main_spectrum.real**2 + main_spectrum.imag**2

        tentative = self._fix_f0_stonemask(power_spectrum, numerator_i, fft_size, fs, f0s, 2)
        accepted = (tentative > 0.0) & (tentative <= f0s * 2)
        refined = np.zeros_like(f0s)
        if accepted.any():
            rows = np.flatnonzero(accepted)
            refined[rows] = self._fix_f0_stonemask(
                power_spectrum[rows], numerator_i[rows], fft_size, fs, tentative[rows], 6
            )

        # A correction beyond twenty percent is rejected in favour of the initial estimate.
        return np.where(np.abs(refined - f0s) > f0s * 0.2, f0s, refined)

    def _design_low_cut_filter(self, taps, fft_size):
        filter_ = np.zeros(fft_size, dtype=np.float64)
        positions = np.arange(1, taps + 1, dtype=np.float64)
        filter_[:taps] = 0.5 - 0.5 * np.cos(positions * 2.0 * _PI / (taps + 1))
        filter_[:taps] = -filter_[:taps] / filter_[:taps].sum()
        half = (taps - 1) // 2
        filter_[fft_size - half :] = filter_[:half]
        filter_[:taps] = filter_[half : half + taps].copy()
        filter_[0] += 1.0
        return filter_

    def _spectrum_for_estimation(self, x, y_length, actual_fs, fft_size, decimation_ratio):
        y = np.zeros(fft_size, dtype=np.float64)
        if decimation_ratio != 1:
            decimated = self._decimate(x, decimation_ratio)
            y[: decimated.shape[0]] = decimated
        else:
            y[: x.shape[0]] = x

        y[:y_length] -= y[:y_length].mean()
        y[y_length:] = 0.0
        spectrum = np.fft.rfft(y)

        cutoff_in_sample = self._matlab_round(actual_fs / _CUT_OFF)
        low_cut_filter = self._design_low_cut_filter(cutoff_in_sample * 2 + 1, fft_size)
        return spectrum * np.fft.rfft(low_cut_filter)

    def _dio_filtered_signal(self, half_average_length, fft_size, spectrum, y_length):
        """Convolves the spectrum with a Nuttall low-pass whose cutoff follows its own length."""
        low_pass_filter = np.zeros(fft_size, dtype=np.float64)
        low_pass_filter[: half_average_length * 4] = self._nuttall_window(half_average_length * 4)
        signal = np.fft.irfft(spectrum * np.fft.rfft(low_pass_filter), n=fft_size) * fft_size
        index_bias = half_average_length * 2
        return signal[index_bias : index_bias + y_length]

    def _dio_f0_candidate_contour(self, crossings, boundary_f0, temporal_positions):
        f0_length = temporal_positions.shape[0]
        if any(locations.shape[0] < 3 for locations, _ in crossings):
            return np.zeros(f0_length, dtype=np.float64), np.full(f0_length, _MAXIMUM_VALUE)

        interpolated = np.stack(
            [self._interp1(locations, intervals, temporal_positions) for locations, intervals in crossings]
        )
        candidate = interpolated.mean(axis=0)
        score = np.sqrt(((interpolated - candidate) ** 2).sum(axis=0) / 3.0)

        rejected = (
            (candidate > boundary_f0)
            | (candidate < boundary_f0 / 2.0)
            | (candidate > self.f0_ceil)
            | (candidate < self.f0_floor)
        )
        candidate[rejected] = 0.0
        score[rejected] = _MAXIMUM_VALUE
        return candidate, score

    def _dio_candidates_and_scores(
        self,
        boundary_f0_list, actual_fs, y_length, temporal_positions, spectrum, fft_size
    ):
        bands = boundary_f0_list.shape[0]
        candidates = np.empty((bands, temporal_positions.shape[0]), dtype=np.float64)
        scores = np.empty_like(candidates)
        for band, boundary_f0 in enumerate(boundary_f0_list):
            signal = self._dio_filtered_signal(
                self._matlab_round(actual_fs / boundary_f0 / 2.0), fft_size, spectrum, y_length
            )
            crossings = self._four_zero_crossing_intervals(signal, y_length, actual_fs)
            candidate, score = self._dio_f0_candidate_contour(
                crossings, boundary_f0, temporal_positions
            )
            candidates[band] = candidate
            scores[band] = score / (candidate + _SAFE_GUARD_MINIMUM)
        return candidates, scores

    def _dio_best_f0_contour(self, candidates, scores):
        return candidates[np.argmin(scores, axis=0), np.arange(candidates.shape[1])]

    def _dio_fix_step_1(self, best_f0_contour, voice_range_minimum, allowed_range):
        f0_length = best_f0_contour.shape[0]
        f0_base = np.zeros(f0_length, dtype=np.float64)
        f0_base[voice_range_minimum : f0_length - voice_range_minimum] = best_f0_contour[
            voice_range_minimum : f0_length - voice_range_minimum
        ]

        f0_step1 = np.zeros(f0_length, dtype=np.float64)
        body = f0_base[voice_range_minimum:]
        previous = f0_base[voice_range_minimum - 1 : -1]
        within = np.abs((body - previous) / (_SAFE_GUARD_MINIMUM + body)) < allowed_range
        f0_step1[voice_range_minimum:] = np.where(within, body, 0.0)
        return f0_step1

    def _dio_fix_step_2(self, f0_step1, voice_range_minimum):
        f0_step2 = f0_step1.copy()
        centre = (voice_range_minimum - 1) // 2
        f0_length = f0_step1.shape[0]
        if f0_length - centre <= centre:
            return f0_step2
        windows = np.lib.stride_tricks.sliding_window_view(f0_step1, voice_range_minimum)
        silent = (windows == 0.0).any(axis=-1)
        f0_step2[centre : f0_length - centre][silent] = 0.0
        return f0_step2

    def _voiced_section_edges(self, f0):
        onset = (f0[1:] != 0.0) & (f0[:-1] == 0.0)
        offset = (f0[1:] == 0.0) & (f0[:-1] != 0.0)
        return np.flatnonzero(onset) + 1, np.flatnonzero(offset)

    def _dio_select_best_f0(self, current_f0, past_f0, candidates, target_index, allowed_range):
        reference_f0 = (current_f0 * 3.0 - past_f0) / 2.0
        column = candidates[:, target_index]
        best_f0 = float(column[np.argmin(np.abs(reference_f0 - column))])
        if abs(1.0 - best_f0 / reference_f0) > allowed_range:
            return 0.0
        return best_f0

    def _dio_fix_step_3(self, f0_step2, candidates, allowed_range, offsets):
        f0_length = f0_step2.shape[0]
        f0_step3 = f0_step2.copy()
        for index, start in enumerate(offsets):
            limit = f0_length - 1 if index == offsets.shape[0] - 1 else offsets[index + 1]
            for j in range(start, limit):
                f0_step3[j + 1] = self._dio_select_best_f0(
                    f0_step3[j], f0_step3[j - 1], candidates, j + 1, allowed_range
                )
                if f0_step3[j + 1] == 0.0:
                    break
        return f0_step3

    def _dio_fix_step_4(self, f0_step3, candidates, allowed_range, onsets):
        f0_length = f0_step3.shape[0]
        # One trailing zero stands in for the read one past the contour the upstream loop makes.
        f0_step4 = np.zeros(f0_length + 1, dtype=np.float64)
        f0_step4[:f0_length] = f0_step3
        for index in range(onsets.shape[0] - 1, -1, -1):
            limit = 1 if index == 0 else onsets[index - 1]
            for j in range(onsets[index], limit, -1):
                f0_step4[j - 1] = self._dio_select_best_f0(
                    f0_step4[j], f0_step4[j + 1], candidates, j - 1, allowed_range
                )
                if f0_step4[j - 1] == 0.0:
                    break
        return f0_step4[:f0_length]

    def _dio_fix_f0_contour(self, frame_period, candidates, best_f0_contour, allowed_range):
        f0_length = best_f0_contour.shape[0]
        voice_range_minimum = int(0.5 + 1000.0 / frame_period / self.f0_floor) * 2 + 1
        if f0_length <= voice_range_minimum:
            return np.zeros(f0_length, dtype=np.float64)

        contour = self._dio_fix_step_1(best_f0_contour, voice_range_minimum, allowed_range)
        contour = self._dio_fix_step_2(contour, voice_range_minimum)
        onsets, offsets = self._voiced_section_edges(contour)
        contour = self._dio_fix_step_3(contour, candidates, allowed_range, offsets)
        return self._dio_fix_step_4(contour, candidates, allowed_range, onsets)


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
        self._text_normalizers = {}

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
    def english_normalizer(self):
        """
        Returns:
            `wetext.Normalizer` or `None`: The normalizer of the English branch, or `None` when
            `wetext` is not installed.
        """
        if "en" not in self._text_normalizers:
            self._text_normalizers["en"] = load_text_normalizer()
        return self._text_normalizers["en"]

    @property
    def chinese_normalizer(self):
        """
        Returns:
            `wetext.Normalizer` or `None`: The normalizer of the Chinese branch, built the way
            upstream builds it, or `None` when `wetext` is not installed.
        """
        if "zh" not in self._text_normalizers:
            self._text_normalizers["zh"] = load_text_normalizer(remove_erhua=False)
        return self._text_normalizers["zh"]

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

        Each branch opens with the `wetext` text normalizer, which rewrites a date, a currency
        amount, a unit and an abbreviation, and is skipped when `wetext` is not installed. A Chinese
        sentence then loses the spaces that do not sit inside an embedded English word, has its
        corner marks spelled out, its brackets removed, its full stops and dashes replaced by their
        Chinese counterparts and a trailing run of commas turned into a full stop. Any other
        sentence has its remaining digit runs read out by [`number_to_words`]. Text carrying a
        `<|` `|>` marker is returned untouched.

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
            if self.chinese_normalizer is None:
                warn_without_text_normalizer(text)
            else:
                text = self.chinese_normalizer.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "\u3002")
            text = text.replace(" - ", "\uff0c")
            text = remove_bracket(text)
            text = re.sub(r"[\uff0c,\u3001]+$", "\u3002", text)
            pieces = split_paragraph(text, tokenize, "zh", token_max_n=80, token_min_n=60, merge_len=20)
        else:
            text = normalize_english(text, self.english_normalizer, self.number_speller)
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
        """
        target_rate = self.feature_extractor.mel_sampling_rate
        hop_length = self.feature_extractor.hop_length
        waveform = self._resample(
            audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio)),
            sampling_rate,
            target_rate,
        )
        samples = waveform.squeeze(0).numpy().astype("double")
        frame_period = hop_length * 1000 / target_rate
        estimator = CosyVoiceV1WorldEstimator(target_rate)
        contour, time_axis = estimator.harvest(samples, frame_period=frame_period)
        if (contour != 0).sum() < 5:
            contour, time_axis = estimator.dio(samples, frame_period=frame_period)
        contour = estimator.stonemask(samples, time_axis, contour)
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
    "CosyVoiceV1NumberSpeller",
    "CosyVoiceV1Processor",
    "contains_chinese",
    "is_only_punctuation",
    "load_text_normalizer",
    "normalize_english",
    "number_to_words",
    "remove_bracket",
    "replace_blank",
    "replace_corner_mark",
    "spell_out_number",
    "spell_out_two_digits",
    "split_paragraph",
    "warn_without_text_normalizer",
]
