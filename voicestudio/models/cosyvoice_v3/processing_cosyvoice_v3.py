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
"""Processor class for CosyVoice v3."""

import re
from typing import Optional, Union

from ..cosyvoice_v1.processing_cosyvoice_v1 import is_only_punctuation, split_paragraph
from ..cosyvoice_v1.tokenization_cosyvoice_v1 import (
    contains_chinese,
    normalize_english,
    remove_bracket,
    replace_blank,
    replace_corner_mark,
    warn_without_text_normalizer,
)
from ..cosyvoice_v2.processing_cosyvoice_v2 import CosyVoiceV2Processor
from .configuration_cosyvoice_v3 import CosyVoiceV3Config
from .feature_extraction_cosyvoice_v3 import CosyVoiceV3FeatureExtractor
from .tokenization_cosyvoice_v3 import CosyVoiceV3Tokenizer, rewrite_outside_markup
from .weight_conversion import SPEECH_TOKENIZER_FILE


def normalize_english_outside_markup(text: str, english_normalizer, number_speller) -> str:
    """
    Rewrites an English sentence the way upstream's front end reads it out, leaving the markup of the
    added vocabulary alone.

    Args:
        text (`str`):
            Text to rewrite.
        english_normalizer (`wetext.Normalizer` or `None`):
            Normalizer rewriting a date, a currency amount, a unit and an abbreviation. `None` skips
            that step.
        number_speller ([`CosyVoiceV1NumberSpeller`]):
            Engine whose `number_to_words` reads a digit run out.

    Returns:
        `str`: The rewritten text.
    """
    return rewrite_outside_markup(
        text, lambda piece: normalize_english(piece, english_normalizer, number_speller)
    )


class CosyVoiceV3Processor(CosyVoiceV2Processor):
    r"""
    Constructs a CosyVoice v3 processor, which wraps the Qwen2 text tokenizer, the 24 kHz mel
    spectrogram extractor of the flow matching model, the supervised semantic speech tokenizer and
    the speaker encoder into a single object.

    It differs from v2's in the tokenizer's added vocabulary, which gains the end of system marker
    and the ARPAbet and pinyin sets a caller writes inline to override a pronunciation, in the text
    front end, which normalizes and reads a digit run out without disturbing that markup, and in the
    speech
    tokenizer, which is twice as deep and whose weights come out of `speech_tokenizer_v3.onnx`.

    Args:
        feature_extractor ([`CosyVoiceV3FeatureExtractor`]):
            Mel spectrogram extractor of the flow matching model.
        tokenizer ([`Qwen2TokenizerFast`]):
            Text tokenizer, loaded from the `CosyVoice-BlankEN` directory of the released checkpoint.
        speech_token_model_path (`str`, *optional*):
            Path of the `speech_tokenizer_v3.onnx` graph the speech tokenizer is built from.
        speaker_encoder_model_path (`str`, *optional*):
            Path of the CAM++ weights the speaker encoder is built from.
        speaker_info_path (`str`, *optional*):
            Path of a `spk2info.pt`. The released v3 directory ships none.
        kwargs:
            Forwarded to [`CosyVoiceV2Processor`].
    """

    feature_extractor_type = CosyVoiceV3FeatureExtractor
    tokenizer_type = CosyVoiceV3Tokenizer
    model_config_type = CosyVoiceV3Config
    speech_tokenizer_file = SPEECH_TOKENIZER_FILE

    def normalize_text(
        self, text: str, split: bool = True, text_frontend: bool = True
    ) -> Union[str, list[str]]:
        """
        Rewrites a sentence the way upstream's text front end does, then optionally splits it into
        the pieces upstream synthesizes one at a time.

        Each branch opens with the `wetext` text normalizer, which rewrites a date, a currency
        amount, a unit and an abbreviation, and is skipped when `wetext` is not installed. A Chinese
        sentence then loses the spaces that do not sit inside an embedded English word, has its
        corner marks spelled out, its brackets removed, its full stops and dashes replaced by their
        Chinese counterparts and a trailing run of commas turned into a full stop. Any other
        sentence has its remaining digit runs read out by [`number_to_words`], leaving the markup of
        the added vocabulary alone on both counts. Text carrying a `<|` `|>` marker is returned
        untouched.

        Args:
            text (`str`):
                Text to rewrite.
            split (`bool`, *optional*, defaults to `True`):
                Whether the rewritten text is split into pieces.
            text_frontend (`bool`, *optional*, defaults to `True`):
                Whether the front end runs at all. Upstream turns it off to reproduce the samples of
                its demonstration pages.

        Returns:
            `str` or `list[str]`: The rewritten text, or its pieces when `split` is set.
        """
        if text_frontend is False or ("<|" in text and "|>" in text) or text == "":
            return [text] if split else text
        text = text.strip()

        def tokenize(piece: str) -> list[int]:
            return self.tokenizer.encode(piece, add_special_tokens=False)

        if contains_chinese(text):
            if self.chinese_normalizer is None:
                warn_without_text_normalizer(text)
            else:
                text = rewrite_outside_markup(text, self.chinese_normalizer.normalize)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "\u3002")
            text = text.replace(" - ", "\uff0c")
            text = remove_bracket(text)
            text = re.sub(r"[\uff0c,\u3001]+$", "\u3002", text)
            pieces = split_paragraph(text, tokenize, "zh", token_max_n=80, token_min_n=60, merge_len=20)
        else:
            text = normalize_english_outside_markup(text, self.english_normalizer, self.number_speller)
            pieces = split_paragraph(text, tokenize, "en", token_max_n=80, token_min_n=60, merge_len=20)
        pieces = [piece for piece in pieces if not is_only_punctuation(piece)]
        return pieces if split else text


__all__ = ["CosyVoiceV3Processor", "normalize_english_outside_markup"]
