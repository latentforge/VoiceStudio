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
"""Tokenization class for CosyVoice v3."""

import re

from typing import Callable

from ..cosyvoice_v2.tokenization_cosyvoice_v2 import (
    SPECIAL_TOKENS as V2_SPECIAL_TOKENS,
    CosyVoiceV2Tokenizer,
)


# The tokens upstream's `CosyVoice3Tokenizer` adds to the Qwen2 tokenizer, in the order it adds
# them: v2's tokens, then the end of system marker, the ARPAbet phoneme set and the pinyin initial
# and final set, which a caller writes inline to override the pronunciation of a word.
SPECIAL_TOKENS = V2_SPECIAL_TOKENS + [
    "<|endofsystem|>",
    "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
    "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
    "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
    "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
    "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
    "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
    "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
    "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
    "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
    "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
    "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
    "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
    "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
    "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
    "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
    "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
    "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
    "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
    "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
    "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]",
]


# The added vocabulary as it is written inline, longest first so that no token is matched inside a
# longer one.
ADDED_TOKEN_PATTERN = re.compile(
    "|".join(re.escape(token) for token in sorted(SPECIAL_TOKENS, key=len, reverse=True))
)


def rewrite_outside_markup(text: str, rewrite: Callable[[str], str]) -> str:
    """
    Applies a rewrite to a sentence span by span, leaving the spans that hold a token of the added
    vocabulary alone. Their stress digits belong to the token rather than to a number, and the text
    normalizer either asserts on their bracketed shape or spells those digits out.

    Args:
        text (`str`):
            Text to rewrite.
        rewrite (`Callable`):
            Callable rewriting one span. It is not called on a span that is empty or whitespace only.

    Returns:
        `str`: The rewritten text.
    """
    rewritten, position = [], 0

    def span(piece: str) -> str:
        return rewrite(piece) if piece.strip() else piece

    for match in ADDED_TOKEN_PATTERN.finditer(text):
        rewritten.append(span(text[position : match.start()]))
        rewritten.append(match.group())
        position = match.end()
    rewritten.append(span(text[position:]))
    return "".join(rewritten)


class CosyVoiceV3Tokenizer(CosyVoiceV2Tokenizer):
    r"""
    Constructs a CosyVoice v3 tokenizer, which is v2's with the added vocabulary upstream grows:
    the end of system marker, the ARPAbet phoneme set and the pinyin initial and final set, which
    a caller writes inline to override the pronunciation of a word.

    Args:
        kwargs:
            Forwarded to [`CosyVoiceV2Tokenizer`].
    """

    added_special_tokens = SPECIAL_TOKENS


__all__ = [
    "ADDED_TOKEN_PATTERN",
    "SPECIAL_TOKENS",
    "CosyVoiceV3Tokenizer",
    "rewrite_outside_markup",
]
