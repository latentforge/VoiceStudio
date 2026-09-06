"""Processor class for CosyVoice v3."""

import re
from typing import Callable, Optional, Union

from ..cosyvoice_v1.processing_cosyvoice_v1 import (
    contains_chinese,
    is_only_punctuation,
    normalize_english,
    remove_bracket,
    replace_blank,
    replace_corner_mark,
    split_paragraph,
    warn_without_text_normalizer,
)
from ..cosyvoice_v2.processing_cosyvoice_v2 import (
    SPECIAL_TOKENS as V2_SPECIAL_TOKENS,
    CosyVoiceV2FeatureExtractor,
    CosyVoiceV2Processor,
)
from .configuration_cosyvoice_v3 import CosyVoiceV3Config
from .weight_conversion import SPEECH_TOKENIZER_FILE


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


class CosyVoiceV3FeatureExtractor(CosyVoiceV2FeatureExtractor):
    r"""
    Constructs a CosyVoice v3 feature extractor, which is v2's unchanged: the flow matching model of
    both versions is conditioned on the same 24 kHz, 80 bin log mel spectrogram.

    Args:
        kwargs:
            Forwarded to [`CosyVoiceV2FeatureExtractor`].
    """


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
    model_config_type = CosyVoiceV3Config
    speech_tokenizer_file = SPEECH_TOKENIZER_FILE

    @staticmethod
    def add_special_tokens(tokenizer, tokens: Optional[list[str]] = None) -> int:
        r"""
        Adds upstream's v3 special tokens to a tokenizer, in upstream's order.

        Args:
            tokenizer (`PreTrainedTokenizerBase`):
                Tokenizer to extend.
            tokens (`list[str]`, *optional*):
                Tokens to add. Defaults to [`SPECIAL_TOKENS`].

        Returns:
            `int`: The number of tokens the tokenizer did not already carry.
        """
        return CosyVoiceV2Processor.add_special_tokens(
            tokenizer, tokens=SPECIAL_TOKENS if tokens is None else tokens
        )

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


__all__ = [
    "ADDED_TOKEN_PATTERN",
    "SPECIAL_TOKENS",
    "CosyVoiceV3FeatureExtractor",
    "CosyVoiceV3Processor",
    "normalize_english_outside_markup",
    "rewrite_outside_markup",
]
