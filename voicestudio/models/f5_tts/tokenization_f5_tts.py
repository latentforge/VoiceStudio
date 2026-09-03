"""Tokenization class for F5-TTS."""

import os

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PUNCTUATION_TRANSLATION = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})


def is_chinese(character: str) -> bool:
    r"""
    Args:
        character (`str`):
            Single character to test.

    Returns:
        `bool`: Whether the character falls in the common Chinese range.
    """
    return "㄀" <= character <= "鿿"


def convert_char_to_pinyin(text_list: list[str], polyphone: bool = True) -> list[list[str]]:
    r"""
    Splits every string into the character sequence the model is trained on, replacing runs of Chinese characters
    with their tone numbered pinyin and inserting a space in front of a multi character segment that does not
    already follow one.

    Chinese text needs the `rjieba` word segmenter and the `pypinyin` grapheme to phoneme converter. Without them,
    a string holding no Chinese character is split into its characters, which leaves out the space the segmenter
    would insert in front of a multi character segment following a character other than a space, a colon or a
    quote.

    Args:
        text_list (`list[str]`):
            Strings to convert.
        polyphone (`bool`, *optional*, defaults to `True`):
            Whether runs of Chinese characters are converted with tone sandhi applied.

    Returns:
        `list[list[str]]`: One character list per input string.

    Raises:
        ImportError: If a string holds Chinese characters and `rjieba` or `pypinyin` is not installed.
    """
    try:
        import rjieba
        from pypinyin import Style, lazy_pinyin

        segmenter_available = True
    except ImportError:
        segmenter_available = False

    final_text_list = []
    for text in text_list:
        char_list = []
        text = text.translate(PUNCTUATION_TRANSLATION)

        if not segmenter_available:
            if any(is_chinese(character) for character in text):
                raise ImportError(
                    "Converting Chinese text to pinyin needs the `rjieba` and `pypinyin` packages, which are not "
                    "installed."
                )
            final_text_list.append(list(text))
            continue

        for segment in rjieba.cut(text):
            segment_byte_len = len(bytes(segment, "UTF-8"))
            if segment_byte_len == len(segment):
                if char_list and segment_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(segment)
            elif polyphone and segment_byte_len == 3 * len(segment):
                pinyin = lazy_pinyin(segment, style=Style.TONE3, tone_sandhi=True)
                for index, character in enumerate(segment):
                    if is_chinese(character):
                        char_list.append(" ")
                    char_list.append(pinyin[index])
            else:
                for character in segment:
                    if ord(character) < 256:
                        char_list.extend(character)
                    elif is_chinese(character):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(character, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(character)
        final_text_list.append(char_list)

    return final_text_list


class F5TTSTokenizer(PreTrainedTokenizer):
    r"""
    Constructs an F5-TTS tokenizer. Id `0` is the filler the model's text embedding reserves for positions that
    carry no character, and every line of the vocabulary file takes the id one past its line number. The first line,
    a single space, doubles as the unknown token. Chinese text is replaced by tone numbered pinyin before lookup.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer
    to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file, one token per line.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            Token standing for the filler id `0`. It is not part of the vocabulary file.
        unk_token (`str`, *optional*, defaults to `" "`):
            Token every out of vocabulary character maps onto.
        polyphone (`bool`, *optional*, defaults to `True`):
            Whether runs of Chinese characters are converted with tone sandhi applied.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file: str,
        pad_token: str = "<pad>",
        unk_token: str = " ",
        polyphone: bool = True,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.polyphone = polyphone

        self.encoder = {pad_token: 0}
        with open(vocab_file, "r", encoding="utf-8") as vocab_handle:
            for index, line in enumerate(vocab_handle):
                self.encoder[line[:-1]] = index + 1
        self.decoder = {index: token for token, index in self.encoder.items()}

        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text: str) -> list[str]:
        return convert_char_to_pinyin([text], polyphone=self.polyphone)[0]

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder[str(self.unk_token)])

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, str(self.unk_token))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(token for token in tokens if token != str(self.pad_token))

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        r"""
        Args:
            save_directory (`str`):
                Directory the vocabulary file is written to.
            filename_prefix (`str`, *optional*):
                Prefix prepended to the vocabulary file name.

        Returns:
            `tuple[str]`: Path of the written vocabulary file.

        Raises:
            ValueError: If `save_directory` is not a directory.
        """
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path ({save_directory}) should be a directory.")
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, index in sorted(self.encoder.items(), key=lambda item: item[1]):
                if index > 0:
                    writer.write(token + "\n")
        return (vocab_file,)


__all__ = ["F5TTSTokenizer", "convert_char_to_pinyin"]
