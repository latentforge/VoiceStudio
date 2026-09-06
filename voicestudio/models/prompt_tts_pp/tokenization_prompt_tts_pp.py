# Copyright 2024 LY Corporation and the LatentForge team. All rights reserved.
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
"""Tokenization class for PromptTTS++."""

import json
import os

from transformers.tokenization_python import PreTrainedTokenizer
from transformers.utils import logging, requires_backends


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

PAD = "_"
BOS = "^"
EOS = "$"

# The Montreal Forced Aligner English phoneme set, plus its "spn", "sil" and "sp" labels.
PHONEMES = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "spn",
    "sil",
    "sp",
]

SYMBOLS = [PAD, BOS, EOS] + PHONEMES


class PromptTTSPPTokenizer(PreTrainedTokenizer):
    """
    Constructs a PromptTTS++ tokenizer, which turns English text into the phoneme ids of the Montreal Forced
    Aligner symbol set the model was aligned with.

    Args:
        vocab_file (`str`, *optional*):
            Path to a vocabulary file. Defaults to the symbol table of the original implementation.
        bos_token (`str`, *optional*, defaults to `"^"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"$"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"_"`):
            The token used for padding, for example when batching sequences of different lengths.
        phonemize (`bool`, *optional*, defaults to `True`):
            Whether to run grapheme to phoneme conversion, which requires the `g2p_en` backend. Pass `False` to
            tokenize a whitespace separated phoneme sequence directly.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        bos_token: str = BOS,
        eos_token: str = EOS,
        pad_token: str = PAD,
        phonemize: bool = True,
        **kwargs,
    ):
        if vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)
        else:
            self.encoder = {symbol: index for index, symbol in enumerate(SYMBOLS)}
        self.decoder = {index: symbol for symbol, index in self.encoder.items()}
        self.phonemize = phonemize
        self._g2p = None

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            phonemize=phonemize,
            special_tokens_pattern="none",
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    @property
    def g2p(self):
        """The `g2p_en` grapheme to phoneme converter, built on first use."""
        if self._g2p is None:
            requires_backends(self, "g2p_en")
            import g2p_en

            self._g2p = g2p_en.G2p()
        return self._g2p

    def get_vocab(self) -> dict[str, int]:
        """
        Returns:
            `dict[str, int]`: The vocabulary, including the tokens added after instantiation.
        """
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text: str) -> list[str]:
        """
        Args:
            text (`str`):
                Text to phonemize, or a whitespace separated phoneme sequence when the tokenizer was built with
                `phonemize=False`.

        Returns:
            `list[str]`: The phonemes of `text` that the symbol table holds. Sentence punctuation becomes the
            silence phoneme and anything else the symbol table does not hold is dropped, as the original
            implementation does.
        """
        tokens = self.g2p(text) if self.phonemize else text.split()
        tokens = ["sil" if token in [",", "."] else token for token in tokens]
        return [token for token in tokens if token in self.encoder]

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder[self.pad_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.pad_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Args:
            tokens (`list[str]`):
                Phonemes to join.

        Returns:
            `str`: The whitespace separated phoneme sequence.
        """
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Args:
            token_ids_0 (`list[int]`):
                Phoneme ids the special tokens are added to.
            token_ids_1 (`list[int]`, *optional*):
                Unused, PromptTTS++ takes a single sequence.

        Returns:
            `list[int]`: The phoneme ids, surrounded by the beginning and end of sequence tokens.
        """
        if token_ids_1 is not None:
            raise ValueError("PromptTTS++ takes a single phoneme sequence.")
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        """
        Args:
            token_ids_0 (`list[int]`):
                Phoneme ids.
            token_ids_1 (`list[int]`, *optional*):
                Unused, PromptTTS++ takes a single sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether the ids already carry the beginning and end of sequence tokens.

        Returns:
            `list[int]`: A mask that is 1 on the special tokens and 0 on the phonemes.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        return [1] + [0] * len(token_ids_0) + [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        """
        Args:
            save_directory (`str`):
                Directory to save the vocabulary to.
            filename_prefix (`str`, *optional*):
                Prefix of the vocabulary file name.

        Returns:
            `tuple[str]`: Path of the saved vocabulary file.

        Raises:
            OSError: If `save_directory` is not a directory.
        """
        if not os.path.isdir(save_directory):
            raise OSError(f"Vocabulary path ({save_directory}) should be a directory")
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
        )
        with open(vocab_file, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.get_vocab(), ensure_ascii=False))
        return (vocab_file,)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_g2p"] = None
        return state


__all__ = ["PromptTTSPPTokenizer"]
