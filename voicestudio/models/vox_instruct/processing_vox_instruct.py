# coding=utf-8
# Copyright 2024 The VoxInstruct Authors and the HuggingFace Inc. team. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Processor class for VoxInstruct."""

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


class VoxInstructProcessor(ProcessorMixin):
    r"""
    Constructs a VoxInstruct processor, wrapping an mT5 tokenizer and a [`VoxInstructFeatureExtractor`] into a single
    processor.

    It lays out the flat token sequence both stages read. Every sequence is
    `<bos> <language> <semantic...> <eos> <acoustic...> <eos + 1>`, with the padding token at zero, the language ids
    next, then the semantic tokens and then the first EnCodec codebook, and it carries the segment ids that tell the
    semantic span from the acoustic one.

    Args:
        feature_extractor ([`VoxInstructFeatureExtractor`]):
            Reads a speech prompt at the two sampling rates the tokenizers need.
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The mT5 tokenizer of the instruction text.
        max_text_len (`int`, *optional*, defaults to 512):
            Length the instruction is padded or truncated to.
        max_len (`int`, *optional*, defaults to 2048):
            Length the flat token sequence is truncated to.
        num_language_ids (`int`, *optional*, defaults to 2):
            Number of language identity tokens.
        semantic_vocab_size (`int`, *optional*, defaults to 500):
            Number of semantic tokens.
        acoustic_vocab_size (`int`, *optional*, defaults to 1024):
            Size of a single EnCodec codebook.
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of EnCodec codebooks.
        language_mapping (`dict[str, int]`, *optional*):
            Maps a language name onto its identity index. Defaults to `{"en": 0, "zh": 1}`.
    """

    attributes = ["feature_extractor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        max_text_len: int = 512,
        max_len: int = 2048,
        num_language_ids: int = 2,
        semantic_vocab_size: int = 500,
        acoustic_vocab_size: int = 1024,
        num_codebooks: int = 8,
        language_mapping: dict[str, int] | None = None,
        **kwargs,
    ):
        self.max_text_len = max_text_len
        self.max_len = max_len
        self.num_language_ids = num_language_ids
        self.semantic_vocab_size = semantic_vocab_size
        self.acoustic_vocab_size = acoustic_vocab_size
        self.num_codebooks = num_codebooks
        self.language_mapping = language_mapping if language_mapping is not None else {"en": 0, "zh": 1}
        super().__init__(feature_extractor, tokenizer, **kwargs)

    @property
    def semantic_token_offset(self) -> int:
        """Token id of the first semantic token."""
        return 1 + self.num_language_ids

    @property
    def acoustic_token_offset(self) -> int:
        """Token id of the first acoustic token."""
        return 1 + self.num_language_ids + self.semantic_vocab_size

    @property
    def bos_token_id(self) -> int:
        """Token id opening every sequence."""
        return 1 + self.num_language_ids + self.semantic_vocab_size + self.acoustic_vocab_size

    @property
    def semantic_eos_token_id(self) -> int:
        """Token id closing the semantic span."""
        return self.bos_token_id + 1

    @property
    def acoustic_eos_token_id(self) -> int:
        """Token id closing the acoustic span."""
        return self.bos_token_id + 2

    def _encode_text(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes the instructions and pads them to `max_text_len`, keeping the end of sequence token last."""
        input_ids = torch.zeros((len(text), self.max_text_len), dtype=torch.long)
        attention_mask = torch.zeros((len(text), self.max_text_len), dtype=torch.long)
        for index, instruction in enumerate(text):
            ids = self.tokenizer(instruction.strip().capitalize(), return_tensors="pt").input_ids[0]
            if ids.shape[0] >= self.max_text_len:
                ids = ids[: self.max_text_len].clone()
                ids[-1] = self.tokenizer.eos_token_id
            input_ids[index, : ids.shape[0]] = ids
            attention_mask[index, : ids.shape[0]] = 1
        return input_ids, attention_mask

    def _language_id(self, language) -> int:
        """Resolves a language name or index onto its identity index."""
        if isinstance(language, str):
            return self.language_mapping[language]
        return int(language)

    def _build_sequence(self, language_id, semantic_ids, acoustic_ids):
        """Lays out one flat sequence, its per codebook grid and its segment ids."""
        codebooks = self.num_codebooks
        language = np.asarray([language_id + 1])
        opening = np.asarray([self.bos_token_id])
        semantic_end = np.asarray([self.semantic_eos_token_id])
        acoustic_end = np.asarray([self.acoustic_eos_token_id])

        acoustic = np.asarray(acoustic_ids) + self.acoustic_token_offset
        if semantic_ids is None:
            semantic = np.zeros((0,), dtype=np.int64)
        else:
            semantic = np.asarray(semantic_ids) + self.semantic_token_offset

        flat = np.concatenate([opening, language, semantic, semantic_end, acoustic[:, 0], acoustic_end])
        grid = np.concatenate(
            [
                np.stack([opening] * codebooks, axis=1),
                np.stack([language] * codebooks, axis=1),
                np.stack([semantic] * codebooks, axis=1),
                np.stack([semantic_end] * codebooks, axis=1),
                acoustic,
                np.stack([acoustic_end] * codebooks, axis=1),
            ]
        )
        segments = np.asarray([1] * (semantic.shape[0] + 3) + [2] * (acoustic.shape[0] + 1))
        return flat[: self.max_len], grid[: self.max_len], segments[: self.max_len]

    def _build_prompt(self, language_id, semantic_ids):
        """Lays out the prompt the autoregressive stage is primed with."""
        codebooks = self.num_codebooks
        language = np.asarray([language_id + 1])
        opening = np.asarray([self.bos_token_id])
        if semantic_ids is None:
            semantic = np.zeros((0,), dtype=np.int64)
        else:
            semantic = np.asarray(semantic_ids) + self.semantic_token_offset

        flat = np.concatenate([opening, language, semantic])
        segments = np.ones((flat.shape[0],), dtype=np.int64)
        return flat, segments, codebooks

    def __call__(
        self,
        text: str | list[str] | None = None,
        language: str | int | list | None = None,
        semantic_ids=None,
        acoustic_ids=None,
        audio=None,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        r"""
        Args:
            text (`str` or `list[str]`, *optional*):
                Free form instructions. A transcript quoted inside the instruction is what makes the model speak it.
            language (`str`, `int` or `list`, *optional*):
                Language name present in `language_mapping`, or its index. Defaults to the first index.
            semantic_ids (`list` or `np.ndarray`, *optional*):
                Semantic tokens of the target speech, one sequence of shape `(num_frames,)` per sample. Passing them
                together with `acoustic_ids` produces a training batch.
            acoustic_ids (`list` or `np.ndarray`, *optional*):
                EnCodec codes of the target speech, one array of shape `(num_frames, num_codebooks)` per sample.
            audio (`np.ndarray`, `torch.Tensor` or list of them, *optional*):
                Speech prompt whose voice is carried over. It is tokenized by the model, not here.
            sampling_rate (`int`, *optional*):
                Sampling rate of `audio`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                Only `"pt"` is supported.

        Returns:
            [`~feature_extraction_utils.BatchFeature`] carrying `text_input_ids` and `text_attention_mask`,
            `language_ids`, and either the training tensors `input_ids`, `segment_ids`, `attention_mask`, `labels`,
            `nar_input_ids` and `nar_labels`, or the speech prompt tensors `input_values`, `padding_mask` and
            `semantic_input_values`.

        Raises:
            ValueError: If `text` is missing, or if only one of `semantic_ids` and `acoustic_ids` is given.
        """
        if text is None:
            raise ValueError("VoxInstruct is conditioned on an instruction, so `text` is required.")
        if isinstance(text, str):
            text = [text]
        if acoustic_ids is None and semantic_ids is not None:
            raise ValueError("`semantic_ids` describes a target that also needs `acoustic_ids`.")

        batch_size = len(text)
        if language is None:
            language = [0] * batch_size
        elif isinstance(language, (str, int, np.integer)):
            language = [language] * batch_size
        language_ids = [self._language_id(item) for item in language]

        text_input_ids, text_attention_mask = self._encode_text(text)
        data = {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "language_ids": torch.tensor(language_ids, dtype=torch.long),
        }

        if acoustic_ids is not None:
            if semantic_ids is None:
                semantic_ids = [None] * batch_size
            sequences = [
                self._build_sequence(language_ids[index], semantic_ids[index], acoustic_ids[index])
                for index in range(batch_size)
            ]
            length = max(flat.shape[0] for flat, _, _ in sequences)
            input_ids = torch.zeros((batch_size, length), dtype=torch.long)
            grid = torch.zeros((batch_size, length, self.num_codebooks), dtype=torch.long)
            segment_ids = torch.zeros((batch_size, length), dtype=torch.long)
            attention_mask = torch.zeros((batch_size, length), dtype=torch.long)
            for index, (flat, codes, segments) in enumerate(sequences):
                span = flat.shape[0]
                input_ids[index, :span] = torch.from_numpy(flat)
                grid[index, :span] = torch.from_numpy(codes)
                segment_ids[index, :span] = torch.from_numpy(segments)
                attention_mask[index, :span] = 1
            data.update(
                {
                    "input_ids": input_ids,
                    "segment_ids": segment_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.masked_fill(attention_mask == 0, -100),
                    "nar_input_ids": grid,
                    "nar_labels": grid,
                }
            )
        elif audio is None:
            prompts = [self._build_prompt(language_ids[index], None) for index in range(batch_size)]
            data["input_ids"] = torch.tensor(np.stack([flat for flat, _, _ in prompts]), dtype=torch.long)
            data["segment_ids"] = torch.tensor(np.stack([seg for _, seg, _ in prompts]), dtype=torch.long)

        if audio is not None:
            data.update(self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt"))

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """Forwards to the tokenizer's [`~PreTrainedTokenizer.batch_decode`]."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forwards to the tokenizer's [`~PreTrainedTokenizer.decode`]."""
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["VoxInstructProcessor"]
