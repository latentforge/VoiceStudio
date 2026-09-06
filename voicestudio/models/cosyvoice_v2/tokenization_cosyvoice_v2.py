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
"""Tokenization class for CosyVoice v2."""

from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer


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


class CosyVoiceV2Tokenizer(Qwen2Tokenizer):
    r"""
    Constructs a CosyVoice v2 tokenizer, the Qwen2 tokenizer of the released `CosyVoice-BlankEN`
    directory carrying upstream's added vocabulary: the chat markers, the end of prompt marker and
    the paralinguistic tags a caller writes inline.

    The end of text token is both the end of sequence and the padding token, as upstream sets them.

    Args:
        kwargs:
            Forwarded to [`Qwen2Tokenizer`].
    """

    added_special_tokens = SPECIAL_TOKENS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_special_tokens(
            {
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": list(self.added_special_tokens),
            }
        )


__all__ = ["SPECIAL_TOKENS", "CosyVoiceV2Tokenizer"]
