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
"""Checkpoint conversion for CosyVoice v3."""

from pathlib import Path

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from ..cosyvoice_v2.weight_conversion import TEXT_MODEL_SUBDIR
from .configuration_cosyvoice_v3 import CosyVoiceV3Config


PUBLISHED_CHECKPOINTS = ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512",)

# The recipe every released v3 directory ships and the text model configuration [`build_config`] reads, which
# name the revision the conversion is keyed on the way v2's do.
RELEASED_CONFIG_FILES = ("cosyvoice3.yaml", f"{TEXT_MODEL_SUBDIR}/config.json")

SPEECH_TOKENIZER_FILE = "speech_tokenizer_v3.onnx"


def build_config(directory: "str | Path", **overrides) -> CosyVoiceV3Config:
    r"""
    Builds the [`CosyVoiceV3Config`] of a released CosyVoice v3 directory.

    Args:
        directory (`str` or `os.PathLike`):
            Local directory of the released checkpoint.
        overrides (`dict`, *optional*):
            Configuration fields overriding the released geometry.

    Returns:
        [`CosyVoiceV3Config`]: The configuration.
    """
    text_config = Qwen2Config.from_pretrained(str(Path(directory) / TEXT_MODEL_SUBDIR))
    return CosyVoiceV3Config(text_config=text_config, **overrides)


__all__ = [
    "PUBLISHED_CHECKPOINTS",
    "RELEASED_CONFIG_FILES",
    "SPEECH_TOKENIZER_FILE",
    "build_config",
]
