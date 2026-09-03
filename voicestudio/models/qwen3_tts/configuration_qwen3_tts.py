# Copyright 2026 The Qwen team, Alibaba Group and the LatentForge team. All rights reserved.
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
"""Configuration class for Qwen3-TTS."""

from transformers.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig as _Qwen3TTSConfig,
)
from transformers.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)


def _standardize_talker_rope(talker_config):
    """
    Rewrite a talker configuration mapping that carries `rope_scaling`/`rope_theta` into one carrying
    `rope_parameters`.

    Args:
        talker_config (`dict` or [`Qwen3TTSTalkerConfig`], *optional*):
            The `talker_config` argument as given to [`Qwen3TTSConfig`].

    Returns:
        `dict` or [`Qwen3TTSTalkerConfig`]: The same value, with `rope_scaling` and `rope_theta` replaced by the
        equivalent `rope_parameters` when they are present.
    """
    if not isinstance(talker_config, dict) or "rope_scaling" not in talker_config:
        return talker_config

    talker_config = dict(talker_config)
    rope_scaling = talker_config.pop("rope_scaling")
    rope_theta = talker_config.pop("rope_theta", None)
    if rope_scaling is None:
        return talker_config

    rope_parameters = dict(rope_scaling)
    if rope_theta is not None:
        rope_parameters["rope_theta"] = rope_theta
    # The talker gives every mRoPE section the same position ids, which makes the interleaved and the
    # non-interleaved layout numerically equal, and `Qwen3TTSTalkerAttention` implements the latter.
    rope_parameters["interleaved"] = False
    talker_config["rope_parameters"] = rope_parameters
    return talker_config


class Qwen3TTSConfig(_Qwen3TTSConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSForConditionalGeneration`]. It reads
    the `rope_scaling` and `rope_theta` keys that the published `Qwen/Qwen3-TTS-*` checkpoints record for their
    talker, which [`Qwen3TTSTalkerConfig`] expects as a single `rope_parameters` mapping. Without that the talker
    falls back to a rope base of 500000 and to even mRoPE sections, neither of which is what the checkpoint was
    trained with.

    It takes every argument of the [`Qwen3TTSConfig`] it inherits from.
    """

    def __init__(self, talker_config: dict | None = None, **kwargs):
        super().__init__(talker_config=_standardize_talker_rope(talker_config), **kwargs)


__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "Qwen3TTSTalkerConfig",
]
