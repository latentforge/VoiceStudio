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

from typing import Any

from transformers.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig as _Qwen3TTSConfig,
)
from transformers.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)


def _standardize_rope_parameters(value: Any) -> Any:
    r"""
    Rewrite every mapping that carries `rope_scaling`/`rope_theta` into one carrying `rope_parameters`.

    Args:
        value (`Any`):
            A configuration mapping, or any value nested inside one. Mappings are rewritten at every depth and
            every other value is returned unchanged.

    Returns:
        `Any`: The same value, with `rope_scaling` and `rope_theta` replaced by the equivalent `rope_parameters`
        in each nested mapping that carries either of them and does not already carry `rope_parameters`.
    """
    if not isinstance(value, dict):
        return value

    # `rope_parameters` is the rewrite target, so the `rope_theta` inside it is already in the 5.0 spelling.
    value = {
        key: nested if key == "rope_parameters" else _standardize_rope_parameters(nested)
        for key, nested in value.items()
    }
    if "rope_parameters" in value or not value.keys() & {"rope_scaling", "rope_theta"}:
        return value

    rope_scaling = value.pop("rope_scaling", None)
    rope_theta = value.pop("rope_theta", None)
    rope_parameters = dict(rope_scaling) if rope_scaling is not None else {}
    if rope_theta is not None:
        rope_parameters["rope_theta"] = rope_theta
    if not rope_parameters:
        return value

    rope_parameters.setdefault("rope_type", "default")
    if "mrope_section" in rope_parameters:
        # mRoPE here gives every section the same position ids, which makes the interleaved and the
        # non-interleaved layout numerically equal, and `Qwen3TTSTalkerAttention` implements the latter.
        rope_parameters["interleaved"] = False
    value["rope_parameters"] = rope_parameters
    return value


class Qwen3TTSConfig(_Qwen3TTSConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSForConditionalGeneration`]. It reads
    the `rope_scaling` and `rope_theta` keys that the published `Qwen/Qwen3-TTS-*` checkpoints record, at every
    depth of the configuration, as the single `rope_parameters` mapping the classes it inherits from expect.
    Without that the talker falls back to a rope base of 500000 and to even mRoPE sections, and the code predictor
    falls back to a rope base of 500000, none of which is what the checkpoints were trained with.

    It takes every argument of the [`Qwen3TTSConfig`] it inherits from.
    """

    def __init__(self, **kwargs):
        super().__init__(**_standardize_rope_parameters(kwargs))


__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "Qwen3TTSTalkerConfig",
]
