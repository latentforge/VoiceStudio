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
"""PyTorch Qwen3-TTS model."""

from transformers.conversion_mapping import (
    Concatenate,
    PrefixChange,
    WeightConverter,
    WeightRenaming,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSBasePreTrainedModel,
    Qwen3TTSPreTrainedModel,
    Qwen3TTSTalkerCodePredictorModel,
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    Qwen3TTSTalkerModel,
    Qwen3TTSTalkerTextPreTrainedModel,
)
from transformers.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration as _Qwen3TTSForConditionalGeneration,
)

from .configuration_qwen3_tts import Qwen3TTSConfig


# The published `Qwen/Qwen3-TTS-*` checkpoints keep the talker under a `talker.` prefix, name its codec
# embedding table `codec_embedding` and its text projection `linear_fc1`/`linear_fc2`, and store the code
# predictor's output head as one `nn.Linear` per residual codebook rather than the single fused
# `nn.Linear` that `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` declares.
register_checkpoint_conversion_mapping(
    "Qwen3TTSForConditionalGeneration",
    [
        PrefixChange(prefix_to_remove="talker"),
        WeightRenaming(source_patterns=r"^model\.codec_embedding\.", target_patterns=r"model.embed_tokens."),
        WeightRenaming(
            source_patterns=r"^text_projection\.linear_fc1\.", target_patterns=r"text_projection.linear_1."
        ),
        WeightRenaming(
            source_patterns=r"^text_projection\.linear_fc2\.", target_patterns=r"text_projection.linear_2."
        ),
        WeightConverter(
            source_patterns=r"code_predictor.lm_head.*.weight",
            target_patterns=r"code_predictor.lm_head.weight",
            operations=[Concatenate(dim=0)],
        ),
    ],
    overwrite=True,
)


class Qwen3TTSForConditionalGeneration(_Qwen3TTSForConditionalGeneration):
    config: Qwen3TTSConfig


__all__ = [
    "Qwen3TTSBasePreTrainedModel",
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerCodePredictorModel",
    "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSTalkerTextPreTrainedModel",
]
