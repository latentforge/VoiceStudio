# Copyright 2025 SparkAudio, Xinsheng Wang and the LatentForge team. All rights reserved.
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
"""PyTorch Spark-TTS model."""

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2PreTrainedModel
from transformers.utils import auto_docstring

from .configuration_spark_tts import SparkTTSConfig


@auto_docstring
class SparkTTSPreTrainedModel(Qwen2PreTrainedModel):
    config: SparkTTSConfig


@auto_docstring(
    custom_intro="""
    Spark-TTS, a Qwen2 decoder whose vocabulary is extended with BiCodec semantic and global tokens plus task and
    style control tokens, so that speech synthesis, voice cloning and attribute control are all next-token prediction
    over one flat sequence. Reference audio is turned into global tokens and generated semantic tokens are turned
    back into a waveform by [`SparkTTSBiCodecModel`], which [`SparkTTSProcessor`] holds.

    The training objective is the one [`Qwen2ForCausalLM`] computes: cross entropy of the next token over the joint
    text/BiCodec vocabulary, on the positions `labels` leaves unmasked.
    """
)
class SparkTTSForConditionalGeneration(SparkTTSPreTrainedModel, Qwen2ForCausalLM):
    pass


__all__ = ["SparkTTSForConditionalGeneration", "SparkTTSPreTrainedModel"]
