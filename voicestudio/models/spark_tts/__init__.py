# Copyright (c) 2025 SparkAudio
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

"""SparkTTS model package."""

from .configuration_spark_tts import SparkTTSConfig
from .modeling_spark_tts import (
    BiCodecModel,
    BiCodecOutput,
    BiCodecPreTrainedModel,
    SparkTTSForConditionalGeneration,
    SparkTTSOutput,
    SparkTTSPreTrainedModel,
)
from .processing_spark_tts import SparkTTSProcessor


__all__ = [
    "SparkTTSConfig",
    "BiCodecModel",
    "BiCodecOutput",
    "BiCodecPreTrainedModel",
    "SparkTTSForConditionalGeneration",
    "SparkTTSOutput",
    "SparkTTSPreTrainedModel",
    "SparkTTSProcessor",
]
