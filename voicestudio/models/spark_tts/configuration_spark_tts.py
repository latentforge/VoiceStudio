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
"""Configuration class for Spark-TTS."""

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class SparkTTSConfig(Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`SparkTTSForConditionalGeneration`]. It is used
    to instantiate a Spark-TTS model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B).

    Spark-TTS is a [`Qwen2Model`] whose vocabulary is extended with the BiCodec semantic and global tokens and with
    the task and style control tokens, so this configuration is a [`Qwen2Config`] plus the handful of settings the
    processor needs to prepare reference audio. It inherits every argument of [`Qwen2Config`]; `vocab_size` covers
    ordinary text tokens as well as the added ones. The audio tokenizer is a model of its own, configured by
    [`SparkTTSBiCodecConfig`] and loaded by [`SparkTTSProcessor`].

    Args:
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sample rate, in Hz, of the waveform produced by the audio tokenizer.
        ref_segment_duration (`float`, *optional*, defaults to 6.0):
            Duration, in seconds, of the reference clip the speaker encoder derives global tokens from.
        volume_normalize (`bool`, *optional*, defaults to `True`):
            Whether reference audio is volume-normalized before it is encoded.

    Example:

    ```python
    >>> from voicestudio.models.spark_tts import SparkTTSConfig, SparkTTSForConditionalGeneration

    >>> configuration = SparkTTSConfig()

    >>> model = SparkTTSForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "spark_tts"

    def __init__(
        self,
        sampling_rate: int = 16000,
        ref_segment_duration: float = 6.0,
        volume_normalize: bool = True,
        vocab_size: int = 166000,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        rope_parameters: dict | None = None,
        tie_word_embeddings: bool = True,
        max_window_layers: int = 21,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.ref_segment_duration = ref_segment_duration
        self.volume_normalize = volume_normalize

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rope_parameters=rope_parameters or {"rope_type": "default", "rope_theta": 1000000.0},
            tie_word_embeddings=tie_word_embeddings,
            max_window_layers=max_window_layers,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


__all__ = ["SparkTTSConfig"]
