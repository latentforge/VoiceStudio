# Copyright 2026 Boson AI and the LatentForge team. All rights reserved.
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
"""Configuration class for Higgs TTS 3."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class HiggsTTS3AudioEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of the fused multi-codebook audio embedding/head of
    a [`HiggsTTS3ForConditionalGeneration`]. It does not describe a standalone encoder module: Higgs TTS 3
    encodes and decodes waveforms with the transformers-native [`HiggsAudioV2TokenizerModel`], loaded separately
    from `audio_tokenizer_id` on the parent [`HiggsTTS3Config`].

    Args:
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of residual codebooks produced by the audio tokenizer for one audio frame.
        vocab_size (`int`, *optional*, defaults to 1026):
            Size of each codebook's vocabulary, including the two special beginning-of-codebook/end-of-codebook
            ids appended after the codec's own codebook entries.
        out_dim (`int`, *optional*, defaults to 2560):
            Dimensionality of the fused audio embedding and head. Must match `text_config.hidden_size`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the fused audio output head shares its weight with the fused audio input embedding.
    """

    model_type = "higgs_tts3_encoder"

    def __init__(
        self,
        num_codebooks: int = 8,
        vocab_size: int = 1026,
        out_dim: int = 2560,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.out_dim = out_dim
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class HiggsTTS3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HiggsTTS3ForConditionalGeneration`]. It is
    used to instantiate a Higgs TTS 3 model according to the specified arguments, defining a [`Qwen3Model`] text
    backbone paired with a fused multi-codebook audio embedding and head.

    Args:
        text_config ([`Qwen3Config`], *optional*):
            Configuration of the [`Qwen3Model`] backbone.
        audio_encoder_config ([`HiggsTTS3AudioEncoderConfig`], *optional*):
            Configuration of the fused multi-codebook audio embedding/head.
        audio_token_id (`int`, *optional*, defaults to -100):
            Placeholder id marking audio-frame positions in `input_ids`. Positions carrying this id are looked up
            in the fused audio embedding instead of the text embedding.
        audio_stream_bos_id (`int`, *optional*, defaults to 1024):
            Id, within a codebook's vocabulary, of the beginning-of-codebook special used by the delay pattern.
        audio_stream_eos_id (`int`, *optional*, defaults to 1025):
            Id, within a codebook's vocabulary, of the end-of-codebook special used by the delay pattern.
        audio_tokenizer_id (`str`, *optional*, defaults to `"bosonai/higgs-audio-v2-tokenizer"`):
            Repository id of the [`HiggsAudioV2TokenizerModel`] used to encode reference audio to codes and decode
            generated codes back to a waveform.
        sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate, in Hz, of the waveform produced by the audio tokenizer.

    Example:

    ```python
    >>> from voicestudio.models.higgs_tts3 import HiggsTTS3Config, HiggsTTS3ForConditionalGeneration

    >>> configuration = HiggsTTS3Config()

    >>> model = HiggsTTS3ForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "higgs_tts3"
    sub_configs = {"text_config": Qwen3Config, "audio_encoder_config": HiggsTTS3AudioEncoderConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: Qwen3Config | dict | None = None,
        audio_encoder_config: HiggsTTS3AudioEncoderConfig | dict | None = None,
        audio_token_id: int = -100,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        audio_tokenizer_id: str = "bosonai/higgs-audio-v2-tokenizer",
        sample_rate: int = 24_000,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)
        elif text_config is None:
            text_config = Qwen3Config()
        self.text_config = text_config

        if isinstance(audio_encoder_config, dict):
            audio_encoder_config = HiggsTTS3AudioEncoderConfig(**audio_encoder_config)
        elif audio_encoder_config is None:
            audio_encoder_config = HiggsTTS3AudioEncoderConfig()
        self.audio_encoder_config = audio_encoder_config

        self.audio_token_id = audio_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.audio_tokenizer_id = audio_tokenizer_id
        self.sample_rate = sample_rate
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PreTrainedConfig:
        del decoder
        return self.text_config

    @property
    def num_codebooks(self) -> int:
        return self.audio_encoder_config.num_codebooks

    @property
    def codebook_size(self) -> int:
        return self.audio_encoder_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.audio_encoder_config.out_dim


__all__ = ["HiggsTTS3AudioEncoderConfig", "HiggsTTS3Config"]
