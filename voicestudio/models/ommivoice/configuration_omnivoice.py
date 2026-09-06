# Copyright 2026 Xiaomi Corporation and the LatentForge team. All rights reserved.
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
"""Configuration class for OmniVoice."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig


class OmniVoiceConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`OmniVoiceForConditionalGeneration`]. It is
    used to instantiate an OmniVoice model according to the specified arguments, defining a language model backbone
    (a [`Qwen3Model`] in the released checkpoint) paired with a fused multi-codebook audio embedding and output
    head. Audio frames are predicted by iterative unmasking rather than autoregressively, so the backbone attends
    bidirectionally and never uses a cache.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) architecture.

    Args:
        audio_vocab_size (`int`, *optional*, defaults to 1025):
            Size of each codebook's vocabulary, including the mask id appended after the codec's own entries.
        audio_mask_id (`int`, *optional*, defaults to 1024):
            Id, within a codebook's vocabulary, of the mask token that marks a frame as not yet decoded.
        num_audio_codebook (`int`, *optional*, defaults to 8):
            Number of residual codebooks the audio tokenizer produces for one audio frame.
        audio_codebook_weights (`list[float]`, *optional*, defaults to `[8, 8, 6, 6, 4, 4, 2, 2]`):
            Per-codebook weights of the training loss. They are normalized to sum to one before being applied.
        llm_config ([`PreTrainedConfig`] or `dict`, *optional*):
            Configuration of the language model backbone. Defaults to a [`Qwen3Config`].
        audio_tokenizer_id (`str`, *optional*, defaults to `"eustlb/higgs-audio-v2-tokenizer"`):
            Repository id of the [`HiggsAudioV2TokenizerModel`] that [`OmniVoiceProcessor`] falls back to when a
            checkpoint does not bundle its own `audio_tokenizer` subfolder.

    Example:

    ```python
    >>> from voicestudio.models.ommivoice import OmniVoiceConfig, OmniVoiceForConditionalGeneration

    >>> configuration = OmniVoiceConfig()

    >>> model = OmniVoiceForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        audio_codebook_weights: list[float] | None = None,
        llm_config: PreTrainedConfig | dict | None = None,
        audio_tokenizer_id: str = "eustlb/higgs-audio-v2-tokenizer",
        **kwargs,
    ):
        if isinstance(llm_config, dict):
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        elif llm_config is None:
            llm_config = CONFIG_MAPPING["qwen3"]()
        self.llm_config = llm_config

        self.audio_vocab_size = audio_vocab_size
        self.audio_mask_id = audio_mask_id
        self.num_audio_codebook = num_audio_codebook
        if audio_codebook_weights is None:
            audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]
        if len(audio_codebook_weights) != num_audio_codebook:
            raise ValueError(
                f"`audio_codebook_weights` holds {len(audio_codebook_weights)} weights but `num_audio_codebook` "
                f"is {num_audio_codebook}; one weight per codebook is required."
            )
        self.audio_codebook_weights = audio_codebook_weights
        self.audio_tokenizer_id = audio_tokenizer_id

        super().__init__(**kwargs)

    def get_text_config(self, decoder=None, encoder=None) -> PreTrainedConfig:
        return self.llm_config

    @property
    def num_codebooks(self) -> int:
        return self.num_audio_codebook

    @property
    def codebook_size(self) -> int:
        return self.audio_vocab_size

    @property
    def hidden_size(self) -> int:
        return self.llm_config.hidden_size


__all__ = ["OmniVoiceConfig"]
