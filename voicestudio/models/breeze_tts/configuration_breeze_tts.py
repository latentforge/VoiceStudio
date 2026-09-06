# Copyright 2026 RESONIA, INC., Sesame, The HuggingFace Inc. team and the LatentForge team.
# All rights reserved.
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
"""Configuration class for Breeze TTS 2."""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.csm.configuration_csm import CsmConfig, CsmDepthDecoderConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


@strict
class BreezeTTSDepthDecoderConfig(CsmDepthDecoderConfig):
    r"""
    This is the configuration class to store the configuration of a [`BreezeTTSDepthDecoderModel`]. It is used to
    instantiate the Breeze TTS 2 depth decoder, which predicts codebooks `1 .. num_codebooks - 1` of an audio frame
    from the backbone hidden state that produced codebook 0. Instantiating a configuration with the defaults will
    yield the depth decoder of [BreezeBlue/breeze-tts-2](https://huggingface.co/BreezeBlue/breeze-tts-2).

    Args:
        num_codebooks (`int`, *optional*, defaults to 16):
            Number of codebooks the audio tokenizer produces per frame.
        backbone_hidden_size (`int`, *optional*, defaults to 2048):
            Width of the backbone hidden state spliced in at depth position 0.
        vocab_size (`int`, *optional*, defaults to 2051):
            Size of a single codebook vocabulary.
        hidden_size (`int`, *optional*, defaults to 1024):
            Width of the depth decoder representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Width of the gated MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of depth decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads for grouped query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Width of one attention head.
        max_position_embeddings (`int`, *optional*, defaults to 33):
            Longest depth position index the rotary embedding is built for.
        audio_embed_size (`int`, *optional*):
            Width of the codebook token embedding table, and the width the backbone hidden state is projected to
            before it is spliced in at position 0. Defaults to `backbone_hidden_size`, in which case neither
            projection is created.
        codebook_loss_weights (`list[float]`, *optional*):
            Per-codebook weights the released training code spreads evenly over the `num_codebooks - 1` codebooks
            the depth decoder predicts. The released code computes the expanded weight list but never applies it,
            so the depth decoder loss is unweighted.
    """

    model_type = "breeze_tts_depth_decoder"
    base_config_key = "depth_decoder_config"

    num_codebooks: int | None = 16
    backbone_hidden_size: int = 2048
    vocab_size: int = 2051
    hidden_size: int = 1024
    intermediate_size: int = 8192
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int | None = 2
    head_dim: int | None = 128
    max_position_embeddings: int = 33

    audio_embed_size: int | None = None
    codebook_loss_weights: list[float | int] | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.audio_embed_size is None:
            self.audio_embed_size = self.backbone_hidden_size


@strict
class BreezeTTSConfig(CsmConfig):
    r"""
    This is the configuration class to store the configuration of a [`BreezeTTSForConditionalGeneration`]. It is
    used to instantiate a Breeze TTS 2 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a configuration similar to
    [BreezeBlue/breeze-tts-2](https://huggingface.co/BreezeBlue/breeze-tts-2).

    Args:
        num_codebooks (`int`, *optional*, defaults to 16):
            Number of codebooks the audio tokenizer produces per frame.
        vocab_size (`int`, *optional*, defaults to 2051):
            Size of a single codebook vocabulary. The backbone head has one extra class on top of it, used as the
            end-of-audio class of codebook 0.
        text_vocab_size (`int`, *optional*, defaults to 262158):
            Size of the text vocabulary, including `audio_token_id`, `audio_eos_token_id`, the speaker tokens and
            the instruction delimiters appended to the base tokenizer.
        hidden_size (`int`, *optional*, defaults to 2048):
            Width of the backbone representations.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Width of the backbone gated MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of backbone layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of backbone attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of backbone key/value heads for grouped query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Width of one backbone attention head.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Longest frame position the model is used with.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id in the text vocabulary.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of sequence token id in the text vocabulary.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of sequence token id in the text vocabulary.
        audio_token_id (`int`, *optional*, defaults to 262144):
            Text token id whose positions are replaced by embedded codebook frames.
        audio_eos_token_id (`int`, *optional*, defaults to 262145):
            Text token id marking the end of an audio span.
        codebook_pad_token_id (`int`, *optional*, defaults to 2050):
            Padding id inside a codebook vocabulary.
        codebook_eos_token_id (`int`, *optional*, defaults to 0):
            Codebook id every codebook of the end-of-audio frame is embedded with.
        audio_embed_size (`int`, *optional*):
            Width of the codebook token embedding table shared with the depth decoder. Defaults to `hidden_size`,
            in which case no projection to `hidden_size` is created.
        backbone_model_type (`str`, *optional*, defaults to `"qwen3"`):
            `model_type` of the decoder layer stack the backbone is built from.
        backbone_config (`PreTrainedConfig` or `dict`, *optional*):
            Configuration of that decoder layer stack. Defaults to a `backbone_model_type` config carrying the
            backbone dimensions declared above.
        backbone_model_name_or_path (`str`, *optional*):
            Repository id or local path `backbone_config` is loaded from when it is not given inline.
        text_encoder_config (`PreTrainedConfig`, `dict` or `str`, *optional*):
            Configuration of the text encoder that embeds the text spans of the prompt, or the repository id or
            local path it is loaded from. `None` disables the text encoder and embeds `input_ids` with
            `embed_text_tokens` instead.
        text_encoder_proj_type (`str`, *optional*, defaults to `"linear"`):
            How text encoder hidden states are projected to `hidden_size`. One of `"linear"`, `"mlp"` or
            `"breeze_dimfusion"`.
        text_encoder_feature_layer_idx (`int` or `tuple[int]`, *optional*, defaults to -1):
            Which text encoder layers are concatenated into the projected feature. `-1` uses the last hidden state.
        text_encoder_dimfusion_layer_start_idx (`int`, *optional*, defaults to 1):
            First text encoder hidden state fed to the per-backbone-layer DimFusion projections. Index 0 is the
            embedding output.
        text_encoder_dimfusion_layer_end_idx (`int`, *optional*):
            One past the last text encoder hidden state fed to the DimFusion projections. `None` runs to the last.
        text_encoder_dimfusion_fuse_first_layer (`bool`, *optional*, defaults to `False`):
            Whether the first DimFusion projection is concatenated onto the projected text embedding, halving the
            width `text_encoder_proj` outputs.
        text_encoder_bucket_max_length_ratio (`float`, *optional*, defaults to 4.0):
            Longest-to-shortest length ratio allowed inside one padded text encoder batch.
        text_encoder_lora_config (`dict`, *optional*):
            Description of the LoRA adapters the released training run applied to the text encoder. The released
            checkpoint reports them disabled and already merged into the base weights.
        text_encoder_special_tokens_config (`dict`, *optional*):
            Description of the speaker and instruction tokens the released training run appended to the text
            encoder vocabulary. The released checkpoint reports them already merged into the base embedding.
        depth_header_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight applied to the depth decoder loss when it is summed with the backbone loss during training.
        depth_decoder_config (`BreezeTTSDepthDecoderConfig` or `dict`, *optional*):
            Configuration of the depth decoder.
        codec_config (`PreTrainedConfig` or `dict`, *optional*):
            Configuration of the audio tokenizer whose codebooks the model predicts.

    Example:

    ```python
    >>> from voicestudio.models.breeze_tts import BreezeTTSConfig, BreezeTTSForConditionalGeneration

    >>> configuration = BreezeTTSConfig()

    >>> model = BreezeTTSForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "breeze_tts"
    base_config_key = "breeze_tts_config"
    sub_configs = {
        "codec_config": AutoConfig,
        "depth_decoder_config": BreezeTTSDepthDecoderConfig,
        "backbone_config": AutoConfig,
        "text_encoder_config": AutoConfig,
    }

    num_codebooks: int | None = 16
    vocab_size: int = 2051
    text_vocab_size: int = 262158
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    head_dim: int | None = 128
    max_position_embeddings: int = 2048
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 1
    audio_token_id: int | None = 262144
    audio_eos_token_id: int | list[int] | None = 262145
    codebook_pad_token_id: int | None = 2050
    codebook_eos_token_id: int | list[int] | None = 0

    audio_embed_size: int | None = None
    backbone_model_type: str = "qwen3"
    backbone_config: dict | PreTrainedConfig | None = None
    backbone_model_name_or_path: str | None = None
    text_encoder_config: str | dict | PreTrainedConfig | None = None
    text_encoder_proj_type: str = "linear"
    text_encoder_feature_layer_idx: int | tuple[int, ...] | list[int] = -1
    text_encoder_dimfusion_layer_start_idx: int = 1
    text_encoder_dimfusion_layer_end_idx: int | None = None
    text_encoder_dimfusion_fuse_first_layer: bool = False
    text_encoder_bucket_max_length_ratio: float = 4.0
    text_encoder_lora_config: dict | None = None
    text_encoder_special_tokens_config: dict | None = None
    depth_header_loss_weight: float = 1.0

    def __post_init__(self, **kwargs):
        if self.audio_embed_size is None:
            self.audio_embed_size = self.hidden_size

        if self.depth_decoder_config is None:
            self.depth_decoder_config = BreezeTTSDepthDecoderConfig(
                num_codebooks=self.num_codebooks,
                backbone_hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                audio_embed_size=self.audio_embed_size,
            )
        elif isinstance(self.depth_decoder_config, dict):
            self.depth_decoder_config = BreezeTTSDepthDecoderConfig(**self.depth_decoder_config)

        if self.backbone_config is None and self.backbone_model_name_or_path is not None:
            self.backbone_config = AutoConfig.from_pretrained(self.backbone_model_name_or_path)
        if self.backbone_config is None:
            self.backbone_config = AutoConfig.for_model(
                self.backbone_model_type,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
            )
        elif isinstance(self.backbone_config, dict):
            self.backbone_config = AutoConfig.for_model(**self.backbone_config)

        if isinstance(self.text_encoder_config, str):
            self.text_encoder_config = AutoConfig.from_pretrained(self.text_encoder_config)
        elif isinstance(self.text_encoder_config, dict):
            self.text_encoder_config = AutoConfig.for_model(**self.text_encoder_config)

        if isinstance(self.text_encoder_feature_layer_idx, int):
            self.text_encoder_feature_layer_idx = (self.text_encoder_feature_layer_idx,)
        else:
            self.text_encoder_feature_layer_idx = tuple(self.text_encoder_feature_layer_idx)

        super().__post_init__(**kwargs)
        # `tie_codebooks_embeddings` is what the checkpoints carry, `tie_word_embeddings` is what drives the
        # `_tied_weights_keys` machinery; the codebook embedding table is the only tied parameter.
        self.tie_word_embeddings = bool(self.tie_codebooks_embeddings)

    def validate_token_ids(self):
        """
        Validates the special token ids against the vocabulary each of them indexes: `codebook_*_token_id` index a
        single codebook, every other special token id indexes the text vocabulary.
        """
        for name in self:
            value = getattr(self, name)
            if not name.endswith("_token_id") or not isinstance(value, int):
                continue
            vocab_size = self.vocab_size if name.startswith("codebook_") else self.text_vocab_size
            if not 0 <= value < vocab_size:
                logger.warning_once(
                    f"Model config: {name} must be `None` or an integer within the vocabulary (between 0 and "
                    f"{vocab_size}), got {value}. This may result in unexpected behavior."
                )


__all__ = ["BreezeTTSConfig", "BreezeTTSDepthDecoderConfig"]
