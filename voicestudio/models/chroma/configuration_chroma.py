# coding=utf-8
# Copyright 2025 The FlashLabs team. All rights reserved.
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


from typing import Optional, Dict, Any

from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

# Transformers 4.x.x compatibility: Qwen2_5OmniThinkerConfig may not exist
try:
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    _HAS_QWEN_OMNI = True
except ImportError:
    Qwen2_5OmniThinkerConfig = None
    _HAS_QWEN_OMNI = False

# Transformers 4.x.x compatibility: MimiConfig may not exist
try:
    from transformers.models.mimi.configuration_mimi import MimiConfig
    _HAS_MIMI = True
except ImportError:
    MimiConfig = None
    _HAS_MIMI = False

logger = logging.get_logger(__name__)


class ChromaBackboneConfig(PretrainedConfig):
    model_type = "chroma_backbone"

    def __init__(
        self,
        audio_num_codebooks: Optional[int] = 8,
        vocab_size: Optional[int] = 2051,
        max_position_embeddings: Optional[int] = 2048,
        hidden_size: Optional[int] = 2048,
        intermediate_size: Optional[int] = 8192,
        num_hidden_layers: Optional[int] = 16,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 8,
        hidden_act: Optional[str] = "silu",
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-5,
        use_cache: Optional[bool] = True,
        rope_parameters: Optional[Dict[str, Any]] = None,  # RopeParameters -> Dict
        rope_scaling: Optional[Dict[str, Any]] = None,      # 4.x compatibility
        rope_theta: Optional[float] = 500000.0,
        head_dim: Optional[int] = 64,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        mlp_bias: Optional[bool] = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_num_codebooks = audio_num_codebooks
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        # transformers 4.x compatibility: use rope_scaling instead of rope_parameters
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or rope_parameters


class ChromaDecoderConfig(PretrainedConfig):
    model_type = "chroma_decoder"

    def __init__(
        self,
        audio_num_codebooks: Optional[int] = 8,
        audio_embedding_dim: Optional[int] = 2048,
        vocab_size: Optional[int] = 2051,
        max_position_embeddings: Optional[int] = 33,
        hidden_size: Optional[int] = 1024,
        intermediate_size: Optional[int] = 8192,
        num_hidden_layers: Optional[int] = 4,
        num_attention_heads: Optional[int] = 8,
        num_key_value_heads: Optional[int] = 2,
        hidden_act: Optional[str] = "silu",
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-5,
        use_cache: Optional[bool] = True,
        rope_parameters: Optional[Dict[str, Any]] = None,  # RopeParameters -> Dict
        rope_scaling: Optional[Dict[str, Any]] = None,      # 4.x compatibility
        rope_theta: Optional[float] = 500000.0,
        head_dim: Optional[int] = 128,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_num_codebooks = audio_num_codebooks
        self.audio_embedding_dim = audio_embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim

        # transformers 4.x compatibility: use rope_scaling instead of rope_parameters
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or rope_parameters


class ChromaConfig(PretrainedConfig):
    model_type = "chroma"

    # Transformers 4.x.x compatibility: sub_configs is only supported in 5.x.x
    # Build sub_configs dynamically based on available classes
    sub_configs = {
        "backbone_config": ChromaBackboneConfig,
        "decoder_config": ChromaDecoderConfig
    }
    if _HAS_QWEN_OMNI and Qwen2_5OmniThinkerConfig is not None:
        sub_configs["thinker_config"] = Qwen2_5OmniThinkerConfig
    if _HAS_MIMI and MimiConfig is not None:
        sub_configs["codec_config"] = MimiConfig

    def __init__(
        self,
        thinker_config=None,
        backbone_config=None,
        decoder_config=None,
        codec_config=None,
        codebook_pad_token_id=2050,
        codebook_eos_token_id=0,
        audio_num_codebooks=8,
        text_start_token_id=151665,
        text_end_token_id=151666,
        im_end_token_id=151645,
        audio_frame_freq=1920,
        **kwargs
    ):
        # thinker config (Transformers 4.x.x compatibility)
        if _HAS_QWEN_OMNI and Qwen2_5OmniThinkerConfig is not None:
            if isinstance(thinker_config, dict):
                self.thinker_config = Qwen2_5OmniThinkerConfig(**thinker_config)
            elif isinstance(thinker_config, Qwen2_5OmniThinkerConfig):
                self.thinker_config = thinker_config
            elif thinker_config is None:
                self.thinker_config = Qwen2_5OmniThinkerConfig()
        else:
            # Fallback for Transformers 4.x.x: store as dict or None
            self.thinker_config = thinker_config

        # backbone config
        if isinstance(backbone_config, dict):
            self.backbone_config = ChromaBackboneConfig(**backbone_config)
        elif isinstance(backbone_config, ChromaBackboneConfig):
            self.backbone_config = backbone_config
        elif backbone_config is None:
            self.backbone_config = ChromaBackboneConfig(audio_num_codebooks=audio_num_codebooks)

        # decoder config
        if isinstance(decoder_config, dict):
            self.decoder_config = ChromaDecoderConfig(**decoder_config)
        elif isinstance(decoder_config, ChromaDecoderConfig):
            self.decoder_config = decoder_config
        elif decoder_config is None:
            self.decoder_config = ChromaDecoderConfig(audio_num_codebooks=audio_num_codebooks)

        # codec config (Mimi) - Transformers 4.x.x compatibility
        if _HAS_MIMI and MimiConfig is not None:
            if isinstance(codec_config, dict):
                self.codec_config = MimiConfig(**codec_config)
            elif isinstance(codec_config, MimiConfig):
                self.codec_config = codec_config
            elif codec_config is None:
                self.codec_config = MimiConfig(num_quantizers=audio_num_codebooks, frame_rate=12.5)
        else:
            # Fallback for Transformers 4.x.x: store as dict or None
            self.codec_config = codec_config

        self.audio_num_codebooks = audio_num_codebooks
        self.codebook_pad_token_id = codebook_pad_token_id
        self.codebook_eos_token_id = codebook_eos_token_id
        self.text_start_token_id = text_start_token_id
        self.text_end_token_id = text_end_token_id
        self.im_end_token_id = im_end_token_id
        self.audio_frame_freq = audio_frame_freq
        super().__init__(**kwargs)
