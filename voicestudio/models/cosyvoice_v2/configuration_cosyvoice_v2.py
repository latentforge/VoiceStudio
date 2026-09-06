# coding=utf-8
# Copyright 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du) and the HuggingFace Inc. team. All rights reserved.
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
"""Configuration class for CosyVoice v2."""

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from ..cosyvoice_v1.configuration_cosyvoice_v1 import CosyVoiceV1Config


class CosyVoiceV2Config(CosyVoiceV1Config):
    r"""
    This is the configuration class to store the configuration of a
    [`CosyVoiceV2ForConditionalGeneration`]. CosyVoice v2 keeps the three network layout of
    [`CosyVoiceV1Config`] but replaces the language model with a pretrained Qwen2 decoder, upsamples
    the flow matching encoder by `token_mel_ratio` instead of interpolating it, and makes the flow
    matching estimator and the encoder chunk aware so that they can run on a stream.

    Instantiating a configuration with the defaults yields the geometry of the released
    `FunAudioLLM/CosyVoice2-0.5B` checkpoint.

    Args:
        text_config (`Qwen2Config` or `dict`, *optional*):
            Configuration of the Qwen2 decoder that carries the language model. Defaults to the
            geometry of the `CosyVoice-BlankEN` directory shipped with the checkpoint.
        num_speech_special_tokens (`int`, *optional*, defaults to 3):
            Number of ids appended to the speech vocabulary. The first is the end of speech token and
            the third is the fill token that separates the text and speech groups of a bistream
            sequence.
        mix_ratio (`list[int]`, *optional*, defaults to `[5, 15]`):
            Number of text tokens and of speech tokens in one group of a bistream training sequence.
        token_mel_ratio (`int`, *optional*, defaults to 2):
            Number of mel frames produced per speech token.
        pre_lookahead_len (`int`, *optional*, defaults to 3):
            Number of future speech tokens the flow matching encoder is allowed to look at.
        pre_lookahead_channels (`int`, *optional*, defaults to 512):
            Inner channels of the lookahead convolution of the flow matching encoder.
        flow_encoder_up_num_layers (`int`, *optional*, defaults to 4):
            Number of flow matching encoder layers that run after the upsampling layer.
        estimator_static_chunk_size (`int`, *optional*, defaults to 50):
            Static chunk size of the flow matching estimator attention mask while streaming, in mel
            frames.
        estimator_num_decoding_left_chunks (`int`, *optional*, defaults to -1):
            Number of past chunks the estimator may attend to while streaming. A negative value
            allows all of them.
        noise_length (`int`, *optional*, defaults to 15000):
            Length in mel frames of the fixed noise the Euler solver starts from.
        noise_seed (`int`, *optional*, defaults to 0):
            Seed the fixed noise is drawn with.
        speech_tokenizer_fsmn_kernel_size (`int`, *optional*, defaults to 31):
            Kernel size of the depthwise memory convolution the speech tokenizer attention adds to
            its value projection.
        speech_tokenizer_rope_theta (`float`, *optional*, defaults to 10000.0):
            Base of the rotary embedding of the speech tokenizer attention.
        speech_tokenizer_max_position_embeddings (`int`, *optional*, defaults to 2048):
            Longest sequence the rotary embedding of the speech tokenizer is built for.
        speech_tokenizer_fsq_levels (`list[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 3, 3]`):
            Number of levels of each dimension of the finite scalar quantizer the speech tokenizer
            closes with. Their product is the size of the speech vocabulary.
        kwargs:
            Forwarded to [`CosyVoiceV1Config`], whose defaults are overridden to the v2 geometry.
    """

    model_type = "cosyvoice_v2"
    sub_configs = {"text_config": Qwen2Config}

    def __init__(
        self,
        text_config: "Qwen2Config | dict | None" = None,
        num_speech_special_tokens: int = 3,
        mix_ratio: list[int] | None = None,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        pre_lookahead_channels: int = 512,
        flow_encoder_up_num_layers: int = 4,
        estimator_static_chunk_size: int = 50,
        estimator_num_decoding_left_chunks: int = -1,
        noise_length: int = 15000,
        noise_seed: int = 0,
        speech_tokenizer_fsmn_kernel_size: int = 31,
        speech_tokenizer_rope_theta: float = 10000.0,
        speech_tokenizer_max_position_embeddings: int = 2048,
        speech_tokenizer_fsq_levels: list[int] | None = None,
        **kwargs,
    ):
        defaults = {
            "sample_rate": 24000,
            "speech_vocab_size": 6561,
            "lm_hidden_size": 896,
            "flow_input_size": 512,
            "flow_encoder_hidden_size": 512,
            "flow_encoder_num_heads": 8,
            "flow_encoder_ffn_dim": 2048,
            "flow_encoder_num_layers": 6,
            "flow_encoder_attention_dropout": 0.1,
            "flow_encoder_chunk_size": 25,
            "flow_input_frame_rate": 25,
            "estimator_channels": [256],
            "vocoder_upsample_rates": [8, 5, 3],
            "vocoder_upsample_kernel_sizes": [16, 11, 7],
            "vocoder_source_resblock_kernel_sizes": [7, 7, 11],
            "vocoder_source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "vocoder_mel_loss_n_fft": 1920,
            "vocoder_mel_loss_hop_length": 480,
            "vocoder_mel_loss_win_length": 1920,
            "speech_tokenizer_conv_stride": 2,
            "speech_tokenizer_max_source_positions": None,
        }
        for name, value in defaults.items():
            kwargs.setdefault(name, value)

        if text_config is None:
            text_config = Qwen2Config(
                vocab_size=151936,
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=24,
                num_attention_heads=14,
                num_key_value_heads=2,
                hidden_act="silu",
                max_position_embeddings=32768,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                tie_word_embeddings=True,
                bos_token_id=151643,
                eos_token_id=151645,
            )
        elif isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        self.text_config = text_config

        self.num_speech_special_tokens = num_speech_special_tokens
        self.mix_ratio = [5, 15] if mix_ratio is None else mix_ratio
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_channels = pre_lookahead_channels
        self.flow_encoder_up_num_layers = flow_encoder_up_num_layers
        self.estimator_static_chunk_size = estimator_static_chunk_size
        self.estimator_num_decoding_left_chunks = estimator_num_decoding_left_chunks
        self.noise_length = noise_length
        self.noise_seed = noise_seed
        self.speech_tokenizer_fsmn_kernel_size = speech_tokenizer_fsmn_kernel_size
        self.speech_tokenizer_rope_theta = speech_tokenizer_rope_theta
        self.speech_tokenizer_max_position_embeddings = speech_tokenizer_max_position_embeddings
        self.speech_tokenizer_fsq_levels = (
            [3] * 8 if speech_tokenizer_fsq_levels is None else speech_tokenizer_fsq_levels
        )
        super().__init__(**kwargs)

        # One dtype governs the whole composite. The Qwen2 directory the released checkpoint ships
        # declares bfloat16 in its own config.json, which would otherwise build the language model
        # backbone in bfloat16 while every sibling module stays at the default dtype, and the first
        # matrix multiply across that boundary raises.
        for field in ("dtype", "torch_dtype"):
            if not hasattr(self.text_config, field) or not hasattr(self, field):
                continue
            try:
                setattr(self.text_config, field, getattr(self, field))
            except AttributeError:
                pass

    @property
    def speech_head_size(self) -> int:
        r"""
        Returns:
            `int`: Number of classes the language model head predicts.
        """
        return self.speech_vocab_size + self.num_speech_special_tokens


__all__ = ["CosyVoiceV2Config"]
