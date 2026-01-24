# Copyright 2024 Boson AI. All rights reserved.
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
#
# Modifications by LatentForge:
# - Implement transformers standard ConditionalGenerationModel interface
"""Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio."""

import torch
import torch.nn as nn
import math
import glob
import functools
import os
from copy import deepcopy
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
import numpy as np
from safetensors.torch import load_file
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaMLP,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.utils import logging, ModelOutput, is_flash_attn_2_available

from .common import HiggsAudioPreTrainedModel
from .utils import (
    merge_input_ids_with_audio_features,
    count_parameters,
    revert_delay_pattern,
)
from .custom_modules import PartiallyFrozenLinear, PartiallyFrozenEmbedding
from .cuda_graph_runner import CUDAGraphRunner
from .audio_head import HiggsAudioDecoderProjector
from ...audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer



try:
    from ..._boson.model.higgs_audio import modeling_higgs_audio
    from ..._boson.model.higgs_audio.modeling_higgs_audio import (
        GenerationMode, HiggsAudioPreTrainedModel, HiggsAudioGenerationOutput
    )
except ImportError:
    from voicestudio._boson.model.higgs_audio import modeling_higgs_audio
    from voicestudio._boson.model.higgs_audio.modeling_higgs_audio import (
        GenerationMode, HiggsAudioPreTrainedModel, HiggsAudioGenerationOutput
    )

from .configuration_higgs_audio import HiggsAudioConfig, HiggsAudioEncoderConfig, HiggsAudioGenerationConfig

_HiggsAudioModel = modeling_higgs_audio.HiggsAudioModel

logger = logging.get_logger(__name__)


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """We implement a dual-path FFN decoder layer where the audio tokens and text tokens go through separate FFN layers.

    The audio and text tokens share the text-attention layer, but will be encoded with separate feedforward layers.
    In addition, the audio tokens can be configured to go through separate attention layer.

    Following is an illustration:

     t    t    t    a   a     a    t    t    t
                        |
                        | (shared attention layer)
                        v
    h_t  h_t  h_t  h_a  h_a  h_a  h_t  h_t  h_t
                        |
                        | (separate text/audio hidden states)
                        v
    [h_t  h_t  h_t  h_t  h_t  h_t], [h_a, h_a, h_a]
             |                             |
             | (separate FFNs)             |
             v                             v
    [o_t  o_t  o_t  o_t  o_t  o_t], [o_a, o_a, o_a]
                        |
                        | (reorder)
                        v
    o_t  o_t  o_t  o_a  o_a  o_a  o_t  o_t  o_t

    This has a few advantages:
    1) We are able to use a smaller FFN, or even bypass the FFN for audio tokens. This accelerates the inference speed.
    2) The Audio-FFN introduces more trainable parameters to the model.
       This should have the same effect as the mixture-of-expert layer and we may expect better performance due to parameter scaling.
    3) We can replace the original FFN in LLMs with the dual-path FFN without changing the number of FLOPs.


    """

    def __init__(
        self, config: HiggsAudioConfig, layer_idx: int, fast_forward: bool = False, use_audio_attention: bool = False
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = layer_idx
        # Use LlamaAttention directly - v4.57.3 compatible
        self.self_attn = LlamaAttention(config=text_config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(text_config)

        if not fast_forward:
            if use_audio_attention:
                # Use LlamaAttention directly - v4.57.3 compatible
                self.audio_attn = LlamaAttention(config=text_config, layer_idx=layer_idx + 1)
                self.audio_post_audio_attn_layer_norm = LlamaRMSNorm(
                    text_config.hidden_size, eps=text_config.rms_norm_eps
                )

            self.audio_mlp = LlamaMLP(text_config)
            self.audio_input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
            self.audio_post_attention_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        self.use_audio_attention = use_audio_attention
        self.fast_forward = fast_forward
        if self.fast_forward:
            assert not self.use_audio_attention, (
                "We cannot use audio_attention if the layer is marked as fast-forward."
            )
        self.input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        fast_forward_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_out_mask: Optional[torch.BoolTensor] = None,
        is_decoding_audio_token: Optional[bool] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        is_using_cuda_graph: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids
                IDs of positions in the input sequence
            audio_out_mask
                Mask for identifying the audio tokens. Size (batch_size, sequence_length)
                1 --> location contains audio_out
                0 --> location does not contain audio_out

                When use_cache is True and not in torch compile mode, the audio_out_mask contains audio_out masks for
                all tokens up to the current token.  That means, it has size (batch_size, sequence_length) while
                hidden_states will have size (batch_size, 1). In the torch compile mode, the audio_out_mask will have
                size (batch_size, 1).
            is_decoding_audio_token
                Used in the torch compile mode to determine if the current token is an audio token or not.
            past_key_value (`Cache`, *optional*): cached past key and value projection states. We fetch the corresponding cached key/value via the layer_idx.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            is_using_cuda_graph (`bool`, *optional*):
                Indicates whether the model is running by cuda graph.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        target_length = hidden_states.shape[1]
        use_static_cache = isinstance(past_key_value, StaticCache)
        decode_stage = hidden_states.shape[1] == 1
        if is_using_cuda_graph:
            assert decode_stage and use_static_cache, (
                "The CUDA graph mode should only be used in the decoding stage with static cache."
            )

        # If we are decoding an audio token and the layer is marked as fast-forward,
        # we can skip it.
        if is_decoding_audio_token and self.fast_forward:
            return (hidden_states,)

        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        audio_out_mask_sq = audio_out_mask

        if self.fast_forward and has_audio_out:
            original_hidden_states = hidden_states.clone()
            min_dtype = torch.finfo(hidden_states.dtype).min
            if attention_mask is None:
                attention_mask = ~audio_out_mask

                if self.self_attn.config._attn_implementation != "flash_attention_2":
                    sequence_length = audio_out_mask.shape[1]
                    attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                        attention_mask=attention_mask,
                        sequence_length=sequence_length,
                        target_length=sequence_length,
                        dtype=hidden_states.dtype,
                        min_dtype=min_dtype,
                        device=hidden_states.device,
                        cache_position=cache_position,
                        batch_size=hidden_states.shape[0],
                    )
                    if use_cache:
                        attention_mask = attention_mask[:, :, -target_length:, :]
            elif len(attention_mask.shape) == 2:
                # Attention mask has shape (batch_size, sequence_length)
                # We should be using flash attention 2
                attention_mask = attention_mask * ~audio_out_mask
            elif len(attention_mask.shape) == 4:
                # When using static cache, the attention mask was already preprocessed in the previous layer
                if use_static_cache:
                    attention_mask = fast_forward_attention_mask
                else:
                    if use_cache:
                        attention_mask = attention_mask.masked_fill(
                            audio_out_mask[:, -target_length:].reshape(audio_out_mask.shape[0], 1, target_length, 1)
                            | audio_out_mask.reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]),
                            min_dtype,
                            )
                    else:
                        attention_mask = attention_mask.masked_fill(
                            audio_out_mask.reshape(audio_out_mask.shape[0], 1, audio_out_mask.shape[1], 1)
                            | audio_out_mask.reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]),
                            min_dtype,
                            )
            else:
                raise NotImplementedError(f"Unsupported attention_mask format, attention_mask={attention_mask}")

            if (
                    self.self_attn.config._attn_implementation == "sdpa"
                    and attention_mask is not None
                    and attention_mask.device.type == "cuda"
                    and not output_attentions
            ):
                # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                attention_mask = AttentionMaskConverter._unmask_unattended(attention_mask, min_dtype)

        if has_audio_out and not self.fast_forward:
            # Apply separate layernorm layers for audio tokens and text tokens
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
            else:
                hidden_states = torch.where(
                    audio_out_mask_sq.unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Audio Attention
        if self.use_audio_attention and has_audio_out:
            if use_static_cache:
                assert audio_attention_mask is not None, (
                    "audio_attention_mask should not be None when using static cache."
                )

            if audio_attention_mask is None:
                no_audio_out_mask = (~audio_out_mask)[:, -target_length:].reshape(
                    audio_out_mask.shape[0], 1, target_length, 1
                ) | (~audio_out_mask).reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1])
                min_dtype = torch.finfo(hidden_states.dtype).min

                if attention_mask is None:
                    audio_attention_mask = audio_out_mask

                    if self.audio_attn.config._attn_implementation != "flash_attention_2":
                        sequence_length = audio_out_mask.shape[1]
                        audio_attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                            attention_mask=audio_attention_mask,
                            sequence_length=sequence_length,
                            target_length=sequence_length,
                            dtype=hidden_states.dtype,
                            min_dtype=min_dtype,
                            device=hidden_states.device,
                            cache_position=cache_position,
                            batch_size=hidden_states.shape[0],
                        )
                        if use_cache:
                            audio_attention_mask = audio_attention_mask[:, :, -target_length:, :]
                        audio_attention_mask = audio_attention_mask.masked_fill(no_audio_out_mask, min_dtype)
                elif len(attention_mask.shape) == 2:
                    # Attention mask has shape (batch_size, sequence_length)
                    audio_attention_mask = attention_mask * audio_out_mask
                elif len(attention_mask.shape) == 4:
                    # Attention mask has shape (batch_size, 1, query_length, key_length)
                    # In addition, the attention mask should be inverted. This means "1" (attend_to) --> "0", and "0" --> minimal dtype value.
                    audio_attention_mask = attention_mask.masked_fill(no_audio_out_mask, min_dtype)
                else:
                    raise NotImplementedError(f"Unsupported attention_mask format, attention_mask={attention_mask}")

                if (
                        self.audio_attn.config._attn_implementation == "sdpa"
                        and audio_attention_mask is not None
                        and audio_attention_mask.device.type == "cuda"
                        and not output_attentions
                ):
                    # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
                    # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                    # Details: https://github.com/pytorch/pytorch/issues/110213
                    audio_attention_mask = AttentionMaskConverter._unmask_unattended(audio_attention_mask, min_dtype)

            audio_attention_mask = audio_attention_mask.contiguous()

            audio_attn_output = self.audio_attn(
                hidden_states=hidden_states,
                attention_mask=audio_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # Handle variable return values: (hidden_states, present_key_value) or (hidden_states, attn_weights, present_key_value)
            if len(audio_attn_output) == 2:
                audio_hidden_states, audio_present_key_value = audio_attn_output
                audio_self_attn_weights = None
            else:
                audio_hidden_states, audio_self_attn_weights, audio_present_key_value = audio_attn_output
            audio_hidden_states = residual + audio_hidden_states
            if use_cache:
                residual = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), audio_hidden_states, residual
                )
            else:
                residual = torch.where(audio_out_mask_sq.unsqueeze(-1), audio_hidden_states, residual)
            audio_hidden_states = self.audio_post_audio_attn_layer_norm(audio_hidden_states)
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), audio_hidden_states, hidden_states
                )
            else:
                hidden_states = torch.where(audio_out_mask_sq.unsqueeze(-1), audio_hidden_states, hidden_states)

        # Text Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Handle variable return values: (hidden_states, present_key_value) or (hidden_states, attn_weights, present_key_value)
        if len(attn_output) == 2:
            hidden_states, present_key_value = attn_output
            self_attn_weights = None
        else:
            hidden_states, self_attn_weights, present_key_value = attn_output
        hidden_states = residual + hidden_states

        # Apply Dual-path FFN
        residual = hidden_states

        if has_audio_out and not self.fast_forward:
            if use_cache:
                real_audio_out_mask = audio_out_mask_sq[:, -target_length:]
            else:
                real_audio_out_mask = audio_out_mask_sq

            # Make whole graph in decode stage
            if decode_stage and is_using_cuda_graph:
                assert is_decoding_audio_token is not None, (
                    "is_decoding_audio_token should be present in the decoding stage."
                )
                if is_decoding_audio_token:
                    hidden_states = self.audio_post_attention_layernorm(hidden_states)
                    hidden_states = self.audio_mlp(hidden_states)
                else:
                    hidden_states = self.post_attention_layernorm(hidden_states)
                    hidden_states = self.mlp(hidden_states)
                residual = residual + hidden_states
            else:
                text_hidden_states = self.post_attention_layernorm(hidden_states[~real_audio_out_mask])
                audio_hidden_states = self.audio_post_attention_layernorm(hidden_states[real_audio_out_mask])

                text_hidden_states = self.mlp(text_hidden_states)
                residual[~real_audio_out_mask] += text_hidden_states

                audio_hidden_states = self.audio_mlp(audio_hidden_states)
                residual[real_audio_out_mask] += audio_hidden_states

            hidden_states = residual
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        if self.fast_forward and has_audio_out:
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), original_hidden_states, hidden_states
                )
            else:
                hidden_states = torch.where(audio_out_mask_sq.unsqueeze(-1), original_hidden_states, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            if self.use_audio_attention:
                # The returned attn weights have shape (batch_size, num_heads + num_audio_attn_heads, seq_length, seq_length)
                outputs += (torch.concat([self_attn_weights, audio_self_attn_weights], dim=1),)
            else:
                # The returned attn weights have shape (batch_size, num_heads, seq_length, seq_length)
                outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class HiggsAudioModel(_HiggsAudioModel):
    pass


class HiggsAudioForConditionalGeneration(HiggsAudioPreTrainedModel, GenerationMixin):
    """
    HiggsAudio model for conditional generation following Hugging Face Transformers standard pattern.

    This class wraps HiggsAudioModel and provides all the functionality needed for inference,
    including audio tokenizer integration, KV cache management, and generation.

    Example:
        ```python
        from voicestudio.model.higgs_audio import (
            HiggsAudioForConditionalGeneration,
            HiggsAudioProcessor
        )

        # Load model and processor
        model = HiggsAudioForConditionalGeneration.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base"
        )
        processor = HiggsAudioProcessor.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer"
        )

        # Prepare inputs
        inputs = processor(chat_ml_sample)

        # Generate
        outputs = model.generate(**inputs, max_new_tokens=1024)

        # Decode audio
        audio = processor.decode_audio(outputs.audio_sequences[0])
        ```
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)

        # Ensure _attn_implementation is set to avoid None value errors
        # This gets set by from_pretrained, but we need a fallback for direct instantiation
        if not hasattr(config, '_attn_implementation') or config._attn_implementation is None:
            # Set a default value to prevent KeyError in attention layers
            config._attn_implementation = 'sdpa'  # Use sdpa as default fallback

        # Propagate to text_config as well (used by LlamaAttention layers)
        if hasattr(config, 'text_config'):
            config.text_config._attn_implementation = config._attn_implementation

        # Initialize the base model
        self.model = HiggsAudioModel(config)

        # Audio tokenizer will be set via set_audio_tokenizer()
        self.audio_tokenizer = None
        self.audio_tokenizer_tps = None
        self.samples_per_token = None

        # KV cache management
        self.kv_caches = None

        # Post init
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load a pretrained model with automatic attention implementation detection.

        If no attn_implementation is specified, this will try to use the best available
        attention implementation in the following order:
        1. flash_attention_2 (if installed)
        2. sdpa (if PyTorch >= 2.1.1)
        3. eager (fallback)
        """
        # If attn_implementation is not specified, auto-detect the best one
        if 'attn_implementation' not in kwargs:
            if is_flash_attn_2_available():
                kwargs['attn_implementation'] = 'flash_attention_2'
            else:  # Explicitly set to 'sdpa' to avoid None value
                kwargs['attn_implementation'] = 'sdpa'

        return super(HiggsAudioForConditionalGeneration, cls).from_pretrained(*args, **kwargs)

    def set_audio_tokenizer(self, audio_tokenizer, device: str = "cuda"):
        """
        Set the audio tokenizer for the model.

        Args:
            audio_tokenizer: The audio tokenizer instance
            device: Device to place the audio tokenizer on
        """
        if isinstance(audio_tokenizer, str):
            # Load from path
            self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=device)
        else:
            # Already loaded
            self.audio_tokenizer = audio_tokenizer

        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)

    def prepare_kv_caches(self, kv_cache_lengths: List[int] = [1024, 4096, 8192]):
        """
        Prepare KV caches for different sequence lengths.

        Args:
            kv_cache_lengths: List of cache lengths to prepare
        """
        cache_config = deepcopy(self.config.text_config)
        cache_config.num_hidden_layers = self.config.text_config.num_hidden_layers
        if self.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.config.audio_dual_ffn_layers)

        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(kv_cache_lengths)
        }

    def enable_cuda_graphs(self):
        """
        Enable CUDA graph optimization by capturing graphs for each KV cache length.
        This should be called after prepare_kv_caches().
        """
        if self.kv_caches is None:
            raise ValueError("KV caches must be prepared before enabling CUDA graphs. Call prepare_kv_caches() first.")

        if self.model.device.type != "cuda":
            logger.warning("CUDA graphs are only supported on CUDA devices. Skipping.")
            return

        logger.info("Capturing CUDA graphs for each KV cache length")
        self.model.capture_model(self.kv_caches.values())

    def decode_audio(self, audio_tokens: torch.Tensor) -> np.ndarray:
        """
        Decode audio tokens to waveform.

        Args:
            audio_tokens: Audio tokens of shape (num_codebooks, seq_len)

        Returns:
            Audio waveform as numpy array
        """
        if self.audio_tokenizer is None:
            raise ValueError("Audio tokenizer not set. Call set_audio_tokenizer() first.")

        audio_codebook_size = self.config.audio_codebook_size
        vq_code = revert_delay_pattern(audio_tokens).clip(0, audio_codebook_size - 1)[:, 1:-1]
        wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
        return wv_numpy

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids_start_group_loc: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass of the model. Delegates to the base HiggsAudioModel.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            audio_out_ids_start_group_loc=audio_out_ids_start_group_loc,
            label_ids=label_ids,
            label_audio_ids=label_audio_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.4,#0.7->0.4 mj
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_audio_waveforms: bool = True,
        **kwargs
    ) -> HiggsAudioGenerationOutput:
        audio_eos_token_id = getattr(self.model, 'audio_eos_token_id', None)
        text_eos_token_id = self.config.text_config.eos_token_id

        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = [text_eos_token_id]
            if audio_eos_token_id is not None:
                kwargs["eos_token_id"].append(audio_eos_token_id)
        else:
            if isinstance(kwargs["eos_token_id"], int):
                kwargs["eos_token_id"] = [kwargs["eos_token_id"]]

            if audio_eos_token_id is not None and audio_eos_token_id not in kwargs["eos_token_id"]:
                kwargs["eos_token_id"].append(audio_eos_token_id)
        #eod token 생성시 정지 mj
        """
        Generate audio and text sequences.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            audio_features: Audio features from Whisper encoder
            audio_feature_attention_mask: Attention mask for audio features
            audio_in_ids: Input audio token IDs
            audio_in_ids_start: Start indices for input audio
            audio_out_ids: Output audio token IDs (for continuation)
            audio_out_ids_start: Start indices for output audio
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            return_audio_waveforms: Whether to decode audio tokens to waveforms
            **kwargs: Additional generation parameters

        Returns:
            HiggsAudioGenerationOutput containing generated sequences and audio
        """
        # Note: KV cache reset is not needed here as StaticCache manages its own state
        # Calling reset() inside @torch.inference_mode() causes RuntimeError with inplace ops

        # Prepare generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )

        kwargs.setdefault("return_dict_in_generate", True) #mj

        # Call the base model's generate method
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            generation_config=generation_config,
            past_key_values_buckets=self.kv_caches,
            **kwargs
        )

        # Decode audio waveforms if requested
        if return_audio_waveforms and len(outputs.audio_sequences) > 0 and self.audio_tokenizer is not None:
            audio_waveforms = []
            for audio_tokens in outputs.audio_sequences:
                waveform = self.decode_audio(audio_tokens)
                audio_waveforms.append(waveform)

            # Concatenate all waveforms
            if len(audio_waveforms) > 0:
                outputs.audio_waveforms = np.concatenate(audio_waveforms)
                outputs.sampling_rate = self.audio_tokenizer.sampling_rate
            else:
                outputs.audio_waveforms = None
                outputs.sampling_rate = None

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Prepare inputs for generation. Delegates to the base model.
        """
        return self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        """
        Update model kwargs for generation. Delegates to the base model.
        """
        return self.model._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )


modeling_higgs_audio.HiggsAudioDualFFNDecoderLayer = HiggsAudioDualFFNDecoderLayer
modeling_higgs_audio.HiggsAudioModel = HiggsAudioModel

# Add audio_waveforms and sampling_rate to HiggsAudioGenerationOutput for convenience
HiggsAudioGenerationOutput.audio_waveforms = None
HiggsAudioGenerationOutput.sampling_rate = None
