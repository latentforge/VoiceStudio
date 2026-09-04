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
"""PyTorch CosyVoice v1 model."""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from transformers.activations import ACT2FN
from transformers.conversion_mapping import WeightRenaming, register_checkpoint_conversion_mapping
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.speecht5.modeling_speecht5 import HifiGanResidualBlock
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperAttention, WhisperEncoderLayer
from transformers.utils import auto_docstring

from ..bigvgan.modeling_bigvgan import dynamic_range_compression, mel_spectrogram
from .configuration_cosyvoice_v1 import CosyVoiceV1Config
from .generation_cosyvoice_v1 import CosyVoiceV1GenerationMixin
from .weight_conversion import RELEASED_CONFIG_FILE, build_config, converted_checkpoint, resolve_checkpoint


IGNORE_ID = -1


# A released CosyVoice v1 directory holds one file per network, and [`load_checkpoint`] merges them
# under the name of the submodule each belongs to. What is left is upstream's own module names.
# `BaseEncoder` calls its input projection `embed.out`, its blocks `encoders` and its closing norm
# `after_norm`, and `TransformerEncoderLayer` calls its two norms `norm1`/`norm2` where
# `ConformerEncoderLayer` calls the same two `norm_mha`/`norm_ff`. The vocoder was trained with the
# pre-parametrization spelling of weight norm. Every rule names one leaf rather than a whole block,
# so that the reverse mapping `save_pretrained` builds from it stays exact.
CHECKPOINT_CONVERSION = [
    WeightRenaming(source_patterns=r"\.embed\.out\.0\.", target_patterns=r"\.input_projection\.proj\."),
    WeightRenaming(source_patterns=r"\.embed\.out\.1\.", target_patterns=r"\.input_projection\.layer_norm\."),
    WeightRenaming(source_patterns=r"\.encoders\.(\d+)\.self_attn\.", target_patterns=r"\.layers\.\1\.self_attn\."),
    WeightRenaming(
        source_patterns=r"\.encoders\.(\d+)\.feed_forward\.", target_patterns=r"\.layers\.\1\.feed_forward\."
    ),
    WeightRenaming(
        source_patterns=r"\.encoders\.(\d+)\.norm1\.", target_patterns=r"\.layers\.\1\.self_attn_layer_norm\."
    ),
    WeightRenaming(
        source_patterns=r"\.encoders\.(\d+)\.norm2\.", target_patterns=r"\.layers\.\1\.final_layer_norm\."
    ),
    WeightRenaming(
        source_patterns=r"\.encoders\.(\d+)\.norm_mha\.", target_patterns=r"\.layers\.\1\.self_attn_layer_norm\."
    ),
    WeightRenaming(
        source_patterns=r"\.encoders\.(\d+)\.norm_ff\.", target_patterns=r"\.layers\.\1\.final_layer_norm\."
    ),
    WeightRenaming(source_patterns=r"\.after_norm\.", target_patterns=r"\.layer_norm\."),
    WeightRenaming(source_patterns=r"\.weight_g$", target_patterns=r"\.parametrizations\.weight\.original0"),
    WeightRenaming(source_patterns=r"\.weight_v$", target_patterns=r"\.parametrizations\.weight\.original1"),
]

register_checkpoint_conversion_mapping("CosyVoiceV1ForConditionalGeneration", CHECKPOINT_CONVERSION, overwrite=True)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Builds a boolean mask that is `True` on padding positions.

    Args:
        lengths (`torch.Tensor` of shape `(batch_size,)`):
            Number of valid steps of every sequence.
        max_len (`int`, *optional*, defaults to 0):
            Length of the returned mask. Defaults to the longest sequence.

    Returns:
        `torch.Tensor` of shape `(batch_size, max_len)`: `True` where the sequence is padded.
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    positions = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, max_len)
    return positions >= lengths.unsqueeze(-1)


def build_attention_bias(
    padding_mask: torch.Tensor,
    chunk_size: int,
    dtype: torch.dtype,
    num_cached_steps: int = 0,
) -> torch.Tensor:
    """
    Builds the additive attention bias of a chunked encoder.

    Args:
        padding_mask (`torch.Tensor` of shape `(batch_size, key_length)`):
            `True` on the positions that may be attended to.
        chunk_size (`int`):
            Static chunk size. A value of 1 yields a causal mask, 0 yields full attention.
        dtype (`torch.dtype`):
            Floating point type of the returned bias.
        num_cached_steps (`int`, *optional*, defaults to 0):
            Number of key positions that come from the cache and therefore carry no query.

    Returns:
        `torch.Tensor` of shape `(batch_size, 1, query_length, key_length)`: additive attention bias.
    """
    batch_size, key_length = padding_mask.shape
    query_length = key_length - num_cached_steps
    if chunk_size > 0:
        positions = torch.arange(key_length, device=padding_mask.device)
        block_end = (torch.div(positions, chunk_size, rounding_mode="trunc") + 1) * chunk_size
        visible = positions.unsqueeze(0) < block_end.unsqueeze(1)
        visible = visible[num_cached_steps:]
        mask = padding_mask.unsqueeze(1) & visible.unsqueeze(0)
    else:
        mask = padding_mask.unsqueeze(1).expand(batch_size, query_length, key_length)
    # a query that can see nothing would make the softmax undefined, so it is allowed to see everything
    mask = torch.where(mask.sum(dim=-1, keepdim=True) == 0, torch.ones_like(mask), mask)
    bias = torch.zeros(mask.shape, dtype=dtype, device=padding_mask.device)
    bias = bias.masked_fill(~mask, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


class CosyVoiceV1RelPositionalEmbedding(nn.Module):
    """
    Sinusoidal relative positional embedding covering the positive and the negative range, laid out so
    that the shift trick of Transformer-XL selects the right offsets.

    Args:
        hidden_size (`int`):
            Dimension of the embedding.
        max_source_positions (`int`):
            Length used to precompute the table.
    """

    def __init__(self, hidden_size: int, max_source_positions: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_source_positions
        # The table is a deterministic function of `hidden_size`, so it is built on first use rather
        # than in the constructor, which would leave it uninitialised under meta device loading.
        self.pe = None

    def extend_pe(self, hidden_states: torch.Tensor, pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Grows the table so that it covers `2 * hidden_states.size(1) - 1` relative offsets.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence the table has to cover.
            pe (`torch.Tensor`, *optional*):
                Table computed so far.

        Returns:
            `torch.Tensor` of shape `(1, 2 * sequence_length - 1, hidden_size)`: the table.
        """
        if pe is not None and pe.size(1) >= hidden_states.size(1) * 2 - 1:
            if pe.dtype != hidden_states.dtype or pe.device != hidden_states.device:
                pe = pe.to(dtype=hidden_states.dtype, device=hidden_states.device)
            return pe
        positive = torch.zeros(hidden_states.size(1), self.hidden_size)
        negative = torch.zeros(hidden_states.size(1), self.hidden_size)
        position = torch.arange(0, hidden_states.size(1), dtype=torch.int64).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.int64).float() * -(math.log(10000.0) / self.hidden_size)
        )
        positive[:, 0::2] = torch.sin(position * div_term)
        positive[:, 1::2] = torch.cos(position * div_term)
        negative[:, 0::2] = torch.sin(-1 * position * div_term)
        negative[:, 1::2] = torch.cos(-1 * position * div_term)
        positive = torch.flip(positive, [0]).unsqueeze(0)
        negative = negative[1:].unsqueeze(0)
        pe = torch.cat([positive, negative], dim=1)
        return pe.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor, key_length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence the embedding is built for.
            key_length (`int`, *optional*):
                Number of key positions. Defaults to the sequence length of `hidden_states`.

        Returns:
            `torch.Tensor` of shape `(1, 2 * key_length - 1, hidden_size)`: the relative embedding.
        """
        key_length = hidden_states.size(1) if key_length is None else key_length
        table_length = max(key_length, self.max_len)
        self.pe = self.extend_pe(hidden_states.new_zeros(1, table_length, 1), self.pe)
        start = self.pe.size(1) // 2 - key_length + 1
        end = self.pe.size(1) // 2 + key_length
        return self.pe[:, start:end]


class CosyVoiceV1Attention(nn.Module):
    """
    Multi head self attention with the Transformer-XL relative position parameterisation, extended with
    an incremental key value cache.

    Args:
        hidden_size (`int`):
            Dimension of the queries, keys and values.
        num_heads (`int`):
            Number of attention heads.
        dropout (`float`):
            Dropout applied to the attention probabilities.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        relative_position_embeddings: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, query_length, hidden_size)`):
                Sequence to attend from.
            relative_position_embeddings (`torch.Tensor` of shape `(1, 2 * key_length - 1, hidden_size)`):
                Relative position table covering the keys.
            attention_bias (`torch.Tensor` of shape `(batch_size, 1, query_length, key_length)`, *optional*):
                Additive attention bias.
            past_key_value (`tuple(torch.Tensor)`, *optional*):
                Keys and values of the previous steps, each of shape
                `(batch_size, num_heads, past_length, head_size)`.

        Returns:
            `tuple(torch.Tensor)`: the attention output and the updated key value pair.
        """
        batch_size, query_length, _ = hidden_states.size()

        query = self.linear_q(hidden_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.linear_k(hidden_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.linear_v(hidden_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        present_key_value = (key, value)

        scores = self._apply_relative_embeddings(query, key, relative_position_embeddings)
        if attention_bias is not None:
            scores = scores + attention_bias

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        attn_output = torch.matmul(probs, value)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_length, self.num_heads * self.head_size)
        return self.linear_out(attn_output), present_key_value

    def _apply_relative_embeddings(
        self, query: torch.Tensor, key: torch.Tensor, relative_position_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query (`torch.Tensor` of shape `(batch_size, num_heads, query_length, head_size)`):
                Projected queries.
            key (`torch.Tensor` of shape `(batch_size, num_heads, key_length, head_size)`):
                Projected keys.
            relative_position_embeddings (`torch.Tensor` of shape `(1, 2 * key_length - 1, hidden_size)`):
                Relative position table covering the keys.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_heads, query_length, key_length)`: attention scores.
        """
        proj = self.linear_pos(relative_position_embeddings)
        proj = proj.view(relative_position_embeddings.size(0), -1, self.num_heads, self.head_size)
        proj = proj.transpose(1, 2).transpose(2, 3)

        query = query.transpose(1, 2)
        query_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        query_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        scores_ac = torch.matmul(query_with_bias_u, key.transpose(-2, -1))
        scores_bd = torch.matmul(query_with_bias_v, proj)

        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.size(-1) // 2 + 1]

        return (scores_ac + scores_bd) / math.sqrt(self.head_size)


class CosyVoiceV1FeedForward(nn.Module):
    """
    Position wise feed forward layer.

    Args:
        hidden_size (`int`):
            Input and output dimension.
        ffn_dim (`int`):
            Inner dimension.
        dropout (`float`):
            Dropout applied on the inner activation.
        hidden_act (`str`):
            Activation of the inner layer.
    """

    def __init__(self, hidden_size: int, ffn_dim: int, dropout: float, hidden_act: str):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, ffn_dim)
        self.activation = ACT2FN[hidden_act]
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(ffn_dim, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(hidden_states))))


class CosyVoiceV1EncoderLayer(nn.Module):
    """
    Pre norm self attention block followed by a pre norm feed forward block.

    Args:
        hidden_size (`int`):
            Dimension of the layer.
        num_heads (`int`):
            Number of attention heads.
        ffn_dim (`int`):
            Inner dimension of the feed forward block.
        dropout (`float`):
            Dropout applied on both residual branches.
        attention_dropout (`float`):
            Dropout applied to the attention probabilities.
        hidden_act (`str`):
            Activation of the feed forward block.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, ffn_dim: int, dropout: float, attention_dropout: float, hidden_act: str
    ):
        super().__init__()
        self.self_attn = CosyVoiceV1Attention(hidden_size, num_heads, attention_dropout)
        self.feed_forward = CosyVoiceV1FeedForward(hidden_size, ffn_dim, dropout, hidden_act)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        relative_position_embeddings: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, relative_position_embeddings, attention_bias, past_key_value
        )
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = residual + self.dropout(self.feed_forward(hidden_states))
        return hidden_states, present_key_value


class CosyVoiceV1InputProjection(nn.Module):
    """
    Projects the encoder input to the model dimension without subsampling it.

    Args:
        input_size (`int`):
            Dimension of the input.
        hidden_size (`int`):
            Dimension of the output.
        dropout (`float`):
            Dropout applied after the layer norm.
        activation (`bool`, *optional*, defaults to `False`):
            Whether a ReLU closes the projection.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float, activation: bool = False):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.layer_norm(self.proj(hidden_states)))
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        return hidden_states


class CosyVoiceV1Encoder(nn.Module):
    """
    Stack of [`CosyVoiceV1EncoderLayer`] preceded by a linear input projection and a relative positional
    embedding, and closed by a final layer norm.

    Args:
        input_size (`int`):
            Dimension of the input.
        hidden_size (`int`):
            Dimension of the layers.
        num_heads (`int`):
            Number of attention heads.
        ffn_dim (`int`):
            Inner dimension of the feed forward blocks.
        num_layers (`int`):
            Number of layers.
        dropout (`float`):
            Dropout applied after the input projection and on the residual branches.
        positional_dropout (`float`):
            Dropout applied to the scaled inputs and to the relative positional embeddings.
        attention_dropout (`float`):
            Dropout applied to the attention probabilities.
        hidden_act (`str`):
            Activation of the feed forward blocks.
        chunk_size (`int`):
            Static chunk size of the attention mask.
        max_source_positions (`int`):
            Length used to precompute the relative positional embedding table.
        input_projection_activation (`bool`, *optional*, defaults to `False`):
            Whether the input projection ends with a ReLU.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float,
        positional_dropout: float,
        attention_dropout: float,
        hidden_act: str,
        chunk_size: int,
        max_source_positions: int,
        input_projection_activation: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.embed_scale = math.sqrt(hidden_size)
        self.input_projection = CosyVoiceV1InputProjection(
            input_size, hidden_size, dropout, activation=input_projection_activation
        )
        self.pos_embedding = CosyVoiceV1RelPositionalEmbedding(hidden_size, max_source_positions)
        self.pos_dropout = nn.Dropout(positional_dropout)
        self.layers = nn.ModuleList(
            [
                CosyVoiceV1EncoderLayer(hidden_size, num_heads, ffn_dim, dropout, attention_dropout, hidden_act)
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, input_size)`):
                Encoder input.
            padding_mask (`torch.Tensor` of shape `(batch_size, key_length)`, *optional*):
                `True` on the positions that may be attended to, cached positions included.
            past_key_values (`list(tuple(torch.Tensor))`, *optional*):
                Per layer keys and values of the previous steps.

        Returns:
            `tuple`: the encoder output and the updated per layer key value pairs.
        """
        hidden_states = self.input_projection(hidden_states)
        num_cached_steps = 0 if past_key_values is None else past_key_values[0][0].size(2)
        key_length = hidden_states.size(1) + num_cached_steps

        relative_position_embeddings = self.pos_embedding(hidden_states, key_length)
        hidden_states = self.pos_dropout(hidden_states * self.embed_scale)
        relative_position_embeddings = self.pos_dropout(relative_position_embeddings)

        if padding_mask is None:
            padding_mask = hidden_states.new_ones(hidden_states.size(0), key_length, dtype=torch.bool)
        attention_bias = build_attention_bias(padding_mask, self.chunk_size, hidden_states.dtype, num_cached_steps)

        present_key_values = []
        for index, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[index]
            hidden_states, present_key_value = layer(
                hidden_states, relative_position_embeddings, attention_bias, past_key_value
            )
            present_key_values.append(present_key_value)
        return self.layer_norm(hidden_states), present_key_values


class CosyVoiceV1LabelSmoothingLoss(nn.Module):
    """
    Kullback-Leibler divergence against a smoothed one hot target, ignoring padded targets.

    Args:
        vocab_size (`int`):
            Number of classes.
        smoothing (`float`):
            Probability mass spread over the classes that are not the target.
        normalize_length (`bool`):
            Whether the loss is divided by the number of unmasked targets instead of the batch size.
        ignore_index (`int`, *optional*, defaults to -1):
            Target value that is left out of the loss.
    """

    def __init__(self, vocab_size: int, smoothing: float, normalize_length: bool, ignore_index: int = IGNORE_ID):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.normalize_length = normalize_length
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (`torch.Tensor` of shape `(batch_size, sequence_length, vocab_size)`):
                Unnormalized predictions.
            target (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Targets, padded with `ignore_index`.

        Returns:
            `torch.Tensor`: the scalar loss.
        """
        batch_size = logits.size(0)
        logits = logits.view(-1, self.vocab_size)
        target = target.view(-1)
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        ignore = target == self.ignore_index
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(logits, dim=1), true_dist)
        denominator = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denominator


class CosyVoiceV1SpeechTokenLM(nn.Module):
    """
    Autoregressive model that turns text tokens, a speaker embedding and a speech token prompt into a
    sequence of supervised semantic speech tokens.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        self.config = config
        self.speech_vocab_size = config.speech_vocab_size
        self.sos_index = 0
        self.task_id_index = 1
        self.eos_token_id = config.speech_vocab_size

        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_encoder_input_size)
        self.text_encoder = CosyVoiceV1Encoder(
            input_size=config.text_encoder_input_size,
            hidden_size=config.text_encoder_hidden_size,
            num_heads=config.text_encoder_num_heads,
            ffn_dim=config.text_encoder_ffn_dim,
            num_layers=config.text_encoder_num_layers,
            dropout=config.text_encoder_dropout,
            positional_dropout=config.text_encoder_positional_dropout,
            attention_dropout=config.text_encoder_attention_dropout,
            hidden_act=config.text_encoder_hidden_act,
            chunk_size=config.text_encoder_chunk_size,
            max_source_positions=config.max_source_positions,
        )
        self.text_encoder_affine_layer = nn.Linear(config.text_encoder_hidden_size, config.lm_hidden_size)

        self.llm_embedding = nn.Embedding(2, config.lm_hidden_size)
        self.speech_embedding = nn.Embedding(config.speech_vocab_size, config.lm_hidden_size)
        self.spk_embed_affine_layer = nn.Linear(config.speaker_embedding_dim, config.lm_hidden_size)
        self.llm = CosyVoiceV1Encoder(
            input_size=config.lm_hidden_size,
            hidden_size=config.lm_hidden_size,
            num_heads=config.lm_num_heads,
            ffn_dim=config.lm_ffn_dim,
            num_layers=config.lm_num_layers,
            dropout=config.lm_dropout,
            positional_dropout=config.lm_positional_dropout,
            attention_dropout=config.lm_attention_dropout,
            hidden_act=config.lm_hidden_act,
            chunk_size=config.lm_chunk_size,
            max_source_positions=config.max_source_positions,
            input_projection_activation=config.lm_input_projection_activation,
        )
        self.llm_decoder = nn.Linear(config.lm_hidden_size, config.speech_vocab_size + 1)

    def encode_text(self, input_ids: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, text_length)`):
                Text token ids.
            input_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid text tokens per sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, text_length, lm_hidden_size)`: encoded text.
        """
        hidden_states = self.text_embedding(input_ids)
        padding_mask = ~make_pad_mask(input_lengths, hidden_states.size(1))
        hidden_states, _ = self.text_encoder(hidden_states, padding_mask)
        return self.text_encoder_affine_layer(hidden_states)

    def encode_speaker(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speaker_embedding (`torch.Tensor` of shape `(batch_size, speaker_embedding_dim)`):
                Utterance level speaker embedding.

        Returns:
            `torch.Tensor` of shape `(batch_size, 1, lm_hidden_size)`: projected speaker embedding.
        """
        return self.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1)).unsqueeze(1)

    def build_inputs(
        self,
        text_hidden_states: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_hidden_states: torch.Tensor,
        speech_token_ids: torch.Tensor,
        speech_token_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenates the start of sequence embedding, the speaker embedding, the encoded text, the task
        id embedding and the teacher forced speech token embeddings of every sequence.

        Args:
            text_hidden_states (`torch.Tensor` of shape `(batch_size, text_length, lm_hidden_size)`):
                Encoded text.
            text_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid text tokens per sequence.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, 1, lm_hidden_size)`):
                Projected speaker embedding.
            speech_token_ids (`torch.Tensor` of shape `(batch_size, speech_length)`):
                Target speech tokens.
            speech_token_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid speech tokens per sequence.

        Returns:
            `tuple(torch.Tensor)`: the packed inputs embeds and their lengths.
        """
        sos_embed = self.llm_embedding.weight[self.sos_index].reshape(1, -1)
        task_id_embed = self.llm_embedding.weight[self.task_id_index].reshape(1, -1)
        speech_hidden_states = self.speech_embedding(speech_token_ids)

        sequences = []
        for index in range(text_hidden_states.size(0)):
            sequences.append(
                torch.concat(
                    [
                        sos_embed,
                        speaker_hidden_states[index],
                        text_hidden_states[index, : text_lengths[index]],
                        task_id_embed,
                        speech_hidden_states[index, : speech_token_lengths[index]],
                    ],
                    dim=0,
                )
            )
        lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.int32)
        inputs_embeds = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=IGNORE_ID)
        return inputs_embeds, lengths.to(text_hidden_states.device)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, lm_hidden_size)`):
                Packed language model inputs.
            input_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid steps per sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, speech_vocab_size + 1)`: logits.
        """
        padding_mask = ~make_pad_mask(input_lengths, inputs_embeds.size(1))
        hidden_states, _ = self.llm(inputs_embeds, padding_mask)
        return self.llm_decoder(hidden_states)


class CosyVoiceV1InterpolateRegulator(nn.Module):
    """
    Resamples the encoder output to the mel frame rate with a linear interpolation followed by a stack
    of convolutions.

    Args:
        channels (`int`):
            Number of channels.
        sampling_ratios (`list[int]`):
            One convolution, group norm and Mish block is created per entry.
        groups (`int`, *optional*, defaults to 1):
            Number of groups of the group norms.
    """

    def __init__(self, channels: int, sampling_ratios: list[int], groups: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in sampling_ratios:
            layers.extend([nn.Conv1d(channels, channels, 3, 1, 1), nn.GroupNorm(groups, channels), nn.Mish()])
        layers.append(nn.Conv1d(channels, channels, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor, output_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, channels)`):
                Encoder output.
            output_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Target number of mel frames per sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, max_output_length, channels)`: resampled features.
        """
        mask = (~make_pad_mask(output_lengths)).to(hidden_states).unsqueeze(-1)
        hidden_states = F.interpolate(
            hidden_states.transpose(1, 2).contiguous(), size=int(output_lengths.max()), mode="linear"
        )
        return self.model(hidden_states).transpose(1, 2).contiguous() * mask

    def inference(
        self,
        prompt_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        prompt_mel_length: int,
        mel_length: int,
        input_frame_rate: int,
    ) -> torch.Tensor:
        """
        Interpolates the prompt and the generated part separately so that the boundary between them
        stays on an exact mel frame.

        Args:
            prompt_hidden_states (`torch.Tensor` of shape `(1, prompt_length, channels)`):
                Encoded prompt speech tokens.
            hidden_states (`torch.Tensor` of shape `(1, sequence_length, channels)`):
                Encoded speech tokens to synthesize.
            prompt_mel_length (`int`):
                Number of mel frames of the prompt.
            mel_length (`int`):
                Number of mel frames to generate.
            input_frame_rate (`int`):
                Number of speech tokens per second.

        Returns:
            `torch.Tensor` of shape `(1, prompt_mel_length + mel_length, channels)`: resampled features.
        """
        edge_mel_length = int(20 / input_frame_rate * 22050 / 256)
        if hidden_states.shape[1] > 40:
            head = F.interpolate(
                hidden_states[:, :20].transpose(1, 2).contiguous(), size=edge_mel_length, mode="linear"
            )
            middle = F.interpolate(
                hidden_states[:, 20:-20].transpose(1, 2).contiguous(),
                size=mel_length - edge_mel_length * 2,
                mode="linear",
            )
            tail = F.interpolate(
                hidden_states[:, -20:].transpose(1, 2).contiguous(), size=edge_mel_length, mode="linear"
            )
            resampled = torch.concat([head, middle, tail], dim=2)
        else:
            resampled = F.interpolate(hidden_states.transpose(1, 2).contiguous(), size=mel_length, mode="linear")
        if prompt_hidden_states.shape[1] != 0:
            prompt = F.interpolate(
                prompt_hidden_states.transpose(1, 2).contiguous(), size=prompt_mel_length, mode="linear"
            )
            resampled = torch.concat([prompt, resampled], dim=2)
        return self.model(resampled).transpose(1, 2).contiguous()


class CosyVoiceV1SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal embedding of the flow matching timestep.

    Args:
        dim (`int`):
            Dimension of the embedding, which has to be even.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("CosyVoiceV1SinusoidalPosEmb requires dim to be even")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        if timesteps.ndim < 1:
            timesteps = timesteps.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device).float() * -emb)
        emb = scale * timesteps.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class CosyVoiceV1TimestepEmbedding(nn.Module):
    """
    Two layer projection of the sinusoidal timestep embedding.

    Args:
        in_channels (`int`):
            Dimension of the sinusoidal embedding.
        time_embed_dim (`int`):
            Dimension of the projection.
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(hidden_states)))


class CosyVoiceV1Block1D(nn.Module):
    """
    Masked convolution, group norm and Mish.

    Args:
        dim (`int`):
            Number of input channels.
        dim_out (`int`):
            Number of output channels.
        groups (`int`, *optional*, defaults to 8):
            Number of groups of the group norm.
    """

    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), nn.Mish())

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states * mask) * mask


class CosyVoiceV1ResnetBlock1D(nn.Module):
    """
    Residual block conditioned on the flow matching timestep.

    Args:
        dim (`int`):
            Number of input channels.
        dim_out (`int`):
            Number of output channels.
        time_emb_dim (`int`):
            Dimension of the timestep embedding.
        groups (`int`, *optional*, defaults to 8):
            Number of groups of the group norms.
    """

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = CosyVoiceV1Block1D(dim, dim_out, groups=groups)
        self.block2 = CosyVoiceV1Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.block1(hidden_states, mask)
        residual = residual + self.mlp(time_emb).unsqueeze(-1)
        residual = self.block2(residual, mask)
        return residual + self.res_conv(hidden_states * mask)


class CosyVoiceV1Downsample1D(nn.Module):
    """
    Strided convolution that halves the time axis.

    Args:
        dim (`int`):
            Number of channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class CosyVoiceV1Upsample1D(nn.Module):
    """
    Transposed convolution that doubles the time axis.

    Args:
        dim (`int`):
            Number of channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class CosyVoiceV1EstimatorAttention(nn.Module):
    """
    Self attention of the flow matching estimator, with unbiased query, key and value projections.

    Args:
        query_dim (`int`):
            Dimension of the input.
        num_heads (`int`):
            Number of attention heads.
        head_dim (`int`):
            Dimension of every head.
        dropout (`float`):
            Dropout applied on the output projection.
    """

    def __init__(self, query_dim: int, num_heads: int, head_dim: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)])

    def forward(self, hidden_states: torch.Tensor, attention_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.to_k(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.to_v(hidden_states).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_bias)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        for layer in self.to_out:
            hidden_states = layer(hidden_states)
        return hidden_states


class CosyVoiceV1EstimatorGELU(nn.Module):
    """
    Linear projection followed by an exact GELU.

    Args:
        dim_in (`int`):
            Input dimension.
        dim_out (`int`):
            Output dimension.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states))


class CosyVoiceV1EstimatorFeedForward(nn.Module):
    """
    Feed forward of the flow matching estimator transformer blocks.

    Args:
        dim (`int`):
            Input and output dimension.
        mult (`int`):
            Expansion factor of the inner dimension.
        dropout (`float`):
            Dropout applied between the two linear layers.
    """

    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.ModuleList([CosyVoiceV1EstimatorGELU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class CosyVoiceV1EstimatorTransformerBlock(nn.Module):
    """
    Pre norm self attention block followed by a pre norm feed forward block, used inside the flow
    matching estimator.

    Args:
        dim (`int`):
            Dimension of the block.
        num_heads (`int`):
            Number of attention heads.
        head_dim (`int`):
            Dimension of every attention head.
        dropout (`float`):
            Dropout of the attention output and of the feed forward.
        ffn_mult (`int`):
            Expansion factor of the feed forward inner dimension.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float, ffn_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CosyVoiceV1EstimatorAttention(dim, num_heads, head_dim, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = CosyVoiceV1EstimatorFeedForward(dim, ffn_mult, dropout)

    def forward(self, hidden_states: torch.Tensor, attention_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.attn1(self.norm1(hidden_states), attention_bias) + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class CosyVoiceV1ConditionalDecoder(nn.Module):
    """
    One dimensional UNet with transformer blocks that predicts the flow matching vector field.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        channels = tuple(config.estimator_channels)
        self.in_channels = config.estimator_in_channels
        self.out_channels = config.estimator_out_channels

        self.time_embeddings = CosyVoiceV1SinusoidalPosEmb(config.estimator_in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = CosyVoiceV1TimestepEmbedding(config.estimator_in_channels, time_embed_dim)

        def make_transformer_blocks(dim: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    CosyVoiceV1EstimatorTransformerBlock(
                        dim,
                        config.estimator_num_heads,
                        config.estimator_head_dim,
                        config.estimator_dropout,
                        config.estimator_ffn_mult,
                    )
                    for _ in range(config.estimator_num_blocks)
                ]
            )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = config.estimator_in_channels
        for index in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[index]
            is_last = index == len(channels) - 1
            resnet = CosyVoiceV1ResnetBlock1D(
                input_channel, output_channel, time_embed_dim, config.estimator_group_norm_groups
            )
            downsample = (
                CosyVoiceV1Downsample1D(output_channel)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, make_transformer_blocks(output_channel), downsample]))

        for _ in range(config.estimator_num_mid_blocks):
            resnet = CosyVoiceV1ResnetBlock1D(
                channels[-1], output_channel, time_embed_dim, config.estimator_group_norm_groups
            )
            self.mid_blocks.append(nn.ModuleList([resnet, make_transformer_blocks(output_channel)]))

        channels = channels[::-1] + (channels[0],)
        for index in range(len(channels) - 1):
            input_channel = channels[index] * 2
            output_channel = channels[index + 1]
            is_last = index == len(channels) - 2
            resnet = CosyVoiceV1ResnetBlock1D(
                input_channel, output_channel, time_embed_dim, config.estimator_group_norm_groups
            )
            upsample = (
                CosyVoiceV1Upsample1D(output_channel)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, make_transformer_blocks(output_channel), upsample]))

        self.final_block = CosyVoiceV1Block1D(channels[-1], channels[-1], config.estimator_group_norm_groups)
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        timesteps: torch.Tensor,
        speaker_hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, out_channels, mel_length)`):
                Noisy mel spectrogram.
            mask (`torch.Tensor` of shape `(batch_size, 1, mel_length)`):
                1 on valid frames.
            mu (`torch.Tensor` of shape `(batch_size, out_channels, mel_length)`):
                Encoder output resampled to the mel frame rate.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Flow matching timesteps.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, out_channels)`):
                Projected speaker embedding.
            conditioning (`torch.Tensor` of shape `(batch_size, out_channels, mel_length)`):
                Mel spectrogram prefix used as conditioning.

        Returns:
            `torch.Tensor` of shape `(batch_size, out_channels, mel_length)`: predicted vector field.
        """
        time_emb = self.time_mlp(self.time_embeddings(timesteps).to(timesteps.dtype))

        hidden_states = torch.cat([hidden_states, mu], dim=1)
        speaker_hidden_states = speaker_hidden_states.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        hidden_states = torch.cat([hidden_states, speaker_hidden_states], dim=1)
        hidden_states = torch.cat([hidden_states, conditioning], dim=1)

        skips = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            hidden_states = resnet(hidden_states, mask_down, time_emb)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            attention_bias = self._attention_bias(mask_down, hidden_states.dtype)
            for block in transformer_blocks:
                hidden_states = block(hidden_states, attention_bias)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            skips.append(hidden_states)
            hidden_states = downsample(hidden_states * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            hidden_states = resnet(hidden_states, mask_mid, time_emb)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            attention_bias = self._attention_bias(mask_mid, hidden_states.dtype)
            for block in transformer_blocks:
                hidden_states = block(hidden_states, attention_bias)
            hidden_states = hidden_states.transpose(1, 2).contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = skips.pop()
            hidden_states = torch.cat([hidden_states[:, :, : skip.shape[-1]], skip], dim=1)
            hidden_states = resnet(hidden_states, mask_up, time_emb)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            attention_bias = self._attention_bias(mask_up, hidden_states.dtype)
            for block in transformer_blocks:
                hidden_states = block(hidden_states, attention_bias)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            hidden_states = upsample(hidden_states * mask_up)

        hidden_states = self.final_block(hidden_states, mask_up)
        return self.final_proj(hidden_states * mask_up) * mask

    @staticmethod
    def _attention_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Args:
            mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                1 on valid frames.
            dtype (`torch.dtype`):
                Floating point type of the returned bias.

        Returns:
            `torch.Tensor` of shape `(batch_size, 1, 1, sequence_length)`: additive attention bias.
        """
        bias = (1.0 - mask.to(dtype)) * -1.0e10
        return bias.unsqueeze(1)


class CosyVoiceV1ConditionalCFM(nn.Module):
    """
    Optimal transport conditional flow matching head with a fixed step Euler solver and classifier free
    guidance.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        self.n_feats = config.estimator_out_channels
        self.sigma_min = config.sigma_min
        self.t_scheduler = config.t_scheduler
        self.training_cfg_rate = config.training_cfg_rate
        self.inference_cfg_rate = config.inference_cfg_rate
        self.estimator = CosyVoiceV1ConditionalDecoder(config)

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int,
        speaker_hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        temperature: float = 1.0,
        prompt_len: int = 0,
        cache: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mu (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Encoder output resampled to the mel frame rate.
            mask (`torch.Tensor` of shape `(batch_size, 1, mel_length)`):
                1 on valid frames.
            num_steps (`int`):
                Number of Euler steps.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, n_feats)`):
                Projected speaker embedding.
            conditioning (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Mel spectrogram prefix used as conditioning.
            temperature (`float`, *optional*, defaults to 1.0):
                Scale of the initial noise.
            prompt_len (`int`, *optional*, defaults to 0):
                Number of prompt mel frames kept in the cache.
            cache (`torch.Tensor`, *optional*):
                Noise and encoder output of the prompt and of the overlap of the previous chunk.

        Returns:
            `tuple(torch.Tensor)`: the sampled mel spectrogram and the updated cache.
        """
        if cache is None:
            cache = mu.new_zeros(1, self.n_feats, 0, 2)
        noise = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        if cache_size != 0:
            noise[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        noise_cache = torch.concat([noise[:, :, :prompt_len], noise[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([noise_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, num_steps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(noise, t_span, mu, mask, speaker_hidden_states, conditioning), cache

    def solve_euler(
        self,
        hidden_states: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker_hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Initial noise.
            t_span (`torch.Tensor` of shape `(num_steps + 1,)`):
                Timesteps of the solver.
            mu (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Encoder output resampled to the mel frame rate.
            mask (`torch.Tensor` of shape `(batch_size, 1, mel_length)`):
                1 on valid frames.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, n_feats)`):
                Projected speaker embedding.
            conditioning (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Mel spectrogram prefix used as conditioning.

        Returns:
            `torch.Tensor` of shape `(batch_size, n_feats, mel_length)`: the sampled mel spectrogram.
        """
        timestep, delta = t_span[0].unsqueeze(dim=0), t_span[1] - t_span[0]
        length = hidden_states.size(2)
        dtype = speaker_hidden_states.dtype

        # the second half of every batch carries the unconditional branch, which is left at zero
        hidden_states_in = torch.zeros([2, self.n_feats, length], device=hidden_states.device, dtype=dtype)
        mask_in = torch.zeros([2, 1, length], device=hidden_states.device, dtype=dtype)
        mu_in = torch.zeros([2, self.n_feats, length], device=hidden_states.device, dtype=dtype)
        timestep_in = torch.zeros([2], device=hidden_states.device, dtype=dtype)
        speaker_in = torch.zeros([2, self.n_feats], device=hidden_states.device, dtype=dtype)
        conditioning_in = torch.zeros([2, self.n_feats, length], device=hidden_states.device, dtype=dtype)

        for step in range(1, len(t_span)):
            hidden_states_in[:] = hidden_states
            mask_in[:] = mask
            mu_in[0] = mu
            timestep_in[:] = timestep.unsqueeze(0)
            speaker_in[0] = speaker_hidden_states
            conditioning_in[0] = conditioning
            vector_field = self.estimator(
                hidden_states_in, mask_in, mu_in, timestep_in, speaker_in, conditioning_in
            )
            conditional, unconditional = torch.split(
                vector_field, [hidden_states.size(0), hidden_states.size(0)], dim=0
            )
            vector_field = (1.0 + self.inference_cfg_rate) * conditional - self.inference_cfg_rate * unconditional
            hidden_states = hidden_states + delta * vector_field
            timestep = timestep + delta
            if step < len(t_span) - 1:
                delta = t_span[step + 1] - timestep
        return hidden_states.float()

    def compute_loss(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        speaker_hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Ground truth mel spectrogram.
            mask (`torch.Tensor` of shape `(batch_size, 1, mel_length)`):
                1 on valid frames.
            mu (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Encoder output resampled to the mel frame rate.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, n_feats)`):
                Projected speaker embedding.
            conditioning (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Mel spectrogram prefix used as conditioning.

        Returns:
            `torch.Tensor`: the conditional flow matching loss.
        """
        batch_size = mu.size(0)
        timesteps = torch.rand([batch_size, 1, 1], device=mu.device, dtype=mu.dtype)
        noise = torch.randn_like(target)

        noisy = (1 - (1 - self.sigma_min) * timesteps) * noise + timesteps * target
        vector_field = target - (1 - self.sigma_min) * noise

        if self.training_cfg_rate > 0:
            keep = torch.rand(batch_size, device=target.device) > self.training_cfg_rate
            mu = mu * keep.view(-1, 1, 1)
            speaker_hidden_states = speaker_hidden_states * keep.view(-1, 1)
            conditioning = conditioning * keep.view(-1, 1, 1)

        prediction = self.estimator(
            noisy, mask, mu, timesteps.squeeze(), speaker_hidden_states, conditioning
        )
        return F.mse_loss(prediction * mask, vector_field * mask, reduction="sum") / (
            torch.sum(mask) * vector_field.shape[1]
        )


class CosyVoiceV1FlowModel(nn.Module):
    """
    Speech token to mel spectrogram model made of an encoder, an interpolating length regulator and a
    conditional flow matching head.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        self.config = config
        self.output_size = config.flow_output_size
        self.input_frame_rate = config.flow_input_frame_rate
        self.input_embedding = nn.Embedding(config.speech_vocab_size, config.flow_input_size)
        self.spk_embed_affine_layer = nn.Linear(config.speaker_embedding_dim, config.flow_output_size)
        self.encoder = CosyVoiceV1Encoder(
            input_size=config.flow_input_size,
            hidden_size=config.flow_encoder_hidden_size,
            num_heads=config.flow_encoder_num_heads,
            ffn_dim=config.flow_encoder_ffn_dim,
            num_layers=config.flow_encoder_num_layers,
            dropout=config.flow_encoder_dropout,
            positional_dropout=config.flow_encoder_positional_dropout,
            attention_dropout=config.flow_encoder_attention_dropout,
            hidden_act=config.flow_encoder_hidden_act,
            chunk_size=config.flow_encoder_chunk_size,
            max_source_positions=config.max_source_positions,
        )
        self.encoder_proj = nn.Linear(config.flow_encoder_hidden_size, config.flow_output_size)
        self.length_regulator = CosyVoiceV1InterpolateRegulator(
            config.flow_output_size, config.length_regulator_sampling_ratios
        )
        self.decoder = CosyVoiceV1ConditionalCFM(config)

    def encode(self, speech_token_ids: torch.Tensor, speech_token_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speech_token_ids (`torch.Tensor` of shape `(batch_size, speech_length)`):
                Supervised semantic speech tokens.
            speech_token_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid speech tokens per sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, speech_length, flow_output_size)`: encoded tokens.
        """
        mask = (~make_pad_mask(speech_token_lengths, speech_token_ids.size(1))).unsqueeze(-1)
        hidden_states = self.input_embedding(torch.clamp(speech_token_ids, min=0)) * mask
        hidden_states, _ = self.encoder(hidden_states, mask.squeeze(-1))
        return self.encoder_proj(hidden_states)

    def forward(
        self,
        speech_token_ids: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        speech_feat: torch.Tensor,
        speech_feat_lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            speech_token_ids (`torch.Tensor` of shape `(batch_size, speech_length)`):
                Supervised semantic speech tokens.
            speech_token_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid speech tokens per sequence.
            speech_feat (`torch.Tensor` of shape `(batch_size, mel_length, flow_output_size)`):
                Ground truth mel spectrogram.
            speech_feat_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid mel frames per sequence.
            speaker_embedding (`torch.Tensor` of shape `(batch_size, speaker_embedding_dim)`):
                Utterance level speaker embedding.

        Returns:
            `torch.Tensor`: the conditional flow matching loss.
        """
        speaker_hidden_states = self.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1))
        hidden_states = self.encode(speech_token_ids, speech_token_lengths)
        hidden_states = self.length_regulator(hidden_states, speech_feat_lengths)

        conditioning = torch.zeros_like(speech_feat)
        for index, length in enumerate(speech_feat_lengths):
            if random.random() < 0.5:
                continue
            prefix = random.randint(0, int(0.3 * length))
            conditioning[index, :prefix] = speech_feat[index, :prefix]
        conditioning = conditioning.transpose(1, 2)

        mask = (~make_pad_mask(speech_feat_lengths, speech_feat.size(1))).to(hidden_states)
        return self.decoder.compute_loss(
            speech_feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            hidden_states.transpose(1, 2).contiguous(),
            speaker_hidden_states,
            conditioning,
        )

    @torch.inference_mode()
    def inference(
        self,
        speech_token_ids: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        prompt_token_lengths: torch.Tensor,
        prompt_feat: torch.Tensor,
        speaker_embedding: torch.Tensor,
        num_steps: int,
        cache: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            speech_token_ids (`torch.Tensor` of shape `(1, speech_length)`):
                Speech tokens to synthesize.
            speech_token_lengths (`torch.Tensor` of shape `(1,)`):
                Number of speech tokens to synthesize.
            prompt_token_ids (`torch.Tensor` of shape `(1, prompt_length)`):
                Speech tokens of the prompt.
            prompt_token_lengths (`torch.Tensor` of shape `(1,)`):
                Number of prompt speech tokens.
            prompt_feat (`torch.Tensor` of shape `(1, prompt_mel_length, flow_output_size)`):
                Mel spectrogram of the prompt.
            speaker_embedding (`torch.Tensor` of shape `(1, speaker_embedding_dim)`):
                Utterance level speaker embedding.
            num_steps (`int`):
                Number of Euler steps.
            cache (`torch.Tensor`, *optional*):
                Flow matching cache carried over from the previous chunk.

        Returns:
            `tuple(torch.Tensor)`: the generated mel spectrogram and the updated cache.
        """
        speaker_hidden_states = self.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1))

        prompt_length, token_length = prompt_token_ids.shape[1], speech_token_ids.shape[1]
        token_ids = torch.concat([prompt_token_ids, speech_token_ids], dim=1)
        token_lengths = prompt_token_lengths + speech_token_lengths
        hidden_states = self.encode(token_ids, token_lengths)

        prompt_mel_length = prompt_feat.shape[1]
        mel_length = int(token_length / self.input_frame_rate * 22050 / 256)
        hidden_states = self.length_regulator.inference(
            hidden_states[:, :prompt_length],
            hidden_states[:, prompt_length:],
            prompt_mel_length,
            mel_length,
            self.input_frame_rate,
        )

        conditioning = torch.zeros(
            [1, prompt_mel_length + mel_length, self.output_size], device=hidden_states.device, dtype=hidden_states.dtype
        )
        conditioning[:, :prompt_mel_length] = prompt_feat
        conditioning = conditioning.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([prompt_mel_length + mel_length]))).to(hidden_states)
        mel, cache = self.decoder(
            mu=hidden_states.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            num_steps=num_steps,
            speaker_hidden_states=speaker_hidden_states,
            conditioning=conditioning,
            prompt_len=prompt_mel_length,
            cache=cache,
        )
        return mel[:, :, prompt_mel_length:].float(), cache


class CosyVoiceV1Snake(nn.Module):
    """
    Periodic activation `x + sin(a * x) ** 2 / a` with a learnable per channel `a`.

    Args:
        num_channels (`int`):
            Number of channels.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_channels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        return hidden_states + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(hidden_states * alpha), 2)


class CosyVoiceV1ResBlock(HifiGanResidualBlock):
    """
    Dilated residual block of the vocoder. Its convolutions are those of a HiFi-GAN residual block, and
    a per channel Snake activation runs in front of each of them where the base class applies a leaky
    ReLU.

    Args:
        channels (`int`):
            Number of channels.
        kernel_size (`int`):
            Kernel size of the convolutions.
        dilations (`list[int]`):
            Dilation of every dilated convolution.
    """

    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__(channels, kernel_size, dilations)
        self.apply_weight_norm()
        self.activations1 = nn.ModuleList([CosyVoiceV1Snake(channels) for _ in dilations])
        self.activations2 = nn.ModuleList([CosyVoiceV1Snake(channels) for _ in dilations])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for index in range(len(self.convs1)):
            residual = self.activations1[index](hidden_states)
            residual = self.convs1[index](residual)
            residual = self.activations2[index](residual)
            residual = self.convs2[index](residual)
            hidden_states = residual + hidden_states
        return hidden_states


class CosyVoiceV1SineGen(nn.Module):
    """
    Generates the harmonic excitation of the neural source filter from an f0 contour.

    Args:
        sampling_rate (`int`):
            Sampling rate of the excitation.
        num_harmonics (`int`):
            Number of harmonics above f0.
        amplitude (`float`):
            Amplitude of the sine waves.
        noise_std (`float`):
            Standard deviation of the noise added on voiced frames.
        voiced_threshold (`float`):
            f0 above which a frame counts as voiced.
    """

    def __init__(
        self, sampling_rate: int, num_harmonics: int, amplitude: float, noise_std: float, voiced_threshold: float
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f0 (`torch.Tensor` of shape `(batch_size, num_samples, 1)`):
                Upsampled f0 contour, zero on unvoiced steps.

        Returns:
            `tuple(torch.Tensor)`: the sine waves of shape `(batch_size, num_samples, num_harmonics + 1)`
            and the voiced mask of shape `(batch_size, num_samples, 1)`.
        """
        f0 = f0.transpose(1, 2)
        harmonics = torch.zeros((f0.size(0), self.num_harmonics + 1, f0.size(-1)), device=f0.device, dtype=f0.dtype)
        for index in range(self.num_harmonics + 1):
            harmonics[:, index : index + 1, :] = f0 * (index + 1) / self.sampling_rate

        theta = 2 * np.pi * (torch.cumsum(harmonics, dim=-1) % 1)
        phase = torch.empty(
            (f0.size(0), self.num_harmonics + 1, 1), device=f0.device, dtype=f0.dtype
        ).uniform_(-np.pi, np.pi)
        phase[:, 0, :] = 0
        sine_waves = self.amplitude * torch.sin(theta + phase)

        voiced = (f0 > self.voiced_threshold).to(f0.dtype)
        noise_amplitude = voiced * self.noise_std + (1 - voiced) * self.amplitude / 3
        sine_waves = sine_waves * voiced + noise_amplitude * torch.randn_like(sine_waves)
        return sine_waves.transpose(1, 2), voiced.transpose(1, 2)


class CosyVoiceV1SourceModule(nn.Module):
    """
    Merges the harmonics of [`CosyVoiceV1SineGen`] into a single excitation signal.

    Args:
        sampling_rate (`int`):
            Sampling rate of the excitation.
        num_harmonics (`int`):
            Number of harmonics above f0.
        amplitude (`float`):
            Amplitude of the sine waves.
        noise_std (`float`):
            Standard deviation of the noise added on voiced frames.
        voiced_threshold (`float`):
            f0 above which a frame counts as voiced.
    """

    def __init__(
        self, sampling_rate: int, num_harmonics: int, amplitude: float, noise_std: float, voiced_threshold: float
    ):
        super().__init__()
        self.amplitude = amplitude
        self.l_sin_gen = CosyVoiceV1SineGen(sampling_rate, num_harmonics, amplitude, noise_std, voiced_threshold)
        self.l_linear = nn.Linear(num_harmonics + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sine_waves, _ = self.l_sin_gen(f0)
        return self.l_tanh(self.l_linear(sine_waves))


class CosyVoiceV1F0Predictor(nn.Module):
    """
    Convolutional network predicting an f0 contour from a mel spectrogram.

    Args:
        in_channels (`int`):
            Number of mel bins.
        hidden_size (`int`):
            Number of channels of the convolutions.
    """

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(5):
            layers.extend([weight_norm(nn.Conv1d(channels, hidden_size, kernel_size=3, padding=1)), nn.ELU()])
            channels = hidden_size
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel (`torch.Tensor` of shape `(batch_size, in_channels, mel_length)`):
                Mel spectrogram.

        Returns:
            `torch.Tensor` of shape `(batch_size, mel_length)`: the predicted f0 contour.
        """
        hidden_states = self.condnet(mel).transpose(1, 2)
        return torch.abs(self.classifier(hidden_states).squeeze(-1))


@auto_docstring(
    custom_intro="""
    Output of [`CosyVoiceV1HiFTGenerator.compute_loss`].
    """
)
@dataclass
class CosyVoiceV1VocoderOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`):
        Sum of the terms of the vocoder objective that do not need a discriminator.
    mel_loss (`torch.FloatTensor` of shape `(1,)`):
        Log mel spectrogram distance between the generated and the ground truth waveform, weighted
        by `config.vocoder_mel_loss_coeff`.
    f0_loss (`torch.FloatTensor` of shape `(1,)`):
        L1 distance between the predicted and the extracted f0 contour.
    waveform (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        Generated waveform.
    f0 (`torch.FloatTensor` of shape `(batch_size, mel_length)`):
        Predicted f0 contour.
    """

    loss: Optional[torch.FloatTensor] = None
    mel_loss: Optional[torch.FloatTensor] = None
    f0_loss: Optional[torch.FloatTensor] = None
    waveform: Optional[torch.FloatTensor] = None
    f0: Optional[torch.FloatTensor] = None


class CosyVoiceV1HiFTGenerator(nn.Module):
    """
    HiFTNet vocoder, a neural source filter fed into an inverse short time Fourier transform head.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        self.config = config
        self.n_fft = config.vocoder_istft_n_fft
        self.hop_length = config.vocoder_istft_hop_length
        self.leaky_relu_slope = config.vocoder_leaky_relu_slope
        self.audio_limit = config.vocoder_audio_limit
        self.num_kernels = len(config.vocoder_resblock_kernel_sizes)
        self.num_upsamples = len(config.vocoder_upsample_rates)

        self.f0_predictor = CosyVoiceV1F0Predictor(config.vocoder_in_channels, config.f0_predictor_hidden_size)
        self.m_source = CosyVoiceV1SourceModule(
            config.sample_rate,
            config.vocoder_num_harmonics,
            config.vocoder_source_amplitude,
            config.vocoder_source_noise_std,
            config.vocoder_voiced_threshold,
        )
        upsample_scale = int(np.prod(config.vocoder_upsample_rates) * config.vocoder_istft_hop_length)
        self.f0_upsamp = nn.Upsample(scale_factor=upsample_scale)

        base_channels = config.vocoder_base_channels
        self.conv_pre = weight_norm(nn.Conv1d(config.vocoder_in_channels, base_channels, 7, 1, padding=3))

        self.ups = nn.ModuleList(
            [
                weight_norm(
                    nn.ConvTranspose1d(
                        base_channels // (2**index),
                        base_channels // (2 ** (index + 1)),
                        kernel_size,
                        rate,
                        padding=(kernel_size - rate) // 2,
                    )
                )
                for index, (rate, kernel_size) in enumerate(
                    zip(config.vocoder_upsample_rates, config.vocoder_upsample_kernel_sizes)
                )
            ]
        )

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + list(config.vocoder_upsample_rates)[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for index, (rate, kernel_size, dilations) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                config.vocoder_source_resblock_kernel_sizes,
                config.vocoder_source_resblock_dilation_sizes,
            )
        ):
            channels = base_channels // (2 ** (index + 1))
            if rate == 1:
                self.source_downs.append(nn.Conv1d(self.n_fft + 2, channels, 1, 1))
            else:
                self.source_downs.append(
                    nn.Conv1d(self.n_fft + 2, channels, int(rate) * 2, int(rate), padding=int(rate) // 2)
                )
            self.source_resblocks.append(CosyVoiceV1ResBlock(channels, kernel_size, dilations))

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = base_channels // (2 ** (index + 1))
            for kernel_size, dilations in zip(
                config.vocoder_resblock_kernel_sizes, config.vocoder_resblock_dilation_sizes
            ):
                self.resblocks.append(CosyVoiceV1ResBlock(channels, kernel_size, dilations))

        self.conv_post = weight_norm(nn.Conv1d(channels, self.n_fft + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))

    def _stft_window(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (`torch.Tensor`):
                Tensor whose device and floating point type the window is built for.

        Returns:
            `torch.Tensor` of shape `(vocoder_istft_n_fft,)`: the analysis window.
        """
        return torch.hann_window(self.n_fft, periodic=True, device=tensor.device, dtype=tensor.dtype)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self._stft_window(waveform),
            return_complex=True,
        )
        spectrogram = torch.view_as_real(spectrogram)
        return torch.cat([spectrogram[..., 0], spectrogram[..., 1]], dim=1)

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imaginary = magnitude * torch.sin(phase)
        return torch.istft(
            torch.complex(real, imaginary),
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self._stft_window(magnitude),
        )

    def decode(self, mel: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel (`torch.Tensor` of shape `(batch_size, vocoder_in_channels, mel_length)`):
                Mel spectrogram.
            source (`torch.Tensor` of shape `(batch_size, 1, num_samples)`):
                Harmonic excitation.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_samples)`: the generated waveform.
        """
        source_stft = self._stft(source.squeeze(1))

        hidden_states = self.conv_pre(mel)
        for index in range(self.num_upsamples):
            hidden_states = F.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = self.ups[index](hidden_states)
            if index == self.num_upsamples - 1:
                hidden_states = self.reflection_pad(hidden_states)

            source_hidden_states = self.source_downs[index](source_stft)
            source_hidden_states = self.source_resblocks[index](source_hidden_states)
            hidden_states = hidden_states + source_hidden_states

            residual = None
            for kernel_index in range(self.num_kernels):
                block = self.resblocks[index * self.num_kernels + kernel_index]
                residual = block(hidden_states) if residual is None else residual + block(hidden_states)
            hidden_states = residual / self.num_kernels

        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        magnitude = torch.exp(hidden_states[:, : self.n_fft // 2 + 1, :])
        phase = torch.sin(hidden_states[:, self.n_fft // 2 + 1 :, :])
        waveform = self._istft(magnitude, phase)
        return torch.clamp(waveform, -self.audio_limit, self.audio_limit)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel (`torch.Tensor` of shape `(batch_size, vocoder_in_channels, mel_length)`):
                Mel spectrogram.

        Returns:
            `tuple(torch.Tensor)`: the generated waveform and the predicted f0 contour.
        """
        f0 = self.f0_predictor(mel)
        source = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        source = self.m_source(source).transpose(1, 2)
        return self.decode(mel, source), f0

    def mel_loss(self, waveform: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Measures the log mel spectrogram distance the generator is regressed onto. Its filter bank runs
        up to the Nyquist frequency rather than to the `fmax` of the mel the model consumes.

        Args:
            waveform (`torch.Tensor` of shape `(batch_size, num_samples)`):
                Generated waveform.
            labels (`torch.Tensor` of shape `(batch_size, num_samples)`):
                Ground truth waveform.

        Returns:
            `torch.Tensor`: the distance, weighted by `config.vocoder_mel_loss_coeff`.
        """
        resolution = {
            "sampling_rate": self.config.sample_rate,
            "n_fft": self.config.vocoder_mel_loss_n_fft,
            "hop_length": self.config.vocoder_mel_loss_hop_length,
            "win_length": self.config.vocoder_mel_loss_win_length,
            "num_mel_bins": self.config.vocoder_mel_loss_num_mel_bins,
            "fmin": self.config.vocoder_mel_loss_fmin,
            "fmax": self.config.vocoder_mel_loss_fmax,
            "centered": False,
        }
        return self.config.vocoder_mel_loss_coeff * F.l1_loss(
            dynamic_range_compression(mel_spectrogram(waveform, **resolution)),
            dynamic_range_compression(mel_spectrogram(labels, **resolution)),
        )

    def compute_loss(
        self, mel: torch.Tensor, labels: torch.Tensor, pitch_feat: torch.Tensor
    ) -> CosyVoiceV1VocoderOutput:
        """
        Runs the generator and scores it with the terms of the vocoder objective that need no
        discriminator, the weighted mel spectrogram reconstruction loss and the L1 loss between the
        predicted and the extracted f0 contour. The adversarial, feature matching and true positive
        rate terms are not implemented.

        Args:
            mel (`torch.Tensor` of shape `(batch_size, vocoder_in_channels, mel_length)`):
                Mel spectrogram of the ground truth waveform.
            labels (`torch.Tensor` of shape `(batch_size, num_samples)`):
                Ground truth waveform.
            pitch_feat (`torch.Tensor` of shape `(batch_size, mel_length)`):
                Extracted f0 contour, which [`~CosyVoiceV1Processor.compute_f0`] produces.

        Returns:
            [`CosyVoiceV1VocoderOutput`]: the loss, its two terms, the generated waveform and the
            predicted f0 contour.
        """
        waveform, f0 = self(mel)
        reconstruction_loss = self.mel_loss(waveform, labels)
        f0_loss = F.l1_loss(f0, pitch_feat)
        return CosyVoiceV1VocoderOutput(
            loss=reconstruction_loss + f0_loss,
            mel_loss=reconstruction_loss,
            f0_loss=f0_loss,
            waveform=waveform,
            f0=f0,
        )

    @torch.inference_mode()
    def inference(self, mel: torch.Tensor, cache_source: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel (`torch.Tensor` of shape `(batch_size, vocoder_in_channels, mel_length)`):
                Mel spectrogram.
            cache_source (`torch.Tensor`, *optional*):
                Excitation of the previous chunk, reused to avoid a phase discontinuity.

        Returns:
            `tuple(torch.Tensor)`: the generated waveform and the excitation used to produce it.
        """
        f0 = self.f0_predictor(mel)
        source = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        source = self.m_source(source).transpose(1, 2)
        if cache_source is not None and cache_source.shape[2] != 0:
            source[:, :, : cache_source.shape[2]] = cache_source
        return self.decode(mel, source), source


def build_speaker_nonlinear(num_channels: int, affine: bool = True) -> nn.Sequential:
    """
    Builds the normalization and activation pair the speaker encoder puts in front of every
    convolution.

    Args:
        num_channels (`int`):
            Channels the batch normalization runs over.
        affine (`bool`, *optional*, defaults to `True`):
            Whether the batch normalization is followed by a ReLU and carries a scale and a shift.

    Returns:
        `nn.Sequential`: The pair.
    """
    nonlinear = nn.Sequential()
    nonlinear.add_module("batchnorm", nn.BatchNorm1d(num_channels, affine=affine))
    if affine:
        nonlinear.add_module("relu", nn.ReLU(inplace=True))
    return nonlinear


class CosyVoiceV1SpeakerResBlock(nn.Module):
    """
    Two dimensional residual block of the speaker encoder front end, which strides over the mel axis
    only and leaves the time axis alone.

    Args:
        in_channels (`int`):
            Channels of the input.
        out_channels (`int`):
            Channels of the output.
        stride (`int`, *optional*, defaults to 1):
            Stride over the mel axis.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_mel_bins, num_frames)`):
                Input feature map.

        Returns:
            `torch.Tensor` of shape `(batch_size, out_channels, num_mel_bins // stride, num_frames)`:
            the output feature map.
        """
        residual = F.relu(self.bn1(self.conv1(hidden_states)))
        residual = self.bn2(self.conv2(residual))
        return F.relu(residual + self.shortcut(hidden_states))


class CosyVoiceV1SpeakerFrontEnd(nn.Module):
    """
    Two dimensional convolutional front end of the speaker encoder, which reduces the mel axis by
    eight and flattens what is left into the channels of a one dimensional sequence.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        channels = config.speaker_encoder_front_end_channels
        self.conv1 = nn.Conv2d(1, channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer1 = self._make_layer(channels, channels, config.speaker_encoder_front_end_num_blocks[0])
        self.layer2 = self._make_layer(channels, channels, config.speaker_encoder_front_end_num_blocks[1])
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.out_channels = channels * (config.speaker_encoder_num_mel_bins // 8)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        blocks = [CosyVoiceV1SpeakerResBlock(in_channels, out_channels, stride=2)]
        blocks += [CosyVoiceV1SpeakerResBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        return nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Filter bank features.

        Returns:
            `torch.Tensor` of shape `(batch_size, out_channels, num_frames)`: the flattened feature map.
        """
        hidden_states = F.relu(self.bn1(self.conv1(features.unsqueeze(1))))
        hidden_states = self.layer2(self.layer1(hidden_states))
        hidden_states = F.relu(self.bn2(self.conv2(hidden_states)))
        batch_size, channels, num_bins, num_frames = hidden_states.shape
        return hidden_states.reshape(batch_size, channels * num_bins, num_frames)


class CosyVoiceV1TDNNLayer(nn.Module):
    """
    Time delay layer, a strided one dimensional convolution followed by a normalization and an
    activation.

    Args:
        in_channels (`int`):
            Channels of the input.
        out_channels (`int`):
            Channels of the output.
        kernel_size (`int`):
            Kernel size of the convolution.
        stride (`int`, *optional*, defaults to 1):
            Stride of the convolution.
        dilation (`int`, *optional*, defaults to 1):
            Dilation of the convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.nonlinear = build_speaker_nonlinear(out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_frames)`):
                Input sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, out_channels, num_frames // stride)`: the output.
        """
        return self.nonlinear(self.linear(hidden_states))


class CosyVoiceV1CAMLayer(nn.Module):
    """
    Context aware masking layer, which gates a local convolution by a mask read off the utterance
    level average and the segment level averages of the same input.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
        bottleneck_channels (`int`):
            Channels of the input, which the dense layer has already bottlenecked.
        out_channels (`int`):
            Channels of the output.
        kernel_size (`int`):
            Kernel size of the local convolution.
        dilation (`int`):
            Dilation of the local convolution.
    """

    def __init__(
        self,
        config: CosyVoiceV1Config,
        bottleneck_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.segment_length = config.speaker_encoder_segment_length
        reduced = bottleneck_channels // config.speaker_encoder_reduction
        self.linear_local = nn.Conv1d(
            bottleneck_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            bias=False,
        )
        self.linear1 = nn.Conv1d(bottleneck_channels, reduced, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(reduced, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def segment_pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Averages the sequence over segments and holds each average for the whole segment.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, num_frames)`):
                Input sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, num_frames)`: the held averages.
        """
        segments = F.avg_pool1d(
            hidden_states, kernel_size=self.segment_length, stride=self.segment_length, ceil_mode=True
        )
        shape = segments.shape
        segments = segments.unsqueeze(-1).expand(*shape, self.segment_length).reshape(*shape[:-1], -1)
        return segments[..., : hidden_states.shape[-1]]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, bottleneck_channels, num_frames)`):
                Input sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, out_channels, num_frames)`: the gated output.
        """
        local = self.linear_local(hidden_states)
        context = hidden_states.mean(-1, keepdim=True) + self.segment_pooling(hidden_states)
        context = self.relu(self.linear1(context))
        return local * self.sigmoid(self.linear2(context))


class CosyVoiceV1CAMDenseTDNNLayer(nn.Module):
    """
    One layer of a densely connected time delay block: a bottleneck projection followed by a context
    aware masking layer, whose output the block appends to its input.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
        in_channels (`int`):
            Channels of the input, which grows by `speaker_encoder_growth_rate` per layer.
        kernel_size (`int`):
            Kernel size of the context aware masking convolution.
        dilation (`int`):
            Dilation of that convolution.
    """

    def __init__(self, config: CosyVoiceV1Config, in_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        bottleneck = config.speaker_encoder_bottleneck_size * config.speaker_encoder_growth_rate
        self.nonlinear1 = build_speaker_nonlinear(in_channels)
        self.linear1 = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
        self.nonlinear2 = build_speaker_nonlinear(bottleneck)
        self.cam_layer = CosyVoiceV1CAMLayer(
            config, bottleneck, config.speaker_encoder_growth_rate, kernel_size, dilation
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_frames)`):
                Input sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, growth_rate, num_frames)`: the channels this layer
            contributes.
        """
        hidden_states = self.linear1(self.nonlinear1(hidden_states))
        return self.cam_layer(self.nonlinear2(hidden_states))


class CosyVoiceV1CAMDenseTDNNBlock(nn.ModuleList):
    """
    Densely connected block of [`CosyVoiceV1CAMDenseTDNNLayer`], each layer reading every channel the
    ones before it produced.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
        num_layers (`int`):
            Number of layers.
        in_channels (`int`):
            Channels entering the first layer.
        kernel_size (`int`):
            Kernel size of every context aware masking convolution.
        dilation (`int`):
            Dilation of every context aware masking convolution.
    """

    def __init__(self, config: CosyVoiceV1Config, num_layers: int, in_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        for index in range(num_layers):
            layer = CosyVoiceV1CAMDenseTDNNLayer(
                config, in_channels + index * config.speaker_encoder_growth_rate, kernel_size, dilation
            )
            self.add_module(f"tdnnd{index + 1}", layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_frames)`):
                Input sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, in_channels + num_layers * growth_rate, num_frames)`:
            the input with every layer's channels appended.
        """
        for layer in self:
            hidden_states = torch.cat([hidden_states, layer(hidden_states)], dim=1)
        return hidden_states


class CosyVoiceV1TransitLayer(nn.Module):
    """
    Normalization and activation followed by a pointwise convolution, which halves the channels a
    dense block produced.

    Args:
        in_channels (`int`):
            Channels of the input.
        out_channels (`int`):
            Channels of the output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.nonlinear = build_speaker_nonlinear(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_frames)`):
                Input sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, out_channels, num_frames)`: the output.
        """
        return self.linear(self.nonlinear(hidden_states))


class CosyVoiceV1SpeakerStatsPool(nn.Module):
    """Concatenates the mean and the standard deviation of a sequence over time."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, num_frames)`):
                Input sequence.

        Returns:
            `torch.Tensor` of shape `(batch_size, 2 * channels)`: the statistics.
        """
        return torch.cat([hidden_states.mean(dim=-1), hidden_states.std(dim=-1, unbiased=True)], dim=-1)


class CosyVoiceV1DenseLayer(nn.Module):
    """
    Pointwise convolution followed by an unaffine normalization, which projects the pooled statistics
    onto the speaker embedding.

    Args:
        in_channels (`int`):
            Channels of the pooled statistics.
        out_channels (`int`):
            Size of the speaker embedding.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.nonlinear = build_speaker_nonlinear(out_channels, affine=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels)`):
                Pooled statistics.

        Returns:
            `torch.Tensor` of shape `(batch_size, out_channels)`: the speaker embedding.
        """
        if hidden_states.dim() == 2:
            return self.nonlinear(self.linear(hidden_states.unsqueeze(-1)).squeeze(-1))
        return self.nonlinear(self.linear(hidden_states))


class CosyVoiceV1SpeakerEncoder(nn.Module):
    """
    CAM++ speaker encoder, a two dimensional convolutional front end feeding densely connected time
    delay blocks whose layers are gated by context aware masks, closed by mean and standard deviation
    pooling and a projection onto the speaker embedding.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        self.head = CosyVoiceV1SpeakerFrontEnd(config)
        growth_rate = config.speaker_encoder_growth_rate

        self.xvector = nn.Sequential()
        self.xvector.add_module(
            "tdnn", CosyVoiceV1TDNNLayer(self.head.out_channels, config.speaker_encoder_init_channels, 5, stride=2)
        )
        channels = config.speaker_encoder_init_channels
        blocks = zip(
            config.speaker_encoder_num_layers,
            config.speaker_encoder_kernel_sizes,
            config.speaker_encoder_dilations,
        )
        for index, (num_layers, kernel_size, dilation) in enumerate(blocks):
            self.xvector.add_module(
                f"block{index + 1}",
                CosyVoiceV1CAMDenseTDNNBlock(config, num_layers, channels, kernel_size, dilation),
            )
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(f"transit{index + 1}", CosyVoiceV1TransitLayer(channels, channels // 2))
            channels //= 2
        self.xvector.add_module("out_nonlinear", build_speaker_nonlinear(channels))
        self.xvector.add_module("stats", CosyVoiceV1SpeakerStatsPool())
        self.xvector.add_module("dense", CosyVoiceV1DenseLayer(channels * 2, config.speaker_embedding_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (`torch.Tensor` of shape `(batch_size, num_frames, num_mel_bins)`):
                Mean subtracted kaldi filter bank features.

        Returns:
            `torch.Tensor` of shape `(batch_size, speaker_embedding_dim)`: the speaker embedding.
        """
        return self.xvector(self.head(features.permute(0, 2, 1)))


def build_speech_tokenizer_encoder_config(config: CosyVoiceV1Config) -> WhisperConfig:
    r"""
    Builds the [`WhisperConfig`] the speech tokenizer encoder layers read their geometry from.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.

    Returns:
        [`WhisperConfig`]: The configuration.
    """
    return WhisperConfig(
        num_mel_bins=config.speech_tokenizer_num_mel_bins,
        d_model=config.speech_tokenizer_hidden_size,
        encoder_attention_heads=config.speech_tokenizer_num_heads,
        encoder_ffn_dim=config.speech_tokenizer_ffn_dim,
        encoder_layers=config.speech_tokenizer_num_layers,
        activation_function="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )


class CosyVoiceV1SpeechTokenizerAttention(WhisperAttention):
    """
    Self attention of the speech tokenizer encoder, which scales the query and the key by the fourth
    root of the head dimension apiece where [`WhisperAttention`] scales the query alone by its square
    root, and takes its padding mask as a boolean over positions rather than as an additive bias.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__(
            embed_dim=config.speech_tokenizer_hidden_size, num_heads=config.speech_tokenizer_num_heads
        )
        self.scaling = self.head_dim**-0.25

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> tuple[torch.Tensor, None]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input sequence.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask that is `True` on padding positions.
            kwargs:
                Ignored.

        Returns:
            `tuple(torch.Tensor, None)`: the attention output of shape
            `(batch_size, sequence_length, hidden_size)`, and `None` in place of the attention
            weights [`WhisperEncoderLayer`] discards.
        """
        batch_size, length, _ = hidden_states.shape
        shape = (batch_size, length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).permute(0, 2, 1, 3) * self.scaling
        key = self.k_proj(hidden_states).view(shape).permute(0, 2, 3, 1) * self.scaling
        value = self.v_proj(hidden_states).view(shape).permute(0, 2, 1, 3)
        scores = query @ key
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        context = scores.softmax(dim=-1) @ value
        return self.out_proj(context.permute(0, 2, 1, 3).reshape(batch_size, length, -1)), None


class CosyVoiceV1SpeechTokenizerLayer(WhisperEncoderLayer):
    """
    One encoder layer of the speech tokenizer, which is a Whisper encoder layer around
    [`CosyVoiceV1SpeechTokenizerAttention`].

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__(build_speech_tokenizer_encoder_config(config))
        self.self_attn = CosyVoiceV1SpeechTokenizerAttention(config)


class CosyVoiceV1SpeechTokenizerQuantizer(nn.Module):
    """
    Vector quantizer of the speech tokenizer, which L2 normalizes the encoder output and reads the
    nearest codebook entry off it by euclidean distance.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        self.embedding = nn.Embedding(config.speech_vocab_size, config.speech_tokenizer_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Encoder output.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length)`: the speech token ids.
        """
        hidden_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        codebook = self.embedding.weight
        distance = flat.pow(2).sum(dim=-1, keepdim=True) - (2 * flat) @ codebook.transpose(0, 1)
        distance = distance + codebook.pow(2).sum(dim=-1)
        return distance.argmin(dim=-1).view(hidden_states.shape[:-1])


class CosyVoiceV1SpeechTokenizer(nn.Module):
    """
    Supervised semantic speech tokenizer of CosyVoice v1, which is the first six blocks of a Whisper
    large encoder closed by [`CosyVoiceV1SpeechTokenizerQuantizer`]. Its two opening convolutions
    halve the mel frame rate once, so one token stands for two mel frames.

    Args:
        config ([`CosyVoiceV1Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__()
        hidden_size = config.speech_tokenizer_hidden_size
        self.strides = (config.speech_tokenizer_conv_stride, 2)
        self.conv1 = nn.Conv1d(
            config.speech_tokenizer_num_mel_bins, hidden_size, 3, stride=self.strides[0], padding=1
        )
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, stride=self.strides[1], padding=1)
        self.embed_positions = (
            None
            if config.speech_tokenizer_max_source_positions is None
            else nn.Embedding(config.speech_tokenizer_max_source_positions, hidden_size)
        )
        self.layers = nn.ModuleList(
            [CosyVoiceV1SpeechTokenizerLayer(config) for _ in range(config.speech_tokenizer_num_layers)]
        )
        self.quantizer = CosyVoiceV1SpeechTokenizerQuantizer(config)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Mel frames of each utterance.

        Returns:
            `torch.Tensor` of shape `(batch_size,)`: speech tokens of each utterance.
        """
        for stride in self.strides:
            input_lengths = (input_lengths - 1) // stride + 1
        return input_lengths

    def embed(self, input_features: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_features (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Log mel spectrogram.
            input_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Mel frames of each utterance.

        Returns:
            `tuple(torch.Tensor)`: the encoder input of shape
            `(batch_size, sequence_length, hidden_size)` and the mask that is `True` on its padding
            positions.
        """
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states)).permute(0, 2, 1)
        length = hidden_states.shape[1]
        if self.embed_positions is not None:
            hidden_states = hidden_states + self.embed_positions.weight[:length]
        lengths = self.output_lengths(input_lengths).clamp(max=length)
        positions = torch.arange(length, device=hidden_states.device)
        return hidden_states, positions >= lengths.to(hidden_states.device)[:, None]

    def encode(self, input_features: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Log mel spectrogram.
            input_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Mel frames of each utterance.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: the encoder output.
        """
        hidden_states, padding_mask = self.embed(input_features, input_lengths)
        for layer in self.layers:
            hidden_states = layer(hidden_states, padding_mask)
        return hidden_states

    def forward(self, input_features: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Log mel spectrogram.
            input_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Mel frames of each utterance.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length)`: the speech token ids.
        """
        return self.quantizer(self.encode(input_features, input_lengths))


@auto_docstring(
    custom_intro="""
    Output of [`CosyVoiceV1ForConditionalGeneration`].
    """
)
@dataclass
class CosyVoiceV1Output(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Label smoothed cross entropy over the speech tokens, returned when `labels` is provided.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, speech_vocab_size + 1)`):
        Speech token scores.
    accuracy (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Fraction of correctly predicted speech tokens, returned when `labels` is provided.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    accuracy: Optional[torch.FloatTensor] = None


@auto_docstring
class CosyVoiceV1PreTrainedModel(PreTrainedModel):
    config: CosyVoiceV1Config
    base_model_prefix = "cosyvoice_v1"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    @classmethod
    def _released_checkpoint(cls, source, **kwargs) -> "tuple[CosyVoiceV1Config, Path] | None":
        r"""
        Locates a released CosyVoice v1 directory, fetching the recipe that names its revision rather than the
        three networks, which the conversion fetches for itself.

        Args:
            source (`str` or `os.PathLike`, *optional*):
                Repository id or local directory.
            kwargs (`dict`, *optional*):
                Fields of `weight_conversion.DOWNLOAD_KWARGS` selecting a revision and a cache.

        Returns:
            `tuple[CosyVoiceV1Config, Path]` or `None`: The configuration and the local directory naming the
            revision the released files are read from, or `None` when `source` holds no released checkpoint.
        """
        directory = resolve_checkpoint(source, (RELEASED_CONFIG_FILE,), **kwargs)
        if directory is None:
            return None
        return build_config(directory), directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Repository id or local directory. The directories the CosyVoice authors published hold
                one file per network rather than a single checkpoint, and are read as they are.
            model_args:
                Forwarded to [`~PreTrainedModel.from_pretrained`].
            kwargs:
                Forwarded to [`~PreTrainedModel.from_pretrained`].

        Returns:
            [`CosyVoiceV1PreTrainedModel`]: The loaded model.
        """
        try:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        except OSError:
            released = cls._released_checkpoint(pretrained_model_name_or_path, **kwargs)
            if released is None:
                raise
        config, directory = released
        converted = converted_checkpoint(pretrained_model_name_or_path, directory, config, download_kwargs=kwargs)
        return super().from_pretrained(converted, *model_args, **kwargs)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, CosyVoiceV1Attention):
            nn.init.xavier_uniform_(module.pos_bias_u)
            nn.init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, CosyVoiceV1ConditionalDecoder):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(submodule.weight, nonlinearity="relu")
                    if submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0)
                elif isinstance(submodule, nn.GroupNorm):
                    nn.init.constant_(submodule.weight, 1)
                    nn.init.constant_(submodule.bias, 0)


def build_speech_token_labels(
    text_token_lengths: torch.Tensor,
    speech_token_ids: torch.Tensor,
    speech_token_lengths: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """
    Builds the language model targets of a batch.

    The start of sequence embedding, the speaker embedding and every encoded text position are masked
    out with `IGNORE_ID`, then the speech tokens of the utterance are predicted, then the end of speech
    token.

    Args:
        text_token_lengths (`torch.Tensor` of shape `(batch_size,)`):
            Number of text tokens per sequence.
        speech_token_ids (`torch.Tensor` of shape `(batch_size, speech_length)`):
            Target speech tokens.
        speech_token_lengths (`torch.Tensor` of shape `(batch_size,)`):
            Number of valid speech tokens per sequence.
        eos_token_id (`int`):
            Id of the end of speech token, which is `speech_vocab_size`.

    Returns:
        `torch.Tensor` of shape `(batch_size, sequence_length)`: the targets.
    """
    targets = []
    for index in range(speech_token_ids.size(0)):
        prefix = [IGNORE_ID] * (2 + int(text_token_lengths[index]))
        tokens = speech_token_ids[index, : speech_token_lengths[index]].tolist()
        targets.append(torch.tensor(prefix + tokens + [eos_token_id]))
    return nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=IGNORE_ID).to(speech_token_ids.device)


@auto_docstring(
    custom_intro="""
    CosyVoice v1, made of an autoregressive text to speech token language model, a conditional flow
    matching model turning speech tokens into a mel spectrogram, and a HiFTNet vocoder.

    The three networks are trained one at a time upstream, so `forward` optimizes the language model
    objective only. [`CosyVoiceV1FlowModel.forward`] returns the flow matching objective and
    [`CosyVoiceV1HiFTGenerator.compute_loss`] returns the terms of the vocoder objective that need no
    discriminator.
    """
)
class CosyVoiceV1ForConditionalGeneration(CosyVoiceV1GenerationMixin, CosyVoiceV1PreTrainedModel):
    def __init__(self, config: CosyVoiceV1Config):
        super().__init__(config)
        self.llm = CosyVoiceV1SpeechTokenLM(config)
        self.flow = CosyVoiceV1FlowModel(config)
        self.hift = CosyVoiceV1HiFTGenerator(config)
        self.criterion = CosyVoiceV1LabelSmoothingLoss(
            config.speech_vocab_size + 1, config.label_smoothing, config.length_normalized_loss
        )
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        speech_token_ids: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> CosyVoiceV1Output:
        r"""
        input_ids (`torch.Tensor` of shape `(batch_size, text_length)`):
            Text token ids.
        input_lengths (`torch.Tensor` of shape `(batch_size,)`):
            Number of valid text tokens per sequence.
        speech_token_ids (`torch.Tensor` of shape `(batch_size, speech_length)`):
            Teacher forced speech tokens.
        speech_token_lengths (`torch.Tensor` of shape `(batch_size,)`):
            Number of valid speech tokens per sequence.
        speaker_embedding (`torch.Tensor` of shape `(batch_size, speaker_embedding_dim)`):
            Utterance level speaker embedding.
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Speech token targets, padded with -1 on the positions that carry no target. They are
            built by [`build_speech_token_labels`].
        """
        text_hidden_states = self.llm.encode_text(input_ids, input_lengths)
        speaker_hidden_states = self.llm.encode_speaker(speaker_embedding)
        inputs_embeds, lm_input_lengths = self.llm.build_inputs(
            text_hidden_states, input_lengths, speaker_hidden_states, speech_token_ids, speech_token_lengths
        )
        logits = self.llm(inputs_embeds, lm_input_lengths)

        loss, accuracy = None, None
        if labels is not None:
            loss = self.criterion(logits, labels)
            predictions = logits.view(-1, self.config.speech_vocab_size + 1).argmax(-1).view_as(labels)
            keep = labels != IGNORE_ID
            accuracy = (predictions.masked_select(keep) == labels.masked_select(keep)).sum() / keep.sum()

        return CosyVoiceV1Output(loss=loss, logits=logits, accuracy=accuracy)


__all__ = [
    "CosyVoiceV1Attention",
    "CosyVoiceV1ConditionalCFM",
    "CosyVoiceV1ConditionalDecoder",
    "CosyVoiceV1Encoder",
    "CosyVoiceV1FlowModel",
    "CosyVoiceV1ForConditionalGeneration",
    "CosyVoiceV1HiFTGenerator",
    "CosyVoiceV1LabelSmoothingLoss",
    "CosyVoiceV1Output",
    "CosyVoiceV1PreTrainedModel",
    "CosyVoiceV1SpeakerEncoder",
    "CosyVoiceV1SpeechTokenLM",
    "CosyVoiceV1SpeechTokenizer",
    "CosyVoiceV1VocoderOutput",
    "build_speech_token_labels",
    "make_pad_mask",
]
