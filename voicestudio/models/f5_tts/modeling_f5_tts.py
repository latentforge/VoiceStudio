# MIT License
#
# Copyright (c) 2024 Yushen CHEN
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""PyTorch F5-TTS model."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, auto_docstring, logging

from ..bigvgan import BigVGANConfig, BigVGANModel
from ..vocos import VocosModel
from .configuration_f5_tts import F5TTSConfig
from .generation_f5_tts import F5TTSGenerationMixin


logger = logging.get_logger(__name__)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0):
    r"""
    Builds the concatenated cosine and sine table the text embedding is offset by.

    Args:
        dim (`int`):
            Dimensionality of the table. Must be even.
        end (`int`):
            Number of positions to precompute.
        theta (`float`, *optional*, defaults to 10000.0):
            Base of the geometric frequency progression.
        theta_rescale_factor (`float`, *optional*, defaults to 1.0):
            NTK aware rescaling of `theta`, for extending the table beyond the trained length.

    Returns:
        `torch.Tensor`: Table of shape `(end, dim)` holding the cosines in its first half and the sines in its
        second half.
    """
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    positions = torch.arange(end, device=freqs.device)
    freqs = torch.outer(positions, freqs).float()
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def get_pos_embed_indices(start: torch.Tensor, length: int, max_pos: int, scale: float = 1.0):
    r"""
    Builds per sample position indices into a precomputed position table, clamped to its length.

    Args:
        start (`torch.Tensor`):
            Per sample first position, of shape `(batch_size,)`.
        length (`int`):
            Number of positions to emit per sample.
        max_pos (`int`):
            Number of rows the position table holds.
        scale (`float`, *optional*, defaults to 1.0):
            Stride between consecutive positions.

    Returns:
        `torch.Tensor`: Indices of shape `(batch_size, length)`.
    """
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    positions = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    return torch.where(positions < max_pos, positions, max_pos - 1)


def lengths_to_mask(lengths: torch.Tensor, length: int | None = None) -> torch.Tensor:
    r"""
    Turns per sample lengths into a boolean keep mask.

    Args:
        lengths (`torch.Tensor`):
            Per sample length, of shape `(batch_size,)`.
        length (`int`, *optional*):
            Width of the mask. Defaults to the largest entry of `lengths`.

    Returns:
        `torch.Tensor`: Boolean mask of shape `(batch_size, length)`, `True` on valid positions.
    """
    if length is None:
        length = int(lengths.amax())
    positions = torch.arange(length, device=lengths.device)
    return positions[None, :] < lengths[:, None]


def mask_from_frac_lengths(lengths: torch.Tensor, frac_lengths: torch.Tensor) -> torch.Tensor:
    r"""
    Draws one random contiguous span per sample and returns it as a boolean mask.

    Args:
        lengths (`torch.Tensor`):
            Per sample sequence length, of shape `(batch_size,)`.
        frac_lengths (`torch.Tensor`):
            Per sample span length as a fraction of the sequence length, of shape `(batch_size,)`.

    Returns:
        `torch.Tensor`: Boolean mask of shape `(batch_size, max(lengths))`, `True` inside the drawn span.
    """
    span_lengths = (frac_lengths * lengths).long()
    max_start = lengths - span_lengths
    start = (max_start * torch.rand_like(frac_lengths)).long().clamp(min=0)
    end = start + span_lengths

    max_seq_len = int(lengths.max())
    positions = torch.arange(max_seq_len, device=start.device).long()
    return (positions[None, :] >= start[:, None]) & (positions[None, :] < end[:, None])


def build_attention_mask(padding_mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    r"""
    Turns a boolean padding mask into the additive attention mask the attention interfaces take.

    Args:
        padding_mask (`torch.Tensor`, *optional*):
            Boolean mask of shape `(batch_size, sequence_length)`, `True` on valid positions.
        dtype (`torch.dtype`):
            Dtype of the produced mask.

    Returns:
        `torch.Tensor`: Additive mask of shape `(batch_size, 1, 1, sequence_length)`, or `None` when
        `padding_mask` is `None`.
    """
    if padding_mask is None:
        return None
    mask = torch.zeros(padding_mask.shape, dtype=dtype, device=padding_mask.device)
    return mask.masked_fill(~padding_mask, torch.finfo(dtype).min)[:, None, None, :]


def deinterleave_head_dim(hidden_states: torch.Tensor) -> torch.Tensor:
    r"""
    Reorders a head dimension laid out as interleaved rotary pairs `(x0, x1, x2, x3, ...)` into the half split
    layout `(x0, x2, ..., x1, x3, ...)` that [`~transformers.models.llama.modeling_llama.apply_rotary_pos_emb`]
    expects.

    Args:
        hidden_states (`torch.Tensor`):
            Tensor whose last dimension is a head dimension.

    Returns:
        `torch.Tensor`: Tensor of the same shape with its last dimension reordered.
    """
    return torch.cat((hidden_states[..., 0::2], hidden_states[..., 1::2]), dim=-1)


class F5TTSGlobalResponseNorm(nn.Module):
    r"""
    Constructs the global response normalization of a ConvNeXt V2 block.

    Args:
        dim (`int`):
            Number of channels to normalize.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global_features = torch.norm(hidden_states, p=2, dim=1, keepdim=True)
        normalized_features = global_features / (global_features.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (hidden_states * normalized_features) + self.beta + hidden_states


class F5TTSConvNeXtV2Block(nn.Module):
    r"""
    Constructs the ConvNeXt V2 block the text encoder is built from.

    Args:
        dim (`int`):
            Number of channels in and out of the block.
        intermediate_dim (`int`):
            Dimensionality of the pointwise expansion.
        dilation (`int`, *optional*, defaults to 1):
            Dilation of the depthwise convolution.
    """

    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = F5TTSGlobalResponseNorm(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.grn(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        return residual + hidden_states


class F5TTSConvPositionEmbedding(nn.Module):
    r"""
    Constructs the depthwise convolutional position embedding added to the speech input embedding.

    Args:
        dim (`int`):
            Number of channels in and out of the module.
        kernel_size (`int`, *optional*, defaults to 31):
            Width of both convolutions. Must be odd.
        groups (`int`, *optional*, defaults to 16):
            Number of convolution groups.
    """

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"`kernel_size` must be odd, got {kernel_size}.")
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.masked_layer_indices = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)
        hidden_states = hidden_states.permute(0, 2, 1)

        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask, 0.0)
        for index, layer in enumerate(self.conv1d):
            hidden_states = layer(hidden_states)
            if padding_mask is not None and index in self.masked_layer_indices:
                hidden_states = hidden_states.masked_fill(~padding_mask, 0.0)

        return hidden_states.permute(0, 2, 1)


class F5TTSUNetRMSNorm(nn.Module):
    r"""
    Constructs the L2 normalization based RMS norm the `"unett"` backbone wraps its attention and feed forward
    blocks in.

    Args:
        dim (`int`):
            Number of channels to normalize.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(hidden_states, dim=-1) * self.scale * self.g


class F5TTSAdaLayerNorm(nn.Module):
    r"""
    Constructs the adaptive layer norm that modulates a transformer block from the flow time step.

    Args:
        dim (`int`):
            Number of channels to normalize.
        eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the layer normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, hidden_states: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class F5TTSAdaLayerNormFinal(nn.Module):
    r"""
    Constructs the adaptive layer norm applied before the output projection.

    Args:
        dim (`int`):
            Number of channels to normalize.
        eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the layer normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, hidden_states: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]


class F5TTSFeedForward(nn.Module):
    r"""
    Constructs the position wise feed forward of a transformer block.

    Args:
        dim (`int`):
            Number of channels in and out of the block.
        mult (`int`, *optional*, defaults to 4):
            Expansion factor of the inner dimension.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout applied after the activation.
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(approximate="tanh")),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden_states)


class F5TTSAttention(nn.Module):
    r"""
    Constructs the self attention of a backbone layer.

    Args:
        config ([`F5TTSConfig`]):
            Model configuration.
    """

    def __init__(self, config: F5TTSConfig):
        super().__init__()
        self.config = config
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.inner_dim = self.heads * self.head_dim
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.pe_attn_head = config.pe_attn_head
        self.is_causal = False

        self.to_q = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_k = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_v = nn.Linear(config.hidden_size, self.inner_dim)

        if config.qk_norm == "rms_norm":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, config.hidden_size), nn.Dropout(config.dropout)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.to_q(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.to_k(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.to_v(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        rotary_heads = self.heads if self.pe_attn_head is None else self.pe_attn_head
        rotary_query, rotary_key = apply_rotary_pos_emb(
            deinterleave_head_dim(query_states[:, :rotary_heads]),
            deinterleave_head_dim(key_states[:, :rotary_heads]),
            cos,
            sin,
        )
        if rotary_heads < self.heads:
            query_states = torch.cat((rotary_query, query_states[:, rotary_heads:]), dim=1)
            key_states = torch.cat((rotary_key, key_states[:, rotary_heads:]), dim=1)
        else:
            query_states, key_states = rotary_query, rotary_key

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.to_out[0](attn_output)
        attn_output = self.to_out[1](attn_output)

        if padding_mask is not None:
            attn_output = attn_output.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

        return attn_output, attn_weights


class F5TTSDecoderLayer(GradientCheckpointingLayer):
    r"""
    Constructs one layer of the `"dit"` backbone, an adaptive layer norm modulated self attention block followed by
    an adaptive layer norm modulated feed forward block.

    Args:
        config ([`F5TTSConfig`]):
            Model configuration.
    """

    def __init__(self, config: F5TTSConfig):
        super().__init__()
        self.attn_norm = F5TTSAdaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = F5TTSAttention(config)
        self.ff_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.ff = F5TTSFeedForward(dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, emb=timestep_embedding)

        attn_output, _ = self.attn(
            normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **kwargs,
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        normed = self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(normed)
        return hidden_states


class F5TTSUNetLayer(GradientCheckpointingLayer):
    r"""
    Constructs one layer of the `"unett"` backbone, an RMS normed self attention block followed by an RMS normed
    feed forward block, optionally preceded by the projection that joins in the skip connection coming from the
    mirrored layer in the first half of the stack.

    Args:
        config ([`F5TTSConfig`]):
            Model configuration.
        has_skip_projection (`bool`, *optional*, defaults to `False`):
            Whether the layer owns the linear that projects the concatenation of its input and its skip connection
            back down to `config.hidden_size`.
    """

    def __init__(self, config: F5TTSConfig, has_skip_projection: bool = False):
        super().__init__()
        skip_proj = (
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False) if has_skip_projection else None
        )
        self.layer = nn.ModuleList(
            [
                skip_proj,
                F5TTSUNetRMSNorm(config.hidden_size),
                F5TTSAttention(config),
                F5TTSUNetRMSNorm(config.hidden_size),
                F5TTSFeedForward(dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        skip_connection: torch.Tensor | None = None,
        skip_connect_type: str = "concat",
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        skip_proj, attn_norm, attn, ff_norm, ff = self.layer

        if skip_connection is not None:
            if skip_connect_type == "concat":
                hidden_states = skip_proj(torch.cat((hidden_states, skip_connection), dim=-1))
            elif skip_connect_type == "add":
                hidden_states = hidden_states + skip_connection

        attn_output, _ = attn(
            attn_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output
        return hidden_states + ff(ff_norm(hidden_states))


class F5TTSSinusPositionEmbedding(nn.Module):
    r"""
    Constructs the sinusoidal embedding of the flow time step.

    Args:
        dim (`int`):
            Dimensionality of the embedding. Must be even.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device).float() * -emb)
        emb = scale * timestep.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class F5TTSTimestepEmbedding(nn.Module):
    r"""
    Constructs the conditioning embedding of the flow time step.

    Args:
        dim (`int`):
            Dimensionality of the produced embedding.
        freq_embed_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the intermediate sinusoidal embedding.
    """

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = F5TTSSinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(time_hidden)


class F5TTSTextEmbedding(nn.Module):
    r"""
    Constructs the character embedding, its sinusoidal position offset and the ConvNeXt V2 text encoder.

    Args:
        config ([`F5TTSConfig`]):
            Model configuration.
    """

    def __init__(self, config: F5TTSConfig):
        super().__init__()
        self.text_embed = nn.Embedding(config.text_vocab_size + 1, config.text_dim)
        self.mask_padding = config.text_mask_padding
        self.average_upsampling = config.text_average_upsampling
        self.extra_modeling = config.text_conv_layers > 0

        if self.extra_modeling:
            self.precompute_max_pos = config.text_max_positions
            self.freqs_cis = nn.Buffer(
                precompute_freqs_cis(config.text_dim, self.precompute_max_pos), persistent=False
            )
            self.text_blocks = nn.Sequential(
                *[
                    F5TTSConvNeXtV2Block(config.text_dim, config.text_dim * config.text_conv_mult)
                    for _ in range(config.text_conv_layers)
                ]
            )

    def average_upsample_text_by_mask(
        self, text_embeds: torch.Tensor, text_mask: torch.Tensor, target_lengths: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Repeats every text position a near equal number of times so the encoded text spans the speech length.

        Args:
            text_embeds (`torch.Tensor`):
                Encoded text of shape `(batch_size, sequence_length, text_dim)`.
            text_mask (`torch.Tensor`):
                Boolean mask of shape `(batch_size, sequence_length)`, `True` on real characters.
            target_lengths (`torch.Tensor`):
                Per sample speech length, of shape `(batch_size,)`.

        Returns:
            `torch.Tensor`: Upsampled text of the same shape as `text_embeds`.
        """
        batch_size = text_embeds.shape[0]
        text_lengths = text_mask.sum(dim=1)
        upsampled_text = torch.zeros_like(text_embeds)

        for index in range(batch_size):
            text_len = int(text_lengths[index].item())
            audio_len = int(target_lengths[index].item())

            if text_len == 0 or audio_len <= 0:
                continue

            valid_ind = torch.where(text_mask[index])[0]
            valid_data = text_embeds[index, valid_ind, :]

            base_repeat = audio_len // text_len
            remainder = audio_len % text_len

            indices = []
            for position in range(text_len):
                indices.extend([position] * (base_repeat + (1 if position >= text_len - remainder else 0)))

            indices = torch.tensor(indices[:audio_len], device=text_embeds.device, dtype=torch.long)
            upsampled_text[index, :audio_len, :] = valid_data[indices]

        return upsampled_text

    def forward(self, input_ids: torch.Tensor, seq_len, drop_text: bool = False) -> torch.Tensor:
        r"""
        Args:
            input_ids (`torch.Tensor`):
                Character ids of shape `(batch_size, text_length)`, padded with the filler id `0`.
            seq_len (`int` or `torch.Tensor`):
                Speech length the text is curtailed and padded to, either shared or per sample.
            drop_text (`bool`, *optional*, defaults to `False`):
                Whether to zero the text, for the unconditional branch of classifier free guidance.

        Returns:
            `torch.Tensor`: Encoded text of shape `(batch_size, seq_len, text_dim)`.
        """
        valid_pos_mask = None
        if torch.is_tensor(seq_len):
            seq_len = seq_len.to(device=input_ids.device, dtype=torch.long)
            max_seq_len = int(seq_len.max().item())
        else:
            max_seq_len = int(seq_len)

        input_ids = input_ids[:, :max_seq_len]
        input_ids = F.pad(input_ids, (0, max_seq_len - input_ids.shape[1]), value=0)

        if torch.is_tensor(seq_len):
            positions = torch.arange(max_seq_len, device=input_ids.device).unsqueeze(0)
            valid_pos_mask = positions < seq_len.unsqueeze(1)
            input_ids = input_ids.masked_fill(~valid_pos_mask, 0)

        if self.mask_padding:
            text_mask = input_ids == 0

        if drop_text:
            input_ids = torch.zeros_like(input_ids)

        text_embeds = self.text_embed(input_ids)
        if valid_pos_mask is not None:
            text_embeds = text_embeds.masked_fill(~valid_pos_mask.unsqueeze(-1), 0.0)

        if self.extra_modeling:
            freqs = self.freqs_cis[:max_seq_len, :]
            if valid_pos_mask is not None:
                freqs = freqs.unsqueeze(0) * valid_pos_mask.unsqueeze(-1).to(freqs.dtype)
            text_embeds = text_embeds + freqs

            if self.mask_padding:
                text_embeds = text_embeds.masked_fill(
                    text_mask.unsqueeze(-1).expand(-1, -1, text_embeds.size(-1)), 0.0
                )
                for block in self.text_blocks:
                    text_embeds = block(text_embeds)
                    text_embeds = text_embeds.masked_fill(
                        text_mask.unsqueeze(-1).expand(-1, -1, text_embeds.size(-1)), 0.0
                    )
            else:
                text_embeds = self.text_blocks(text_embeds)

        if self.average_upsampling:
            if torch.is_tensor(seq_len):
                target_lengths = seq_len.to(device=text_embeds.device, dtype=torch.long)
            else:
                target_lengths = torch.full(
                    (text_embeds.shape[0],), int(seq_len), device=text_embeds.device, dtype=torch.long
                )
            text_embeds = self.average_upsample_text_by_mask(text_embeds, ~text_mask, target_lengths)

        return text_embeds


class F5TTSUNetTextEmbedding(F5TTSTextEmbedding):
    r"""
    Constructs the character embedding and text encoder of the `"unett"` backbone, which offsets the embedding with
    a clamped lookup into the sinusoidal table and never average upsamples.

    Args:
        config ([`F5TTSConfig`]):
            Model configuration.
    """

    def forward(self, input_ids: torch.Tensor, seq_len: int, drop_text: bool = False) -> torch.Tensor:
        r"""
        Args:
            input_ids (`torch.Tensor`):
                Character ids of shape `(batch_size, text_length)`, padded with the filler id `0`.
            seq_len (`int`):
                Speech length the text is curtailed and padded to.
            drop_text (`bool`, *optional*, defaults to `False`):
                Whether to zero the text, for the unconditional branch of classifier free guidance.

        Returns:
            `torch.Tensor`: Encoded text of shape `(batch_size, seq_len, text_dim)`.
        """
        input_ids = input_ids[:, :seq_len]
        batch_size, text_len = input_ids.shape[0], input_ids.shape[1]
        input_ids = F.pad(input_ids, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = input_ids == 0

        if drop_text:
            input_ids = torch.zeros_like(input_ids)

        text_embeds = self.text_embed(input_ids)

        if self.extra_modeling:
            batch_start = torch.zeros((batch_size,), dtype=torch.long, device=input_ids.device)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_embeds = text_embeds + self.freqs_cis[pos_idx]

            if self.mask_padding:
                text_embeds = text_embeds.masked_fill(
                    text_mask.unsqueeze(-1).expand(-1, -1, text_embeds.size(-1)), 0.0
                )
                for block in self.text_blocks:
                    text_embeds = block(text_embeds)
                    text_embeds = text_embeds.masked_fill(
                        text_mask.unsqueeze(-1).expand(-1, -1, text_embeds.size(-1)), 0.0
                    )
            else:
                text_embeds = self.text_blocks(text_embeds)

        return text_embeds


class F5TTSInputEmbedding(nn.Module):
    r"""
    Constructs the projection that mixes the noised speech, the masked conditioning speech and the encoded text
    into the backbone dimension.

    Args:
        config ([`F5TTSConfig`]):
            Model configuration.
    """

    def __init__(self, config: F5TTSConfig):
        super().__init__()
        self.proj = nn.Linear(config.mel_dim * 2 + config.text_dim, config.hidden_size)
        self.conv_pos_embed = F5TTSConvPositionEmbedding(dim=config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        text_embeds: torch.Tensor,
        drop_audio_cond: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if drop_audio_cond:
            conditioning_features = torch.zeros_like(conditioning_features)

        hidden_states = self.proj(torch.cat((hidden_states, conditioning_features, text_embeds), dim=-1))
        return self.conv_pos_embed(hidden_states, padding_mask=padding_mask) + hidden_states


class F5TTSRotaryEmbedding(LlamaRotaryEmbedding):
    pass


@auto_docstring
class F5TTSPreTrainedModel(PreTrainedModel):
    config: F5TTSConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["F5TTSDecoderLayer", "F5TTSUNetLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _keys_to_ignore_on_load_unexpected = [r"rotary_embed\.inv_freq"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, F5TTSUNetRMSNorm):
            init.ones_(module.g)
        elif isinstance(module, F5TTSGlobalResponseNorm):
            init.zeros_(module.gamma)
            init.zeros_(module.beta)
        elif isinstance(module, F5TTSTextEmbedding):
            if module.extra_modeling:
                init.copy_(
                    module.freqs_cis,
                    precompute_freqs_cis(module.freqs_cis.shape[-1], module.precompute_max_pos),
                )
        elif isinstance(module, (F5TTSDecoderLayer, F5TTSAdaLayerNormFinal)):
            adaptive_norm = module.attn_norm if isinstance(module, F5TTSDecoderLayer) else module
            init.zeros_(adaptive_norm.linear.weight)
            init.zeros_(adaptive_norm.linear.bias)
        elif isinstance(module, F5TTSModel):
            init.zeros_(module.proj_out.weight)
            init.zeros_(module.proj_out.bias)
        else:
            super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    The F5-TTS diffusion transformer backbone. It reads the noised speech, the masked conditioning speech, the
    character sequence and the flow time step, and predicts the vector field of the conditional flow.
    """
)
class F5TTSModel(F5TTSPreTrainedModel):
    def __init__(self, config: F5TTSConfig):
        super().__init__(config)
        self.time_embed = F5TTSTimestepEmbedding(config.hidden_size)
        self.text_embed = F5TTSTextEmbedding(config)
        self.input_embed = F5TTSInputEmbedding(config)
        self.rotary_embed = F5TTSRotaryEmbedding(config=config)

        self.transformer_blocks = nn.ModuleList(
            [F5TTSDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.long_skip_connection = (
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
            if config.long_skip_connection
            else None
        )
        self.norm_out = F5TTSAdaLayerNormFinal(config.hidden_size, eps=config.layer_norm_eps)
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

        self.text_cond = None
        self.text_uncond = None

        self.post_init()

    def clear_cache(self):
        r"""Drops the encoded text kept across the steps of one sampling run."""
        self.text_cond, self.text_uncond = None, None

    def get_input_embed(
        self,
        hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        input_ids: torch.Tensor,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Encodes the text and mixes it with the noised and conditioning speech.

        Args:
            hidden_states (`torch.Tensor`):
                Noised speech of shape `(batch_size, sequence_length, mel_dim)`.
            conditioning_features (`torch.Tensor`):
                Masked conditioning speech of shape `(batch_size, sequence_length, mel_dim)`.
            input_ids (`torch.Tensor`):
                Character ids of shape `(batch_size, text_length)`.
            drop_audio_cond (`bool`, *optional*, defaults to `False`):
                Whether to zero the conditioning speech.
            drop_text (`bool`, *optional*, defaults to `False`):
                Whether to zero the text.
            cache (`bool`, *optional*, defaults to `True`):
                Whether the encoded text is reused across sampling steps.
            padding_mask (`torch.Tensor`, *optional*):
                Boolean mask of shape `(batch_size, sequence_length)`, `True` on valid speech frames.

        Returns:
            `torch.Tensor`: Input embedding of shape `(batch_size, sequence_length, hidden_size)`.
        """
        if self.text_uncond is None or self.text_cond is None or not cache:
            seq_len = hidden_states.shape[1] if padding_mask is None else padding_mask.sum(dim=1)
            text_embeds = self.text_embed(input_ids, seq_len=seq_len, drop_text=drop_text)
            if cache:
                if drop_text:
                    self.text_uncond = text_embeds
                else:
                    self.text_cond = text_embeds

        if cache:
            text_embeds = self.text_uncond if drop_text else self.text_cond

        return self.input_embed(
            hidden_states,
            conditioning_features,
            text_embeds,
            drop_audio_cond=drop_audio_cond,
            padding_mask=padding_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        input_ids: torch.Tensor,
        timestep: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor`):
                Noised speech of shape `(batch_size, sequence_length, mel_dim)`.
            conditioning_features (`torch.Tensor`):
                Masked conditioning speech of shape `(batch_size, sequence_length, mel_dim)`.
            input_ids (`torch.Tensor`):
                Character ids of shape `(batch_size, text_length)`, padded with the filler id `0`.
            timestep (`torch.Tensor`):
                Flow time step in `[0, 1]`, either a scalar or of shape `(batch_size,)`.
            padding_mask (`torch.Tensor`, *optional*):
                Boolean mask of shape `(batch_size, sequence_length)`, `True` on valid speech frames.
            drop_audio_cond (`bool`, *optional*, defaults to `False`):
                Whether to zero the conditioning speech.
            drop_text (`bool`, *optional*, defaults to `False`):
                Whether to zero the text.
            cfg_infer (`bool`, *optional*, defaults to `False`):
                Whether to stack the conditional and unconditional branches of classifier free guidance into one
                batch of size `2 * batch_size`.
            cache (`bool`, *optional*, defaults to `False`):
                Whether the encoded text is reused across sampling steps.

        Returns:
            `torch.Tensor`: Predicted vector field of shape `(batch_size, sequence_length, mel_dim)`, or
            `(2 * batch_size, sequence_length, mel_dim)` when `cfg_infer` is set.
        """
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        if timestep.ndim == 0:
            timestep = timestep.repeat(batch_size)

        timestep_embedding = self.time_embed(timestep)
        if cfg_infer:
            embeds_cond = self.get_input_embed(
                hidden_states,
                conditioning_features,
                input_ids,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                padding_mask=padding_mask,
            )
            embeds_uncond = self.get_input_embed(
                hidden_states,
                conditioning_features,
                input_ids,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                padding_mask=padding_mask,
            )
            hidden_states = torch.cat((embeds_cond, embeds_uncond), dim=0)
            timestep_embedding = torch.cat((timestep_embedding, timestep_embedding), dim=0)
            padding_mask = torch.cat((padding_mask, padding_mask), dim=0) if padding_mask is not None else None
        else:
            hidden_states = self.get_input_embed(
                hidden_states,
                conditioning_features,
                input_ids,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                padding_mask=padding_mask,
            )

        position_ids = torch.arange(seq_len, device=hidden_states.device)[None, :]
        position_embeddings = self.rotary_embed(hidden_states, position_ids)
        attention_mask = (
            build_attention_mask(padding_mask, hidden_states.dtype) if self.config.attn_mask_enabled else None
        )

        residual = hidden_states if self.long_skip_connection is not None else None

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                timestep_embedding,
                position_embeddings,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **kwargs,
            )

        if self.long_skip_connection is not None:
            hidden_states = self.long_skip_connection(torch.cat((hidden_states, residual), dim=-1))

        hidden_states = self.norm_out(hidden_states, timestep_embedding)
        return self.proj_out(hidden_states)


@auto_docstring(
    custom_intro="""
    The E2-TTS flat UNet transformer backbone shipped alongside F5-TTS. It prepends the flow time step embedding to
    the sequence and joins every second half layer onto its mirrored first half counterpart.
    """
)
class F5TTSUNetModel(F5TTSPreTrainedModel):
    def __init__(self, config: F5TTSConfig):
        super().__init__(config)
        self.skip_connect_type = config.skip_connect_type
        self.depth = config.num_hidden_layers

        self.time_embed = F5TTSTimestepEmbedding(config.hidden_size)
        self.text_embed = F5TTSUNetTextEmbedding(config)
        self.input_embed = F5TTSInputEmbedding(config)
        self.rotary_embed = F5TTSRotaryEmbedding(config=config)

        self.layers = nn.ModuleList(
            [
                F5TTSUNetLayer(
                    config,
                    has_skip_projection=(
                        config.skip_connect_type == "concat" and idx >= config.num_hidden_layers // 2
                    ),
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm_out = F5TTSUNetRMSNorm(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

        self.text_cond = None
        self.text_uncond = None

        self.post_init()

    def clear_cache(self):
        r"""Drops the encoded text kept across the steps of one sampling run."""
        self.text_cond, self.text_uncond = None, None

    def get_input_embed(
        self,
        hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        input_ids: torch.Tensor,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
    ) -> torch.Tensor:
        r"""
        Encodes the text and mixes it with the noised and conditioning speech.

        Args:
            hidden_states (`torch.Tensor`):
                Noised speech of shape `(batch_size, sequence_length, mel_dim)`.
            conditioning_features (`torch.Tensor`):
                Masked conditioning speech of shape `(batch_size, sequence_length, mel_dim)`.
            input_ids (`torch.Tensor`):
                Character ids of shape `(batch_size, text_length)`.
            drop_audio_cond (`bool`, *optional*, defaults to `False`):
                Whether to zero the conditioning speech.
            drop_text (`bool`, *optional*, defaults to `False`):
                Whether to zero the text.
            cache (`bool`, *optional*, defaults to `True`):
                Whether the encoded text is reused across sampling steps.

        Returns:
            `torch.Tensor`: Input embedding of shape `(batch_size, sequence_length, hidden_size)`.
        """
        seq_len = hidden_states.shape[1]
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(input_ids, seq_len, drop_text=True)
                text_embeds = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(input_ids, seq_len, drop_text=False)
                text_embeds = self.text_cond
        else:
            text_embeds = self.text_embed(input_ids, seq_len, drop_text=drop_text)

        return self.input_embed(
            hidden_states, conditioning_features, text_embeds, drop_audio_cond=drop_audio_cond
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        input_ids: torch.Tensor,
        timestep: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor`):
                Noised speech of shape `(batch_size, sequence_length, mel_dim)`.
            conditioning_features (`torch.Tensor`):
                Masked conditioning speech of shape `(batch_size, sequence_length, mel_dim)`.
            input_ids (`torch.Tensor`):
                Character ids of shape `(batch_size, text_length)`, padded with the filler id `0`.
            timestep (`torch.Tensor`):
                Flow time step in `[0, 1]`, either a scalar or of shape `(batch_size,)`.
            padding_mask (`torch.Tensor`, *optional*):
                Boolean mask of shape `(batch_size, sequence_length)`, `True` on valid speech frames.
            drop_audio_cond (`bool`, *optional*, defaults to `False`):
                Whether to zero the conditioning speech.
            drop_text (`bool`, *optional*, defaults to `False`):
                Whether to zero the text.
            cfg_infer (`bool`, *optional*, defaults to `False`):
                Whether to stack the conditional and unconditional branches of classifier free guidance into one
                batch of size `2 * batch_size`.
            cache (`bool`, *optional*, defaults to `False`):
                Whether the encoded text is reused across sampling steps.

        Returns:
            `torch.Tensor`: Predicted vector field of shape `(batch_size, sequence_length, mel_dim)`, or
            `(2 * batch_size, sequence_length, mel_dim)` when `cfg_infer` is set.
        """
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        if timestep.ndim == 0:
            timestep = timestep.repeat(batch_size)

        timestep_embedding = self.time_embed(timestep)
        if cfg_infer:
            embeds_cond = self.get_input_embed(
                hidden_states, conditioning_features, input_ids, drop_audio_cond=False, drop_text=False, cache=cache
            )
            embeds_uncond = self.get_input_embed(
                hidden_states, conditioning_features, input_ids, drop_audio_cond=True, drop_text=True, cache=cache
            )
            hidden_states = torch.cat((embeds_cond, embeds_uncond), dim=0)
            timestep_embedding = torch.cat((timestep_embedding, timestep_embedding), dim=0)
            padding_mask = torch.cat((padding_mask, padding_mask), dim=0) if padding_mask is not None else None
        else:
            hidden_states = self.get_input_embed(
                hidden_states,
                conditioning_features,
                input_ids,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
            )

        hidden_states = torch.cat([timestep_embedding.unsqueeze(1), hidden_states], dim=1)
        if padding_mask is not None:
            padding_mask = F.pad(padding_mask, (1, 0), value=True)

        position_ids = torch.arange(seq_len + 1, device=hidden_states.device)[None, :]
        position_embeddings = self.rotary_embed(hidden_states, position_ids)
        attention_mask = (
            build_attention_mask(padding_mask, hidden_states.dtype) if self.config.attn_mask_enabled else None
        )

        skips = []
        for index, layer in enumerate(self.layers):
            is_first_half = (index + 1) <= (self.depth // 2)
            if is_first_half:
                skips.append(hidden_states)
                skip_connection = None
            else:
                skip_connection = skips.pop()

            hidden_states = layer(
                hidden_states,
                position_embeddings,
                skip_connection=skip_connection,
                skip_connect_type=self.skip_connect_type,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **kwargs,
            )

        hidden_states = self.norm_out(hidden_states)[:, 1:, :]
        return self.proj_out(hidden_states)


@dataclass
@auto_docstring(
    custom_intro="""
    Output of [`F5TTSForConditionalGeneration`].
    """
)
class F5TTSOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Conditional flow matching loss, the mean squared error between the predicted vector field and the target
        velocity over the frames of the randomly drawn infilling span.
    vector_field (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`):
        Predicted vector field of the conditional flow.
    span_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*, returned when `labels` is
    provided):
        Boolean mask of the infilling span the loss was taken over.
    conditioning_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`, *optional*,
    returned when `labels` is provided):
        Conditioning speech the span was masked out of.
    """

    loss: torch.FloatTensor | None = None
    vector_field: torch.FloatTensor | None = None
    span_mask: torch.BoolTensor | None = None
    conditioning_features: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    F5-TTS, a conditional flow matching text to speech model over log mel spectrograms. Training draws a random
    infilling span, noises the target spectrogram along the straight optimal transport path and regresses the
    backbone's vector field onto the target velocity inside that span. Generation integrates the same vector field
    from Gaussian noise with a fixed step solver, keeping the reference speech outside the generated span.
    """
)
class F5TTSForConditionalGeneration(F5TTSPreTrainedModel, F5TTSGenerationMixin):
    _tied_weights_keys = None

    def __init__(self, config: F5TTSConfig):
        super().__init__(config)
        self.model = F5TTSModel(config) if config.backbone == "dit" else F5TTSUNetModel(config)
        vocoder_class = BigVGANModel if isinstance(config.vocoder_config, BigVGANConfig) else VocosModel
        self.vocoder = vocoder_class(config.vocoder_config)
        self.mel_dim = config.mel_dim
        self.post_init()
        self.freeze_vocoder()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Loads an F5-TTS or E2-TTS checkpoint, from a published repository as it stands or from a directory
        [`~weight_conversion.convert`] wrote. A published checkpoint holds the backbone alone, so the vocoder of
        the mel front end it was trained against is read out of that vocoder's own repository and composed in.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                `"SWivid/F5-TTS"`, `"SWivid/E2-TTS"`, any key of `PUBLISHED_CHECKPOINTS`, or any repository id or
                directory holding one of the two layouts.
            model_args (`tuple`, *optional*):
                Positional arguments of [`~PreTrainedModel.from_pretrained`].
            kwargs (`dict`, *optional*):
                Keyword arguments of [`~PreTrainedModel.from_pretrained`]. `subfolder` names which of the
                checkpoints a published repository holds to load, and defaults to the entry of
                `DEFAULT_CHECKPOINTS` that repository names.

        Returns:
            [`F5TTSForConditionalGeneration`]: The loaded model, with the vocoder frozen again after loading
            replaced the parameters created by `__init__`.
        """
        from .weight_conversion import converted_checkpoint, is_published_layout

        subfolder = kwargs.get("subfolder") or None
        if (
            pretrained_model_name_or_path is not None
            and kwargs.get("config") is None
            and kwargs.get("state_dict") is None
            and is_published_layout(pretrained_model_name_or_path, subfolder)
        ):
            kwargs.pop("subfolder", None)
            pretrained_model_name_or_path = converted_checkpoint(pretrained_model_name_or_path, subfolder=subfolder)
        outputs = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = outputs[0] if isinstance(outputs, tuple) else outputs
        model.freeze_vocoder()
        return outputs

    def get_backbone(self):
        return self.model

    def freeze_vocoder(self):
        """Freezes the vocoder, which upstream loads pretrained and never optimizes."""
        for parameter in self.vocoder.parameters():
            parameter.requires_grad = False

    def vocode(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        r"""
        Turns a log mel spectrogram into a waveform.

        Args:
            mel_spectrogram (`torch.FloatTensor` of shape `(batch_size, mel_dim, num_frames)`):
                Log mel spectrogram to vocode.

        Returns:
            `torch.Tensor`: Waveform of shape `(batch_size, num_samples)`.
        """
        return self.vocoder(input_features=mel_spectrogram).audio_values

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> F5TTSOutput:
        r"""
        input_ids (`torch.Tensor` of shape `(batch_size, text_length)`):
            Character ids of the reference transcription followed by the text to speak, padded with the filler id
            `0`.
        input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`, *optional*):
            Noised log mel spectrogram to evaluate the vector field at. Required when `labels` is not given, and
            supplied by the sampler.
        conditioning_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`, *optional*):
            Conditioning log mel spectrogram, zeroed on the frames to be generated. Required when `labels` is not
            given, and supplied by the sampler.
        timestep (`torch.FloatTensor`, *optional*):
            Flow time step in `[0, 1]`, either a scalar or of shape `(batch_size,)`. Required when `labels` is not
            given, and supplied by the sampler.
        attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask over the spectrogram frames, `True` on valid frames.
        labels (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`, *optional*):
            Ground truth log mel spectrogram. Passing it runs one conditional flow matching training step and
            returns its loss.
        drop_audio_cond (`bool`, *optional*, defaults to `False`):
            Whether to zero the conditioning speech, for the unconditional branch of classifier free guidance.
        drop_text (`bool`, *optional*, defaults to `False`):
            Whether to zero the text, for the unconditional branch of classifier free guidance.
        cfg_infer (`bool`, *optional*, defaults to `False`):
            Whether to stack the conditional and unconditional branches into one batch of size `2 * batch_size`.
        cache (`bool`, *optional*, defaults to `False`):
            Whether the encoded text is reused across sampling steps.

        Returns:
            [`F5TTSOutput`]
        """
        if labels is None:
            if input_features is None or conditioning_features is None or timestep is None:
                raise ValueError(
                    "Without `labels`, `input_features`, `conditioning_features` and `timestep` are all required."
                )
            vector_field = self.model(
                input_features,
                conditioning_features,
                input_ids,
                timestep,
                padding_mask=attention_mask,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cfg_infer=cfg_infer,
                cache=cache,
                **kwargs,
            )
            return F5TTSOutput(vector_field=vector_field)

        batch_size, seq_len = labels.shape[0], labels.shape[1]
        device, dtype = labels.device, labels.dtype

        if attention_mask is None:
            lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
            attention_mask = lengths_to_mask(lengths, length=seq_len)
        else:
            lengths = attention_mask.sum(dim=1)

        frac_lengths = torch.zeros((batch_size,), device=device).float().uniform_(*self.config.frac_lengths_mask)
        span_mask = mask_from_frac_lengths(lengths, frac_lengths) & attention_mask

        noise = torch.randn_like(labels)
        timestep = torch.rand((batch_size,), dtype=dtype, device=device)
        expanded_timestep = timestep.unsqueeze(-1).unsqueeze(-1)
        noisy_features = (1 - expanded_timestep) * noise + expanded_timestep * labels
        target_velocity = labels - noise

        conditioning_features = torch.where(span_mask[..., None], torch.zeros_like(labels), labels)

        drop_audio_cond = torch.rand(()).item() < self.config.audio_drop_prob
        if torch.rand(()).item() < self.config.cond_drop_prob:
            drop_audio_cond, drop_text = True, True
        else:
            drop_text = False

        vector_field = self.model(
            noisy_features,
            conditioning_features,
            input_ids,
            timestep,
            padding_mask=attention_mask,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            **kwargs,
        )

        loss = F.mse_loss(vector_field, target_velocity, reduction="none")
        loss = loss[span_mask].mean()

        return F5TTSOutput(
            loss=loss,
            vector_field=vector_field,
            span_mask=span_mask,
            conditioning_features=conditioning_features,
        )


__all__ = [
    "F5TTSForConditionalGeneration",
    "F5TTSModel",
    "F5TTSOutput",
    "F5TTSPreTrainedModel",
    "F5TTSUNetModel",
]
