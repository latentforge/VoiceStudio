# Copyright 2024 LY Corporation and the LatentForge team. All rights reserved.
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
"""PyTorch PromptTTS++ model."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.fastspeech2_conformer.modeling_fastspeech2_conformer import (
    FastSpeech2ConformerAttention,
    FastSpeech2ConformerConvolutionModule,
    FastSpeech2ConformerMultiLayeredConv1d,
)
from transformers.utils import auto_docstring, logging

from .configuration_prompt_tts_pp import PromptTTSPPBigVGanConfig, PromptTTSPPConfig


logger = logging.get_logger(__name__)


def sequence_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """
    Builds a boolean mask that is `True` at every position below the corresponding length.

    Args:
        lengths (`torch.Tensor` of shape `(batch_size,)`):
            Length of each sequence in the batch.
        max_length (`int`, *optional*):
            Length to pad the mask to. Defaults to the largest entry of `lengths`.

    Returns:
        `torch.Tensor` of shape `(batch_size, max_length)`: The mask.
    """
    if max_length is None:
        max_length = int(lengths.max())
    positions = torch.arange(int(max_length), dtype=lengths.dtype, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def generate_alignment_path(durations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Expands per-phoneme durations into a hard monotonic alignment matrix.

    Args:
        durations (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Number of frames each phoneme spans.
        mask (`torch.Tensor` of shape `(batch_size, sequence_length, num_frames)`):
            Mask that is nonzero on the valid phoneme/frame pairs.

    Returns:
        `torch.Tensor` of shape `(batch_size, sequence_length, num_frames)`: The alignment matrix.
    """
    batch_size, sequence_length, num_frames = mask.shape
    cumulative_durations = torch.cumsum(durations, dim=1)
    path = sequence_mask(cumulative_durations.view(batch_size * sequence_length), num_frames).to(mask.dtype)
    path = path.view(batch_size, sequence_length, num_frames)
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    return path * mask


def to_log_scale(durations: torch.Tensor) -> torch.Tensor:
    """
    Takes the logarithm of the nonzero entries of `durations` and leaves the zeros untouched.

    Args:
        durations (`torch.Tensor`):
            Durations in frames.

    Returns:
        `torch.Tensor`: Log durations of the same shape.
    """
    return torch.where(durations != 0, durations.clamp_min(torch.finfo(durations.dtype).tiny).log(), durations)


def mdn_loss(
    log_pi: torch.Tensor,
    log_sigma: torch.Tensor,
    mu: torch.Tensor,
    target: torch.Tensor,
    log_pi_min: float = -7.0,
    log_sigma_min: float = -7.0,
    reduce: bool = True,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the negative log likelihood of `target` under a mixture of diagonal Gaussians.

    Args:
        log_pi (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians)` or
            `(batch_size, sequence_length, num_gaussians, dim)`):
            Log mixture weights. A four dimensional tensor selects the dimension-wise formulation, where each
            output dimension is modelled by its own one dimensional mixture.
        log_sigma (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians, dim)`):
            Log standard deviation of each mixture component.
        mu (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians, dim)`):
            Mean of each mixture component.
        target (`torch.Tensor` of shape `(batch_size, sequence_length, dim)`):
            Target values.
        log_pi_min (`float`, *optional*, defaults to -7.0):
            Lower clamp on `log_pi`, for numerical stability.
        log_sigma_min (`float`, *optional*, defaults to -7.0):
            Lower clamp on `log_sigma`, for numerical stability.
        reduce (`bool`, *optional*, defaults to `True`):
            Whether to average the per-step losses over the sequence.
        mask (`torch.Tensor` of shape `(batch_size, sequence_length, 1)`, *optional*):
            Boolean mask that is `True` on the valid steps.

    Returns:
        `torch.Tensor`: The negative log likelihood, of shape `(batch_size,)` when `reduce` is `True` and
        `(batch_size, sequence_length)` (or `(batch_size, sequence_length, dim)` in the dimension-wise
        formulation) otherwise.
    """
    dim_wise = log_pi.dim() == 4

    log_sigma = torch.clamp(log_sigma, min=log_sigma_min)
    log_pi = torch.clamp(log_pi, min=log_pi_min)

    target = target.unsqueeze(2).expand_as(log_sigma)

    # Center the target and clamp it within five standard deviations, for numerical stability.
    centered_target = target - mu
    scale = torch.exp(log_sigma)
    edge = 5 * scale
    centered_target = torch.where(centered_target > edge, edge, centered_target)
    centered_target = torch.where(centered_target < -edge, -edge, centered_target)

    distribution = torch.distributions.Normal(loc=0, scale=scale)
    log_prob = distribution.log_prob(centered_target)

    if dim_wise:
        loss = log_prob + log_pi
    else:
        # A diagonal covariance turns the joint log density into the sum of the per-dimension ones.
        loss = torch.sum(log_prob, dim=3) + log_pi

    if mask is not None:
        if loss.dim() == 4:
            mask_expand = ~mask.unsqueeze(-1).expand_as(loss)
        else:
            mask_expand = ~mask.expand_as(loss)
        loss = loss.masked_fill(mask_expand, -float("inf"))

    loss = -torch.logsumexp(loss, dim=2)

    if reduce:
        return torch.mean(loss, dim=1)
    return loss


def mdn_get_most_probable_sigma_and_mu(
    log_pi: torch.Tensor, log_sigma: torch.Tensor, mu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Selects the standard deviation and mean of the mixture component with the largest weight.

    Args:
        log_pi (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians)` or
            `(batch_size, sequence_length, num_gaussians, dim)`):
            Log mixture weights.
        log_sigma (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians, dim)`):
            Log standard deviation of each mixture component.
        mu (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians, dim)`):
            Mean of each mixture component.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The standard deviation and the mean of the most probable
        component, both of shape `(batch_size, sequence_length, dim)`.
    """
    dim_wise = log_pi.dim() == 4
    num_gaussians = mu.shape[2]
    _, max_component = torch.max(log_pi, dim=2)

    one_hot = F.one_hot(max_component, num_gaussians)
    if dim_wise:
        one_hot = one_hot.transpose(2, 3)
    else:
        one_hot = one_hot.unsqueeze(3).expand_as(mu)

    max_mu = torch.sum(mu * one_hot, dim=2)
    max_sigma = torch.exp(torch.sum(log_sigma * one_hot, dim=2))
    return max_sigma, max_mu


def mdn_sample_sigma_and_mu(
    log_pi: torch.Tensor, log_sigma: torch.Tensor, mu: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a mixture component and returns its standard deviation and mean.

    Args:
        log_pi (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians)` or
            `(batch_size, sequence_length, num_gaussians, dim)`):
            Log mixture weights.
        log_sigma (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians, dim)`):
            Log standard deviation of each mixture component.
        mu (`torch.Tensor` of shape `(batch_size, sequence_length, num_gaussians, dim)`):
            Mean of each mixture component.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The standard deviation and the mean of the sampled component,
        both of shape `(batch_size, sequence_length, dim)`.
    """
    dim_wise = log_pi.dim() == 4

    if dim_wise:
        probabilities = log_pi.exp().squeeze(1).transpose(1, 2)
    else:
        probabilities = log_pi.exp()

    selected_indices = torch.distributions.Categorical(probs=probabilities).sample()
    one_hot = F.one_hot(selected_indices, probabilities.shape[-1])

    if dim_wise:
        one_hot = one_hot.unsqueeze(1).transpose(2, 3)
    else:
        one_hot = one_hot.unsqueeze(3).expand_as(mu)

    sampled_mu = torch.sum(mu * one_hot, dim=2)
    sampled_sigma = torch.exp(torch.sum(log_sigma * one_hot, dim=2))
    return sampled_sigma, sampled_mu


class PromptTTSPPRelPositionalEncoding(nn.Module):
    r"""
    Constructs the relative positional encoding of the conformer encoder.

    Both variants of https://github.com/espnet/espnet/pull/2816 are parameter free and therefore
    indistinguishable in a checkpoint, so the variant is read from the configuration.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.input_scale = math.sqrt(self.embed_dim)
        self.rel_pos_type = config.rel_pos_type
        self.max_len = config.max_source_positions
        self.dropout = nn.Dropout(p=config.encoder_positional_dropout_rate)
        self.pos_enc = nn.Buffer(self.build_pos_enc(self.max_len), persistent=False)

    def build_pos_enc(self, length: int, device=None, dtype=None) -> torch.Tensor:
        """
        Builds the sinusoidal table the encoding slices its positional embedding out of.

        Args:
            length (`int`):
                Number of positions to tabulate.
            device (`torch.device`, *optional*):
                Device to build the table on.
            dtype (`torch.dtype`, *optional*):
                Data type of the table.

        Returns:
            `torch.Tensor`: The table, of shape `(1, length, embed_dim)` for the legacy variant and
            `(1, 2 * length - 1, embed_dim)` for the new one.
        """
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.embed_dim)
        )
        if self.rel_pos_type == "legacy":
            # Positions run backwards, so the table's values depend on `length` itself.
            position = torch.arange(length - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
            pos_enc = torch.zeros(length, self.embed_dim)
            pos_enc[:, 0::2] = torch.sin(position * div_term)
            pos_enc[:, 1::2] = torch.cos(position * div_term)
            pos_enc = pos_enc.unsqueeze(0)
        else:
            position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
            pos_enc_positive = torch.zeros(length, self.embed_dim)
            pos_enc_negative = torch.zeros(length, self.embed_dim)
            pos_enc_positive[:, 0::2] = torch.sin(position * div_term)
            pos_enc_positive[:, 1::2] = torch.cos(position * div_term)
            pos_enc_negative[:, 0::2] = torch.sin(-1 * position * div_term)
            pos_enc_negative[:, 1::2] = torch.cos(-1 * position * div_term)
            pos_enc_positive = torch.flip(pos_enc_positive, [0]).unsqueeze(0)
            pos_enc_negative = pos_enc_negative[1:].unsqueeze(0)
            pos_enc = torch.cat([pos_enc_positive, pos_enc_negative], dim=1)
        return pos_enc.to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Encoder input.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: The scaled input and the positional embedding, of shape
            `(1, sequence_length, hidden_size)` for the legacy variant and
            `(1, 2 * sequence_length - 1, hidden_size)` for the new one.
        """
        sequence_length = hidden_states.size(1)
        required = sequence_length if self.rel_pos_type == "legacy" else 2 * sequence_length - 1
        if self.pos_enc.size(1) < required:
            self.pos_enc = self.build_pos_enc(
                sequence_length, device=hidden_states.device, dtype=hidden_states.dtype
            )
        elif self.pos_enc.dtype != hidden_states.dtype or self.pos_enc.device != hidden_states.device:
            self.pos_enc = self.pos_enc.to(device=hidden_states.device, dtype=hidden_states.dtype)

        hidden_states = hidden_states * self.input_scale
        if self.rel_pos_type == "legacy":
            pos_emb = self.pos_enc[:, :sequence_length]
        else:
            center_idx = self.pos_enc.size(1) // 2
            pos_emb = self.pos_enc[:, center_idx - sequence_length + 1 : center_idx + sequence_length]
        return self.dropout(hidden_states), self.dropout(pos_emb)


class PromptTTSPPAttention(FastSpeech2ConformerAttention):
    r"""
    Constructs the conformer self-attention with relative position encoding.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__(config, config.encoder_config)
        self.rel_pos_type = config.rel_pos_type

    def shift_relative_position_tensor(self, pos_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_tensor (`torch.Tensor` of shape `(batch_size, num_heads, time, time)` for the legacy variant and
                `(batch_size, num_heads, time, 2 * time - 1)` for the new one):
                Query/position-embedding scores.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_heads, time, time)`: The shifted scores.
        """
        if self.rel_pos_type != "legacy":
            return super().shift_relative_position_tensor(pos_tensor)

        zero_pad = torch.zeros((*pos_tensor.size()[:3], 1), device=pos_tensor.device, dtype=pos_tensor.dtype)
        pos_tensor_padded = torch.cat([zero_pad, pos_tensor], dim=-1)
        pos_tensor_padded = pos_tensor_padded.view(
            *pos_tensor.size()[:2], pos_tensor.size(3) + 1, pos_tensor.size(2)
        )
        return pos_tensor_padded[:, :, 1:].view_as(pos_tensor)


class PromptTTSPPMultiLayeredConv1d(FastSpeech2ConformerMultiLayeredConv1d):
    r"""
    Constructs the position-wise convolutions of a conformer block.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__(config, config.encoder_config)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input of the position-wise convolutions.
            mask (`torch.Tensor` of shape `(batch_size, sequence_length, 1)`, *optional*):
                Mask that is zero on the padded steps.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: The convolved input.
        """
        if mask is None:
            mask = hidden_states.new_ones(hidden_states.size(0), hidden_states.size(1), 1)
        hidden_states = hidden_states * mask
        hidden_states = torch.relu(self.conv1(hidden_states.transpose(-1, 1))).transpose(-1, 1) * mask
        hidden_states = self.dropout(hidden_states)
        return self.conv2(hidden_states.transpose(-1, 1)).transpose(-1, 1) * mask


class PromptTTSPPConvolutionModule(FastSpeech2ConformerConvolutionModule):
    r"""
    Constructs the convolution module of a conformer block.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__(config, config.encoder_config)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input of the convolution module.
            mask (`torch.Tensor` of shape `(batch_size, sequence_length, 1)`, *optional*):
                Mask that is zero on the padded steps.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: The convolved input.
        """
        if mask is None:
            mask = hidden_states.new_ones(hidden_states.size(0), hidden_states.size(1), 1)
        hidden_states = hidden_states.transpose(1, 2)
        mask = mask.transpose(1, 2)

        hidden_states = self.pointwise_conv1(hidden_states) * mask
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        hidden_states = self.depthwise_conv(hidden_states) * mask
        hidden_states = self.activation(self.norm(hidden_states))

        hidden_states = self.pointwise_conv2(hidden_states) * mask
        return hidden_states.transpose(1, 2)


class PromptTTSPPEncoderLayer(nn.Module):
    r"""
    Constructs a conformer block, that is a macaron feed-forward module, self-attention, a convolution module and
    a second feed-forward module, each of them masked on the padded steps.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.self_attn = PromptTTSPPAttention(config)
        self.feed_forward = PromptTTSPPMultiLayeredConv1d(config)

        self.macaron_style = config.use_macaron_style_in_conformer
        if self.macaron_style:
            self.feed_forward_macaron = PromptTTSPPMultiLayeredConv1d(config)
            self.ff_macaron_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        self.use_cnn_module = config.use_cnn_in_conformer
        if self.use_cnn_module:
            self.conv_module = PromptTTSPPConvolutionModule(config)
            self.conv_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ff_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.encoder_dropout_rate)
        self.normalize_before = config.encoder_normalize_before
        self.concat_after = config.encoder_concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Block input.
            pos_emb (`torch.Tensor`):
                Relative positional embedding.
            attention_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Mask that is zero on the padded steps.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: The block output.
        """
        mask = attention_mask.transpose(1, 2).to(hidden_states.dtype)
        hidden_states = hidden_states * mask

        if self.macaron_style:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.ff_macaron_layer_norm(hidden_states)
            hidden_states = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(hidden_states, mask)
            )
            if not self.normalize_before:
                hidden_states = self.ff_macaron_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        attention_output, _ = self.self_attn(hidden_states, attention_mask=attention_mask, pos_emb=pos_emb)
        attention_output = attention_output * mask

        if self.concat_after:
            hidden_states = residual + self.concat_linear(torch.cat((hidden_states, attention_output), dim=-1))
        else:
            hidden_states = residual + self.dropout(attention_output)
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.use_cnn_module:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.conv_layer_norm(hidden_states)
            hidden_states = residual + self.dropout(self.conv_module(hidden_states, mask)) * mask
            if not self.normalize_before:
                hidden_states = self.conv_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.ff_layer_norm(hidden_states)
        hidden_states = residual + self.ff_scale * self.dropout(self.feed_forward(hidden_states, mask)) * mask
        if not self.normalize_before:
            hidden_states = self.ff_layer_norm(hidden_states)

        if self.use_cnn_module:
            hidden_states = self.final_layer_norm(hidden_states) * mask

        return hidden_states


class PromptTTSPPEncoder(nn.Module):
    r"""
    Constructs the conformer phoneme encoder.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.pos_enc = PromptTTSPPRelPositionalEncoding(config)
        self.layers = nn.ModuleList([PromptTTSPPEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.normalize_before = config.encoder_normalize_before
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Phoneme embeddings.
            attention_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Mask that is zero on the padded phonemes.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: The encoded phonemes.
        """
        hidden_states, pos_emb = self.pos_enc(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_emb, attention_mask)
        if self.normalize_before:
            hidden_states = self.after_norm(hidden_states)
        return hidden_states * attention_mask.transpose(1, 2).to(hidden_states.dtype)


class PromptTTSPPPredictorLayer(nn.Module):
    r"""
    Constructs one convolution layer of a variance predictor.

    Args:
        channels (`int`):
            Number of input and output channels.
        kernel_size (`int`):
            Kernel size of the convolution.
        dropout (`float`):
            Dropout rate applied to the layer output.
        layer_norm_eps (`float`):
            Epsilon of the layer normalization.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float, layer_norm_eps: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(channels, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Layer input.
            mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Mask that is zero on the padded steps.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, sequence_length)`: The layer output.
        """
        hidden_states = torch.relu(self.conv(hidden_states))
        hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        return self.dropout(hidden_states) * mask


class PromptTTSPPVariancePredictor(nn.Module):
    r"""
    Constructs a variance predictor, a stack of masked convolutions followed by a projection.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
        out_channels (`int`):
            Number of predicted channels.
        num_layers (`int`):
            Number of convolution layers.
        kernel_size (`int`):
            Kernel size of the convolutions.
        dropout (`float`):
            Dropout rate of the convolutions.
        detach (`bool`, *optional*, defaults to `False`):
            Whether to detach the input, cutting the gradient flowing back into the encoder.
    """

    def __init__(
        self,
        config: PromptTTSPPConfig,
        out_channels: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        detach: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PromptTTSPPPredictorLayer(config.hidden_size, kernel_size, dropout, config.variance_layer_norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.out_layer = nn.Conv1d(config.hidden_size, out_channels, 1)
        self.detach = detach

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, hidden_size, sequence_length)`):
                Predictor input.
            mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Mask that is zero on the padded steps.

        Returns:
            `torch.Tensor` of shape `(batch_size, out_channels, sequence_length)`: The prediction.
        """
        if self.detach:
            hidden_states = hidden_states.detach()
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return self.out_layer(hidden_states) * mask


class PromptTTSPPMDNLayer(nn.Module):
    r"""
    Constructs a mixture density network layer mapping its input to the parameters of a mixture of Gaussians with
    diagonal covariance.

    Args:
        in_dim (`int`):
            Dimensionality of the input.
        out_dim (`int`):
            Dimensionality of the modelled variable.
        num_gaussians (`int`):
            Number of mixture components.
        dim_wise (`bool`, *optional*, defaults to `True`):
            Whether each output dimension is modelled by its own one dimensional mixture.
    """

    def __init__(self, in_dim: int, out_dim: int, num_gaussians: int, dim_wise: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.num_gaussians = num_gaussians
        self.dim_wise = dim_wise

        self.log_pi = nn.Linear(in_dim, out_dim * num_gaussians if dim_wise else num_gaussians)
        self.log_sigma = nn.Linear(in_dim, out_dim * num_gaussians)
        self.mu = nn.Linear(in_dim, out_dim * num_gaussians)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, in_dim)`):
                Layer input.

        Returns:
            `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: The log mixture weights, of shape
            `(batch_size, sequence_length, num_gaussians)` or `(batch_size, sequence_length, num_gaussians,
            out_dim)`, and the log standard deviations and means, both of shape `(batch_size, sequence_length,
            num_gaussians, out_dim)`.
        """
        batch_size = hidden_states.shape[0]
        if self.dim_wise:
            log_pi = self.log_pi(hidden_states).view(batch_size, -1, self.num_gaussians, self.out_dim)
            log_pi = F.log_softmax(log_pi, dim=2)
        else:
            log_pi = F.log_softmax(self.log_pi(hidden_states), dim=2)
        log_sigma = self.log_sigma(hidden_states).view(batch_size, -1, self.num_gaussians, self.out_dim)
        mu = self.mu(hidden_states).view(batch_size, -1, self.num_gaussians, self.out_dim)
        return log_pi, log_sigma, mu


class PromptTTSPPDurationPredictor(nn.Module):
    r"""
    Constructs the duration predictor, a stack of masked convolutions followed by a mixture density network over
    the log duration of each phoneme.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PromptTTSPPPredictorLayer(
                    config.hidden_size,
                    config.duration_predictor_kernel_size,
                    config.duration_predictor_dropout,
                    config.variance_layer_norm_eps,
                )
                for _ in range(config.duration_predictor_layers)
            ]
        )
        self.out_layer = PromptTTSPPMDNLayer(
            config.hidden_size, 1, config.duration_predictor_num_gaussians, dim_wise=True
        )
        self.detach = config.stop_gradient_from_duration_predictor
        self.disable_amp = config.disable_mdn_autocast

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, hidden_size, sequence_length)`):
                Encoded phonemes.
            mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Mask that is zero on the padded phonemes.

        Returns:
            `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: The mixture parameters of the log duration.
        """
        if self.detach:
            hidden_states = hidden_states.detach()
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        # The mixture density network is kept in full precision, autocast destabilizes its training.
        if self.disable_amp:
            hidden_states = hidden_states.float()
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                return self.out_layer(hidden_states.transpose(-1, -2))
        return self.out_layer(hidden_states.transpose(-1, -2))

    def predict_log_durations(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, hidden_size, sequence_length)`):
                Encoded phonemes.
            mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Mask that is zero on the padded phonemes.

        Returns:
            `torch.Tensor` of shape `(batch_size, 1, sequence_length)`: The mean of the log normal duration of
            the most probable mixture component.
        """
        log_pi, log_sigma, mu = self(hidden_states, mask)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        log_durations = mu + sigma.pow(2).clamp_min(1e-14) / 2
        return log_durations.transpose(-1, -2)


class PromptTTSPPFramePriorNetwork(nn.Module):
    r"""
    Constructs the frame prior network, a stack of residual convolutions run on the length regulated features.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.input_scale = math.sqrt(config.hidden_size)
        self.max_len = config.max_source_positions
        self.pos_enc = nn.Buffer(self.build_pos_enc(config.hidden_size, self.max_len), persistent=False)
        self.pos_dropout = nn.Dropout(config.frame_prior_positional_dropout)
        self.norm_emb = nn.LayerNorm(config.hidden_size, eps=config.variance_layer_norm_eps)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    config.frame_prior_kernel_size,
                    padding=config.frame_prior_kernel_size // 2,
                )
                for _ in range(config.frame_prior_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(config.hidden_size, eps=config.variance_layer_norm_eps)
                for _ in range(config.frame_prior_layers)
            ]
        )
        self.dropout = nn.Dropout(config.frame_prior_dropout)

    @staticmethod
    def build_pos_enc(embed_dim: int, length: int, device=None, dtype=None) -> torch.Tensor:
        """
        Builds the absolute sinusoidal positional encoding added to the length regulated features.

        Args:
            embed_dim (`int`):
                Dimensionality of the encoding.
            length (`int`):
                Number of positions to tabulate.
            device (`torch.device`, *optional*):
                Device to build the table on.
            dtype (`torch.dtype`, *optional*):
                Data type of the table.

        Returns:
            `torch.Tensor` of shape `(1, length, embed_dim)`: The encoding.
        """
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
        pos_enc = torch.zeros(length, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0).to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, hidden_size, num_frames)`):
                Length regulated features.
            mask (`torch.Tensor` of shape `(batch_size, 1, num_frames)`):
                Mask that is zero on the padded frames.

        Returns:
            `torch.Tensor` of shape `(batch_size, hidden_size, num_frames)`: The frame level features.
        """
        num_frames = hidden_states.size(-1)
        if self.pos_enc.size(1) < num_frames:
            self.pos_enc = self.build_pos_enc(
                hidden_states.size(1), num_frames, device=hidden_states.device, dtype=hidden_states.dtype
            )
        elif self.pos_enc.dtype != hidden_states.dtype or self.pos_enc.device != hidden_states.device:
            self.pos_enc = self.pos_enc.to(device=hidden_states.device, dtype=hidden_states.dtype)

        hidden_states = hidden_states * mask
        hidden_states = hidden_states.transpose(1, 2) * self.input_scale + self.pos_enc[:, :num_frames]
        hidden_states = self.pos_dropout(hidden_states)
        hidden_states = self.norm_emb(hidden_states).transpose(1, 2)

        for conv, norm in zip(self.convs, self.norms):
            residual = self.dropout(F.gelu(conv(hidden_states * mask)))
            hidden_states = norm((hidden_states + residual).transpose(1, 2)).transpose(1, 2)

        return hidden_states * mask


class PromptTTSPPVarianceAdaptor(nn.Module):
    r"""
    Constructs the variance adaptor: a mixture density duration predictor, length regulation, a frame prior
    network and a pitch predictor whose two channels are log continuous f0 and voicing.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.duration_predictor = PromptTTSPPDurationPredictor(config)
        self.pitch_predictor = PromptTTSPPVariancePredictor(
            config,
            out_channels=2,
            num_layers=config.pitch_predictor_layers,
            kernel_size=config.pitch_predictor_kernel_size,
            dropout=config.pitch_predictor_dropout,
            detach=config.stop_gradient_from_pitch_predictor,
        )
        self.pitch_embed = nn.Conv1d(1, config.hidden_size, config.pitch_embed_kernel_size)

        self.use_energy_predictor = config.use_energy_predictor
        if self.use_energy_predictor:
            self.energy_predictor = PromptTTSPPVariancePredictor(
                config,
                out_channels=1,
                num_layers=config.energy_predictor_layers,
                kernel_size=config.energy_predictor_kernel_size,
                dropout=config.energy_predictor_dropout,
                detach=config.stop_gradient_from_energy_predictor,
            )
            self.energy_embed = nn.Conv1d(1, config.hidden_size, config.energy_embed_kernel_size)

        self.frame_prior_network = PromptTTSPPFramePriorNetwork(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        phoneme_mask: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        log_f0: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
    ) -> tuple:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, hidden_size, sequence_length)`):
                Encoded phonemes with the style embedding added.
            phoneme_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                Mask that is zero on the padded phonemes.
            frame_mask (`torch.Tensor` of shape `(batch_size, 1, num_frames)`, *optional*):
                Mask that is zero on the padded frames. Required when `durations` is given.
            durations (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
                Ground truth durations. When `None`, the predicted durations are used and both log continuous f0
                and voicing are taken from the pitch predictor rather than from `log_f0`.
            log_f0 (`torch.Tensor` of shape `(batch_size, 1, num_frames)`, *optional*):
                Ground truth log continuous f0, embedded back into the frame level features.
            energy (`torch.Tensor` of shape `(batch_size, 1, num_frames)`, *optional*):
                Ground truth energy, embedded back into the frame level features.

        Returns:
            `tuple`: The frame level features, the frame mask, the mixture parameters of the log duration, the
            predicted log continuous f0, the predicted voicing and the predicted energy.
        """
        duration_outputs = self.duration_predictor(hidden_states, phoneme_mask)

        if durations is None:
            log_durations = self.duration_predictor.predict_log_durations(hidden_states, phoneme_mask)
            durations = log_durations.exp().round().clamp_min(1).long() * phoneme_mask.long()
            frame_lengths = durations.squeeze(1).sum(dim=-1)
            frame_mask = sequence_mask(frame_lengths).unsqueeze(1).to(hidden_states.dtype)
            durations = durations.to(hidden_states.dtype)

        path_mask = phoneme_mask.unsqueeze(-1) * frame_mask.unsqueeze(2)
        alignment_path = generate_alignment_path(durations.squeeze(1), path_mask.squeeze(1))
        hidden_states = hidden_states @ alignment_path

        hidden_states = self.frame_prior_network(hidden_states, frame_mask)

        log_f0_pred, vuv_pred = self.pitch_predictor(hidden_states, frame_mask).split(1, dim=1)
        pitch_embedding = self.pitch_embed(log_f0 if log_f0 is not None else log_f0_pred) * frame_mask

        energy_pred = None
        energy_embedding = 0
        if self.use_energy_predictor:
            energy_pred = self.energy_predictor(hidden_states, frame_mask)
            energy_embedding = self.energy_embed(energy if energy is not None else energy_pred) * frame_mask

        hidden_states = hidden_states + pitch_embedding + energy_embedding

        return hidden_states, frame_mask, duration_outputs, log_f0_pred, vuv_pred, energy_pred


class PromptTTSPPReferenceEncoder(nn.Module):
    r"""
    Constructs the reference encoder of the global style token module, a stack of strided 2D convolutions
    summarized by a gated recurrent unit.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.conv_layers = config.reference_encoder_conv_layers
        self.conv_stride = config.reference_encoder_conv_stride
        kernel_size = config.reference_encoder_conv_kernel_size
        padding = (kernel_size - 1) // 2

        convs = []
        for i in range(self.conv_layers):
            in_channels = 1 if i == 0 else config.reference_encoder_conv_channels[i - 1]
            out_channels = config.reference_encoder_conv_channels[i]
            convs += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=self.conv_stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.convs = nn.Sequential(*convs)

        gru_in_units = config.num_mel_bins
        for _ in range(self.conv_layers):
            gru_in_units = (gru_in_units - kernel_size + 2 * padding) // self.conv_stride + 1
        gru_in_units *= config.reference_encoder_conv_channels[-1]
        self.gru = nn.GRU(
            gru_in_units,
            config.reference_encoder_gru_units,
            config.reference_encoder_gru_layers,
            batch_first=True,
        )

    def forward(self, spectrogram: torch.Tensor, spectrogram_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            spectrogram (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Reference mel spectrogram.
            spectrogram_lengths (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                Number of valid frames of each reference spectrogram.

        Returns:
            `torch.Tensor` of shape `(batch_size, gru_units, 1)`: The reference embedding.
        """
        batch_size = spectrogram.size(0)
        hidden_states = self.convs(spectrogram.transpose(1, 2).unsqueeze(1)).transpose(1, 2)
        hidden_states = hidden_states.contiguous().view(batch_size, hidden_states.size(1), -1)
        self.gru.flatten_parameters()
        if spectrogram_lengths is None:
            _, reference_embedding = self.gru(hidden_states)
        else:
            # Lengths of the sub-sampled features the convolutions produced.
            packed_lengths = torch.ceil(
                spectrogram_lengths.float() / (self.conv_stride**self.conv_layers)
            ).long()
            packed_lengths = torch.clamp(packed_lengths, 1)
            packed = nn.utils.rnn.pack_padded_sequence(
                hidden_states, packed_lengths.to("cpu"), batch_first=True, enforce_sorted=False
            )
            _, reference_embedding = self.gru(packed)
        return reference_embedding[-1].unsqueeze(-1)


class PromptTTSPPStyleTokenAttention(nn.Module):
    r"""
    Constructs the attention of the style token layer, which queries the style tokens with the reference
    embedding.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        if config.gst_token_dim % config.gst_heads != 0:
            raise ValueError("gst_token_dim must be divisible by gst_heads.")
        self.num_heads = config.gst_heads
        self.head_dim = config.gst_token_dim // config.gst_heads
        self.linear_q = nn.Linear(config.reference_encoder_gru_units, config.gst_token_dim)
        self.linear_k = nn.Linear(self.head_dim, config.gst_token_dim)
        self.linear_v = nn.Linear(self.head_dim, config.gst_token_dim)
        self.linear_out = nn.Linear(config.gst_token_dim, config.gst_token_dim)

    def forward(self, reference_embedding: torch.Tensor, style_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reference_embedding (`torch.Tensor` of shape `(batch_size, 1, gru_units)`):
                Reference embedding used as the query.
            style_tokens (`torch.Tensor` of shape `(batch_size, gst_tokens, gst_token_dim // gst_heads)`):
                Style tokens used as keys and values.

        Returns:
            `torch.Tensor` of shape `(batch_size, 1, gst_token_dim)`: The style embedding.
        """
        batch_size = reference_embedding.shape[0]
        query = self.linear_q(reference_embedding).view(batch_size, -1, self.num_heads, self.head_dim)
        key = self.linear_k(style_tokens).view(batch_size, -1, self.num_heads, self.head_dim)
        value = self.linear_v(style_tokens).view(batch_size, -1, self.num_heads, self.head_dim)
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))

        # The scores are scaled by the full model width rather than by the head width.
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dim * self.num_heads)
        scores = F.softmax(scores, dim=-1)
        attention_output = scores @ value
        attention_output = attention_output.transpose(-1, -2).contiguous().view(batch_size, 1, -1)
        return self.linear_out(attention_output)


class PromptTTSPPStyleEncoder(nn.Module):
    r"""
    Constructs the global style token encoder, which turns a reference mel spectrogram into a style embedding.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.reference_encoder = PromptTTSPPReferenceEncoder(config)
        self.style_tokens = nn.Parameter(torch.randn(config.gst_tokens, config.gst_token_dim // config.gst_heads))
        self.attention = PromptTTSPPStyleTokenAttention(config)

    def forward(self, spectrogram: torch.Tensor, spectrogram_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            spectrogram (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Reference mel spectrogram.
            spectrogram_lengths (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                Number of valid frames of each reference spectrogram.

        Returns:
            `torch.Tensor` of shape `(batch_size, gst_token_dim, 1)`: The style embedding.
        """
        reference_embedding = self.reference_encoder(spectrogram, spectrogram_lengths)
        batch_size = reference_embedding.size(0)
        style_tokens = torch.tanh(self.style_tokens).unsqueeze(0).expand(batch_size, -1, -1)
        style_embedding = self.attention(reference_embedding.transpose(-1, -2), style_tokens)
        return style_embedding.squeeze(1).unsqueeze(-1)


class PromptTTSPPPromptEncoder(nn.Module):
    r"""
    Constructs the prompt encoder, a BERT encoder whose pooled classification token is mapped to a style
    embedding by a multilayer perceptron.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.bert = BertModel(config.prompt_encoder_config)
        self.adapter = nn.Sequential(
            nn.Linear(config.prompt_encoder_config.hidden_size, config.prompt_adapter_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.prompt_adapter_hidden_size, config.prompt_adapter_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.prompt_adapter_hidden_size, config.hidden_size),
        )

    def freeze_bert(self):
        """
        Freezes every BERT parameter but the attention of its last layer, which is the only part upstream trains.

        The layers left frozen keep their dropout, since upstream trains the last layer's attention through it.
        """
        for parameter in self.bert.parameters():
            parameter.requires_grad = False
        for parameter in self.bert.encoder.layer[-1].attention.parameters():
            parameter.requires_grad = True

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Style prompt token ids.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask that is zero on the padded prompt tokens.

        Returns:
            `torch.Tensor` of shape `(batch_size, hidden_size, 1)`: The prompt embedding.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.adapter(outputs.last_hidden_state[:, 0, :]).unsqueeze(-1)


class PromptTTSPPDenoiserResidualBlock(nn.Module):
    r"""
    Constructs one residual block of the denoiser, a gated dilated convolution conditioned on the frame level
    features and on the diffusion step embedding.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
        dilation (`int`):
            Dilation of the block's convolution.
    """

    def __init__(self, config: PromptTTSPPConfig, dilation: int):
        super().__init__()
        channels = config.denoiser_channels
        kernel_size = config.denoiser_kernel_size
        self.dilated_conv = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size,
            padding=(kernel_size * dilation - dilation) // 2,
            dilation=dilation,
        )
        self.diffusion_projection = nn.Linear(channels, channels)
        self.conditioner_projection = nn.Conv1d(config.hidden_size, 2 * channels, 1)
        self.output_projection = nn.Conv1d(channels, 2 * channels, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        diffusion_step: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, denoiser_channels, num_frames)`):
                Block input.
            conditioning (`torch.Tensor` of shape `(batch_size, hidden_size, num_frames)`):
                Frame level features.
            diffusion_step (`torch.Tensor` of shape `(batch_size, denoiser_channels)`):
                Diffusion step embedding.
            mask (`torch.Tensor` of shape `(batch_size, 1, num_frames)`, *optional*):
                Mask that is zero on the padded frames.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: The block output and its skip connection.
        """
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioning = self.conditioner_projection(conditioning)

        gated = self.dilated_conv(hidden_states + diffusion_step) + conditioning
        gate, filters = torch.chunk(gated, 2, dim=1)
        gated = torch.sigmoid(gate) * torch.tanh(filters)

        gated = self.output_projection(gated)
        if mask is not None:
            gated = gated * mask
        residual, skip = torch.chunk(gated, 2, dim=1)
        return (hidden_states + residual) / math.sqrt(2.0), skip


class PromptTTSPPDenoiser(nn.Module):
    r"""
    Constructs the denoiser of the diffusion decoder, a non-causal stack of gated dilated convolutions predicting
    the noise added to a mel spectrogram.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        channels = config.denoiser_channels
        self.channels = channels
        self.input_projection = nn.Conv1d(config.num_mel_bins, channels, 1)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 4), nn.Mish(), nn.Linear(channels * 4, channels))
        self.residual_layers = nn.ModuleList(
            [
                PromptTTSPPDenoiserResidualBlock(config, 2 ** (i % config.denoiser_dilation_cycle_length))
                for i in range(config.denoiser_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(channels, channels, 1)
        self.output_projection = nn.Conv1d(channels, config.num_mel_bins, 1)

    def get_diffusion_step_embedding(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            diffusion_step (`torch.Tensor` of shape `(batch_size,)`):
                Diffusion step of each item in the batch.

        Returns:
            `torch.Tensor` of shape `(batch_size, denoiser_channels)`: The sinusoidal step embedding.
        """
        half_dim = self.channels // 2
        exponent = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=diffusion_step.device) * -exponent)
        embedding = diffusion_step[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        diffusion_step: torch.Tensor,
        conditioning: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Noisy mel spectrogram.
            diffusion_step (`torch.Tensor` of shape `(batch_size,)`):
                Diffusion step of each item in the batch.
            conditioning (`torch.Tensor` of shape `(batch_size, hidden_size, num_frames)`):
                Frame level features.
            mask (`torch.Tensor` of shape `(batch_size, 1, num_frames)`, *optional*):
                Mask that is zero on the padded frames.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`: The predicted noise.
        """
        hidden_states = F.relu(self.input_projection(hidden_states))
        diffusion_step = self.mlp(self.get_diffusion_step_embedding(diffusion_step))

        skips = []
        for layer in self.residual_layers:
            hidden_states, skip = layer(hidden_states, conditioning, diffusion_step, mask)
            skips.append(skip)

        hidden_states = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.residual_layers))
        hidden_states = F.relu(self.skip_projection(hidden_states))
        return self.output_projection(hidden_states)


class PromptTTSPPDiffusionDecoder(nn.Module):
    r"""
    Constructs the denoising diffusion decoder that generates the mel spectrogram from the frame level features.

    Args:
        config ([`PromptTTSPPConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPConfig):
        super().__init__()
        self.num_mel_bins = config.num_mel_bins
        self.num_diffusion_steps = config.num_diffusion_steps
        self.norm_scale = config.diffusion_norm_scale
        self.min_value = config.diffusion_min_value
        self.max_value = config.diffusion_max_value
        self.denoise_fn = PromptTTSPPDenoiser(config)

        num_steps = config.num_diffusion_steps
        if config.beta_schedule == "linear":
            positions = torch.arange(num_steps, dtype=torch.float64) / (num_steps - 1)
            betas = config.beta_start + (config.beta_end - config.beta_start) * positions
        else:
            steps = num_steps + 1
            positions = torch.arange(steps, dtype=torch.float64) * (steps / (steps - 1))
            alphas_cumprod = torch.cos(
                ((positions / steps) + config.cosine_beta_shift) / (1 + config.cosine_beta_shift) * math.pi * 0.5
            ).pow(2)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = (1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])).clamp(0, 0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt().float())
        self.register_buffer("log_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).log().float())
        self.register_buffer("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt().float())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1).sqrt().float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer(
            "posterior_log_variance_clipped", posterior_variance.clamp_min(1e-20).log().float()
        )
        self.register_buffer(
            "posterior_mean_coef1", (betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)).float()
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)).float(),
        )

    @staticmethod
    def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """
        Args:
            values (`torch.Tensor` of shape `(num_diffusion_steps,)`):
                Schedule to index into.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Diffusion step of each item in the batch.
            shape (`torch.Size`):
                Shape the result is broadcast against.

        Returns:
            `torch.Tensor`: The gathered values, shaped for broadcasting against `shape`.
        """
        batch_size = timesteps.shape[0]
        return values.gather(-1, timesteps).reshape(batch_size, *((1,) * (len(shape) - 1)))

    def normalize(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram (`torch.Tensor`):
                Mel spectrogram.

        Returns:
            `torch.Tensor`: The spectrogram scaled into the diffusion process' value range.
        """
        if self.norm_scale is not None:
            return spectrogram / self.norm_scale
        return (spectrogram - self.min_value) / (self.max_value - self.min_value) * 2 - 1

    def denormalize(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram (`torch.Tensor`):
                Mel spectrogram in the diffusion process' value range.

        Returns:
            `torch.Tensor`: The rescaled spectrogram.
        """
        if self.norm_scale is not None:
            return spectrogram * self.norm_scale
        return (spectrogram + 1) / 2 * (self.max_value - self.min_value) + self.min_value

    def q_sample(self, spectrogram: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Normalized mel spectrogram.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Diffusion step of each item in the batch.
            noise (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Gaussian noise to add.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`: The noised spectrogram.
        """
        return (
            self.extract(self.sqrt_alphas_cumprod, timesteps, spectrogram.shape) * spectrogram
            + self.extract(self.sqrt_one_minus_alphas_cumprod, timesteps, spectrogram.shape) * noise
        )

    def forward(
        self, conditioning: torch.Tensor, spectrogram: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            conditioning (`torch.Tensor` of shape `(batch_size, num_frames, hidden_size)`):
                Frame level features.
            spectrogram (`torch.Tensor` of shape `(batch_size, num_frames, num_mel_bins)`):
                Ground truth mel spectrogram.
            mask (`torch.Tensor` of shape `(batch_size, 1, num_frames)`, *optional*):
                Mask that is zero on the padded frames.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: The sampled noise and the noise the denoiser predicted, both of
            shape `(batch_size, num_frames, num_mel_bins)`.
        """
        batch_size = conditioning.shape[0]
        conditioning = conditioning.transpose(1, 2)

        timesteps = torch.randint(
            0, self.num_diffusion_steps, (batch_size,), device=conditioning.device
        ).long()
        spectrogram = self.normalize(spectrogram).transpose(1, 2)

        noise = torch.randn_like(spectrogram)
        noisy_spectrogram = self.q_sample(spectrogram, timesteps, noise)
        predicted_noise = self.denoise_fn(noisy_spectrogram, timesteps, conditioning, mask=mask)

        return noise.transpose(1, 2), predicted_noise.transpose(1, 2)

    def predict_start_from_noise(
        self, spectrogram: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spectrogram (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Noisy mel spectrogram.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Diffusion step of each item in the batch.
            noise (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Predicted noise.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`: The denoised spectrogram.
        """
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, timesteps, spectrogram.shape) * spectrogram
            - self.extract(self.sqrt_recipm1_alphas_cumprod, timesteps, spectrogram.shape) * noise
        )

    @torch.no_grad()
    def p_sample(
        self,
        spectrogram: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            spectrogram (`torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`):
                Mel spectrogram of the current diffusion step.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Diffusion step of each item in the batch.
            conditioning (`torch.Tensor` of shape `(batch_size, hidden_size, num_frames)`):
                Frame level features.
            clip_denoised (`bool`, *optional*, defaults to `True`):
                Whether to clip the denoised spectrogram to `[-1, 1]`.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_mel_bins, num_frames)`: The spectrogram of the previous
            diffusion step.
        """
        predicted_noise = self.denoise_fn(spectrogram, timesteps, conditioning)
        denoised = self.predict_start_from_noise(spectrogram, timesteps, predicted_noise)
        if clip_denoised:
            denoised = denoised.clamp(-1.0, 1.0)

        model_mean = (
            self.extract(self.posterior_mean_coef1, timesteps, spectrogram.shape) * denoised
            + self.extract(self.posterior_mean_coef2, timesteps, spectrogram.shape) * spectrogram
        )
        model_log_variance = self.extract(self.posterior_log_variance_clipped, timesteps, spectrogram.shape)

        noise = torch.randn_like(spectrogram)
        nonzero_mask = (1 - (timesteps == 0).float()).reshape(
            spectrogram.shape[0], *((1,) * (spectrogram.dim() - 1))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conditioning (`torch.Tensor` of shape `(batch_size, num_frames, hidden_size)`):
                Frame level features.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_frames, num_mel_bins)`: The generated mel spectrogram.
        """
        batch_size, num_frames, _ = conditioning.shape
        conditioning = conditioning.transpose(1, 2)
        spectrogram = torch.randn(
            (batch_size, self.num_mel_bins, num_frames), device=conditioning.device, dtype=conditioning.dtype
        )
        for step in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full((batch_size,), step, device=conditioning.device, dtype=torch.long)
            spectrogram = self.p_sample(spectrogram, timesteps, conditioning)
        return self.denormalize(spectrogram.transpose(1, 2))


@dataclass
class PromptTTSPPModelOutput(ModelOutput):
    r"""
    Output of [`PromptTTSPPModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`):
            Frame level features conditioning the diffusion decoder.
        frame_attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`):
            Mask that is zero on the padded frames.
        style_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size, 1)`):
            Style embedding added to the encoded phonemes.
        prompt_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size, 1)`, *optional*):
            Style embedding predicted from the style prompt, before the style mixture density network.
        duration_outputs (`tuple(torch.FloatTensor)`, *optional*):
            Mixture parameters of the log duration of each phoneme.
        style_mdn_outputs (`tuple(torch.FloatTensor)`, *optional*):
            Mixture parameters of the style embedding predicted from the prompt embedding.
        log_f0 (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`):
            Predicted log continuous f0.
        vuv (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`):
            Predicted voicing.
        energy (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`, *optional*):
            Predicted energy. `None` unless the variance adaptor carries an energy branch.
    """

    last_hidden_state: torch.FloatTensor | None = None
    frame_attention_mask: torch.FloatTensor | None = None
    style_embedding: torch.FloatTensor | None = None
    prompt_embedding: torch.FloatTensor | None = None
    duration_outputs: tuple[torch.FloatTensor] | None = None
    style_mdn_outputs: tuple[torch.FloatTensor] | None = None
    log_f0: torch.FloatTensor | None = None
    vuv: torch.FloatTensor | None = None
    energy: torch.FloatTensor | None = None


@dataclass
class PromptTTSPPOutput(ModelOutput):
    r"""
    Output of [`PromptTTSPPForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Sum of the diffusion, duration, pitch, voicing, style and energy losses.
        spectrogram_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Scaled L1 loss between the sampled and the predicted diffusion noise.
        duration_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Negative log likelihood of the log durations under the duration mixture density network.
        pitch_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            L1 loss of the log continuous f0.
        vuv_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            L1 loss of the voicing.
        style_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Negative log likelihood of the reference style embedding under the style mixture density network.
        energy_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            L1 loss of the energy. `None` unless the variance adaptor carries an energy branch.
        spectrogram (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`, *optional*):
            Generated mel spectrogram. Only returned when no `labels` are given.
        log_f0 (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`):
            Predicted log continuous f0, which the f0 aware vocoder needs alongside the spectrogram.
        vuv (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`):
            Predicted voicing, which the f0 aware vocoder needs alongside the spectrogram.
        frame_attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`):
            Mask that is zero on the padded frames.
        style_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size, 1)`):
            Style embedding the generation was conditioned on.
    """

    loss: torch.FloatTensor | None = None
    spectrogram_loss: torch.FloatTensor | None = None
    duration_loss: torch.FloatTensor | None = None
    pitch_loss: torch.FloatTensor | None = None
    vuv_loss: torch.FloatTensor | None = None
    style_loss: torch.FloatTensor | None = None
    energy_loss: torch.FloatTensor | None = None
    spectrogram: torch.FloatTensor | None = None
    log_f0: torch.FloatTensor | None = None
    vuv: torch.FloatTensor | None = None
    frame_attention_mask: torch.FloatTensor | None = None
    style_embedding: torch.FloatTensor | None = None


@auto_docstring
class PromptTTSPPPreTrainedModel(PreTrainedModel):
    config: PromptTTSPPConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _supports_sdpa = False
    _supports_flash_attn = False

    def post_init(self):
        super().post_init()
        self.freeze_prompt_encoder()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Args:
            args:
                Forwarded to [`~PreTrainedModel.from_pretrained`].
            kwargs:
                Forwarded to [`~PreTrainedModel.from_pretrained`].

        Returns:
            [`PromptTTSPPPreTrainedModel`]: The loaded model, with the prompt encoder frozen again after loading
            replaced the parameters created by `__init__`.
        """
        outputs = super().from_pretrained(*args, **kwargs)
        model = outputs[0] if isinstance(outputs, tuple) else outputs
        model.freeze_prompt_encoder()
        return outputs

    def freeze_prompt_encoder(self):
        """
        Freezes the prompt encoder the way upstream training does, when `freeze_prompt_encoder` is set.

        Loading a checkpoint rebuilds every floating point parameter with `requires_grad=True`, so the flags set
        while the model is built do not survive `from_pretrained` and have to be set again on the loaded model.
        """
        if not self.config.freeze_prompt_encoder:
            return
        for module in self.modules():
            if isinstance(module, PromptTTSPPPromptEncoder):
                module.freeze_bert()

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        super()._init_weights(module)
        if isinstance(module, PromptTTSPPAttention):
            init.xavier_uniform_(module.pos_bias_u)
            init.xavier_uniform_(module.pos_bias_v)
        elif isinstance(module, PromptTTSPPStyleEncoder):
            init.normal_(module.style_tokens, mean=0.0, std=1.0)
        elif isinstance(module, PromptTTSPPDenoiserResidualBlock):
            init.kaiming_normal_(module.dilated_conv.weight)
            init.kaiming_normal_(module.conditioner_projection.weight)
            init.kaiming_normal_(module.output_projection.weight)
        elif isinstance(module, PromptTTSPPDenoiser):
            init.kaiming_normal_(module.input_projection.weight)
            init.kaiming_normal_(module.skip_projection.weight)
            init.zeros_(module.output_projection.weight)
        elif isinstance(module, PromptTTSPPRelPositionalEncoding):
            init.copy_(module.pos_enc, module.build_pos_enc(module.max_len))
        elif isinstance(module, PromptTTSPPFramePriorNetwork):
            init.copy_(module.pos_enc, module.build_pos_enc(module.embed_dim, module.max_len))


@auto_docstring(
    custom_intro="""
    The PromptTTS++ acoustic model, that is everything but the mel spectrogram decoder: the conformer phoneme
    encoder, the style and prompt encoders, and the variance adaptor producing the frame level features.
    """
)
class PromptTTSPPModel(PromptTTSPPPreTrainedModel):
    def __init__(self, config: PromptTTSPPConfig):
        super().__init__(config)
        self.phoneme_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.scale_phoneme_embedding = config.scale_phoneme_embedding
        self.embedding_scale = math.sqrt(config.hidden_size)
        self.encoder = PromptTTSPPEncoder(config)
        self.style_encoder = PromptTTSPPStyleEncoder(config)
        self.prompt_encoder = PromptTTSPPPromptEncoder(config)
        self.style_mdn = (
            PromptTTSPPMDNLayer(
                config.hidden_size, config.hidden_size, config.style_num_gaussians, dim_wise=True
            )
            if config.use_style_mdn
            else None
        )
        self.variance_adaptor = PromptTTSPPVarianceAdaptor(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.phoneme_embedding

    def set_input_embeddings(self, value):
        self.phoneme_embedding = value

    def sample_style_embedding(
        self,
        log_pi: torch.Tensor,
        log_sigma: torch.Tensor,
        mu: torch.Tensor,
        noise_scale: float = 1.0,
        use_max: bool = True,
    ) -> torch.Tensor:
        """
        Draws a style embedding from the style mixture density network.

        Args:
            log_pi (`torch.Tensor` of shape `(batch_size, 1, num_gaussians, hidden_size)`):
                Log mixture weights.
            log_sigma (`torch.Tensor` of shape `(batch_size, 1, num_gaussians, hidden_size)`):
                Log standard deviation of each mixture component.
            mu (`torch.Tensor` of shape `(batch_size, 1, num_gaussians, hidden_size)`):
                Mean of each mixture component.
            noise_scale (`float`, *optional*, defaults to 1.0):
                Scale of the Gaussian noise added to the selected component's mean.
            use_max (`bool`, *optional*, defaults to `True`):
                Whether to take the most probable component rather than sampling one.

        Returns:
            `torch.Tensor` of shape `(batch_size, hidden_size, 1)`: The style embedding.
        """
        if use_max:
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        else:
            sigma, mu = mdn_sample_sigma_and_mu(log_pi, log_sigma, mu)

        style_embedding = mu + sigma * torch.randn_like(sigma) * noise_scale
        if self.config.normalize_style_embedding:
            style_embedding = F.normalize(style_embedding, dim=-1)
        return style_embedding.transpose(-1, -2)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        prompt_input_ids: torch.LongTensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        reference_spectrogram: torch.FloatTensor | None = None,
        reference_spectrogram_lengths: torch.LongTensor | None = None,
        duration_labels: torch.FloatTensor | None = None,
        pitch_labels: torch.FloatTensor | None = None,
        energy_labels: torch.FloatTensor | None = None,
        spectrogram_attention_mask: torch.Tensor | None = None,
        use_max_style: bool = True,
        style_noise_scale: float = 1.0,
    ) -> PromptTTSPPModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Phoneme ids, obtained with [`PromptTTSPPTokenizer`].
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask that is zero on the padded phonemes.
        prompt_input_ids (`torch.LongTensor` of shape `(batch_size, prompt_length)`, *optional*):
            Style prompt token ids, obtained with the processor's BERT tokenizer. Required unless a reference
            spectrogram is given.
        prompt_attention_mask (`torch.Tensor` of shape `(batch_size, prompt_length)`, *optional*):
            Mask that is zero on the padded prompt tokens.
        reference_spectrogram (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, num_frames)`, *optional*):
            Normalized reference mel spectrogram the style embedding is read from. During training this is the
            target spectrogram itself.
        reference_spectrogram_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of valid frames of each reference spectrogram.
        duration_labels (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
            Ground truth durations in frames. Passing them switches the variance adaptor to teacher forcing.
        pitch_labels (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`, *optional*):
            Ground truth log continuous f0, embedded back into the frame level features under teacher forcing.
        energy_labels (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`, *optional*):
            Ground truth energy, embedded back into the frame level features under teacher forcing.
        spectrogram_attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask that is zero on the padded frames. Required under teacher forcing.
        use_max_style (`bool`, *optional*, defaults to `True`):
            Whether the style embedding takes the most probable component of the style mixture density network
            rather than a sampled one.
        style_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale of the Gaussian noise added to the sampled style embedding.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        phoneme_mask = attention_mask.unsqueeze(1).to(self.dtype)

        hidden_states = self.phoneme_embedding(input_ids)
        if self.scale_phoneme_embedding:
            hidden_states = hidden_states * self.embedding_scale
        hidden_states = hidden_states * phoneme_mask.transpose(1, 2)

        hidden_states = self.encoder(hidden_states, phoneme_mask).transpose(1, 2)

        prompt_embedding = None
        style_mdn_outputs = None
        style_embedding = None

        if reference_spectrogram is not None:
            style_embedding = self.style_encoder(reference_spectrogram, reference_spectrogram_lengths)
            if self.config.normalize_style_embedding:
                style_embedding = F.normalize(style_embedding, dim=1)

        if prompt_input_ids is not None:
            prompt_embedding = self.prompt_encoder(prompt_input_ids, prompt_attention_mask)
            if self.config.normalize_style_embedding:
                prompt_embedding = F.normalize(prompt_embedding, dim=1)
            if self.style_mdn is not None:
                with torch.autocast(
                    device_type=prompt_embedding.device.type, enabled=not self.config.disable_mdn_autocast
                ):
                    style_mdn_outputs = self.style_mdn(prompt_embedding.transpose(-1, -2))
            if style_embedding is None:
                style_embedding = (
                    self.sample_style_embedding(
                        *style_mdn_outputs, noise_scale=style_noise_scale, use_max=use_max_style
                    )
                    if style_mdn_outputs is not None
                    else prompt_embedding
                )

        if style_embedding is None:
            raise ValueError("One of prompt_input_ids or reference_spectrogram must be given.")

        hidden_states = hidden_states + style_embedding

        frame_mask = None
        if spectrogram_attention_mask is not None:
            frame_mask = spectrogram_attention_mask.unsqueeze(1).to(hidden_states.dtype)
        if duration_labels is not None and frame_mask is None:
            raise ValueError("spectrogram_attention_mask must be given alongside duration_labels.")

        (
            hidden_states,
            frame_mask,
            duration_outputs,
            log_f0,
            vuv,
            energy,
        ) = self.variance_adaptor(
            hidden_states,
            phoneme_mask,
            frame_mask=frame_mask,
            durations=duration_labels,
            log_f0=pitch_labels,
            energy=energy_labels,
        )

        return PromptTTSPPModelOutput(
            last_hidden_state=hidden_states.transpose(1, 2),
            frame_attention_mask=frame_mask,
            style_embedding=style_embedding,
            prompt_embedding=prompt_embedding,
            duration_outputs=duration_outputs,
            style_mdn_outputs=style_mdn_outputs,
            log_f0=log_f0,
            vuv=vuv,
            energy=energy,
        )


@auto_docstring(
    custom_intro="""
    PromptTTS++, a prompt based text to speech model whose diffusion decoder generates a mel spectrogram from a
    phoneme sequence and a natural language description of the speaker.
    """
)
class PromptTTSPPForConditionalGeneration(PromptTTSPPPreTrainedModel):
    def __init__(self, config: PromptTTSPPConfig):
        super().__init__(config)
        self.model = PromptTTSPPModel(config)
        self.decoder = PromptTTSPPDiffusionDecoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.phoneme_embedding

    def set_input_embeddings(self, value):
        self.model.phoneme_embedding = value

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        prompt_input_ids: torch.LongTensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        reference_spectrogram: torch.FloatTensor | None = None,
        reference_spectrogram_lengths: torch.LongTensor | None = None,
        labels: torch.FloatTensor | None = None,
        spectrogram_attention_mask: torch.Tensor | None = None,
        duration_labels: torch.FloatTensor | None = None,
        pitch_labels: torch.FloatTensor | None = None,
        vuv_labels: torch.FloatTensor | None = None,
        energy_labels: torch.FloatTensor | None = None,
        use_max_style: bool = True,
        style_noise_scale: float = 1.0,
    ) -> PromptTTSPPOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Phoneme ids, obtained with [`PromptTTSPPTokenizer`].
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask that is zero on the padded phonemes.
        prompt_input_ids (`torch.LongTensor` of shape `(batch_size, prompt_length)`, *optional*):
            Style prompt token ids, obtained with the processor's BERT tokenizer.
        prompt_attention_mask (`torch.Tensor` of shape `(batch_size, prompt_length)`, *optional*):
            Mask that is zero on the padded prompt tokens.
        reference_spectrogram (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, num_frames)`, *optional*):
            Normalized reference mel spectrogram the style embedding is read from. Defaults to `labels` when
            training, which is what the reference encoder is trained on.
        reference_spectrogram_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Number of valid frames of each reference spectrogram.
        labels (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, num_frames)`, *optional*):
            Normalized target mel spectrogram. Passing it switches the model to teacher forcing and returns the
            training loss instead of a generated spectrogram.
        spectrogram_attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask that is zero on the padded frames. Required alongside `labels`.
        duration_labels (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
            Ground truth durations in frames. Required alongside `labels`.
        pitch_labels (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`, *optional*):
            Ground truth log continuous f0. Required alongside `labels`.
        vuv_labels (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`, *optional*):
            Ground truth voicing. Required alongside `labels`.
        energy_labels (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`, *optional*):
            Ground truth energy. Only used when the variance adaptor carries an energy branch.
        use_max_style (`bool`, *optional*, defaults to `True`):
            Whether the style embedding takes the most probable component of the style mixture density network
            rather than a sampled one.
        style_noise_scale (`float`, *optional*, defaults to 1.0):
            Scale of the Gaussian noise added to the sampled style embedding.

        Example:

        ```python
        >>> import torch
        >>> from voicestudio.models.prompt_tts_pp import PromptTTSPPForConditionalGeneration, PromptTTSPPProcessor
        >>> from voicestudio.models.prompt_tts_pp.weight_conversion import convert

        >>> convert(output_dir="prompt-tts-pp-converted")
        >>> processor = PromptTTSPPProcessor.from_pretrained("prompt-tts-pp-converted")
        >>> model = PromptTTSPPForConditionalGeneration.from_pretrained("prompt-tts-pp-converted")

        >>> inputs = processor(
        ...     text="This is a text to speech demo.",
        ...     style_prompt="A man speaks slowly in a low tone.",
        ... )
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> spectrogram, f0 = processor.postprocess(outputs)
        ```
        """
        if labels is not None:
            if reference_spectrogram is None:
                reference_spectrogram = labels
                if reference_spectrogram_lengths is None and spectrogram_attention_mask is not None:
                    reference_spectrogram_lengths = spectrogram_attention_mask.sum(dim=-1)
            if duration_labels is None or pitch_labels is None or vuv_labels is None:
                raise ValueError(
                    "duration_labels, pitch_labels and vuv_labels must be given alongside labels."
                )
            if prompt_input_ids is None:
                raise ValueError("prompt_input_ids must be given alongside labels, the style loss needs it.")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            reference_spectrogram=reference_spectrogram,
            reference_spectrogram_lengths=reference_spectrogram_lengths,
            duration_labels=duration_labels if labels is not None else None,
            pitch_labels=pitch_labels if labels is not None else None,
            energy_labels=energy_labels if labels is not None else None,
            spectrogram_attention_mask=spectrogram_attention_mask if labels is not None else None,
            use_max_style=use_max_style,
            style_noise_scale=style_noise_scale,
        )

        frame_mask = outputs.frame_attention_mask
        conditioning = outputs.last_hidden_state

        if labels is None:
            spectrogram = self.decoder.sample(conditioning) * frame_mask.transpose(1, 2)
            return PromptTTSPPOutput(
                spectrogram=spectrogram,
                log_f0=outputs.log_f0,
                vuv=outputs.vuv,
                frame_attention_mask=frame_mask,
                style_embedding=outputs.style_embedding,
            )

        noise, predicted_noise = self.decoder(conditioning, labels.transpose(1, 2), mask=frame_mask)
        noise = noise.transpose(1, 2) * frame_mask
        predicted_noise = predicted_noise.transpose(1, 2) * frame_mask

        num_frames = frame_mask.sum()
        spectrogram_loss = (
            (noise - predicted_noise).abs().sum() / num_frames / self.config.spectrogram_loss_scale
        )

        phoneme_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        ).unsqueeze(-1).bool()
        log_durations = to_log_scale(duration_labels).transpose(-1, -2)
        with torch.autocast(device_type=log_durations.device.type, enabled=not self.config.disable_mdn_autocast):
            duration_loss = mdn_loss(
                *outputs.duration_outputs, log_durations, reduce=False, mask=phoneme_mask
            )
            duration_loss = duration_loss.masked_select(phoneme_mask).mean()

        pitch_loss = (outputs.log_f0 - pitch_labels).abs().sum() / num_frames
        vuv_loss = (outputs.vuv - vuv_labels).abs().sum() / num_frames

        if outputs.style_mdn_outputs is not None:
            with torch.autocast(
                device_type=log_durations.device.type, enabled=not self.config.disable_mdn_autocast
            ):
                style_loss = mdn_loss(
                    *outputs.style_mdn_outputs, outputs.style_embedding.detach().transpose(-1, -2)
                ).mean()
        else:
            style_loss = (outputs.style_embedding.detach() - outputs.prompt_embedding).pow(2).mean()

        loss = spectrogram_loss + duration_loss + pitch_loss + vuv_loss + style_loss

        energy_loss = None
        if outputs.energy is not None:
            energy_loss = (outputs.energy - energy_labels).abs().sum() / num_frames
            loss = loss + energy_loss

        return PromptTTSPPOutput(
            loss=loss,
            spectrogram_loss=spectrogram_loss,
            duration_loss=duration_loss,
            pitch_loss=pitch_loss,
            vuv_loss=vuv_loss,
            style_loss=style_loss,
            energy_loss=energy_loss,
            log_f0=outputs.log_f0,
            vuv=outputs.vuv,
            frame_attention_mask=frame_mask,
            style_embedding=outputs.style_embedding,
        )


def build_anti_alias_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """
    Builds the Kaiser windowed sinc lowpass filter of the anti aliased activation.

    Args:
        cutoff (`float`):
            Cutoff frequency, as a fraction of the sample rate.
        half_width (`float`):
            Width of the transition band, as a fraction of the sample rate.
        kernel_size (`int`):
            Length of the filter.

    Returns:
        `torch.Tensor` of shape `(1, 1, kernel_size)`: The filter, normalized to unit sum.
    """
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    taps = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (taps / taps.sum()).view(1, 1, kernel_size)


class PromptTTSPPSnakeActivation(nn.Module):
    r"""
    Constructs the anti aliased snake activation of BigVGAN, which upsamples its input, applies the periodic
    `x + sin(alpha * x) ** 2 / alpha` nonlinearity, and lowpass filters the result back down, so that the
    harmonics the nonlinearity creates stay below the Nyquist frequency.

    Args:
        config ([`PromptTTSPPBigVGanConfig`]):
            Model configuration.
        channels (`int`):
            Number of channels, each of which owns its own `alpha`.
    """

    def __init__(self, config: PromptTTSPPBigVGanConfig, channels: int):
        super().__init__()
        ratio = config.anti_alias_ratio
        kernel_size = config.anti_alias_kernel_size
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.filter = nn.Buffer(self.build_filter(), persistent=False)

        self.upsample_pad = kernel_size // ratio - 1
        self.upsample_trim_left = self.upsample_pad * ratio + (kernel_size - ratio) // 2
        self.upsample_trim_right = self.upsample_pad * ratio + (kernel_size - ratio + 1) // 2
        self.downsample_pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.downsample_pad_right = kernel_size // 2

    def build_filter(self) -> torch.Tensor:
        """
        Returns:
            `torch.Tensor` of shape `(1, 1, kernel_size)`: The resampling filter of both directions.
        """
        return build_anti_alias_filter(0.5 / self.ratio, 0.6 / self.ratio, self.kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Activation input.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, sequence_length)`: The activated input.
        """
        channels = hidden_states.shape[1]
        taps = self.filter.to(hidden_states.dtype).expand(channels, -1, -1)

        hidden_states = F.pad(hidden_states, (self.upsample_pad, self.upsample_pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, taps, stride=self.ratio, groups=channels
        )
        hidden_states = hidden_states[..., self.upsample_trim_left : -self.upsample_trim_right]

        alpha = self.alpha.exp()
        hidden_states = hidden_states + (1.0 / (alpha + 1e-9)) * (hidden_states * alpha).sin().pow(2)

        hidden_states = F.pad(
            hidden_states, (self.downsample_pad_left, self.downsample_pad_right), mode="replicate"
        )
        return F.conv1d(hidden_states, taps, stride=self.ratio, groups=channels)


class PromptTTSPPAmpLayer(nn.Module):
    r"""
    Constructs one residual layer of an anti aliased multi-periodicity block, a dilated convolution and a plain
    one, each preceded by a snake activation.

    Args:
        config ([`PromptTTSPPBigVGanConfig`]):
            Model configuration.
        channels (`int`):
            Number of channels of the layer.
        kernel_size (`int`):
            Kernel size of both convolutions.
        dilation (`int`):
            Dilation of the first convolution.
    """

    def __init__(self, config: PromptTTSPPBigVGanConfig, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size * dilation - dilation) // 2,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, dilation=1)
        self.activation1 = PromptTTSPPSnakeActivation(config, channels)
        self.activation2 = PromptTTSPPSnakeActivation(config, channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Layer input.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, sequence_length)`: The layer output.
        """
        residual = self.conv1(self.activation1(hidden_states))
        residual = self.conv2(self.activation2(residual))
        return hidden_states + residual

    def apply_weight_norm(self):
        """Reparameterizes the layer's convolutions by weight and direction, as training does."""
        weight_norm = nn.utils.parametrizations.weight_norm
        weight_norm(self.conv1)
        weight_norm(self.conv2)

    def remove_weight_norm(self):
        """Folds the weight norm reparameterization of the layer's convolutions back into plain weights."""
        nn.utils.parametrize.remove_parametrizations(self.conv1, "weight")
        nn.utils.parametrize.remove_parametrizations(self.conv2, "weight")


class PromptTTSPPAmpBlock(nn.Module):
    r"""
    Constructs an anti aliased multi-periodicity block, the stack of residual layers that follows one upsampling
    layer for a single kernel size.

    Args:
        config ([`PromptTTSPPBigVGanConfig`]):
            Model configuration.
        channels (`int`):
            Number of channels of the block.
        kernel_size (`int`):
            Kernel size of the block's convolutions.
        dilations (`list[int]`):
            Dilation of each layer of the block.
    """

    def __init__(self, config: PromptTTSPPBigVGanConfig, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [PromptTTSPPAmpLayer(config, channels, kernel_size, dilation) for dilation in dilations]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Block input.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, sequence_length)`: The block output.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def apply_weight_norm(self):
        """Reparameterizes the block's convolutions by weight and direction, as training does."""
        for layer in self.layers:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        """Folds the weight norm reparameterization of the block's convolutions back into plain weights."""
        for layer in self.layers:
            layer.remove_weight_norm()


class PromptTTSPPSourceModule(nn.Module):
    r"""
    Constructs the harmonic source module of the neural source filter, which turns a sample level fundamental
    frequency into the excitation signal the upsampling stack is conditioned on: a sine wave per harmonic, muted
    on the unvoiced frames and mixed with noise, merged by a single linear layer.

    Args:
        config ([`PromptTTSPPBigVGanConfig`]):
            Model configuration.
    """

    def __init__(self, config: PromptTTSPPBigVGanConfig):
        super().__init__()
        self.sampling_rate = config.sampling_rate
        self.harmonic_num = config.harmonic_num
        self.sine_amplitude = config.sine_amplitude
        self.noise_std = config.noise_std
        self.voiced_threshold = config.voiced_threshold
        self.linear = nn.Linear(config.harmonic_num + 1, 1)

    @torch.no_grad()
    def sine_waves(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0 (`torch.Tensor` of shape `(batch_size, num_samples, 1)`):
                Fundamental frequency in Hz, zero on the unvoiced samples.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_samples, harmonic_num + 1)`: The source signal.
        """
        harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype)
        f0 = f0[:, :, :1] * harmonics

        radians = (f0 / self.sampling_rate) % 1
        # A random initial phase per harmonic, except for the fundamental.
        initial_phase = torch.rand(radians.shape[0], radians.shape[2], device=radians.device, dtype=radians.dtype)
        initial_phase[:, 0] = 0
        radians[:, 0, :] = radians[:, 0, :] + initial_phase

        # The cumulative phase wraps at every full turn, which keeps the sum from losing precision.
        wrapped = torch.cumsum(radians, dim=1) % 1
        shift = torch.zeros_like(radians)
        shift[:, 1:, :] = ((wrapped[:, 1:, :] - wrapped[:, :-1, :]) < 0) * -1.0
        sines = torch.sin(torch.cumsum(radians + shift, dim=1) * 2 * math.pi) * self.sine_amplitude

        voiced = (f0[:, :, :1] > self.voiced_threshold).to(f0.dtype)
        noise_amplitude = voiced * self.noise_std + (1 - voiced) * self.sine_amplitude / 3
        return sines * voiced + noise_amplitude * torch.randn_like(sines)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0 (`torch.Tensor` of shape `(batch_size, num_samples, 1)`):
                Fundamental frequency in Hz, zero on the unvoiced samples.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_samples, 1)`: The excitation signal.
        """
        return torch.tanh(self.linear(self.sine_waves(f0)))


@auto_docstring(
    custom_intro="""
    The f0 aware BigVGAN vocoder of PromptTTS++, which turns a log mel spectrogram and the fundamental frequency
    predicted alongside it into a waveform.
    """
)
class PromptTTSPPBigVGan(PreTrainedModel):
    config: PromptTTSPPBigVGanConfig
    config_class = PromptTTSPPBigVGanConfig
    base_model_prefix = "vocoder"
    main_input_name = "spectrogram"
    supports_gradient_checkpointing = False
    _supports_sdpa = False
    _supports_flash_attn = False

    def __init__(self, config: PromptTTSPPBigVGanConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.hop_length = math.prod(config.upsample_rates)
        self.source = PromptTTSPPSourceModule(config)

        self.conv_pre = nn.Conv1d(
            config.model_in_dim, config.upsample_initial_channel, kernel_size=7, stride=1, padding=3
        )

        self.upsampler = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        for i, (rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    channels,
                    kernel_size=kernel_size,
                    stride=rate,
                    padding=rate // 2 + rate % 2,
                    output_padding=rate % 2,
                )
            )
            if i + 1 < len(config.upsample_rates):
                stride = math.prod(config.upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(1, channels, kernel_size=stride * 2, stride=stride, padding=stride // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, channels, 1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(
                nn.ModuleList(
                    [
                        PromptTTSPPAmpBlock(config, channels, kernel_size, dilations)
                        for kernel_size, dilations in zip(
                            config.resblock_kernel_sizes, config.resblock_dilation_sizes
                        )
                    ]
                )
            )

        self.post_activation = PromptTTSPPSnakeActivation(config, channels)
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        super()._init_weights(module)
        if isinstance(module, PromptTTSPPSnakeActivation):
            init.zeros_(module.alpha)
            init.copy_(module.filter, module.build_filter())

    def apply_weight_norm(self):
        """Reparameterizes the vocoder's convolutions by weight and direction, as training does."""
        weight_norm = nn.utils.parametrizations.weight_norm
        weight_norm(self.conv_pre)
        for layer in self.upsampler:
            weight_norm(layer)
        for blocks in self.resblocks:
            for block in blocks:
                block.apply_weight_norm()
        weight_norm(self.conv_post)

    def remove_weight_norm(self):
        """Folds the weight norm reparameterization of the vocoder's convolutions back into plain weights."""
        nn.utils.parametrize.remove_parametrizations(self.conv_pre, "weight")
        for layer in self.upsampler:
            nn.utils.parametrize.remove_parametrizations(layer, "weight")
        for blocks in self.resblocks:
            for block in blocks:
                block.remove_weight_norm()
        nn.utils.parametrize.remove_parametrizations(self.conv_post, "weight")

    @auto_docstring
    def forward(self, spectrogram: torch.FloatTensor, f0: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        spectrogram (`torch.FloatTensor` of shape `(batch_size, model_in_dim, num_frames)`):
            Log mel spectrogram on the scale the vocoder was trained on, that is the model's prediction after
            [`~PromptTTSPPFeatureExtractor.denormalize`].
        f0 (`torch.FloatTensor` of shape `(batch_size, 1, num_frames)`):
            Fundamental frequency in Hz of each frame, zero on the unvoiced ones.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_samples)`: The waveform.
        """
        excitation = F.interpolate(f0, scale_factor=float(self.hop_length), mode="nearest").transpose(-1, -2)
        excitation = self.source(excitation).transpose(-1, -2)

        hidden_states = self.conv_pre(spectrogram)
        for upsample, noise_conv, blocks in zip(self.upsampler, self.noise_convs, self.resblocks):
            hidden_states = upsample(hidden_states) + noise_conv(excitation)
            hidden_states = sum(block(hidden_states) for block in blocks) / self.num_kernels

        hidden_states = self.post_activation(hidden_states)
        return torch.tanh(self.conv_post(hidden_states)).squeeze(1)


__all__ = [
    "PromptTTSPPBigVGan",
    "PromptTTSPPForConditionalGeneration",
    "PromptTTSPPModel",
    "PromptTTSPPPreTrainedModel",
]
