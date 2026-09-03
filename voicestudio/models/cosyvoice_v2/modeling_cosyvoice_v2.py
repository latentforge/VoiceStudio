# coding=utf-8
# Copyright 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du, Bofan Zhou) and the HuggingFace Inc. team.
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
"""PyTorch CosyVoice v2 model."""

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers.conversion_mapping import WeightRenaming, register_checkpoint_conversion_mapping
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2RotaryEmbedding, apply_rotary_pos_emb
from transformers.utils import auto_docstring

from ..cosyvoice_v1.modeling_cosyvoice_v1 import (
    CHECKPOINT_CONVERSION,
    CosyVoiceV1Block1D,
    CosyVoiceV1ConditionalCFM,
    CosyVoiceV1ConditionalDecoder,
    CosyVoiceV1EncoderLayer,
    CosyVoiceV1HiFTGenerator,
    CosyVoiceV1InputProjection,
    CosyVoiceV1LabelSmoothingLoss,
    CosyVoiceV1PreTrainedModel,
    CosyVoiceV1RelPositionalEmbedding,
    CosyVoiceV1ResnetBlock1D,
    CosyVoiceV1SpeechTokenizer,
    CosyVoiceV1SpeechTokenizerAttention,
    CosyVoiceV1SpeechTokenizerLayer,
    build_attention_bias,
    make_pad_mask,
)
from ..cosyvoice_v1.weight_conversion import CHECKPOINT_FILES, load_checkpoint, resolve_checkpoint
from .configuration_cosyvoice_v2 import CosyVoiceV2Config
from .generation_cosyvoice_v2 import CosyVoiceV2GenerationMixin
from .weight_conversion import TEXT_MODEL_SUBDIR, build_config


IGNORE_ID = -1


# `UpsampleConformerEncoder` runs a second, shorter block stack after the upsampling layer, under its
# own input projection, and `Qwen2Encoder` wraps a whole `Qwen2ForCausalLM` whose head CosyVoice never
# reads. Everything else the released directory names the way v1 does.
register_checkpoint_conversion_mapping(
    "CosyVoiceV2ForConditionalGeneration",
    [
        WeightRenaming(source_patterns=r"^llm\.llm\.model\.model\.", target_patterns=r"llm\.model\."),
        WeightRenaming(source_patterns=r"\.up_embed\.out\.0\.", target_patterns=r"\.up_input_projection\.proj\."),
        WeightRenaming(
            source_patterns=r"\.up_embed\.out\.1\.", target_patterns=r"\.up_input_projection\.layer_norm\."
        ),
        WeightRenaming(
            source_patterns=r"\.up_encoders\.(\d+)\.self_attn\.", target_patterns=r"\.up_layers\.\1\.self_attn\."
        ),
        WeightRenaming(
            source_patterns=r"\.up_encoders\.(\d+)\.feed_forward\.",
            target_patterns=r"\.up_layers\.\1\.feed_forward\.",
        ),
        WeightRenaming(
            source_patterns=r"\.up_encoders\.(\d+)\.norm_mha\.",
            target_patterns=r"\.up_layers\.\1\.self_attn_layer_norm\.",
        ),
        WeightRenaming(
            source_patterns=r"\.up_encoders\.(\d+)\.norm_ff\.",
            target_patterns=r"\.up_layers\.\1\.final_layer_norm\.",
        ),
    ]
    + CHECKPOINT_CONVERSION,
    overwrite=True,
)


class CosyVoiceV2CausalConv1d(nn.Conv1d):
    """
    Convolution that only sees the past, obtained by padding the left of the time axis with
    `kernel_size - 1` zeros and using no padding inside the convolution.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Width of the kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.causal_padding = kernel_size - 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(hidden_states, (self.causal_padding, 0), value=0.0))


class CosyVoiceV2Transpose(nn.Module):
    """
    Swaps two axes of its input.

    Args:
        dim0 (`int`):
            First axis.
        dim1 (`int`):
            Second axis.
    """

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.transpose(hidden_states, self.dim0, self.dim1)


class CosyVoiceV2CausalBlock1D(CosyVoiceV1Block1D):
    """
    Masked causal convolution, layer norm and Mish.

    Args:
        dim (`int`):
            Number of input channels.
        dim_out (`int`):
            Number of output channels.
    """

    def __init__(self, dim: int, dim_out: int):
        super().__init__(dim, dim_out)
        self.block = nn.Sequential(
            CosyVoiceV2CausalConv1d(dim, dim_out, 3),
            CosyVoiceV2Transpose(1, 2),
            nn.LayerNorm(dim_out),
            CosyVoiceV2Transpose(1, 2),
            nn.Mish(),
        )


class CosyVoiceV2CausalResnetBlock1D(CosyVoiceV1ResnetBlock1D):
    """
    Residual block conditioned on the flow matching timestep, built from causal blocks.

    Args:
        dim (`int`):
            Number of input channels.
        dim_out (`int`):
            Number of output channels.
        time_emb_dim (`int`):
            Dimension of the timestep embedding.
    """

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int):
        super().__init__(dim, dim_out, time_emb_dim)
        self.block1 = CosyVoiceV2CausalBlock1D(dim, dim_out)
        self.block2 = CosyVoiceV2CausalBlock1D(dim_out, dim_out)


class CosyVoiceV2PreLookaheadLayer(nn.Module):
    """
    Residual pair of convolutions that mixes a fixed number of future frames into the present one,
    then keeps the result causal.

    Args:
        in_channels (`int`):
            Number of channels in and out of the layer.
        channels (`int`):
            Inner number of channels.
        pre_lookahead_len (`int`):
            Number of future frames the first convolution sees.
    """

    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, in_channels)`):
                Layer input.
            context (`torch.Tensor` of shape `(batch_size, pre_lookahead_len, in_channels)`, *optional*):
                Future frames the layer may look at instead of the zero padding it uses otherwise.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, in_channels)`: the layer output.

        Raises:
            ValueError: If `context` is given while the module is in training mode, or if its length
                is not `pre_lookahead_len`.
        """
        outputs = hidden_states.transpose(1, 2).contiguous()
        if context is None or context.size(1) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        else:
            if self.training:
                raise ValueError("context is only accepted in inference mode")
            if context.size(1) != self.pre_lookahead_len:
                raise ValueError(
                    f"context must hold {self.pre_lookahead_len} frames, got {context.size(1)}."
                )
            outputs = torch.concat([outputs, context.transpose(1, 2).contiguous()], dim=2)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs).transpose(1, 2).contiguous()
        return outputs + hidden_states


class CosyVoiceV2Upsample1D(nn.Module):
    """
    Nearest neighbour upsampling followed by a causal convolution, which stretches the token rate to
    the mel frame rate.

    Args:
        channels (`int`):
            Number of channels in and out of the layer.
        stride (`int`):
            Upsampling factor.
    """

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(channels, channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = F.interpolate(hidden_states, scale_factor=float(self.stride), mode="nearest")
        return self.conv(F.pad(outputs, (self.stride * 2, 0), value=0.0))


class CosyVoiceV2UpsampleEncoder(nn.Module):
    """
    Flow matching encoder of CosyVoice v2. A lookahead convolution and a first stack of encoder layers
    run at the speech token rate, the sequence is then upsampled by `token_mel_ratio` and a second,
    shorter stack runs at the mel frame rate.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__()
        hidden_size = config.flow_encoder_hidden_size
        self.hidden_size = hidden_size
        self.chunk_size = config.flow_encoder_chunk_size
        self.embed_scale = math.sqrt(hidden_size)
        self.pre_lookahead_len = config.pre_lookahead_len

        self.input_projection = CosyVoiceV1InputProjection(
            config.flow_input_size, hidden_size, config.flow_encoder_dropout
        )
        self.pos_embedding = CosyVoiceV1RelPositionalEmbedding(hidden_size, config.max_source_positions)
        self.pos_dropout = nn.Dropout(config.flow_encoder_positional_dropout)
        self.pre_lookahead_layer = CosyVoiceV2PreLookaheadLayer(
            hidden_size, config.pre_lookahead_channels, config.pre_lookahead_len
        )

        def make_layers(num_layers: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    CosyVoiceV1EncoderLayer(
                        hidden_size,
                        config.flow_encoder_num_heads,
                        config.flow_encoder_ffn_dim,
                        config.flow_encoder_dropout,
                        config.flow_encoder_attention_dropout,
                        config.flow_encoder_hidden_act,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.layers = make_layers(config.flow_encoder_num_layers)
        self.up_layer = CosyVoiceV2Upsample1D(hidden_size, config.token_mel_ratio)
        self.up_input_projection = CosyVoiceV1InputProjection(
            config.flow_input_size, hidden_size, config.flow_encoder_dropout
        )
        self.up_layers = make_layers(config.flow_encoder_up_num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def embed(self, hidden_states: torch.Tensor, key_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.input_projection(hidden_states)
        relative_position_embeddings = self.pos_embedding(hidden_states, key_length)
        return (
            self.pos_dropout(hidden_states * self.embed_scale),
            self.pos_dropout(relative_position_embeddings),
        )

    def up_embed(self, hidden_states: torch.Tensor, key_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.up_input_projection(hidden_states)
        relative_position_embeddings = self.pos_embedding(hidden_states, key_length)
        return (
            self.pos_dropout(hidden_states * self.embed_scale),
            self.pos_dropout(relative_position_embeddings),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, flow_input_size)`):
                Embedded speech tokens.
            input_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid speech tokens per sequence.
            context (`torch.Tensor` of shape `(batch_size, pre_lookahead_len, flow_input_size)`, *optional*):
                Speech tokens the lookahead convolution may read beyond the end of `hidden_states`.
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the attention masks are restricted to `flow_encoder_chunk_size` chunks.

        Returns:
            `tuple(torch.Tensor)`: the encoder output at the mel frame rate and its padding mask.
        """
        length = hidden_states.size(1)
        padding_mask = ~make_pad_mask(input_lengths, length)
        hidden_states, relative_position_embeddings = self.embed(hidden_states, length)
        if context is not None and context.size(1) != 0:
            context = self.pos_dropout(self.input_projection(context) * self.embed_scale)

        attention_bias = build_attention_bias(
            padding_mask, self.chunk_size if streaming else 0, hidden_states.dtype
        )
        hidden_states = self.pre_lookahead_layer(hidden_states, context=context)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, relative_position_embeddings, attention_bias)

        hidden_states = self.up_layer(hidden_states.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        up_lengths = input_lengths * self.up_layer.stride
        up_length = hidden_states.size(1)
        up_padding_mask = ~make_pad_mask(up_lengths, up_length)
        hidden_states, relative_position_embeddings = self.up_embed(hidden_states, up_length)
        attention_bias = build_attention_bias(
            up_padding_mask,
            self.chunk_size * self.up_layer.stride if streaming else 0,
            hidden_states.dtype,
        )
        for layer in self.up_layers:
            hidden_states, _ = layer(hidden_states, relative_position_embeddings, attention_bias)

        return self.layer_norm(hidden_states), up_padding_mask


class CosyVoiceV2ConditionalDecoder(CosyVoiceV1ConditionalDecoder):
    """
    One dimensional UNet that predicts the flow matching vector field, built from causal convolutions
    so that it can be run chunk by chunk on a stream.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        self.static_chunk_size = config.estimator_static_chunk_size
        channels = tuple(config.estimator_channels)
        time_embed_dim = channels[0] * 4

        output_channel = config.estimator_in_channels
        for index in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[index]
            is_last = index == len(channels) - 1
            self.down_blocks[index][0] = CosyVoiceV2CausalResnetBlock1D(
                input_channel, output_channel, time_embed_dim
            )
            if is_last:
                self.down_blocks[index][2] = CosyVoiceV2CausalConv1d(output_channel, output_channel, 3)

        for index in range(len(self.mid_blocks)):
            self.mid_blocks[index][0] = CosyVoiceV2CausalResnetBlock1D(
                channels[-1], output_channel, time_embed_dim
            )

        up_channels = channels[::-1] + (channels[0],)
        for index in range(len(up_channels) - 1):
            input_channel = up_channels[index] * 2
            output_channel = up_channels[index + 1]
            is_last = index == len(up_channels) - 2
            self.up_blocks[index][0] = CosyVoiceV2CausalResnetBlock1D(
                input_channel, output_channel, time_embed_dim
            )
            if is_last:
                self.up_blocks[index][2] = CosyVoiceV2CausalConv1d(output_channel, output_channel, 3)

        self.final_block = CosyVoiceV2CausalBlock1D(up_channels[-1], up_channels[-1])

    def _attention_bias(self, mask: torch.Tensor, dtype: torch.dtype, streaming: bool = False) -> torch.Tensor:
        """
        Args:
            mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
                1 on valid frames.
            dtype (`torch.dtype`):
                Floating point type of the returned bias.
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the bias restricts attention to `estimator_static_chunk_size` chunks.

        Returns:
            `torch.Tensor` of shape `(batch_size, 1, query_length, key_length)`: additive attention bias.
        """
        return build_attention_bias(
            mask.squeeze(1).bool(), self.static_chunk_size if streaming else 0, dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        timesteps: torch.Tensor,
        speaker_hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, out_channels, mel_length)`):
                Noisy mel spectrogram.
            mask (`torch.Tensor` of shape `(batch_size, 1, mel_length)`):
                1 on valid frames.
            mu (`torch.Tensor` of shape `(batch_size, out_channels, mel_length)`):
                Encoder output at the mel frame rate.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Flow matching timesteps.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, out_channels)`):
                Projected speaker embedding.
            conditioning (`torch.Tensor` of shape `(batch_size, out_channels, mel_length)`):
                Mel spectrogram prefix used as conditioning.
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the attention masks are restricted to chunks.

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
            attention_bias = self._attention_bias(mask_down, hidden_states.dtype, streaming)
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
            attention_bias = self._attention_bias(mask_mid, hidden_states.dtype, streaming)
            for block in transformer_blocks:
                hidden_states = block(hidden_states, attention_bias)
            hidden_states = hidden_states.transpose(1, 2).contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = skips.pop()
            hidden_states = torch.cat([hidden_states[:, :, : skip.shape[-1]], skip], dim=1)
            hidden_states = resnet(hidden_states, mask_up, time_emb)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            attention_bias = self._attention_bias(mask_up, hidden_states.dtype, streaming)
            for block in transformer_blocks:
                hidden_states = block(hidden_states, attention_bias)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            hidden_states = upsample(hidden_states * mask_up)

        hidden_states = self.final_block(hidden_states, mask_up)
        return self.final_proj(hidden_states * mask_up) * mask


class CosyVoiceV2ConditionalCFM(CosyVoiceV1ConditionalCFM):
    """
    Conditional flow matching head that starts every sampling run from the same fixed noise, so that
    consecutive chunks of a stream stay consistent with one another.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        self.estimator = CosyVoiceV2ConditionalDecoder(config)
        self.noise_length = config.noise_length
        self.noise_seed = config.noise_seed
        self._noise: Optional[torch.Tensor] = None

    def fixed_noise(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Args:
            length (`int`):
                Number of mel frames to return.
            device (`torch.device`):
                Device of the returned tensor.
            dtype (`torch.dtype`):
                Floating point type of the returned tensor.

        Returns:
            `torch.Tensor` of shape `(1, n_feats, length)`: the prefix of the fixed noise.

        Raises:
            ValueError: If `length` exceeds `noise_length`.
        """
        if self._noise is None:
            # Built outside inference mode so that the cached tensor stays usable by autograd even
            # when the first sampling run creates it, and drawn from a saved generator state so that
            # seeding it does not disturb the caller's stream.
            with torch.inference_mode(False):
                state = torch.random.get_rng_state()
                try:
                    torch.manual_seed(self.noise_seed)
                    self._noise = torch.randn([1, self.n_feats, self.noise_length])
                finally:
                    torch.random.set_rng_state(state)
        if length > self.noise_length:
            raise ValueError(
                f"the fixed noise covers {self.noise_length} mel frames, {length} were requested."
            )
        return self._noise[:, :, :length].to(device=device, dtype=dtype)

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int,
        speaker_hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        temperature: float = 1.0,
        streaming: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            mu (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Encoder output at the mel frame rate.
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
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the estimator attends within chunks only.

        Returns:
            `torch.Tensor` of shape `(batch_size, n_feats, mel_length)`: the sampled mel spectrogram.
        """
        noise = self.fixed_noise(mu.size(2), mu.device, mu.dtype) * temperature
        t_span = torch.linspace(0, 1, num_steps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(noise, t_span, mu, mask, speaker_hidden_states, conditioning, streaming)

    def solve_euler(
        self,
        hidden_states: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker_hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Initial noise.
            t_span (`torch.Tensor` of shape `(num_steps + 1,)`):
                Timesteps of the solver.
            mu (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Encoder output at the mel frame rate.
            mask (`torch.Tensor` of shape `(batch_size, 1, mel_length)`):
                1 on valid frames.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, n_feats)`):
                Projected speaker embedding.
            conditioning (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Mel spectrogram prefix used as conditioning.
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the estimator attends within chunks only.

        Returns:
            `torch.Tensor` of shape `(batch_size, n_feats, mel_length)`: the sampled mel spectrogram.
        """
        timestep, delta = t_span[0].unsqueeze(dim=0), t_span[1] - t_span[0]
        length = hidden_states.size(2)
        dtype = speaker_hidden_states.dtype

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
                hidden_states_in, mask_in, mu_in, timestep_in, speaker_in, conditioning_in, streaming
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
        streaming: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            target (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Ground truth mel spectrogram.
            mask (`torch.Tensor` of shape `(batch_size, 1, mel_length)`):
                1 on valid frames.
            mu (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Encoder output at the mel frame rate.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, n_feats)`):
                Projected speaker embedding.
            conditioning (`torch.Tensor` of shape `(batch_size, n_feats, mel_length)`):
                Mel spectrogram prefix used as conditioning.
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the estimator attends within chunks only.

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
            noisy, mask, mu, timesteps.squeeze(), speaker_hidden_states, conditioning, streaming
        )
        return F.mse_loss(prediction * mask, vector_field * mask, reduction="sum") / (
            torch.sum(mask) * vector_field.shape[1]
        )


class CosyVoiceV2FlowModel(nn.Module):
    """
    Speech token to mel spectrogram model of CosyVoice v2. The encoder upsamples the token sequence to
    the mel frame rate itself, so no length regulator is involved.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__()
        self.config = config
        self.output_size = config.flow_output_size
        self.token_mel_ratio = config.token_mel_ratio
        self.pre_lookahead_len = config.pre_lookahead_len
        self.input_embedding = nn.Embedding(config.speech_vocab_size, config.flow_input_size)
        self.spk_embed_affine_layer = nn.Linear(config.speaker_embedding_dim, config.flow_output_size)
        self.encoder = CosyVoiceV2UpsampleEncoder(config)
        self.encoder_proj = nn.Linear(config.flow_encoder_hidden_size, config.flow_output_size)
        self.decoder = CosyVoiceV2ConditionalCFM(config)

    def forward(
        self,
        speech_token_ids: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        speech_feat: torch.Tensor,
        speech_feat_lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        streaming: bool = False,
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
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the encoder and the estimator attend within chunks only. Upstream draws it per
                batch with probability one half.

        Returns:
            `torch.Tensor`: the conditional flow matching loss.
        """
        speaker_hidden_states = self.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1))

        token_mask = (~make_pad_mask(speech_token_lengths, speech_token_ids.size(1))).unsqueeze(-1)
        hidden_states = self.input_embedding(torch.clamp(speech_token_ids, min=0)) * token_mask
        hidden_states, up_padding_mask = self.encoder(
            hidden_states, speech_token_lengths, streaming=streaming
        )
        hidden_states = self.encoder_proj(hidden_states)

        conditioning = torch.zeros_like(speech_feat)
        for index, length in enumerate(speech_feat_lengths):
            if random.random() < 0.5:
                continue
            prefix = random.randint(0, int(0.3 * length))
            conditioning[index, :prefix] = speech_feat[index, :prefix]
        conditioning = conditioning.transpose(1, 2)

        mask = up_padding_mask.to(hidden_states)
        return self.decoder.compute_loss(
            speech_feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            hidden_states.transpose(1, 2).contiguous(),
            speaker_hidden_states,
            conditioning,
            streaming=streaming,
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
        streaming: bool = False,
        finalize: bool = True,
    ) -> torch.Tensor:
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
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the encoder and the estimator attend within chunks only.
            finalize (`bool`, *optional*, defaults to `True`):
                Whether this is the last chunk of the utterance. When it is not, the last
                `pre_lookahead_len` tokens are handed to the encoder as lookahead context instead of
                being encoded.

        Returns:
            `torch.Tensor` of shape `(1, flow_output_size, mel_length)`: the generated mel spectrogram.
        """
        speaker_hidden_states = self.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1))

        token_ids = torch.concat([prompt_token_ids, speech_token_ids], dim=1)
        token_lengths = prompt_token_lengths + speech_token_lengths
        token_mask = (~make_pad_mask(token_lengths, token_ids.size(1))).unsqueeze(-1).to(speaker_embedding)
        hidden_states = self.input_embedding(torch.clamp(token_ids, min=0)) * token_mask

        if finalize:
            hidden_states, _ = self.encoder(hidden_states, token_lengths, streaming=streaming)
        else:
            context = hidden_states[:, -self.pre_lookahead_len :]
            hidden_states, _ = self.encoder(
                hidden_states[:, : -self.pre_lookahead_len], token_lengths, context=context, streaming=streaming
            )
        prompt_mel_length = prompt_feat.shape[1]
        mel_length = hidden_states.shape[1] - prompt_mel_length
        hidden_states = self.encoder_proj(hidden_states)

        conditioning = torch.zeros(
            [1, prompt_mel_length + mel_length, self.output_size],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        conditioning[:, :prompt_mel_length] = prompt_feat
        conditioning = conditioning.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([prompt_mel_length + mel_length]))).to(hidden_states)
        mel = self.decoder(
            mu=hidden_states.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            num_steps=num_steps,
            speaker_hidden_states=speaker_hidden_states,
            conditioning=conditioning,
            streaming=streaming,
        )
        return mel[:, :, prompt_mel_length:].float()


class CosyVoiceV2SpeechTokenLM(nn.Module):
    """
    Autoregressive model that turns text tokens and a speech token prompt into a sequence of
    supervised semantic speech tokens, carried by a pretrained Qwen2 decoder.

    The text is read through the decoder's own embedding table rather than a separate text encoder,
    and the speaker embedding is not part of the language model conditioning at all.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__()
        self.config = config
        self.speech_vocab_size = config.speech_vocab_size
        self.head_size = config.speech_head_size
        self.sos_index = 0
        self.task_id_index = 1
        self.eos_token_id = config.speech_vocab_size
        self.fill_token_id = config.speech_vocab_size + 2
        self.stop_token_ids = [config.speech_vocab_size + index for index in range(3)]
        self.mix_ratio = config.mix_ratio

        self.model = Qwen2Model(config.text_config)
        self.llm_embedding = nn.Embedding(2, config.lm_hidden_size)
        self.speech_embedding = nn.Embedding(self.head_size, config.lm_hidden_size)
        self.llm_decoder = nn.Linear(config.lm_hidden_size, self.head_size)

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, text_length)`):
                Text token ids.

        Returns:
            `torch.Tensor` of shape `(batch_size, text_length, lm_hidden_size)`: text embeddings.
        """
        return self.model.embed_tokens(input_ids)

    def build_inputs(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        speech_token_ids: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        bistream: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Packs one training sequence per batch entry and the targets that go with it.

        In the unistream layout the sequence is the start of sequence embedding, the text, the task id
        embedding and the teacher forced speech tokens. In the bistream layout the text and the speech
        tokens are interleaved in groups of `mix_ratio[0]` and `mix_ratio[1]`, each group predicting
        its speech tokens and then a fill token, with the tail group closed by the end of speech token.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, text_length)`):
                Text token ids.
            input_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid text tokens per sequence.
            speech_token_ids (`torch.Tensor` of shape `(batch_size, speech_length)`):
                Target speech tokens.
            speech_token_lengths (`torch.Tensor` of shape `(batch_size,)`):
                Number of valid speech tokens per sequence.
            bistream (`bool`, *optional*, defaults to `False`):
                Whether the interleaved layout is used. Upstream draws it per sample with probability
                one half, and only when the speech to text ratio exceeds `mix_ratio[1] / mix_ratio[0]`.

        Returns:
            `tuple(torch.Tensor)`: the packed inputs embeds, their lengths and the targets.
        """
        text_ratio, speech_ratio = self.mix_ratio
        sos_embed = self.llm_embedding.weight[self.sos_index].reshape(1, -1)
        task_id_embed = self.llm_embedding.weight[self.task_id_index].reshape(1, -1)
        text_embeds = self.embed_text(input_ids)
        speech_embeds = self.speech_embedding(speech_token_ids)

        sequences, targets = [], []
        for index in range(input_ids.size(0)):
            text_length = int(input_lengths[index])
            speech_length = int(speech_token_lengths[index])
            tokens = speech_token_ids[index, :speech_length].tolist()
            if bistream and speech_length / max(text_length, 1) > speech_ratio / text_ratio:
                target = [IGNORE_ID]
                parts = [sos_embed]
                for group in range(math.ceil((text_length + 1) / text_ratio)):
                    text_slice = slice(group * text_ratio, (group + 1) * text_ratio)
                    speech_slice = slice(group * speech_ratio, (group + 1) * speech_ratio)
                    group_text_length = len(range(*text_slice.indices(text_length)))
                    if group_text_length == text_ratio:
                        target += [IGNORE_ID] * (text_ratio - 1)
                        target += tokens[speech_slice]
                        target.append(self.fill_token_id)
                        parts.append(text_embeds[index, text_slice])
                        parts.append(speech_embeds[index, speech_slice])
                    else:
                        target += [IGNORE_ID] * group_text_length
                        target += tokens[group * speech_ratio :]
                        target.append(self.eos_token_id)
                        parts.append(text_embeds[index, group * text_ratio : text_length])
                        parts.append(task_id_embed)
                        parts.append(speech_embeds[index, group * speech_ratio : speech_length])
                sequences.append(torch.concat(parts, dim=0))
                targets.append(torch.tensor(target))
            else:
                sequences.append(
                    torch.concat(
                        [
                            sos_embed,
                            text_embeds[index, :text_length],
                            task_id_embed,
                            speech_embeds[index, :speech_length],
                        ],
                        dim=0,
                    )
                )
                targets.append(
                    torch.tensor([IGNORE_ID] * (1 + text_length) + tokens + [self.eos_token_id])
                )

        lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.int32)
        inputs_embeds = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=IGNORE_ID)
        labels = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=IGNORE_ID)
        return inputs_embeds, lengths.to(input_ids.device), labels.to(input_ids.device)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Args:
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, lm_hidden_size)`):
                Packed language model inputs.
            input_lengths (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                Number of valid steps per sequence. Left out while decoding one step at a time.
            past_key_values (`Cache`, *optional*):
                Keys and values of the previous steps.
            use_cache (`bool`, *optional*, defaults to `False`):
                Whether the updated cache is returned.

        Returns:
            `tuple`: the speech token logits and the updated cache.
        """
        attention_mask = None
        if input_lengths is not None:
            attention_mask = ~make_pad_mask(input_lengths, inputs_embeds.size(1))
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return self.llm_decoder(outputs.last_hidden_state), outputs.past_key_values


class CosyVoiceV2SineGen(nn.Module):
    """
    Generates the harmonic excitation of the neural source filter from an f0 contour.

    The phase is integrated at the f0 frame rate rather than at the waveform rate: the per sample
    angular increments are decimated by `upsample_scale`, accumulated, and interpolated back up. This
    is the generator upstream selects for every sampling rate other than 22050 Hz.

    Args:
        sampling_rate (`int`):
            Sampling rate of the excitation.
        upsample_scale (`int`):
            Ratio between the waveform rate and the f0 frame rate.
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
        self,
        sampling_rate: int,
        upsample_scale: int,
        num_harmonics: int,
        amplitude: float,
        noise_std: float,
        voiced_threshold: float,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.num_harmonics = num_harmonics
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def _sine(self, f0_harmonics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0_harmonics (`torch.Tensor` of shape `(batch_size, num_samples, num_harmonics + 1)`):
                f0 and its overtones.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_samples, num_harmonics + 1)`: the sine waves.
        """
        rad_values = (f0_harmonics / self.sampling_rate) % 1
        phase_offset = torch.rand(
            f0_harmonics.shape[0], f0_harmonics.shape[2], device=f0_harmonics.device
        )
        phase_offset[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + phase_offset

        rad_values = F.interpolate(
            rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear"
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=float(self.upsample_scale),
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f0 (`torch.Tensor` of shape `(batch_size, num_samples, 1)`):
                Upsampled f0 contour, zero on unvoiced steps.

        Returns:
            `tuple(torch.Tensor)`: the sine waves of shape `(batch_size, num_samples, num_harmonics + 1)`
            and the voiced mask of shape `(batch_size, num_samples, 1)`.
        """
        harmonics = torch.arange(1, self.num_harmonics + 2, device=f0.device, dtype=f0.dtype)
        sine_waves = self._sine(f0 * harmonics) * self.amplitude

        voiced = (f0 > self.voiced_threshold).to(f0.dtype)
        noise_amplitude = voiced * self.noise_std + (1 - voiced) * self.amplitude / 3
        sine_waves = sine_waves * voiced + noise_amplitude * torch.randn_like(sine_waves)
        return sine_waves, voiced


class CosyVoiceV2SourceModule(nn.Module):
    """
    Merges the harmonics of [`CosyVoiceV2SineGen`] into a single excitation signal.

    Args:
        sampling_rate (`int`):
            Sampling rate of the excitation.
        upsample_scale (`int`):
            Ratio between the waveform rate and the f0 frame rate.
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
        self,
        sampling_rate: int,
        upsample_scale: int,
        num_harmonics: int,
        amplitude: float,
        noise_std: float,
        voiced_threshold: float,
    ):
        super().__init__()
        self.amplitude = amplitude
        self.l_sin_gen = CosyVoiceV2SineGen(
            sampling_rate, upsample_scale, num_harmonics, amplitude, noise_std, voiced_threshold
        )
        self.l_linear = nn.Linear(num_harmonics + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sine_waves, _ = self.l_sin_gen(f0)
        return self.l_tanh(self.l_linear(sine_waves))


class CosyVoiceV2HiFTGenerator(CosyVoiceV1HiFTGenerator):
    """
    HiFTNet vocoder of CosyVoice v2, which is the v1 vocoder at 24 kHz with a third upsampling stage
    and the interpolating sine generator upstream selects for every rate other than 22050 Hz.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        self.m_source = CosyVoiceV2SourceModule(
            config.sample_rate,
            int(np.prod(config.vocoder_upsample_rates) * config.vocoder_istft_hop_length),
            config.vocoder_num_harmonics,
            config.vocoder_source_amplitude,
            config.vocoder_source_noise_std,
            config.vocoder_voiced_threshold,
        )


class CosyVoiceV2SpeechTokenizerRotaryEmbedding(Qwen2RotaryEmbedding):
    """
    Rotary position embedding of the speech tokenizer attention.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(
            Qwen2Config(
                hidden_size=config.speech_tokenizer_hidden_size,
                num_attention_heads=config.speech_tokenizer_num_heads,
                max_position_embeddings=config.speech_tokenizer_max_position_embeddings,
                rope_theta=config.speech_tokenizer_rope_theta,
            )
        )


class CosyVoiceV2SpeechTokenizerAttention(CosyVoiceV1SpeechTokenizerAttention):
    """
    Self attention of the speech tokenizer encoder of CosyVoice v2, which drops v1's position table
    for a rotary embedding and adds a memory block to its output: a depthwise convolution over the
    masked value projection, which is what a scalar attention memory network contributes.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        hidden_size = config.speech_tokenizer_hidden_size
        kernel_size = config.speech_tokenizer_fsmn_kernel_size
        self.fsmn_block = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)
        self.padding = ((kernel_size - 1) // 2, kernel_size - 1 - (kernel_size - 1) // 2)

    def memory(self, value: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            value (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Value projection.
            padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
                Mask that is `True` on padding positions.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: the memory block
            output.
        """
        keep = (~padding_mask).unsqueeze(-1).to(value.dtype)
        value = value * keep
        memory = self.fsmn_block(F.pad(value.transpose(1, 2), self.padding)).transpose(1, 2)
        return (memory + value) * keep

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input sequence.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask that is `True` on padding positions. Upstream leaves the attention scores
                unmasked and zeroes the padded rows of the attention output instead, which is what
                this reproduces.
            position_embeddings (`tuple(torch.Tensor)`, *optional*):
                Cosine and sine of the rotary embedding, each of shape
                `(batch_size, sequence_length, head_dim)`.
            kwargs:
                Ignored.

        Returns:
            `tuple(torch.Tensor, None)`: the attention output of shape
            `(batch_size, sequence_length, hidden_size)`, and `None` in place of the attention
            weights [`WhisperEncoderLayer`] discards.
        """
        batch_size, length, _ = hidden_states.shape
        shape = (batch_size, length, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states)
        memory = self.memory(value, attention_mask)
        query = self.q_proj(hidden_states).view(shape).permute(0, 2, 1, 3)
        key = self.k_proj(hidden_states).view(shape).permute(0, 2, 1, 3)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        scores = (query * self.scaling) @ (key.transpose(2, 3) * self.scaling)
        context = scores.softmax(dim=-1) @ value.view(shape).permute(0, 2, 1, 3)
        context = context.masked_fill(attention_mask[:, None, :, None], 0.0)
        return self.out_proj(context.permute(0, 2, 1, 3).reshape(batch_size, length, -1)) + memory, None


class CosyVoiceV2SpeechTokenizerLayer(CosyVoiceV1SpeechTokenizerLayer):
    """
    One encoder layer of the speech tokenizer of CosyVoice v2, which is v1's around
    [`CosyVoiceV2SpeechTokenizerAttention`].

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        self.self_attn = CosyVoiceV2SpeechTokenizerAttention(config)


class CosyVoiceV2SpeechTokenizerQuantizer(nn.Module):
    """
    Finite scalar quantizer of the speech tokenizer of CosyVoice v2, which projects the encoder
    output onto one dimension per entry of `speech_tokenizer_fsq_levels`, rounds each to its own
    grid, and reads the token id off the mixed radix number the rounded dimensions spell.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    # Fraction the bounding tangent is pulled in by, so that a saturated projection still rounds to
    # the outermost level rather than past it.
    eps = 1e-3

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__()
        self.levels = config.speech_tokenizer_fsq_levels
        self.project_in = nn.Linear(config.speech_tokenizer_hidden_size, len(self.levels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Encoder output.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length)`: the speech token ids.
        """
        levels = torch.tensor(self.levels, dtype=hidden_states.dtype, device=hidden_states.device)
        half_range = (levels - 1) * (1 - self.eps) / 2
        offset = torch.where(levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_range).atanh()
        half_width = levels // 2
        basis = torch.cumprod(torch.cat([torch.ones_like(levels[:1]), levels[:-1]]), dim=0)

        projected = self.project_in(hidden_states)
        codes = torch.round(torch.tanh(projected + shift) * half_range - offset)
        return (((codes / half_width) * half_width + half_width) * basis).sum(dim=-1).to(torch.int32)


class CosyVoiceV2SpeechTokenizer(CosyVoiceV1SpeechTokenizer):
    """
    Supervised semantic speech tokenizer of CosyVoice v2, which strides both of its opening
    convolutions, so one token stands for four mel frames, and closes with
    [`CosyVoiceV2SpeechTokenizerQuantizer`] instead of a codebook.

    Args:
        config ([`CosyVoiceV2Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CosyVoiceV2SpeechTokenizerLayer(config) for _ in range(config.speech_tokenizer_num_layers)]
        )
        self.quantizer = CosyVoiceV2SpeechTokenizerQuantizer(config)
        self.rotary_emb = CosyVoiceV2SpeechTokenizerRotaryEmbedding(config)

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
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, padding_mask, position_embeddings=position_embeddings)
        return hidden_states


@auto_docstring(
    custom_intro="""
    Output of [`CosyVoiceV2ForConditionalGeneration`].
    """
)
@dataclass
class CosyVoiceV2Output(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Label smoothed cross entropy over the speech tokens, returned when `labels` is provided.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, speech_head_size)`):
        Speech token scores.
    accuracy (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Fraction of correctly predicted speech tokens, returned when `labels` is provided.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    accuracy: Optional[torch.FloatTensor] = None


@auto_docstring
class CosyVoiceV2PreTrainedModel(CosyVoiceV1PreTrainedModel):
    config: CosyVoiceV2Config
    base_model_prefix = "cosyvoice_v2"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = [r"llm\.model\.rotary_emb\.inv_freq", r"llm\.model\.lm_head\."]

    @classmethod
    def _released_checkpoint(cls, source, **kwargs) -> "tuple[CosyVoiceV2Config, dict[str, torch.Tensor]] | None":
        r"""
        Reads a released CosyVoice v2 directory, whose Qwen2 sub directory is fetched alongside the
        three network files because the configuration is built from it.

        Args:
            source (`str` or `os.PathLike`, *optional*):
                Repository id or local directory.
            kwargs (`dict`, *optional*):
                Fields of `weight_conversion.DOWNLOAD_KWARGS` selecting a revision and a cache.

        Returns:
            `tuple[CosyVoiceV2Config, dict[str, torch.Tensor]]` or `None`: The configuration and the
            merged tensors, or `None` when `source` holds no released checkpoint.
        """
        directory = resolve_checkpoint(
            source, tuple(CHECKPOINT_FILES.values()), (f"{TEXT_MODEL_SUBDIR}/*",), **kwargs
        )
        if directory is None:
            return None
        return build_config(directory), load_checkpoint(directory)


@auto_docstring(
    custom_intro="""
    CosyVoice v2, made of a Qwen2 based text to speech token language model, a chunk aware conditional
    flow matching model turning speech tokens into a mel spectrogram, and a HiFTNet vocoder.

    The three networks are trained one at a time upstream, so `forward` optimizes the language model
    objective only. [`CosyVoiceV2FlowModel.forward`] returns the flow matching objective and
    [`CosyVoiceV2HiFTGenerator.forward`] returns the waveform and the f0 contour the vocoder objective
    is computed from.
    """
)
class CosyVoiceV2ForConditionalGeneration(CosyVoiceV2GenerationMixin, CosyVoiceV2PreTrainedModel):
    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        self.llm = CosyVoiceV2SpeechTokenLM(config)
        self.flow = CosyVoiceV2FlowModel(config)
        self.hift = CosyVoiceV2HiFTGenerator(config)
        self.criterion = CosyVoiceV1LabelSmoothingLoss(
            config.speech_head_size, config.label_smoothing, config.length_normalized_loss
        )
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        speech_token_ids: torch.Tensor,
        speech_token_lengths: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        bistream: bool = False,
    ) -> CosyVoiceV2Output:
        r"""
        input_ids (`torch.Tensor` of shape `(batch_size, text_length)`):
            Text token ids.
        input_lengths (`torch.Tensor` of shape `(batch_size,)`):
            Number of valid text tokens per sequence.
        speech_token_ids (`torch.Tensor` of shape `(batch_size, speech_length)`):
            Teacher forced speech tokens.
        speech_token_lengths (`torch.Tensor` of shape `(batch_size,)`):
            Number of valid speech tokens per sequence.
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Speech token targets padded with -1. When left out they are built alongside the
            inputs, which is what upstream does.
        bistream (`bool`, *optional*, defaults to `False`):
            Whether the interleaved training layout is used.
        """
        inputs_embeds, lengths, built_labels = self.llm.build_inputs(
            input_ids, input_lengths, speech_token_ids, speech_token_lengths, bistream=bistream
        )
        logits, _ = self.llm(inputs_embeds, lengths)

        if labels is None:
            labels = built_labels
        loss = self.criterion(logits, labels)
        predictions = logits.view(-1, self.config.speech_head_size).argmax(-1).view_as(labels)
        keep = labels != IGNORE_ID
        accuracy = (predictions.masked_select(keep) == labels.masked_select(keep)).sum() / keep.sum()
        return CosyVoiceV2Output(loss=loss, logits=logits, accuracy=accuracy)


__all__ = [
    "CosyVoiceV2CausalBlock1D",
    "CosyVoiceV2CausalConv1d",
    "CosyVoiceV2CausalResnetBlock1D",
    "CosyVoiceV2ConditionalCFM",
    "CosyVoiceV2ConditionalDecoder",
    "CosyVoiceV2FlowModel",
    "CosyVoiceV2ForConditionalGeneration",
    "CosyVoiceV2HiFTGenerator",
    "CosyVoiceV2Output",
    "CosyVoiceV2PreLookaheadLayer",
    "CosyVoiceV2PreTrainedModel",
    "CosyVoiceV2SineGen",
    "CosyVoiceV2SourceModule",
    "CosyVoiceV2SpeechTokenLM",
    "CosyVoiceV2SpeechTokenizer",
    "CosyVoiceV2Upsample1D",
    "CosyVoiceV2UpsampleEncoder",
]
