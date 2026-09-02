# Copyright 2025 SparkAudio, Xinsheng Wang and the LatentForge team. All rights reserved.
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
"""PyTorch Spark-TTS model."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.modeling_utils import PreTrainedAudioTokenizerBase
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.dac.modeling_dac import DacResidualUnit, Snake1d
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import maybe_autocast

from .configuration_spark_tts import SparkTTSBiCodecConfig, SparkTTSConfig


@auto_docstring(custom_intro="Codes and reconstruction produced by a full BiCodec analysis/synthesis pass.")
@dataclass
class SparkTTSBiCodecOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor`, *optional*):
        Sum of the commitment and codebook terms of the semantic quantizer.
    audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
        Reconstructed waveform.
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames)`):
        Semantic token indices.
    global_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, num_speaker_tokens)`):
        Global speaker token indices.
    predicted_features (`torch.FloatTensor` of shape `(batch_size, hidden_size, num_frames)`):
        Postnet prediction of the self-supervised features the semantic encoder consumed.
    speaker_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        Utterance-level speaker embedding pooled by the ECAPA-TDNN encoder.
    conditioning_embedding (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        Quantized speaker embedding the prenet and the wave generator are conditioned on.
    perplexity (`torch.FloatTensor`):
        Perplexity of the semantic code distribution over the batch.
    active_codes (`torch.FloatTensor`):
        Number of semantic codes counted as in use.
    """

    loss: torch.FloatTensor | None = None
    audio_values: torch.FloatTensor | None = None
    audio_codes: torch.LongTensor | None = None
    global_codes: torch.LongTensor | None = None
    predicted_features: torch.FloatTensor | None = None
    speaker_embedding: torch.FloatTensor | None = None
    conditioning_embedding: torch.FloatTensor | None = None
    perplexity: torch.FloatTensor | None = None
    active_codes: torch.FloatTensor | None = None


@auto_docstring(custom_intro="Discrete codes produced by the BiCodec analysis path.")
@dataclass
class SparkTTSBiCodecEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames)`):
        Semantic token indices.
    global_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, num_speaker_tokens)`):
        Global speaker token indices.
    """

    audio_codes: torch.LongTensor | None = None
    global_codes: torch.LongTensor | None = None


@auto_docstring(custom_intro="Waveform produced by the BiCodec synthesis path.")
@dataclass
class SparkTTSBiCodecDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
        Reconstructed waveform.
    """

    audio_values: torch.FloatTensor | None = None


class SparkTTSFactorizedVectorQuantizer(nn.Module):
    """
    Single-codebook vector quantizer of the semantic stream. The latent is projected down to `codebook_dim` before
    the lookup and back up afterwards, and both the latent and the codebook are L2-normalized so that the euclidean
    nearest neighbour is the cosine nearest neighbour.
    """

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim
        self.commitment_weight = config.commitment_weight
        self.codebook_loss_weight = config.codebook_loss_weight
        self.decay = config.codebook_ema_decay
        self.threshold_ema_dead_code = config.threshold_ema_dead_code

        if config.hidden_size != config.codebook_dim:
            self.in_proj = nn.Conv1d(config.hidden_size, config.codebook_dim, kernel_size=1)
            self.out_proj = nn.Conv1d(config.codebook_dim, config.hidden_size, kernel_size=1)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        self.register_buffer("cluster_size", torch.zeros(config.codebook_size))

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encodings = latents.transpose(1, 2).reshape(-1, self.codebook_dim)
        codebook = self.codebook.weight

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        distances = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        codes = (-distances).max(1)[1].reshape(latents.shape[0], -1)
        return self.codebook(codes).transpose(1, 2), codes

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, codes = self.decode_latents(self.in_proj(hidden_states))
        return codes

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.codebook(codes).transpose(1, 2))

    def forward(self, hidden_states: torch.Tensor):
        projected_latents = self.in_proj(hidden_states)
        quantized, codes = self.decode_latents(projected_latents)

        one_hot = F.one_hot(codes, self.codebook_size).type(projected_latents.dtype)
        code_usage = one_hot.sum(0).sum(0)
        probabilities = one_hot.reshape(-1, self.codebook_size).mean(0)
        perplexity = torch.exp(-torch.sum(probabilities * torch.log(probabilities + 1e-10)))

        active_codes = (code_usage > 0).sum()
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(code_usage, alpha=1 - self.decay)
            active_codes = (self.cluster_size > self.threshold_ema_dead_code).sum()

        commitment_loss = (
            F.mse_loss(projected_latents, quantized.detach(), reduction="none").mean([1, 2]) * self.commitment_weight
        )
        codebook_loss = (
            F.mse_loss(quantized, projected_latents.detach(), reduction="none").mean([1, 2])
            * self.codebook_loss_weight
        )

        # noop in the forward pass, straight-through gradient estimator in the backward pass
        quantized = projected_latents + (quantized - projected_latents).detach()
        quantized = self.out_proj(quantized)

        return quantized, codes, commitment_loss.mean() + codebook_loss.mean(), perplexity, active_codes.float()


class SparkTTSFiniteScalarQuantizer(nn.Module):
    """
    Finite scalar quantizer of one residual stage of the global stream. Every latent dimension is bounded, rounded to
    one of `levels[dim]` values with a straight-through estimator, and folded into a single mixed-radix index.
    """

    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels_per_dim = list(levels)
        self.codebook_size = math.prod(self.levels_per_dim)
        self.levels = nn.Buffer(self._compute_levels(), persistent=False)
        self.basis = nn.Buffer(self._compute_basis(), persistent=False)
        self.codebook = nn.Buffer(self._compute_codebook(), persistent=False)

    def _compute_levels(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(self.levels_per_dim, dtype=torch.int32, device=device)

    def _compute_basis(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.cumprod(
            torch.tensor([1] + self.levels_per_dim[:-1], device=device), dim=0, dtype=torch.int32
        )

    def _compute_codebook(self, device: torch.device | None = None) -> torch.Tensor:
        levels = self._compute_levels(device=device)
        indices = torch.arange(self.codebook_size, device=device).unsqueeze(-1)
        level_indices = (indices // self._compute_basis(device=device)) % levels
        half_width = levels // 2
        return (level_indices - half_width) / half_width

    def bound(self, hidden_states: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_range = (self.levels - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_range).atanh()
        return (hidden_states + shift).tanh() * half_range - offset

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = hidden_states.dtype
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            hidden_states = hidden_states.float()
            half_width = self.levels // 2
            bounded = self.bound(hidden_states)
            codes = (bounded + (bounded.round() - bounded).detach()) / half_width
            indices = ((codes * half_width + half_width) * self.basis).sum(dim=-1).to(torch.int32)
        return codes.to(original_dtype), indices


class SparkTTSResidualFiniteScalarQuantizer(nn.Module):
    """
    Residual stack of finite scalar quantizers applied to the channel-first speaker latents. Stage `i` quantizes what
    the previous stages left over, on a grid scaled by `(levels - 1) ** -i`.
    """

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        levels = list(config.fsq_levels)
        codebook_dim = len(levels)
        dim = config.speaker_latent_dim

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()

        self.levels_per_dim = levels
        self.num_quantizers = config.fsq_num_quantizers
        self.layers = nn.ModuleList(
            [SparkTTSFiniteScalarQuantizer(levels) for _ in range(config.fsq_num_quantizers)]
        )
        self.scales = nn.Buffer(self._compute_scales(), persistent=False)

    def _compute_scales(self, device: torch.device | None = None) -> torch.Tensor:
        levels = torch.tensor(self.levels_per_dim, dtype=torch.float32, device=device)
        return torch.stack([(levels - 1) ** -index for index in range(self.num_quantizers)])

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(1, 2)
        quantized = 0.0
        for index, (layer, scale) in enumerate(zip(self.layers, self.scales)):
            quantized = quantized + layer.codebook[codes[..., index].long()] * scale
        return self.project_out(quantized)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.project_in(hidden_states.transpose(1, 2))

        quantized_out = 0.0
        residual = hidden_states
        all_codes = []
        for layer, scale in zip(self.layers, self.scales):
            quantized, codes = layer(residual / scale)
            quantized = quantized * scale
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_codes.append(codes)

        quantized_out = self.project_out(quantized_out)
        return quantized_out.transpose(1, 2), torch.stack(all_codes, dim=-1).transpose(1, 2)


class SparkTTSAdaLayerNorm(nn.Module):
    """Layer normalization whose scale and shift are predicted from a conditioning vector."""

    def __init__(self, condition_dim: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Linear(condition_dim, embedding_dim)
        self.shift = nn.Linear(condition_dim, embedding_dim)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale(condition)
        shift = self.shift(condition)
        hidden_states = F.layer_norm(hidden_states, (self.dim,), eps=self.eps)
        return hidden_states * scale.unsqueeze(1) + shift.unsqueeze(1)


class SparkTTSConvNeXtBlock(nn.Module):
    """ConvNeXt block over a 1D time axis: depthwise convolution, normalization, and an inverted-bottleneck MLP."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        condition_dim: int | None = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = condition_dim is not None
        if condition_dim is not None:
            self.norm = SparkTTSAdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma_init_value = layer_scale_init_value
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).transpose(1, 2)
        hidden_states = self.norm(hidden_states, condition) if self.adanorm else self.norm(hidden_states)
        hidden_states = self.pwconv2(self.act(self.pwconv1(hidden_states)))
        if self.gamma is not None:
            hidden_states = self.gamma * hidden_states
        return residual + hidden_states.transpose(1, 2)


class SparkTTSVocosBackbone(nn.Module):
    """
    Stack of [`SparkTTSConvNeXtBlock`]s preserving the temporal resolution. Takes channel-first input and returns
    time-major output.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: float | None = None,
        condition_dim: int | None = None,
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = condition_dim is not None
        if condition_dim is not None:
            self.norm = SparkTTSAdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.layers = nn.ModuleList(
            [
                SparkTTSConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    condition_dim=condition_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embed(hidden_states).transpose(1, 2)
        hidden_states = self.norm(hidden_states, condition) if self.adanorm else self.norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states, condition)
        return self.final_layer_norm(hidden_states.transpose(1, 2))


class SparkTTSSamplingBlock(nn.Module):
    """
    Resampling block that sums a learned (transposed) convolution with parameter-free repeat/average paths. A scale of
    1 leaves the corresponding path out entirely, so the block is a no-op with no parameters.
    """

    def __init__(self, dim: int, groups: int = 1, upsample_scale: int = 1, downsample_scale: int = 1):
        super().__init__()
        self.upsample_scale = upsample_scale
        self.downsample_scale = downsample_scale

        if upsample_scale > 1:
            self.upsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(
                    dim,
                    dim,
                    kernel_size=upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    groups=groups,
                ),
            )

        if downsample_scale > 1:
            self.downsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=2 * downsample_scale,
                    stride=downsample_scale,
                    padding=downsample_scale // 2 + downsample_scale % 2,
                    groups=groups,
                ),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)

        if self.upsample_scale > 1:
            repeated = hidden_states.repeat_interleave(self.upsample_scale, dim=2)
            merged = repeated + self.upsampler(hidden_states)
        else:
            repeated = hidden_states
            merged = hidden_states

        if self.downsample_scale > 1:
            convolved = self.downsampler(merged)
            pooled_merged = F.avg_pool1d(merged, self.downsample_scale, stride=self.downsample_scale)
            pooled_repeated = F.avg_pool1d(repeated, self.downsample_scale, stride=self.downsample_scale)
        else:
            convolved = merged
            pooled_merged = merged
            pooled_repeated = repeated

        return convolved + pooled_repeated + pooled_merged


class SparkTTSResamplingLayer(nn.Module):
    """A [`SparkTTSSamplingBlock`] followed by a short [`SparkTTSVocosBackbone`]."""

    def __init__(self, config: SparkTTSBiCodecConfig, upsample_scale: int = 1, downsample_scale: int = 1):
        super().__init__()
        self.sampler = SparkTTSSamplingBlock(
            dim=config.vocos_dim,
            groups=config.vocos_dim,
            upsample_scale=upsample_scale,
            downsample_scale=downsample_scale,
        )
        self.backbone = SparkTTSVocosBackbone(
            input_channels=config.vocos_dim,
            dim=config.vocos_dim,
            intermediate_dim=config.vocos_intermediate_dim,
            num_layers=config.resampling_num_layers,
            layer_scale_init_value=config.layer_scale_init_value,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.sampler(hidden_states))


class SparkTTSSemanticEncoder(nn.Module):
    """Maps averaged self-supervised speech features to the latent the semantic quantizer operates on."""

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        self.backbone = SparkTTSVocosBackbone(
            input_channels=config.semantic_model_config.hidden_size,
            dim=config.vocos_dim,
            intermediate_dim=config.vocos_intermediate_dim,
            num_layers=config.encoder_num_layers,
            layer_scale_init_value=config.layer_scale_init_value,
        )
        self.resample_layers = nn.ModuleList(
            [SparkTTSResamplingLayer(config, downsample_scale=ratio) for ratio in config.encoder_sample_ratios]
        )
        self.project = nn.Linear(config.vocos_dim, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(hidden_states)
        for layer in self.resample_layers:
            hidden_states = layer(hidden_states)
        return self.project(hidden_states).transpose(1, 2)


class SparkTTSSemanticDecoder(nn.Module):
    """
    Mirror of [`SparkTTSSemanticEncoder`] used for both the prenet, which conditions on the speaker embedding through
    adaptive layer normalization, and the unconditioned postnet.
    """

    def __init__(
        self,
        config: SparkTTSBiCodecConfig,
        num_layers: int,
        sample_ratios: list[int],
        condition_dim: int | None = None,
    ):
        super().__init__()
        self.linear_pre = nn.Linear(config.hidden_size, config.vocos_dim)
        self.resample_layers = nn.ModuleList(
            [SparkTTSResamplingLayer(config, upsample_scale=ratio) for ratio in sample_ratios]
        )
        self.backbone = SparkTTSVocosBackbone(
            input_channels=config.vocos_dim,
            dim=config.vocos_dim,
            intermediate_dim=config.vocos_intermediate_dim,
            num_layers=num_layers,
            layer_scale_init_value=config.layer_scale_init_value,
            condition_dim=condition_dim,
        )
        self.linear = nn.Linear(config.vocos_dim, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.linear_pre(hidden_states.transpose(1, 2))
        for layer in self.resample_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.backbone(hidden_states.transpose(1, 2), condition=condition)
        return self.linear(hidden_states).transpose(1, 2)


class SparkTTSWaveGeneratorBlock(nn.Module):
    """Upsampling block of the wave generator: a transposed convolution followed by three dilated residual units."""

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.snake = Snake1d(input_dim)
        self.conv_t = nn.ConvTranspose1d(
            input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size - stride) // 2
        )
        self.res_unit1 = DacResidualUnit(output_dim, dilation=1)
        self.res_unit2 = DacResidualUnit(output_dim, dilation=3)
        self.res_unit3 = DacResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t(self.snake(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class SparkTTSWaveGenerator(nn.Module):
    """Turns the conditioned semantic latent into a waveform by repeated upsampling."""

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        channels = config.wave_generator_hidden_size
        self.conv_in = nn.Conv1d(config.hidden_size, channels, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            [
                SparkTTSWaveGeneratorBlock(channels // 2**index, channels // 2 ** (index + 1), kernel_size, stride)
                for index, (kernel_size, stride) in enumerate(
                    zip(config.upsample_kernel_sizes, config.upsample_rates)
                )
            ]
        )
        output_dim = channels // 2 ** len(config.upsample_rates)
        self.snake_out = Snake1d(output_dim)
        self.conv_out = nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return torch.tanh(self.conv_out(self.snake_out(hidden_states)))


class SparkTTSConv1dReluBn(nn.Module):
    """Convolution, ReLU, then batch normalization, in that order."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(hidden_states)))


class SparkTTSRes2Conv1dReluBn(nn.Module):
    """
    Res2Net-style convolution: the channels are split into `scale` groups and every group but the first is convolved
    after being summed with the previous group's output, which widens the receptive field within a single layer.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        scale: int = 4,
    ):
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f"Channel count {channels} is not divisible by the Res2Net scale {scale}.")
        self.scale = scale
        self.width = channels // scale
        self.num_blocks = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation)
                for _ in range(self.num_blocks)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.num_blocks)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = []
        splits = torch.split(hidden_states, self.width, 1)
        current = splits[0]
        for index, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if index >= 1:
                current = current + splits[index]
            current = bn(F.relu(conv(current)))
            outputs.append(current)
        if self.scale != 1:
            outputs.append(splits[self.num_blocks])
        return torch.cat(outputs, dim=1)


class SparkTTSSqueezeExcite(nn.Module):
    """Channel gating from the time-averaged activation."""

    def __init__(self, channels: int, bottleneck_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(channels, bottleneck_dim)
        self.linear2 = nn.Linear(bottleneck_dim, channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = hidden_states.mean(dim=2)
        gate = torch.sigmoid(self.linear2(F.relu(self.linear1(gate))))
        return hidden_states * gate.unsqueeze(2)


class SparkTTSSERes2Block(nn.Module):
    """Residual SE-Res2Net block of the ECAPA-TDNN speaker encoder."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        scale: int,
        se_bottleneck_dim: int,
    ):
        super().__init__()
        self.conv1 = SparkTTSConv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0)
        self.res2conv = SparkTTSRes2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale)
        self.conv2 = SparkTTSConv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0)
        self.se = SparkTTSSqueezeExcite(channels, bottleneck_dim=se_bottleneck_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.res2conv(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return residual + self.se(hidden_states)


class SparkTTSAttentiveStatisticsPooling(nn.Module):
    """
    Pools a frame sequence into a mean/standard-deviation pair with per-channel attention weights. The attention
    scores see the global mean and standard deviation alongside the frame itself.
    """

    def __init__(self, in_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context_mean = torch.mean(hidden_states, dim=-1, keepdim=True).expand_as(hidden_states)
        context_std = torch.sqrt(torch.var(hidden_states, dim=-1, keepdim=True) + 1e-7).expand_as(hidden_states)
        attention_input = torch.cat((hidden_states, context_mean, context_std), dim=1)

        attention = torch.tanh(self.linear1(attention_input))
        attention = torch.softmax(self.linear2(attention), dim=2)
        mean = torch.sum(attention * hidden_states, dim=2)
        variance = torch.sum(attention * (hidden_states**2), dim=2) - mean**2
        return torch.cat([mean, torch.sqrt(variance.clamp(min=1e-7))], dim=1)


class SparkTTSEcapaTdnn(nn.Module):
    """
    ECAPA-TDNN speaker encoder. Returns both the pooled speaker embedding and the aggregated frame-level features the
    perceiver resampler consumes.
    """

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        channels = config.speaker_encoder_channels
        self.layer1 = SparkTTSConv1dReluBn(config.num_mel_bins, channels, kernel_size=5, padding=2)
        self.layers = nn.ModuleList(
            [
                SparkTTSSERes2Block(
                    channels,
                    kernel_size=config.speaker_encoder_kernel_size,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    scale=config.speaker_encoder_res2net_scale,
                    se_bottleneck_dim=config.speaker_encoder_se_bottleneck_dim,
                )
                for dilation in config.speaker_encoder_dilations
            ]
        )

        mfa_dim = config.speaker_encoder_mfa_dim
        self.mfa_conv = nn.Conv1d(channels * len(config.speaker_encoder_dilations), mfa_dim, kernel_size=1)
        self.pool = SparkTTSAttentiveStatisticsPooling(
            mfa_dim, bottleneck_dim=config.speaker_encoder_attention_bottleneck_dim
        )
        self.bn = nn.BatchNorm1d(mfa_dim * 2)
        self.linear = nn.Linear(mfa_dim * 2, config.hidden_size)

    def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.layer1(input_features.permute(0, 2, 1))

        block_outputs = []
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            block_outputs.append(hidden_states)

        features = F.relu(self.mfa_conv(torch.cat(block_outputs, dim=1)))
        speaker_embedding = self.linear(self.bn(self.pool(features)))
        return speaker_embedding, features


class SparkTTSPerceiverRMSNorm(nn.Module):
    """Root-mean-square normalization written as an L2 normalization rescaled by `sqrt(dim)`."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(hidden_states, dim=-1) * self.scale * self.gamma


class SparkTTSPerceiverAttention(nn.Module):
    """
    Cross-attention from the learned latents to the frame-level features. The latents are prepended to the keys and
    values, so a latent can attend to the other latents as well as to the context.
    """

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        self.num_heads = config.perceiver_num_attention_heads
        self.head_dim = config.perceiver_head_dim
        inner_dim = self.num_heads * self.head_dim
        dim = config.speaker_latent_dim

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, num_latents, _ = hidden_states.shape
        context = torch.cat((hidden_states, context), dim=-2)

        query = self.to_q(hidden_states)
        key, value = self.to_kv(context).chunk(2, dim=-1)
        query, key, value = (
            tensor.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            for tensor in (query, key, value)
        )

        attn_output = F.scaled_dot_product_attention(query, key, value)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_latents, -1)
        return self.to_out(attn_output)


class SparkTTSPerceiverFeedForward(nn.Module):
    """Gated GELU feed-forward network of the perceiver resampler."""

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        dim = config.speaker_latent_dim
        inner_dim = int(dim * config.perceiver_ffn_multiplier * 2 / 3)
        self.fc1 = nn.Linear(dim, inner_dim * 2)
        self.fc2 = nn.Linear(inner_dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(F.gelu(gate) * hidden_states)


class SparkTTSPerceiverLayer(nn.Module):
    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        self.attn = SparkTTSPerceiverAttention(config)
        self.ff = SparkTTSPerceiverFeedForward(config)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attn(hidden_states, context) + hidden_states
        return self.ff(hidden_states) + hidden_states


class SparkTTSPerceiverResampler(nn.Module):
    """Compresses a variable-length feature sequence into a fixed number of learned latents."""

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        dim = config.speaker_latent_dim
        context_dim = config.speaker_encoder_mfa_dim
        self.proj_context = nn.Linear(context_dim, dim) if context_dim != dim else nn.Identity()
        self.latents = nn.Parameter(torch.randn(config.num_speaker_tokens, dim))
        self.layers = nn.ModuleList([SparkTTSPerceiverLayer(config) for _ in range(config.perceiver_num_layers)])
        self.norm = SparkTTSPerceiverRMSNorm(dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        context = self.proj_context(context)
        latents = self.latents.unsqueeze(0).expand(context.shape[0], -1, -1)
        for layer in self.layers:
            latents = layer(latents, context)
        return self.norm(latents)


class SparkTTSSpeakerEncoder(nn.Module):
    """
    Time-invariant branch of BiCodec: an ECAPA-TDNN over the reference mel spectrogram, resampled to a fixed number of
    latents and quantized into the global tokens, then flattened into the conditioning vector.
    """

    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__()
        self.encoder = SparkTTSEcapaTdnn(config)
        self.perceiver_resampler = SparkTTSPerceiverResampler(config)
        self.quantizer = SparkTTSResidualFiniteScalarQuantizer(config)
        self.project = nn.Linear(config.speaker_latent_dim * config.num_speaker_tokens, config.hidden_size)

    def _resample(self, input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        speaker_embedding, features = self.encoder(input_features)
        latents = self.perceiver_resampler(features.transpose(1, 2)).transpose(1, 2)
        return speaker_embedding, latents

    def encode(self, input_features: torch.Tensor) -> torch.Tensor:
        _, latents = self._resample(input_features)
        return self.quantizer(latents)[1]

    def from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.from_codes(codes).transpose(1, 2)
        return self.project(quantized.reshape(quantized.shape[0], -1))

    def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        speaker_embedding, latents = self._resample(input_features)
        quantized, codes = self.quantizer(latents)
        conditioning_embedding = self.project(quantized.reshape(quantized.shape[0], -1))
        return speaker_embedding, conditioning_embedding, codes


@auto_docstring
class SparkTTSBiCodecPreTrainedModel(PreTrainedAudioTokenizerBase):
    config: SparkTTSBiCodecConfig
    config_class = SparkTTSBiCodecConfig
    base_model_prefix = "bicodec"
    main_input_name = "input_values"
    input_modalities = ("audio",)
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, SparkTTSPerceiverResampler):
            init.normal_(module.latents, mean=0.0, std=std)
        elif isinstance(module, SparkTTSPerceiverRMSNorm):
            init.ones_(module.gamma)
        elif isinstance(module, SparkTTSConvNeXtBlock) and module.gamma is not None:
            init.constant_(module.gamma, module.gamma_init_value)
        elif isinstance(module, Snake1d):
            init.ones_(module.alpha)
        elif isinstance(module, SparkTTSFiniteScalarQuantizer):
            # `levels`/`basis`/`codebook` are non-persistent, so a checkpoint never restores them.
            device = module.levels.device
            init.copy_(module.levels, module._compute_levels(device=device))
            init.copy_(module.basis, module._compute_basis(device=device))
            init.copy_(module.codebook, module._compute_codebook(device=device))
        elif isinstance(module, SparkTTSResidualFiniteScalarQuantizer):
            init.copy_(module.scales, module._compute_scales(device=module.scales.device))


@auto_docstring(
    custom_intro="""
    BiCodec, the audio tokenizer of Spark-TTS. It encodes a waveform into a time-varying semantic token stream and a
    time-invariant global speaker token stream, and reconstructs a waveform from that pair.
    """
)
class SparkTTSBiCodecModel(SparkTTSBiCodecPreTrainedModel):
    def __init__(self, config: SparkTTSBiCodecConfig):
        super().__init__(config)
        self.semantic_model = AutoModel.from_config(config.semantic_model_config)
        self.semantic_encoder = SparkTTSSemanticEncoder(config)
        self.quantizer = SparkTTSFactorizedVectorQuantizer(config)
        self.speaker_encoder = SparkTTSSpeakerEncoder(config)
        self.prenet = SparkTTSSemanticDecoder(
            config,
            num_layers=config.prenet_num_layers,
            sample_ratios=config.prenet_sample_ratios,
            condition_dim=config.hidden_size,
        )
        self.postnet = SparkTTSSemanticDecoder(
            config,
            num_layers=config.postnet_num_layers,
            sample_ratios=config.postnet_sample_ratios,
        )
        self.wave_generator = SparkTTSWaveGenerator(config)

        self.post_init()

    def extract_semantic_features(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Average the hidden states of the self-supervised model at the layers named by
        `config.semantic_hidden_layers`.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Waveform normalized by [`SparkTTSFeatureExtractor`].
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask marking the valid samples of `input_values`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_frames, semantic_hidden_size)`: The averaged features.
        """
        outputs = self.semantic_model(input_values, attention_mask=attention_mask, output_hidden_states=True)
        layers = self.config.semantic_hidden_layers
        return sum(outputs.hidden_states[index] for index in layers) / len(layers)

    @auto_docstring
    @can_return_tuple
    def encode(
        self,
        input_values: torch.Tensor,
        reference_input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SparkTTSBiCodecEncoderOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Waveform to tokenize, normalized by [`SparkTTSFeatureExtractor`].
        reference_input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_frames, num_mel_bins)`):
            Mel spectrogram of the reference clip the global tokens describe.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask marking the valid samples of `input_values`.
        """
        features = self.extract_semantic_features(input_values, attention_mask=attention_mask)
        audio_codes = self.quantizer.encode(self.semantic_encoder(features.transpose(1, 2)))
        global_codes = self.speaker_encoder.encode(reference_input_features)
        return SparkTTSBiCodecEncoderOutput(audio_codes=audio_codes, global_codes=global_codes)

    @auto_docstring
    @can_return_tuple
    def decode(self, audio_codes: torch.Tensor, global_codes: torch.Tensor) -> SparkTTSBiCodecDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames)`):
            Semantic token indices, as produced by [`~SparkTTSBiCodecModel.encode`] or sampled by
            [`SparkTTSForConditionalGeneration`].
        global_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, num_speaker_tokens)`):
            Global speaker token indices.
        """
        quantized = self.quantizer.from_codes(audio_codes)
        conditioning_embedding = self.speaker_encoder.from_codes(global_codes)
        hidden_states = self.prenet(quantized, conditioning_embedding)
        hidden_states = hidden_states + conditioning_embedding.unsqueeze(-1)
        return SparkTTSBiCodecDecoderOutput(audio_values=self.wave_generator(hidden_states))

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_values: torch.Tensor,
        reference_input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SparkTTSBiCodecOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Waveform to reconstruct, normalized by [`SparkTTSFeatureExtractor`].
        reference_input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_frames, num_mel_bins)`):
            Mel spectrogram of the reference clip the global tokens describe.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask marking the valid samples of `input_values`.

        Returns:
            [`SparkTTSBiCodecOutput`]: The reconstruction together with the codes, the postnet feature prediction and
            the quantizer statistics a codec training loop needs.

        Example:

        ```python
        >>> from voicestudio.models.spark_tts import SparkTTSBiCodecModel, SparkTTSFeatureExtractor

        >>> model_id = "SparkAudio/Spark-TTS-0.5B"
        >>> feature_extractor = SparkTTSFeatureExtractor.from_pretrained(model_id)
        >>> model = SparkTTSBiCodecModel.from_pretrained(model_id, subfolder="audio_tokenizer")

        >>> inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        features = self.extract_semantic_features(input_values, attention_mask=attention_mask)
        latents = self.semantic_encoder(features.transpose(1, 2))
        quantized, audio_codes, loss, perplexity, active_codes = self.quantizer(latents)

        speaker_embedding, conditioning_embedding, global_codes = self.speaker_encoder(reference_input_features)

        hidden_states = self.prenet(quantized, conditioning_embedding)
        predicted_features = self.postnet(hidden_states)
        hidden_states = hidden_states + conditioning_embedding.unsqueeze(-1)

        return SparkTTSBiCodecOutput(
            loss=loss,
            audio_values=self.wave_generator(hidden_states),
            audio_codes=audio_codes,
            global_codes=global_codes,
            predicted_features=predicted_features,
            speaker_embedding=speaker_embedding,
            conditioning_embedding=conditioning_embedding,
            perplexity=perplexity,
            active_codes=active_codes,
        )


@auto_docstring
class SparkTTSPreTrainedModel(Qwen2PreTrainedModel):
    config: SparkTTSConfig
    config_class = SparkTTSConfig
    base_model_prefix = "model"


@auto_docstring(
    custom_intro="""
    Spark-TTS, a Qwen2 decoder whose vocabulary is extended with BiCodec semantic and global tokens plus task and
    style control tokens, so that speech synthesis, voice cloning and attribute control are all next-token prediction
    over one flat sequence. Reference audio is turned into global tokens and generated semantic tokens are turned
    back into a waveform by [`SparkTTSBiCodecModel`], which [`SparkTTSProcessor`] holds.
    """
)
class SparkTTSForConditionalGeneration(SparkTTSPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: SparkTTSConfig):
        super().__init__(config)
        self.model = Qwen2Model(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for the next-token prediction loss over the joint text/BiCodec vocabulary. Indices should be in
            `[0, ..., config.text_config.vocab_size]`, and positions set to `-100` are ignored. Can be obtained with
            `output_labels=True` when calling [`SparkTTSProcessor`].

        Example:

        ```python
        >>> from voicestudio.models.spark_tts import SparkTTSForConditionalGeneration, SparkTTSProcessor

        >>> model_id = "SparkAudio/Spark-TTS-0.5B"
        >>> processor = SparkTTSProcessor.from_pretrained(model_id)
        >>> model = SparkTTSForConditionalGeneration.from_pretrained(model_id)

        >>> inputs = processor(text="The sun rises in the east.", gender="female", pitch="moderate", speed="moderate")
        >>> outputs = model.generate(**inputs, max_new_tokens=3000, do_sample=True, top_k=50, top_p=0.95)
        >>> audio_values = processor.decode(outputs, input_length=inputs["input_ids"].shape[-1])
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "SparkTTSBiCodecModel",
    "SparkTTSBiCodecPreTrainedModel",
    "SparkTTSForConditionalGeneration",
    "SparkTTSPreTrainedModel",
]
