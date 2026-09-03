# coding=utf-8
# Copyright 2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li, Bofan Zhou) and
# the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch CosyVoice v3 model."""

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

from transformers.conversion_mapping import WeightRenaming, register_checkpoint_conversion_mapping
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.utils import auto_docstring

from ..cosyvoice_v1.modeling_cosyvoice_v1 import (
    IGNORE_ID,
    CosyVoiceV1HiFTGenerator,
    CosyVoiceV1LabelSmoothingLoss,
    CosyVoiceV1ResBlock,
    make_pad_mask,
)
from ..cosyvoice_v2.modeling_cosyvoice_v2 import (
    CosyVoiceV2ConditionalCFM,
    CosyVoiceV2PreLookaheadLayer,
    CosyVoiceV2PreTrainedModel,
    CosyVoiceV2SpeechTokenLM,
)
from ..cosyvoice_v1.weight_conversion import CHECKPOINT_FILES, resolve_checkpoint
from ..cosyvoice_v2.weight_conversion import TEXT_MODEL_SUBDIR
from ..f5_tts.modeling_f5_tts import (
    F5TTSAdaLayerNormFinal,
    F5TTSDecoderLayer,
    F5TTSRotaryEmbedding,
    F5TTSTimestepEmbedding,
)
from .configuration_cosyvoice_v3 import CosyVoiceV3Config
from .generation_cosyvoice_v3 import CosyVoiceV3GenerationMixin
from .weight_conversion import build_config


# v3 keeps `Qwen2Encoder` and the pre-parametrization spelling of weight norm, and drops the flow
# matching encoder whose block names v1 and v2 have to rename.
register_checkpoint_conversion_mapping(
    "CosyVoiceV3ForConditionalGeneration",
    [
        WeightRenaming(source_patterns=r"^llm\.llm\.model\.model\.", target_patterns=r"llm\.model\."),
        WeightRenaming(source_patterns=r"\.weight_g$", target_patterns=r"\.parametrizations\.weight\.original0"),
        WeightRenaming(source_patterns=r"\.weight_v$", target_patterns=r"\.parametrizations\.weight\.original1"),
    ],
    overwrite=True,
)


def build_chunk_mask(padding_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Builds the square attention mask of a chunked encoder.

    Args:
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            `True` on the positions that carry content.
        chunk_size (`int`):
            Static chunk size. 0 lets every position see every other one.

    Returns:
        `torch.Tensor` of shape `(batch_size, sequence_length, sequence_length)`: `True` where a
        query may attend to a key.
    """
    length = padding_mask.size(1)
    if chunk_size > 0:
        positions = torch.arange(length, device=padding_mask.device)
        block_end = (torch.div(positions, chunk_size, rounding_mode="trunc") + 1) * chunk_size
        visible = positions.unsqueeze(0) < block_end.unsqueeze(1)
        return padding_mask.unsqueeze(1) & visible.unsqueeze(0)
    return padding_mask.unsqueeze(1).expand(-1, length, -1)


class CosyVoiceV3TimestepEmbedding(F5TTSTimestepEmbedding):
    """
    Conditioning embedding of the flow matching time step, a sinusoidal embedding followed by a two
    layer projection.

    Args:
        dim (`int`):
            Dimensionality of the produced embedding.
        freq_embed_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the intermediate sinusoidal embedding.
    """


class CosyVoiceV3DecoderLayer(F5TTSDecoderLayer):
    """
    One layer of the flow matching estimator, an adaptive layer norm modulated self attention block
    followed by an adaptive layer norm modulated feed forward block.

    Args:
        config ([`F5TTSConfig`]):
            Configuration of the estimator.
    """


class CosyVoiceV3AdaLayerNormFinal(F5TTSAdaLayerNormFinal):
    """
    Adaptive layer norm applied before the output projection.

    Args:
        dim (`int`):
            Number of channels to normalize.
        eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the layer normalization.
    """


class CosyVoiceV3RotaryEmbedding(F5TTSRotaryEmbedding):
    """
    Rotary position embedding of the flow matching estimator.

    Args:
        config ([`F5TTSConfig`]):
            Configuration of the estimator.
    """


class CosyVoiceV3CausalConvPositionEmbedding(nn.Module):
    """
    Convolutional position embedding added to the estimator's input embedding, left padded so that a
    frame never reads a future one.

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
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0), nn.Mish())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, dim)`):
                Module input.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, dim)`: the position embedding.
        """
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.conv1(F.pad(hidden_states, (self.kernel_size - 1, 0)))
        hidden_states = self.conv2(F.pad(hidden_states, (self.kernel_size - 1, 0)))
        return hidden_states.permute(0, 2, 1)


class CosyVoiceV3InputEmbedding(nn.Module):
    """
    Projection that mixes the noised mel spectrogram, the conditioning mel spectrogram, the encoded
    speech tokens and the speaker embedding into the estimator's width.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__()
        estimator = config.estimator_config
        self.proj = nn.Linear(
            estimator.mel_dim * 2 + estimator.text_dim + config.flow_output_size, estimator.hidden_size
        )
        self.conv_pos_embed = CosyVoiceV3CausalConvPositionEmbedding(estimator.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        mu: torch.Tensor,
        speaker_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, mel_length, mel_dim)`):
                Noised mel spectrogram.
            conditioning (`torch.Tensor` of shape `(batch_size, mel_length, mel_dim)`):
                Mel spectrogram prefix used as conditioning.
            mu (`torch.Tensor` of shape `(batch_size, mel_length, mel_dim)`):
                Encoded speech tokens at the mel frame rate.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, flow_output_size)`):
                Projected speaker embedding.

        Returns:
            `torch.Tensor` of shape `(batch_size, mel_length, hidden_size)`: the input embedding.
        """
        speaker = speaker_hidden_states.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        hidden_states = self.proj(torch.cat([hidden_states, conditioning, mu, speaker], dim=-1))
        return self.conv_pos_embed(hidden_states) + hidden_states


class CosyVoiceV3ConditionalDecoder(nn.Module):
    """
    Diffusion transformer that predicts the flow matching vector field. It is the F5-TTS diffusion
    transformer at the same size, reading a speaker embedding in place of a character sequence, and
    with its attention restricted to fixed chunks while streaming.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__()
        estimator = config.estimator_config
        self.static_chunk_size = config.estimator_static_chunk_size
        self.time_embed = CosyVoiceV3TimestepEmbedding(estimator.hidden_size)
        self.input_embed = CosyVoiceV3InputEmbedding(config)
        self.rotary_embed = CosyVoiceV3RotaryEmbedding(config=estimator)
        self.transformer_blocks = nn.ModuleList(
            [CosyVoiceV3DecoderLayer(estimator) for _ in range(estimator.num_hidden_layers)]
        )
        self.norm_out = CosyVoiceV3AdaLayerNormFinal(estimator.hidden_size, eps=estimator.layer_norm_eps)
        self.proj_out = nn.Linear(estimator.hidden_size, estimator.mel_dim)

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
            hidden_states (`torch.Tensor` of shape `(batch_size, mel_dim, mel_length)`):
                Noised mel spectrogram.
            mask (`torch.Tensor` of shape `(batch_size, 1, mel_length)`):
                1 on valid frames.
            mu (`torch.Tensor` of shape `(batch_size, mel_dim, mel_length)`):
                Encoded speech tokens at the mel frame rate.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Flow matching timesteps.
            speaker_hidden_states (`torch.Tensor` of shape `(batch_size, flow_output_size)`):
                Projected speaker embedding.
            conditioning (`torch.Tensor` of shape `(batch_size, mel_dim, mel_length)`):
                Mel spectrogram prefix used as conditioning.
            streaming (`bool`, *optional*, defaults to `False`):
                Whether the attention is restricted to `estimator_static_chunk_size` chunks.

        Returns:
            `torch.Tensor` of shape `(batch_size, mel_dim, mel_length)`: predicted vector field.
        """
        hidden_states = hidden_states.transpose(1, 2)
        mu = mu.transpose(1, 2)
        conditioning = conditioning.transpose(1, 2)
        batch_size, length = hidden_states.shape[0], hidden_states.shape[1]
        if timesteps.ndim == 0:
            timesteps = timesteps.repeat(batch_size)

        time_emb = self.time_embed(timesteps)
        hidden_states = self.input_embed(hidden_states, conditioning, mu, speaker_hidden_states)

        position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_embed(hidden_states, position_ids)

        chunk_mask = build_chunk_mask(
            mask.squeeze(1).bool(), self.static_chunk_size if streaming else 0
        )
        attention_bias = torch.zeros(chunk_mask.shape, dtype=hidden_states.dtype, device=chunk_mask.device)
        attention_bias = attention_bias.masked_fill(~chunk_mask, torch.finfo(hidden_states.dtype).min)
        # The output masking upstream applies keeps the last row of the chunk mask, not the padding
        # mask, so a streaming run zeroes whatever the final chunk cannot see.
        padding_mask = chunk_mask[:, -1]

        for layer in self.transformer_blocks:
            hidden_states = layer(
                hidden_states,
                time_emb,
                position_embeddings,
                attention_mask=attention_bias.unsqueeze(1),
                padding_mask=padding_mask,
            )

        hidden_states = self.norm_out(hidden_states, time_emb)
        return self.proj_out(hidden_states).transpose(1, 2)


class CosyVoiceV3ConditionalCFM(CosyVoiceV2ConditionalCFM):
    """
    Conditional flow matching head of CosyVoice v3, which is v2's with the one dimensional UNet
    estimator replaced by a diffusion transformer.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__(config)
        self.estimator = CosyVoiceV3ConditionalDecoder(config)


class CosyVoiceV3FlowModel(nn.Module):
    """
    Speech token to mel spectrogram model of CosyVoice v3. There is no encoder: a lookahead
    convolution runs over the embedded speech tokens, the result is repeated `token_mel_ratio` times
    to reach the mel frame rate, and the diffusion transformer takes it from there.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__()
        self.config = config
        self.output_size = config.flow_output_size
        self.token_mel_ratio = config.token_mel_ratio
        self.pre_lookahead_len = config.pre_lookahead_len
        self.input_embedding = nn.Embedding(config.speech_vocab_size, config.flow_input_size)
        self.spk_embed_affine_layer = nn.Linear(config.speaker_embedding_dim, config.flow_output_size)
        self.pre_lookahead_layer = CosyVoiceV2PreLookaheadLayer(
            config.flow_input_size, config.pre_lookahead_channels, config.pre_lookahead_len
        )
        self.decoder = CosyVoiceV3ConditionalCFM(config)

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
                Whether the estimator attends within chunks only. Upstream draws it per batch with
                probability one half.

        Returns:
            `torch.Tensor`: the conditional flow matching loss.
        """
        speaker_hidden_states = self.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1))

        token_mask = (~make_pad_mask(speech_token_lengths, speech_token_ids.size(1))).float().unsqueeze(-1)
        hidden_states = self.input_embedding(torch.clamp(speech_token_ids, min=0)) * token_mask
        hidden_states = self.pre_lookahead_layer(hidden_states)
        hidden_states = hidden_states.repeat_interleave(self.token_mel_ratio, dim=1)
        mask = token_mask.repeat_interleave(self.token_mel_ratio, dim=1).squeeze(dim=-1)

        conditioning = torch.zeros_like(speech_feat)
        for index, length in enumerate(speech_feat_lengths):
            if random.random() < 0.5:
                continue
            prefix = random.randint(0, int(0.3 * length))
            conditioning[index, :prefix] = speech_feat[index, :prefix]
        conditioning = conditioning.transpose(1, 2)

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
                Whether the estimator attends within chunks only.
            finalize (`bool`, *optional*, defaults to `True`):
                Whether this is the last chunk of the utterance.

        Returns:
            `torch.Tensor` of shape `(1, flow_output_size, mel_length)`: the generated mel spectrogram.
        """
        speaker_hidden_states = self.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1))

        token_ids = torch.concat([prompt_token_ids, speech_token_ids], dim=1)
        token_lengths = prompt_token_lengths + speech_token_lengths
        token_mask = (~make_pad_mask(token_lengths, token_ids.size(1))).unsqueeze(-1).to(speaker_embedding)
        hidden_states = self.input_embedding(torch.clamp(token_ids, min=0)) * token_mask

        if finalize:
            hidden_states = self.pre_lookahead_layer(hidden_states)
        else:
            hidden_states = self.pre_lookahead_layer(
                hidden_states[:, : -self.pre_lookahead_len],
                context=hidden_states[:, -self.pre_lookahead_len :],
            )
        hidden_states = hidden_states.repeat_interleave(self.token_mel_ratio, dim=1)
        prompt_mel_length = prompt_feat.shape[1]
        mel_length = hidden_states.shape[1] - prompt_mel_length

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


class CosyVoiceV3SpeechTokenLM(CosyVoiceV2SpeechTokenLM):
    """
    Autoregressive speech token model of CosyVoice v3.

    It differs from v2's in where its three control vectors live. v2 keeps a separate two entry table
    for the start of sequence and task id vectors; v3 grows the speech token table by
    `num_speech_special_tokens` and takes them from there, so `llm_embedding` is that same table
    under another name and every sequence packing routine is inherited unchanged.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        nn.Module.__init__(self)
        self.config = config
        self.speech_vocab_size = config.speech_vocab_size
        self.head_size = config.speech_head_size
        self.sos_index = config.speech_sos_token_id
        self.task_id_index = config.speech_task_token_id
        self.eos_token_id = config.speech_eos_token_id
        self.fill_token_id = config.speech_fill_token_id
        self.stop_token_ids = [
            config.speech_vocab_size + index for index in range(config.num_speech_special_tokens)
        ]
        self.mix_ratio = config.mix_ratio

        self.model = Qwen2Model(config.text_config)
        self.speech_embedding = nn.Embedding(self.head_size, config.lm_hidden_size)
        self.llm_decoder = nn.Linear(config.lm_hidden_size, self.head_size, bias=False)

    @property
    def llm_embedding(self) -> nn.Embedding:
        r"""
        Returns:
            `nn.Embedding`: The table the start of sequence and task id vectors are read from, which
            in v3 is the speech token table itself.
        """
        return self.speech_embedding


class CosyVoiceV3CausalConv1d(nn.Conv1d):
    """
    Vocoder convolution that pads on one side only, keeping its output length equal to its input's.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Width of the kernel.
        dilation (`int`, *optional*, defaults to 1):
            Dilation of the kernel.
        causal_type (`str`, *optional*, defaults to `"left"`):
            Which side the padding goes on. `"right"` lets the convolution look ahead instead.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1,
        causal_type: str = "left",
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation)
        if causal_type not in ("left", "right"):
            raise ValueError(f"`causal_type` must be 'left' or 'right', got {causal_type}.")
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        self.causal_type = causal_type

    def forward(self, hidden_states: torch.Tensor, cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cache is None or cache.size(2) == 0:
            cache = hidden_states.new_zeros(
                hidden_states.shape[0], hidden_states.shape[1], self.causal_padding
            )
        if self.causal_type == "left":
            hidden_states = torch.cat([cache, hidden_states], dim=2)
        else:
            hidden_states = torch.cat([hidden_states, cache], dim=2)
        return super().forward(hidden_states)


class CosyVoiceV3CausalConv1dUpsample(nn.Conv1d):
    """
    Nearest neighbour upsampling followed by a left padded convolution.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Width of the kernel.
        stride (`int`):
            Upsampling factor.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.causal_padding = kernel_size - 1
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, hidden_states: torch.Tensor, cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.upsample(hidden_states)
        if cache is None or cache.size(2) == 0:
            hidden_states = F.pad(hidden_states, (self.causal_padding, 0), value=0.0)
        else:
            hidden_states = torch.cat([cache, hidden_states], dim=2)
        return super().forward(hidden_states)


class CosyVoiceV3CausalConv1dDownsample(nn.Conv1d):
    """
    Strided convolution that pads on the left only.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Width of the kernel.
        stride (`int`):
            Downsampling factor.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.causal_padding = stride - 1

    def forward(self, hidden_states: torch.Tensor, cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cache is None or cache.size(2) == 0:
            hidden_states = F.pad(hidden_states, (self.causal_padding, 0), value=0.0)
        else:
            hidden_states = torch.cat([cache, hidden_states], dim=2)
        return super().forward(hidden_states)


class CosyVoiceV3ResBlock(CosyVoiceV1ResBlock):
    """
    Residual block of the vocoder, built from left padded convolutions.

    Args:
        channels (`int`):
            Number of channels.
        kernel_size (`int`):
            Width of the kernels.
        dilations (`list[int]`):
            Dilation of each residual layer.
    """

    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__(channels, kernel_size, dilations)
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    CosyVoiceV3CausalConv1d(channels, channels, kernel_size, dilation=dilation)
                )
                for dilation in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(CosyVoiceV3CausalConv1d(channels, channels, kernel_size, dilation=1))
                for _ in dilations
            ]
        )


class CosyVoiceV3SineGen(nn.Module):
    """
    Harmonic excitation generator of the causal vocoder.

    It is the interpolating generator v2 uses, with two changes upstream makes when the vocoder is
    causal: the phase is interpolated back up with nearest neighbour rather than linearly, and the
    per harmonic phase offsets and the additive noise are drawn once and reused, so that consecutive
    chunks of a stream stay consistent with one another.

    Upstream draws those two tensors from whatever global random state is current when the module is
    built, which makes its vocoder irreproducible across processes. They are drawn here from a
    generator seeded with `source_noise_seed`, which is a deliberate deviation.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__()
        self.sampling_rate = config.sample_rate
        self.upsample_scale = int(
            np.prod(config.vocoder_upsample_rates) * config.vocoder_istft_hop_length
        )
        self.num_harmonics = config.vocoder_num_harmonics
        self.amplitude = config.vocoder_source_amplitude
        self.noise_std = config.vocoder_source_noise_std
        self.voiced_threshold = config.vocoder_voiced_threshold
        self.noise_length = config.source_noise_length
        self.noise_seed = config.source_noise_seed
        self._phase: Optional[torch.Tensor] = None
        self._noise: Optional[torch.Tensor] = None

    def _draw(self):
        if self._phase is not None:
            return
        # Built outside inference mode so the cached tensors stay usable by autograd whichever call
        # creates them, and from a private generator so seeding does not disturb the caller.
        with torch.inference_mode(False):
            generator = torch.Generator().manual_seed(self.noise_seed)
            phase = torch.rand(1, self.num_harmonics + 1, generator=generator)
            phase[:, 0] = 0
            self._phase = phase
            self._noise = torch.rand(
                1, self.noise_length, self.num_harmonics + 1, generator=generator
            )

    def _sine(self, f0_harmonics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0_harmonics (`torch.Tensor` of shape `(batch_size, num_samples, num_harmonics + 1)`):
                f0 and its overtones.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_samples, num_harmonics + 1)`: the sine waves.
        """
        self._draw()
        rad_values = (f0_harmonics / self.sampling_rate) % 1
        rad_values[:, 0, :] = rad_values[:, 0, :] + self._phase.to(rad_values)

        rad_values = F.interpolate(
            rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear"
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=float(self.upsample_scale),
            mode="nearest",
        ).transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f0 (`torch.Tensor` of shape `(batch_size, num_samples, 1)`):
                Upsampled f0 contour, zero on unvoiced steps.

        Returns:
            `tuple(torch.Tensor)`: the sine waves and the voiced mask.
        """
        harmonics = torch.arange(1, self.num_harmonics + 2, device=f0.device, dtype=f0.dtype)
        sine_waves = self._sine(f0 * harmonics) * self.amplitude

        voiced = (f0 > self.voiced_threshold).to(f0.dtype)
        noise_amplitude = voiced * self.noise_std + (1 - voiced) * self.amplitude / 3
        self._draw()
        noise = noise_amplitude * self._noise[:, : sine_waves.shape[1]].to(sine_waves)
        return sine_waves * voiced + noise, voiced


class CosyVoiceV3SourceModule(nn.Module):
    """
    Merges the harmonics of [`CosyVoiceV3SineGen`] into a single excitation signal.

    Upstream's counterpart also returns a second noise signal drawn from its own fixed tensor of
    about 29 megabytes. The vocoder discards it, so it is not built here.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__()
        self.amplitude = config.vocoder_source_amplitude
        self.l_sin_gen = CosyVoiceV3SineGen(config)
        self.l_linear = nn.Linear(config.vocoder_num_harmonics + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sine_waves, _ = self.l_sin_gen(f0)
        return self.l_tanh(self.l_linear(sine_waves))


class CosyVoiceV3F0Predictor(nn.Module):
    """
    Convolutional network predicting an f0 contour from a mel spectrogram, built from one lookahead
    convolution followed by four left padded ones.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__()
        channels = config.f0_predictor_hidden_size
        layers = [
            weight_norm(
                CosyVoiceV3CausalConv1d(config.vocoder_in_channels, channels, 4, causal_type="right")
            ),
            nn.ELU(),
        ]
        for _ in range(4):
            layers += [weight_norm(CosyVoiceV3CausalConv1d(channels, channels, 3)), nn.ELU()]
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels, 1)

    def forward(self, mel: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        """
        Args:
            mel (`torch.Tensor` of shape `(batch_size, vocoder_in_channels, mel_length)`):
                Mel spectrogram.
            finalize (`bool`, *optional*, defaults to `True`):
                Whether this is the last chunk. When it is not, the frames the lookahead convolution
                would read past the end are handed to it as its cache instead.

        Returns:
            `torch.Tensor` of shape `(batch_size, mel_length)`: the f0 contour.
        """
        lookahead = self.condnet[0].causal_padding
        if finalize:
            hidden_states = self.condnet[0](mel)
        else:
            hidden_states = self.condnet[0](mel[:, :, :-lookahead], mel[:, :, -lookahead:])
        for index in range(1, len(self.condnet)):
            hidden_states = self.condnet[index](hidden_states)
        return torch.abs(self.classifier(hidden_states.transpose(1, 2)).squeeze(-1))


class CosyVoiceV3HiFTGenerator(CosyVoiceV1HiFTGenerator):
    """
    Causal HiFTNet vocoder of CosyVoice v3. Every convolution pads on one side only, so a chunk can
    be rendered without the frames that follow it.

    Args:
        config ([`CosyVoiceV3Config`]):
            Model configuration.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__(config)
        self.conv_pre_look_right = config.vocoder_conv_pre_look_right
        self.upsample_rates = list(config.vocoder_upsample_rates)
        base_channels = config.vocoder_base_channels

        self.f0_predictor = CosyVoiceV3F0Predictor(config)
        self.m_source = CosyVoiceV3SourceModule(config)
        self.conv_pre = weight_norm(
            CosyVoiceV3CausalConv1d(
                config.vocoder_in_channels, base_channels, config.vocoder_conv_pre_look_right + 1,
                causal_type="right",
            )
        )
        self.ups = nn.ModuleList(
            [
                weight_norm(
                    CosyVoiceV3CausalConv1dUpsample(
                        base_channels // (2**index), base_channels // (2 ** (index + 1)),
                        kernel_size, rate,
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
                self.source_downs.append(CosyVoiceV3CausalConv1d(self.n_fft + 2, channels, 1))
            else:
                self.source_downs.append(
                    CosyVoiceV3CausalConv1dDownsample(self.n_fft + 2, channels, int(rate) * 2, int(rate))
                )
            self.source_resblocks.append(CosyVoiceV3ResBlock(channels, kernel_size, dilations))

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = base_channels // (2 ** (index + 1))
            for kernel_size, dilations in zip(
                config.vocoder_resblock_kernel_sizes, config.vocoder_resblock_dilation_sizes
            ):
                self.resblocks.append(CosyVoiceV3ResBlock(channels, kernel_size, dilations))

        self.conv_post = weight_norm(CosyVoiceV3CausalConv1d(channels, self.n_fft + 2, 7))

    def decode(self, mel: torch.Tensor, source: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        """
        Args:
            mel (`torch.Tensor` of shape `(batch_size, vocoder_in_channels, mel_length)`):
                Mel spectrogram.
            source (`torch.Tensor` of shape `(batch_size, 1, num_samples)`):
                Harmonic excitation.
            finalize (`bool`, *optional*, defaults to `True`):
                Whether this is the last chunk of the utterance.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_samples)`: the generated waveform.
        """
        source_stft = self._stft(source.squeeze(1))
        if finalize:
            hidden_states = self.conv_pre(mel)
        else:
            lookahead = self.conv_pre_look_right
            hidden_states = self.conv_pre(mel[:, :, :-lookahead], mel[:, :, -lookahead:])
            source_stft = source_stft[:, :, : -int(np.prod(self.upsample_rates) * lookahead)]

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
        if not finalize:
            waveform = waveform[:, : -int(np.prod(self.upsample_rates) * self.hop_length)]
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

    @torch.inference_mode()
    def inference(self, mel: torch.Tensor, finalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel (`torch.Tensor` of shape `(batch_size, vocoder_in_channels, mel_length)`):
                Mel spectrogram.
            finalize (`bool`, *optional*, defaults to `True`):
                Whether this is the last chunk of the utterance.

        Returns:
            `tuple(torch.Tensor)`: the generated waveform and the excitation used to produce it.
        """
        # The f0 contour is taken in double precision, which upstream notes is what keeps a chunked
        # run stable. Upstream leaves the module cast; the original dtype is restored here.
        dtype = next(self.f0_predictor.parameters()).dtype
        self.f0_predictor.to(torch.float64)
        f0 = self.f0_predictor(mel.to(torch.float64), finalize=finalize).to(mel)
        self.f0_predictor.to(dtype)

        source = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        source = self.m_source(source).transpose(1, 2)
        if finalize:
            waveform = self.decode(mel, source, finalize=True)
        else:
            trim = self.f0_predictor.condnet[0].causal_padding
            waveform = self.decode(mel[:, :, :-trim], source, finalize=False)
        return waveform, source


@auto_docstring(
    custom_intro="""
    Output of [`CosyVoiceV3ForConditionalGeneration`].
    """
)
@dataclass
class CosyVoiceV3Output(ModelOutput):
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
class CosyVoiceV3PreTrainedModel(CosyVoiceV2PreTrainedModel):
    config: CosyVoiceV3Config
    base_model_prefix = "cosyvoice_v3"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = [
        r"llm\.model\.rotary_emb\.inv_freq",
        r"llm\.model\.lm_head\.",
        r"flow\.decoder\.estimator\.rotary_embed\.",
    ]

    @classmethod
    def _released_checkpoint(cls, source, **kwargs) -> "tuple[CosyVoiceV3Config, Path] | None":
        r"""
        Locates a released CosyVoice v3 directory, whose Qwen2 sub directory is fetched alongside the
        three network files because the configuration is built from it.

        Args:
            source (`str` or `os.PathLike`, *optional*):
                Repository id or local directory.
            kwargs (`dict`, *optional*):
                Fields of `weight_conversion.DOWNLOAD_KWARGS` selecting a revision and a cache.

        Returns:
            `tuple[CosyVoiceV3Config, Path]` or `None`: The configuration and the local directory
            holding the released files, or `None` when `source` holds no released checkpoint.
        """
        directory = resolve_checkpoint(
            source, tuple(CHECKPOINT_FILES.values()), (f"{TEXT_MODEL_SUBDIR}/*",), **kwargs
        )
        if directory is None:
            return None
        return build_config(directory), directory

    def _init_weights(self, module):
        # The rotary embedding computes its frequencies in its constructor and registers them as two
        # non persistent buffers, which meta device initialisation materialises as uninitialised
        # memory rather than by rerunning that computation. They are rebuilt here because the
        # estimator is a plain module, so nothing else reaches them.
        if isinstance(module, CosyVoiceV3RotaryEmbedding):
            rope_init_fn = module.compute_default_rope_parameters
            if module.rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type]
            inv_freq, module.attention_scaling = rope_init_fn(module.config)
            with torch.no_grad():
                module.inv_freq.copy_(inv_freq)
                module.original_inv_freq.copy_(inv_freq)
        else:
            super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    CosyVoice v3, made of a Qwen2 based text to speech token language model, a diffusion transformer
    flow matching model turning speech tokens into a mel spectrogram, and a causal HiFTNet vocoder.

    The three networks are trained one at a time upstream, so `forward` optimizes the language model
    objective only. [`CosyVoiceV3FlowModel.forward`] returns the flow matching objective and
    [`CosyVoiceV3HiFTGenerator.forward`] returns the waveform and the f0 contour the vocoder
    objective is computed from.
    """
)
class CosyVoiceV3ForConditionalGeneration(CosyVoiceV3GenerationMixin, CosyVoiceV3PreTrainedModel):
    def __init__(self, config: CosyVoiceV3Config):
        super().__init__(config)
        self.llm = CosyVoiceV3SpeechTokenLM(config)
        self.flow = CosyVoiceV3FlowModel(config)
        self.hift = CosyVoiceV3HiFTGenerator(config)
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
    ) -> CosyVoiceV3Output:
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
        return CosyVoiceV3Output(loss=loss, logits=logits, accuracy=accuracy)


__all__ = [
    "CosyVoiceV3AdaLayerNormFinal",
    "CosyVoiceV3CausalConv1d",
    "CosyVoiceV3CausalConv1dDownsample",
    "CosyVoiceV3CausalConv1dUpsample",
    "CosyVoiceV3CausalConvPositionEmbedding",
    "CosyVoiceV3ConditionalCFM",
    "CosyVoiceV3ConditionalDecoder",
    "CosyVoiceV3DecoderLayer",
    "CosyVoiceV3F0Predictor",
    "CosyVoiceV3FlowModel",
    "CosyVoiceV3ForConditionalGeneration",
    "CosyVoiceV3HiFTGenerator",
    "CosyVoiceV3InputEmbedding",
    "CosyVoiceV3Output",
    "CosyVoiceV3PreTrainedModel",
    "CosyVoiceV3ResBlock",
    "CosyVoiceV3RotaryEmbedding",
    "CosyVoiceV3SineGen",
    "CosyVoiceV3SourceModule",
    "CosyVoiceV3SpeechTokenLM",
    "CosyVoiceV3TimestepEmbedding",
    "build_chunk_mask",
]
