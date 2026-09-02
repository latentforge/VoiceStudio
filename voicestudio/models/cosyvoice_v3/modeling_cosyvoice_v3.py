# coding=utf-8
# Copyright 2025 Alibaba Inc and The HuggingFace Inc. team. All rights reserved.
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

import itertools
import math
import operator

import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from ..cosyvoice_v1.modeling_cosyvoice_v1 import (
    CosyVoiceV1ConditionalCFM,
    CosyVoiceV1PreTrainedModel,
    CosyVoiceV1SineGen,
    CosyVoiceV1Snake,
    _make_pad_mask,
)
from ..cosyvoice_v2.modeling_cosyvoice_v2 import CosyVoiceV2LLM, CosyVoiceV2LLMOutput
from .configuration_cosyvoice_v3 import (
    CosyVoiceV3Config,
    CosyVoiceV3FlowConfig,
    CosyVoiceV3HiftConfig,
    CosyVoiceV3LLMConfig,
)


class CosyVoiceV3LLM(CosyVoiceV2LLM):
    r"""
    The CosyVoice v3 speech-token language model. Identical Qwen2-backbone architecture to [`CosyVoiceV2LLM`],
    except the start/task/fill/end-of-speech ids live inside the (further extended) speech-token embedding table
    instead of a separate two-entry embedding, matching the original repository's `CosyVoice3LM(Qwen2LM)`.
    """

    config_class = CosyVoiceV3LLMConfig
    _keys_to_ignore_on_load_unexpected = [r"llm\.model\.lm_head\.weight"]

    def __init__(self, config: CosyVoiceV3LLMConfig):
        CosyVoiceV1PreTrainedModel.__init__(self, config)
        vocab = config.speech_token_size + 200
        self.speech_token_size = config.speech_token_size
        self.sos_eos = config.speech_token_size
        self.eos_token_id = config.speech_token_size + 1
        self.task_id = config.speech_token_size + 2
        self.fill_token_id = config.speech_token_size + 3

        self.llm = Qwen2Model(config)
        self.llm_decoder = nn.Linear(config.hidden_size, vocab, bias=False)
        self.speech_embedding = nn.Embedding(vocab, config.hidden_size)
        self.post_init()

    def forward(
        self,
        text_token: torch.LongTensor,
        speech_token: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ) -> CosyVoiceV2LLMOutput:
        """
        Args:
            text_token (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Text token ids, tokenized by the Qwen2 backbone's tokenizer.
            speech_token (`torch.LongTensor` of shape `(batch_size, speech_sequence_length)`):
                Discrete speech token ids.
            labels (`torch.LongTensor` of shape `(batch_size, speech_sequence_length + 1)`, *optional*):
                Target speech token ids (including the trailing end-of-speech token). When given, a cross-entropy
                loss is computed.

        Returns:
            [`CosyVoiceV2LLMOutput`]
        """
        text_emb = self.llm.embed_tokens(text_token)
        speech_emb = self.speech_embedding(speech_token)
        sos_emb = self.speech_embedding.weight[self.sos_eos].reshape(1, 1, -1).expand(text_emb.size(0), -1, -1)
        task_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1).expand(text_emb.size(0), -1, -1)

        lm_input = torch.cat([sos_emb, text_emb, task_emb, speech_emb], dim=1)
        hidden_states = self.llm(inputs_embeds=lm_input).last_hidden_state
        logits = self.llm_decoder(hidden_states)

        loss = None
        if labels is not None:
            prefix_len = lm_input.size(1) - speech_emb.size(1)
            target = torch.full((logits.size(0), lm_input.size(1)), -100, dtype=torch.long, device=logits.device)
            target[:, prefix_len - 1 :] = labels
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
        return CosyVoiceV2LLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        text_token: torch.LongTensor,
        prompt_speech_token: torch.LongTensor | None = None,
        max_new_tokens: int = 2000,
        min_new_tokens: int = 0,
        top_k: int = 25,
    ) -> torch.LongTensor:
        """See [`~CosyVoiceV2LLM.generate`]."""
        device = text_token.device
        text_emb = self.llm.embed_tokens(text_token)
        sos_emb = self.speech_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)
        prompt_emb = (
            self.speech_embedding(prompt_speech_token)
            if prompt_speech_token is not None and prompt_speech_token.size(1) > 0
            else torch.zeros(1, 0, text_emb.size(-1), device=device, dtype=text_emb.dtype)
        )
        lm_input = torch.cat([sos_emb, text_emb, task_emb, prompt_emb], dim=1)

        past_key_values = None
        generated = []
        for step in range(max_new_tokens):
            output = self.llm(inputs_embeds=lm_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            logits = self.llm_decoder(output.last_hidden_state[:, -1])
            logits[:, self.speech_token_size + 1 :] = -float("inf")
            if step < min_new_tokens:
                logits[:, self.eos_token_id] = -float("inf")
            top_logits, top_indices = logits.topk(top_k, dim=-1)
            probs = torch.softmax(top_logits, dim=-1)
            next_token = top_indices.gather(-1, torch.multinomial(probs, 1))
            token_id = next_token.item()
            if token_id == self.eos_token_id:
                break
            generated.append(token_id)
            lm_input = self.speech_embedding(next_token)
        return torch.tensor(generated, device=device, dtype=torch.long).unsqueeze(0)


class CosyVoiceV3DiTRotaryEmbedding(nn.Module):
    """Interleaved-pair rotary embedding matching `x_transformers.RotaryEmbedding` (no xpos scaling)."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1).reshape(seq_len, -1)
        return freqs.unsqueeze(0)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def _apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    rot_dim = freqs.shape[-1]
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    cos, sin = freqs.cos(), freqs.sin()
    t = t * cos + _rotate_half_interleaved(t) * sin
    return torch.cat((t, t_unrotated), dim=-1)


class CosyVoiceV3DiTAttention(nn.Module):
    """Non-fused Q/K/V self-attention with rotary position embeddings, matching `DiT.modules.Attention` +
    `AttnProcessor`."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(hidden_size, inner_dim)
        self.to_k = nn.Linear(hidden_size, inner_dim)
        self.to_v = nn.Linear(hidden_size, inner_dim)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, hidden_size), nn.Dropout(0.0)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None, rope: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = _apply_rotary_pos_emb(self.to_q(hidden_states), rope)
        key = _apply_rotary_pos_emb(self.to_k(hidden_states), rope)
        value = self.to_v(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        hidden_states = self.to_out[1](self.to_out[0](hidden_states))
        return hidden_states


class CosyVoiceV3DiTFeedForward(nn.Module):
    """Plain (non-gated) tanh-approximate GELU MLP, matching `DiT.modules.FeedForward`."""

    def __init__(self, hidden_size: int, ff_mult: int):
        super().__init__()
        inner_dim = hidden_size * ff_mult
        project_in = nn.Sequential(nn.Linear(hidden_size, inner_dim), nn.GELU(approximate="tanh"))
        self.ff = nn.Sequential(project_in, nn.Dropout(0.0), nn.Linear(inner_dim, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden_states)


class CosyVoiceV3DiTAdaLayerNormZero(nn.Module):
    """Six-way AdaLN-Zero modulation (attention shift/scale/gate + feed-forward shift/scale/gate), matching
    `DiT.modules.AdaLayerNormZero`."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 6)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, time_emb: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(self.silu(time_emb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class CosyVoiceV3DiTAdaLayerNormZeroFinal(nn.Module):
    """Two-way AdaLN-Zero modulation for the final projection, matching `DiT.modules.AdaLayerNormZero_Final`."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(time_emb)).chunk(2, dim=1)
        return self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]


class CosyVoiceV3DiTBlock(nn.Module):
    """AdaLN-Zero-modulated pre-norm self-attention + plain-GELU feed-forward block, matching `DiT.modules.DiTBlock`."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, ff_mult: int):
        super().__init__()
        self.attn_norm = CosyVoiceV3DiTAdaLayerNormZero(hidden_size)
        self.attn = CosyVoiceV3DiTAttention(hidden_size, num_heads, head_dim)
        self.ff_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = CosyVoiceV3DiTFeedForward(hidden_size, ff_mult)

    def forward(
        self, hidden_states: torch.Tensor, time_emb: torch.Tensor, attention_mask: torch.Tensor | None, rope: torch.Tensor
    ) -> torch.Tensor:
        normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, time_emb)
        attn_out = self.attn(normed, attention_mask, rope)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out
        normed = self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return hidden_states + gate_mlp.unsqueeze(1) * self.ff(normed)


class CosyVoiceV3DiTSinusPositionEmbedding(nn.Module):
    """Matches `DiT.modules.SinusPositionEmbedding` (distinct scaling from [`CosyVoiceV1SinusoidalPosEmb`])."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class CosyVoiceV3DiTTimestepEmbedding(nn.Module):
    """Matches `DiT.modules.TimestepEmbedding`."""

    def __init__(self, hidden_size: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = CosyVoiceV3DiTSinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.time_embed(timestep).to(timestep.dtype))


class CosyVoiceV3DiTCausalConvPositionEmbedding(nn.Module):
    """Matches `DiT.modules.CausalConvPositionEmbedding`."""

    def __init__(self, hidden_size: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=groups, padding=0), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=groups, padding=0), nn.Mish())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.pad(hidden_states, (self.kernel_size - 1, 0))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.pad(hidden_states, (self.kernel_size - 1, 0))
        hidden_states = self.conv2(hidden_states)
        return hidden_states.transpose(1, 2)


class CosyVoiceV3DiTInputEmbedding(nn.Module):
    """Projects the concatenated noised input / mu / speaker embedding, then adds a causal conv position
    embedding, matching `DiT.dit.InputEmbedding`."""

    def __init__(self, mel_dim: int, mu_dim: int, hidden_size: int, spk_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + mu_dim + spk_dim, hidden_size)
        self.conv_pos_embed = CosyVoiceV3DiTCausalConvPositionEmbedding(hidden_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mu: torch.Tensor, spks: torch.Tensor) -> torch.Tensor:
        spks = spks.unsqueeze(1).expand(-1, x.size(1), -1)
        hidden_states = self.proj(torch.cat([x, cond, mu, spks], dim=-1))
        return self.conv_pos_embed(hidden_states) + hidden_states


class CosyVoiceV3DiT(nn.Module):
    r"""
    Diffusion-transformer estimator of the conditional-flow-matching velocity field, replacing the CosyVoice
    v1/v2 U-Net estimator, matching the original repository's `cosyvoice.flow.DiT.dit.DiT`. Separate Q/K/V
    projections with interleaved rotary position embeddings (no fused qkv), AdaLN-Zero conditioning on the
    flow-matching timestep applied before both the attention and feed-forward sub-blocks, and a plain (non-gated)
    tanh-GELU feed-forward. Shares the `forward(x, mask, mu, t, spks, cond)` signature of
    [`CosyVoiceV1ConditionalDecoder`], so it plugs into the same [`CosyVoiceV1ConditionalCFM`] wrapper.
    """

    def __init__(self, config: CosyVoiceV3FlowConfig):
        super().__init__()
        hidden_size = config.dit_hidden_size
        mel_dim = config.output_size
        self.time_embed = CosyVoiceV3DiTTimestepEmbedding(hidden_size)
        self.input_embed = CosyVoiceV3DiTInputEmbedding(mel_dim, mel_dim, hidden_size, mel_dim)
        self.rotary_embed = CosyVoiceV3DiTRotaryEmbedding(config.dit_head_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                CosyVoiceV3DiTBlock(hidden_size, config.dit_num_attention_heads, config.dit_head_dim, config.dit_ff_mult)
                for _ in range(config.dit_num_hidden_layers)
            ]
        )
        self.norm_out = CosyVoiceV3DiTAdaLayerNormZeroFinal(hidden_size)
        self.proj_out = nn.Linear(hidden_size, mel_dim)

    def forward(self, x, mask, mu, t, spks, cond, streaming: bool = False) -> torch.Tensor:
        """See [`CosyVoiceV1ConditionalDecoder.forward`]."""
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        seq_len = x.size(1)
        if t.ndim == 0:
            t = t.repeat(x.size(0))

        time_emb = self.time_embed(t)
        hidden_states = self.input_embed(x, cond, mu, spks)
        rope = self.rotary_embed(seq_len)

        padding_mask = mask.squeeze(1).bool()
        attention_mask = padding_mask[:, None, None, :].expand(-1, -1, seq_len, -1)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, time_emb, attention_mask, rope)
        hidden_states = self.norm_out(hidden_states, time_emb)
        return self.proj_out(hidden_states).transpose(1, 2) * mask


class CosyVoiceV3PreLookaheadLayer(nn.Module):
    r"""
    Convolves each mel-encoder frame with a small window of future frames, widening to `channels` then back down
    to `in_channels`, matching the original repository's `cosyvoice.transformer.upsample_encoder.PreLookaheadLayer`.
    Distinct from [`CosyVoiceV2PreLookaheadLayer`], which keeps a single width throughout instead of a
    widen/narrow bottleneck.
    """

    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(in_channels, channels, pre_lookahead_len + 1)
        self.conv2 = nn.Conv1d(channels, in_channels, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.pad(hidden_states, (0, self.pre_lookahead_len))
        hidden_states = F.leaky_relu(self.conv1(hidden_states))
        hidden_states = F.pad(hidden_states, (2, 0))
        hidden_states = self.conv2(hidden_states)
        return hidden_states.transpose(1, 2) + residual


class CosyVoiceV3FlowMatchingModel(CosyVoiceV1PreTrainedModel):
    r"""
    The CosyVoice v3 conditional-flow-matching decoder, matching the original repository's
    `CausalMaskedDiffWithDiT`. Unlike [`CosyVoiceV2FlowMatchingModel`], there is no Conformer text encoder and no
    length regulator: the pre-lookahead-convolved speech-token embedding sequence is upsampled to the mel frame
    rate by plain `repeat_interleave`, then fed directly to [`CosyVoiceV3DiT`] as `mu`.
    """

    config_class = CosyVoiceV3FlowConfig
    # The checkpoint's `x_transformers.RotaryEmbedding` saves its deterministic `inv_freq` buffer in the
    # state dict even though it's recomputed identically from `dit_head_dim` at init (non-persistent here too).
    _keys_to_ignore_on_load_unexpected = [r"decoder\.estimator\.rotary_embed\.inv_freq"]

    def __init__(self, config: CosyVoiceV3FlowConfig):
        super().__init__(config)
        self.token_mel_ratio = config.token_mel_ratio
        self.input_embedding = nn.Embedding(config.vocab_size, config.input_size)
        self.spk_embed_affine_layer = nn.Linear(config.spk_embed_dim, config.output_size)
        self.pre_lookahead_layer = CosyVoiceV3PreLookaheadLayer(
            config.input_size, config.pre_lookahead_channels, config.pre_lookahead_len
        )
        self.decoder = CosyVoiceV1ConditionalCFM(config, CosyVoiceV3DiT(config))
        self.post_init()

    def forward(
        self,
        speech_token: torch.LongTensor,
        speech_token_len: torch.LongTensor,
        embedding: torch.FloatTensor,
        speech_feat: torch.FloatTensor | None = None,
        speech_feat_len: torch.LongTensor | None = None,
        n_timesteps: int = 10,
    ) -> ModelOutput:
        """See [`CosyVoiceV2FlowMatchingModel.forward`]."""
        spk_emb = F.normalize(embedding, dim=1)
        spk_emb = self.spk_embed_affine_layer(spk_emb)

        mask = (~_make_pad_mask(speech_token_len, speech_token.size(1))).to(spk_emb.dtype).unsqueeze(-1)
        token_emb = self.input_embedding(speech_token.clamp(min=0)) * mask
        hidden_states = self.pre_lookahead_layer(token_emb)
        hidden_states = hidden_states.repeat_interleave(self.token_mel_ratio, dim=1)
        target_len = speech_token_len * self.token_mel_ratio

        loss = None
        mel = None
        if speech_feat is not None:
            cond = torch.zeros_like(speech_feat).transpose(1, 2)
            out_mask = (~_make_pad_mask(target_len, speech_feat.size(1))).to(hidden_states).unsqueeze(1)
            loss = self.decoder.compute_loss(
                speech_feat.transpose(1, 2), out_mask, hidden_states.transpose(1, 2), spk_emb, cond
            )
        else:
            out_mask = (~_make_pad_mask(target_len, hidden_states.size(1))).to(hidden_states).unsqueeze(1)
            cond = torch.zeros(hidden_states.size(0), self.config.output_size, hidden_states.size(1), device=hidden_states.device, dtype=hidden_states.dtype)
            mel = self.decoder(hidden_states.transpose(1, 2), out_mask, spk_emb, cond, n_timesteps=n_timesteps)
        return ModelOutput(mel=mel, loss=loss)


class CosyVoiceV3CausalConv1d(nn.Conv1d):
    r"""
    Matches the original repository's `cosyvoice.transformer.convolution.CausalConv1d`: a stride-1,
    zero-internal-padding `Conv1d` whose input is causally padded (entirely on one side, instead of
    symmetrically like a plain `Conv1d(..., padding=...)`) before the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal_type: str = "left",
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation, groups=groups, bias=bias)
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        self.causal_type = causal_type

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pad = (self.causal_padding, 0) if self.causal_type == "left" else (0, self.causal_padding)
        return super().forward(F.pad(hidden_states, pad))


class CosyVoiceV3CausalConv1dUpsample(nn.Conv1d):
    """Matches `cosyvoice.transformer.convolution.CausalConv1dUpsample`: a nearest-neighbor `Upsample` followed
    by a causally-(left-)padded `Conv1d`, replacing the non-causal generator's `ConvTranspose1d` upsample stage
    (same `[out_channels, in_channels, kernel_size]` weight layout as any other `Conv1d`, unlike
    `ConvTranspose1d`'s `[in_channels, out_channels, kernel_size]`)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.causal_padding = kernel_size - 1
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.upsample(hidden_states)
        hidden_states = F.pad(hidden_states, (self.causal_padding, 0))
        return super().forward(hidden_states)


class CosyVoiceV3CausalConv1dDownSample(nn.Conv1d):
    """Matches `cosyvoice.transformer.convolution.CausalConv1dDownSample`: a strided `Conv1d` whose input is
    causally left-padded instead of symmetrically padded."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.causal_padding = stride - 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(hidden_states, (self.causal_padding, 0)))


class CosyVoiceV3ResBlock(nn.Module):
    """Causal counterpart of `CosyVoiceV1ResBlock`: identical `Conv1d`-based weight shapes (so a checkpoint
    trained with either loads into both), but each conv is a [`CosyVoiceV3CausalConv1d`] (left-causal padding)
    instead of a symmetrically-padded plain `Conv1d`, matching `ResBlock(..., causal=True)`."""

    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.utils.parametrizations.weight_norm(CosyVoiceV3CausalConv1d(channels, channels, kernel_size, dilation=d))
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.utils.parametrizations.weight_norm(CosyVoiceV3CausalConv1d(channels, channels, kernel_size, dilation=1))
                for _ in dilations
            ]
        )
        self.activations1 = nn.ModuleList([CosyVoiceV1Snake(channels) for _ in dilations])
        self.activations2 = nn.ModuleList([CosyVoiceV1Snake(channels) for _ in dilations])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, self.activations1, self.activations2):
            xt = conv1(act1(hidden_states))
            xt = conv2(act2(xt))
            hidden_states = hidden_states + xt
        return hidden_states


class CosyVoiceV3SourceModule(nn.Module):
    """Neural-source-filter harmonic excitation module. Reuses [`CosyVoiceV1SineGen`] (the checkpoint's
    `SineGen2` sine generator has no learnable parameters and produces the same sine-plus-noise excitation
    shape, only differing in phase/noise randomization details that don't affect gradients or checkpoint
    loading); `l_linear`/`tanh` merge the harmonics into a single excitation channel, matching
    `SourceModuleHnNSF.l_linear`/`l_tanh`."""

    def __init__(self, config: CosyVoiceV3HiftConfig, upsample_scale: int):
        super().__init__()
        self.sine_gen = CosyVoiceV1SineGen(config.sampling_rate, config.nb_harmonics, config.nsf_alpha, config.nsf_sigma, config.nsf_voiced_threshold)
        self.l_linear = nn.Linear(config.nb_harmonics + 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, f0_upsampled: torch.Tensor) -> torch.Tensor:
        sine_waves, _ = self.sine_gen(f0_upsampled)
        return self.tanh(self.l_linear(sine_waves))


class CosyVoiceV3F0Predictor(nn.Module):
    """Matches `cosyvoice.hifigan.f0_predictor.CausalConvRNNF0Predictor`: a causal-conv counterpart of
    [`CosyVoiceV1F0Predictor`] whose first `condnet` layer has kernel size 4 (right-causal) instead of 3
    (symmetric), with the remaining layers kernel size 3 (left-causal)."""

    def __init__(self, config: CosyVoiceV3HiftConfig):
        super().__init__()
        layers = [
            nn.utils.parametrizations.weight_norm(CosyVoiceV3CausalConv1d(config.in_channels, config.base_channels, 4, causal_type="right")),
            nn.ELU(),
        ]
        for _ in range(config.f0_predictor_num_layers - 1):
            layers += [
                nn.utils.parametrizations.weight_norm(CosyVoiceV3CausalConv1d(config.base_channels, config.base_channels, 3, causal_type="left")),
                nn.ELU(),
            ]
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(config.base_channels, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.condnet(mel).transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class CosyVoiceV3HiFTGenerator(CosyVoiceV1PreTrainedModel):
    r"""
    The CosyVoice v3 vocoder, matching the original repository's `CausalHiFTGenerator` (a causal-convolution
    subclass of the same `HiFTGenerator` base [`CosyVoiceV1HiFTGenerator`] implements). Unlike v1/v2's two-stage
    `[8, 8]` `ConvTranspose1d` upsample path, the real checkpoint has three `[8, 5, 3]` upsample stages built
    from [`CosyVoiceV3CausalConv1dUpsample`] (nearest-neighbor upsample + causal `Conv1d`, a different weight
    layout than `ConvTranspose1d`), causal [`CosyVoiceV3ResBlock`]s, a causal `conv_pre`/`conv_post`, and a
    causal [`CosyVoiceV3F0Predictor`].
    """

    config_class = CosyVoiceV3HiftConfig
    main_input_name = "mel"

    def __init__(self, config: CosyVoiceV3HiftConfig):
        super().__init__(config)
        self.num_upsamples = len(config.upsample_rates)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.lrelu_slope = config.lrelu_slope
        self.audio_limit = config.audio_limit
        self.istft_n_fft = config.istft_n_fft
        self.istft_hop_len = config.istft_hop_len
        self.conv_pre_look_right = config.conv_pre_look_right

        upsample_scale = 1
        for rate in config.upsample_rates:
            upsample_scale *= rate
        upsample_scale *= config.istft_hop_len
        self.f0_predictor = CosyVoiceV3F0Predictor(config)
        self.m_source = CosyVoiceV3SourceModule(config, upsample_scale)
        self.f0_upsample = nn.Upsample(scale_factor=upsample_scale)

        self.conv_pre = nn.utils.parametrizations.weight_norm(
            CosyVoiceV3CausalConv1d(config.in_channels, config.base_channels, config.conv_pre_look_right + 1, causal_type="right")
        )
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    CosyVoiceV3CausalConv1dUpsample(config.base_channels // (2**i), config.base_channels // (2 ** (i + 1)), k, u)
                )
            )

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + list(config.upsample_rates[::-1][:-1])
        cum_rates = list(itertools.accumulate(downsample_rates, operator.mul))
        for i, (u, k, d) in enumerate(zip(cum_rates[::-1], config.source_resblock_kernel_sizes, config.source_resblock_dilation_sizes)):
            out_ch = config.base_channels // (2 ** (i + 1))
            if u == 1:
                self.source_downs.append(CosyVoiceV3CausalConv1d(config.istft_n_fft + 2, out_ch, 1, causal_type="left"))
            else:
                self.source_downs.append(CosyVoiceV3CausalConv1dDownSample(config.istft_n_fft + 2, out_ch, int(u) * 2, int(u)))
            self.source_resblocks.append(CosyVoiceV3ResBlock(out_ch, k, d))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.base_channels // (2 ** (i + 1))
            for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(CosyVoiceV3ResBlock(ch, k, d))
        self.conv_post = nn.utils.parametrizations.weight_norm(
            CosyVoiceV3CausalConv1d(ch, config.istft_n_fft + 2, 7, causal_type="left")
        )
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.post_init()

    def _hann_window(self, like: torch.Tensor) -> torch.Tensor:
        # Deliberately *not* a `register_buffer`/`__init__`-time attribute: `from_pretrained` constructs this
        # module under a meta-device init context, and since nothing in the checkpoint ever populates this
        # non-persistent, non-parameter window, transformers' meta-to-real materialization leaves it as
        # uninitialized memory instead of the real Hann window (silently breaks `torch.istft`'s NOLA/window
        # overlap-add check, with no shape/dtype error to catch it). Computed fresh on every call instead,
        # entirely outside the meta-init context, so it's always real data on the right device.
        return torch.hann_window(self.istft_n_fft, device=like.device, dtype=like.dtype)

    def _stft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = self._hann_window(x)
        spec = torch.stft(x, self.istft_n_fft, self.istft_hop_len, self.istft_n_fft, window=window, return_complex=True)
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        window = self._hann_window(magnitude)
        return torch.istft(torch.complex(real, imag), self.istft_n_fft, self.istft_hop_len, self.istft_n_fft, window=window)

    def decode(self, hidden_states: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        s_real, s_imag = self._stft(source.squeeze(1))
        source_stft = torch.cat([s_real, s_imag], dim=1)

        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = F.leaky_relu(hidden_states, self.lrelu_slope)
            hidden_states = self.ups[i](hidden_states)
            if i == self.num_upsamples - 1:
                hidden_states = self.reflection_pad(hidden_states)

            source_i = self.source_downs[i](source_stft)
            source_i = self.source_resblocks[i](source_i)
            hidden_states = hidden_states + source_i

            summed = None
            for j in range(self.num_kernels):
                block_out = self.resblocks[i * self.num_kernels + j](hidden_states)
                summed = block_out if summed is None else summed + block_out
            hidden_states = summed / self.num_kernels

        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        magnitude = torch.exp(hidden_states[:, : self.istft_n_fft // 2 + 1, :])
        phase = torch.sin(hidden_states[:, self.istft_n_fft // 2 + 1 :, :])
        waveform = self._istft(magnitude, phase)
        return torch.clamp(waveform, -self.audio_limit, self.audio_limit)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Mel spectrogram to render into a waveform.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, waveform_length)`: The synthesized audio.
        """
        f0 = self.f0_predictor(mel)
        f0_upsampled = self.f0_upsample(f0[:, None]).transpose(1, 2)
        source = self.m_source(f0_upsampled)
        source = source.transpose(1, 2)
        return self.decode(mel, source).squeeze(1)


class CosyVoiceV3Model(CosyVoiceV1PreTrainedModel):
    r"""
    The full CosyVoice v3 model: a Qwen2-backbone speech-token language model with an extended speech-token
    vocabulary, a DiT conditional-flow-matching mel decoder, and the causal HiFTNet vocoder.
    """

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__(config)
        self.llm = CosyVoiceV3LLM(config.llm_config)
        self.flow = CosyVoiceV3FlowMatchingModel(config.flow_config)
        self.hift = CosyVoiceV3HiFTGenerator(config.hift_config)
        self.post_init()

    def forward(self, **kwargs) -> CosyVoiceV2LLMOutput:
        return self.llm(**kwargs)

    @torch.no_grad()
    def generate_speech(self, text_token, embedding, prompt_speech_token=None, **kwargs) -> torch.Tensor:
        """See [`~CosyVoiceV2Model.generate_speech`]."""
        speech_token = self.llm.generate(text_token, prompt_speech_token, **kwargs)
        speech_token_len = torch.tensor([speech_token.size(1)], device=speech_token.device)
        flow_out = self.flow(speech_token, speech_token_len, embedding)
        return self.hift(flow_out.mel)


class CosyVoiceV3ForConditionalGeneration(CosyVoiceV1PreTrainedModel):
    r"""
    CosyVoice v3 model with the Qwen2-backbone speech-token language model, the DiT flow-matching decoder, and
    the vocoder, with a `generate` method producing a waveform end-to-end. The trainable `forward` pass is the
    speech-token language model's next-token cross-entropy objective.
    """

    config_class = CosyVoiceV3Config

    def __init__(self, config: CosyVoiceV3Config):
        super().__init__(config)
        self.model = CosyVoiceV3Model(config)
        self.post_init()

    def forward(self, **kwargs) -> CosyVoiceV2LLMOutput:
        return self.model.llm(**kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """See [`~CosyVoiceV3Model.generate_speech`]."""
        return self.model.generate_speech(*args, **kwargs)


__all__ = [
    "CosyVoiceV3ForConditionalGeneration",
    "CosyVoiceV3Model",
    "CosyVoiceV3LLM",
    "CosyVoiceV3FlowMatchingModel",
    "CosyVoiceV3DiT",
    "CosyVoiceV3DiTBlock",
    "CosyVoiceV3HiFTGenerator",
]
