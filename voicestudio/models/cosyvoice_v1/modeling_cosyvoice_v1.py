# coding=utf-8
# Copyright 2024 Alibaba Inc and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerFeedForward,
    Wav2Vec2ConformerRelPositionalEmbedding,
    Wav2Vec2ConformerSelfAttention,
)

from .configuration_cosyvoice_v1 import (
    CosyVoiceV1Config,
    CosyVoiceV1FlowConfig,
    CosyVoiceV1HiftConfig,
    CosyVoiceV1LLMConfig,
    CosyVoiceV1TextEncoderConfig,
)


def _make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Returns a `(batch_size, max_len)` boolean mask, `True` at padding positions."""
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq_range = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, max_len)
    return seq_range >= lengths.unsqueeze(-1)


def _conformer_config(hidden_size, num_attention_heads, intermediate_size, hidden_dropout, attention_dropout):
    return Wav2Vec2ConformerConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        activation_dropout=hidden_dropout,
        hidden_act="swish",
        position_embeddings_type="relative",
        max_source_positions=5000,
    )


class CosyVoiceV1EncoderLayer(nn.Module):
    """
    One pre-norm relative-position self-attention + single feed-forward block, matching the original
    CosyVoice/WeNet `TransformerEncoderLayer` (`normalize_before=True`, no macaron feed-forward, no depthwise
    convolution module) rather than the `transformers` `Wav2Vec2ConformerEncoderLayer`, which always includes
    both.
    """

    def __init__(self, config: Wav2Vec2ConformerConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn = Wav2Vec2ConformerSelfAttention(config)
        self.self_attn_dropout = nn.Dropout(config.attention_dropout)
        self.norm_ff = nn.LayerNorm(config.hidden_size)
        self.feed_forward = Wav2Vec2ConformerFeedForward(config)
        self.ff_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states, attention_mask=None, relative_position_embeddings=None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
        )
        hidden_states = residual + self.self_attn_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.norm_ff(hidden_states)
        hidden_states = residual + self.ff_dropout(self.feed_forward(hidden_states))
        return hidden_states


class CosyVoiceV1RelPositionEncoder(nn.Module):
    """
    A relative-position Transformer stack matching the original CosyVoice/WeNet `TransformerEncoder`: an input
    `Linear` + `LayerNorm` projection (`embed.out`), a stack of [`CosyVoiceV1EncoderLayer`]s (self-attention +
    single feed-forward each, no depthwise convolution, no macaron feed-forward), and a final `LayerNorm`
    (`after_norm`). Accepts either a padding-only boolean mask (bidirectional use, e.g. the text encoder) or a
    precomputed additive attention bias (causal use, e.g. the speech-token LM).
    """

    def __init__(self, input_size: int, config: Wav2Vec2ConformerConfig, num_hidden_layers: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_size, config.hidden_size), nn.LayerNorm(config.hidden_size))
        self.embed_dropout = nn.Dropout(config.hidden_dropout)
        self.embed_positions = Wav2Vec2ConformerRelPositionalEmbedding(config)
        self.layers = nn.ModuleList([CosyVoiceV1EncoderLayer(config) for _ in range(num_hidden_layers)])
        self.after_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, attention_bias=None):
        hidden_states = self.embed(hidden_states)
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
        hidden_states = self.embed_dropout(hidden_states)
        relative_position_embeddings = self.embed_positions(hidden_states)

        layer_attention = attention_bias
        if layer_attention is None and attention_mask is not None:
            layer_attention = attention_mask[:, None, None, :].to(hidden_states.dtype)
            layer_attention = (1.0 - layer_attention) * torch.finfo(hidden_states.dtype).min
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=layer_attention, relative_position_embeddings=relative_position_embeddings
            )
        hidden_states = self.after_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class CosyVoiceV1TextEncoder(nn.Module):
    """Embeds text token ids and contextualizes them with a bidirectional relative-position Transformer encoder."""

    def __init__(self, config: CosyVoiceV1TextEncoderConfig):
        super().__init__()
        self.encoder = CosyVoiceV1RelPositionEncoder(
            config.input_size,
            _conformer_config(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.hidden_dropout,
                config.attention_dropout,
            ),
            config.num_hidden_layers,
        )

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = ~_make_pad_mask(lengths, hidden_states.size(1))
        output = self.encoder(hidden_states, attention_mask=attention_mask)
        return output.last_hidden_state, lengths


@dataclass
class CosyVoiceV1LLMOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Speech-token cross-entropy loss, returned when `labels` is given.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, speech_token_size + 1)`):
            Prediction scores over the speech-token vocabulary (plus the end-of-speech token).
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None


class CosyVoiceV1PreTrainedModel(PreTrainedModel):
    config_class = CosyVoiceV1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CosyVoiceV1LLM(CosyVoiceV1PreTrainedModel):
    r"""
    The CosyVoice v1 speech-token language model. Given text encoder output, a speaker embedding, and previously
    generated speech tokens, autoregressively predicts the next discrete speech token with a relative-position
    Conformer/Transformer decoder.
    """

    config_class = CosyVoiceV1LLMConfig

    def __init__(self, config: CosyVoiceV1LLMConfig, text_encoder_config: CosyVoiceV1TextEncoderConfig):
        super().__init__(config)
        self.speech_token_size = config.speech_token_size
        self.sos_eos = 0
        self.task_id = 1
        self.eos_token_id = config.speech_token_size

        self.text_embedding = nn.Embedding(config.text_token_size, text_encoder_config.input_size)
        self.text_encoder = CosyVoiceV1TextEncoder(text_encoder_config)
        self.text_encoder_affine_layer = nn.Linear(text_encoder_config.hidden_size, config.llm_input_size)

        self.llm_embedding = nn.Embedding(2, config.llm_input_size)
        self.llm = CosyVoiceV1RelPositionEncoder(
            config.llm_input_size,
            _conformer_config(
                config.llm_input_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.hidden_dropout,
                config.attention_dropout,
            ),
            config.num_hidden_layers,
        )
        self.llm_decoder = nn.Linear(config.llm_output_size, config.speech_token_size + 1)

        self.speech_embedding = nn.Embedding(config.speech_token_size, config.llm_input_size)
        self.spk_embed_affine_layer = nn.Linear(config.spk_embed_dim, config.llm_input_size)

        self.post_init()

    def _causal_bias(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        causal = torch.triu(torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device), diagonal=1)
        return causal[None, None, :, :]

    def forward(
        self,
        text_token: torch.LongTensor,
        text_token_len: torch.LongTensor,
        speech_token: torch.LongTensor,
        speech_token_len: torch.LongTensor,
        embedding: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
    ) -> CosyVoiceV1LLMOutput:
        """
        Args:
            text_token (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Text token ids.
            text_token_len (`torch.LongTensor` of shape `(batch_size,)`):
                Number of valid text tokens per example.
            speech_token (`torch.LongTensor` of shape `(batch_size, speech_sequence_length)`):
                Discrete speech token ids.
            speech_token_len (`torch.LongTensor` of shape `(batch_size,)`):
                Number of valid speech tokens per example.
            embedding (`torch.FloatTensor` of shape `(batch_size, spk_embed_dim)`):
                Speaker x-vector embedding.
            labels (`torch.LongTensor` of shape `(batch_size, speech_sequence_length + 1)`, *optional*):
                Target speech token ids (including the trailing end-of-speech token). When given, a cross-entropy
                loss is computed.

        Returns:
            [`CosyVoiceV1LLMOutput`]
        """
        device = text_token.device
        text_emb = self.text_embedding(text_token)
        text_emb, text_token_len = self.text_encoder(text_emb, text_token_len)
        text_emb = self.text_encoder_affine_layer(text_emb)

        spk_emb = F.normalize(embedding, dim=1)
        spk_emb = self.spk_embed_affine_layer(spk_emb).unsqueeze(1)

        sos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1).expand(text_emb.size(0), -1, -1)
        task_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1).expand(text_emb.size(0), -1, -1)
        speech_emb = self.speech_embedding(speech_token)

        lm_input = torch.cat([sos_emb, spk_emb, text_emb, task_emb, speech_emb], dim=1)
        seq_len = lm_input.size(1)
        attn_bias = self._causal_bias(seq_len, lm_input.dtype, device)

        hidden_states = self.llm(lm_input, attention_bias=attn_bias).last_hidden_state
        logits = self.llm_decoder(hidden_states)

        loss = None
        if labels is not None:
            prefix_len = lm_input.size(1) - speech_emb.size(1)
            target = torch.full((logits.size(0), seq_len), -100, dtype=torch.long, device=device)
            target[:, prefix_len - 1 :] = labels
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
        return CosyVoiceV1LLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        text_token: torch.LongTensor,
        text_token_len: torch.LongTensor,
        embedding: torch.FloatTensor,
        prompt_speech_token: torch.LongTensor | None = None,
        max_new_tokens: int = 2000,
        min_new_tokens: int = 0,
        top_k: int = 25,
    ) -> torch.LongTensor:
        """
        Autoregressively samples discrete speech tokens conditioned on `text_token` and `embedding`.

        Args:
            text_token (`torch.LongTensor` of shape `(1, text_sequence_length)`):
                Text token ids.
            text_token_len (`torch.LongTensor` of shape `(1,)`):
                Number of valid text tokens.
            embedding (`torch.FloatTensor` of shape `(1, spk_embed_dim)`):
                Speaker x-vector embedding.
            prompt_speech_token (`torch.LongTensor` of shape `(1, prompt_length)`, *optional*):
                Previously generated speech tokens (e.g. from a reference utterance) to prime decoding.
            max_new_tokens (`int`, *optional*, defaults to 2000):
                Maximum number of speech tokens to sample.
            min_new_tokens (`int`, *optional*, defaults to 0):
                Minimum number of speech tokens to sample before the end-of-speech token is allowed.
            top_k (`int`, *optional*, defaults to 25):
                Number of highest-probability tokens kept for sampling.

        Returns:
            `torch.LongTensor` of shape `(1, generated_length)`: The sampled speech token ids.
        """
        device = text_token.device
        text_emb = self.text_embedding(text_token)
        text_emb, _ = self.text_encoder(text_emb, text_token_len)
        text_emb = self.text_encoder_affine_layer(text_emb)

        spk_emb = F.normalize(embedding, dim=1)
        spk_emb = self.spk_embed_affine_layer(spk_emb).unsqueeze(1)

        sos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        prompt_emb = (
            self.speech_embedding(prompt_speech_token)
            if prompt_speech_token is not None and prompt_speech_token.size(1) > 0
            else torch.zeros(1, 0, text_emb.size(-1), device=device, dtype=text_emb.dtype)
        )
        lm_input = torch.cat([sos_emb, spk_emb, text_emb, task_emb, prompt_emb], dim=1)

        generated = []
        for step in range(max_new_tokens):
            seq_len = lm_input.size(1)
            hidden_states = self.llm(lm_input, attention_bias=self._causal_bias(seq_len, lm_input.dtype, device)).last_hidden_state
            logits = self.llm_decoder(hidden_states[:, -1])
            if step < min_new_tokens:
                logits[:, self.eos_token_id] = -float("inf")
            top_logits, top_indices = logits.topk(top_k, dim=-1)
            probs = torch.softmax(top_logits, dim=-1)
            next_token = top_indices.gather(-1, torch.multinomial(probs, 1))
            token_id = next_token.item()
            if token_id == self.eos_token_id:
                break
            generated.append(token_id)
            lm_input = torch.cat([lm_input, self.speech_embedding(next_token)], dim=1)
        return torch.tensor(generated, device=device, dtype=torch.long).unsqueeze(0)


class CosyVoiceV1SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class CosyVoiceV1TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class CosyVoiceV1Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(min(8, dim_out), dim_out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(x * mask) * mask


class CosyVoiceV1ResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = CosyVoiceV1Block1D(dim, dim_out)
        self.block2 = CosyVoiceV1Block1D(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class CosyVoiceV1EstimatorBlock(nn.Module):
    """Pre-norm self-attention + feed-forward block, film-conditioned on the flow-matching timestep."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float, act_fn: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        act = nn.GELU() if act_fn == "gelu" else nn.SiLU()
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), act, nn.Linear(dim * 4, dim))
        self.time_proj = nn.Linear(dim * 4, dim * 2)

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | None, time_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_proj(time_emb).unsqueeze(1).chunk(2, dim=-1)
        normed = self.norm1(hidden_states) * (1 + scale) + shift
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class CosyVoiceV1ConditionalDecoder(nn.Module):
    """U-Net estimator of the conditional-flow-matching velocity field: predicts `d(mel)/dt` given the current
    noised mel, the flow-matching timestep, the length-regulated encoder output `mu`, and the speaker embedding."""

    def __init__(self, config: CosyVoiceV1FlowConfig):
        super().__init__()
        in_channels = config.output_size * 3
        channels = list(config.decoder_channels)
        time_embed_dim = channels[0] * 4
        self.time_embeddings = CosyVoiceV1SinusoidalPosEmb(in_channels)
        self.time_mlp = CosyVoiceV1TimestepEmbedding(in_channels, time_embed_dim)

        def make_stage(input_channel, output_channel):
            resnet = CosyVoiceV1ResnetBlock1D(input_channel, output_channel, time_embed_dim)
            blocks = nn.ModuleList(
                [
                    CosyVoiceV1EstimatorBlock(output_channel, config.decoder_num_heads, config.decoder_attention_head_dim, config.decoder_dropout, "gelu")
                    for _ in range(config.decoder_n_blocks)
                ]
            )
            return resnet, blocks

        self.down_blocks = nn.ModuleList()
        output_channel = in_channels
        for i, ch in enumerate(channels):
            resnet, blocks = make_stage(output_channel, ch)
            is_last = i == len(channels) - 1
            downsample = nn.Conv1d(ch, ch, 3, stride=1, padding=1) if is_last else nn.Conv1d(ch, ch, 4, stride=2, padding=1)
            self.down_blocks.append(nn.ModuleList([resnet, blocks, downsample]))
            output_channel = ch

        self.mid_blocks = nn.ModuleList(
            [nn.ModuleList(make_stage(channels[-1], channels[-1])) for _ in range(config.decoder_num_mid_blocks)]
        )

        self.up_blocks = nn.ModuleList()
        up_channels = channels[::-1] + [channels[0]]
        for i in range(len(up_channels) - 1):
            resnet, blocks = make_stage(up_channels[i] * 2, up_channels[i + 1])
            is_last = i == len(up_channels) - 2
            upsample = (
                nn.Conv1d(up_channels[i + 1], up_channels[i + 1], 3, padding=1)
                if is_last
                else nn.ConvTranspose1d(up_channels[i + 1], up_channels[i + 1], 4, stride=2, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, blocks, upsample]))

        self.final_block = CosyVoiceV1Block1D(up_channels[-1], up_channels[-1])
        self.final_proj = nn.Conv1d(up_channels[-1], config.output_size, 1)

    def forward(self, x, mask, mu, t, spks, cond) -> torch.Tensor:
        """
        Args:
            x (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Noised mel spectrogram at timestep `t`.
            mask (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
                Mask over the mel sequence dimension.
            mu (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Length-regulated text/speech-token encoder output.
            t (`torch.FloatTensor` of shape `(batch_size,)`):
                Flow-matching timestep in `[0, 1]`.
            spks (`torch.FloatTensor` of shape `(batch_size, spk_embed_dim)`):
                Speaker embedding.
            cond (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Prompt mel spectrogram, zeroed outside the reference span.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`: The predicted velocity field.
        """
        time_emb = self.time_mlp(self.time_embeddings(t).to(t.dtype))
        spk_expanded = spks.unsqueeze(-1).expand(-1, -1, x.size(-1))
        hidden_states = torch.cat([x, mu, spk_expanded], dim=1)

        skips = []
        masks = [mask]
        for resnet, blocks, downsample in self.down_blocks:
            m = masks[-1]
            hidden_states = resnet(hidden_states, m, time_emb)
            attn_mask = None if m.all() else (~m.squeeze(1).bool()).unsqueeze(1).expand(-1, m.size(-1), -1)
            hidden_states_t = hidden_states.transpose(1, 2)
            for block in blocks:
                hidden_states_t = block(hidden_states_t, attn_mask, time_emb)
            hidden_states = hidden_states_t.transpose(1, 2)
            skips.append(hidden_states)
            hidden_states = downsample(hidden_states * m)
            masks.append(m[:, :, ::2] if m.size(-1) > 1 else m)
        masks = masks[:-1]
        m = masks[-1]

        for resnet, blocks in self.mid_blocks:
            hidden_states = resnet(hidden_states, m, time_emb)
            attn_mask = None if m.all() else (~m.squeeze(1).bool()).unsqueeze(1).expand(-1, m.size(-1), -1)
            hidden_states_t = hidden_states.transpose(1, 2)
            for block in blocks:
                hidden_states_t = block(hidden_states_t, attn_mask, time_emb)
            hidden_states = hidden_states_t.transpose(1, 2)

        for resnet, blocks, upsample in self.up_blocks:
            m = masks.pop()
            skip = skips.pop()
            hidden_states = torch.cat([hidden_states[:, :, : skip.shape[-1]], skip], dim=1)
            hidden_states = resnet(hidden_states, m, time_emb)
            attn_mask = None if m.all() else (~m.squeeze(1).bool()).unsqueeze(1).expand(-1, m.size(-1), -1)
            hidden_states_t = hidden_states.transpose(1, 2)
            for block in blocks:
                hidden_states_t = block(hidden_states_t, attn_mask, time_emb)
            hidden_states = hidden_states_t.transpose(1, 2)
            hidden_states = upsample(hidden_states * m)
        hidden_states = self.final_block(hidden_states, m)
        return self.final_proj(hidden_states * m) * mask


class CosyVoiceV1ConditionalCFM(nn.Module):
    """Conditional flow matching: trains a velocity-field estimator with a linear (rectified-flow) probability
    path and samples from it with a fixed-step Euler ODE solver."""

    def __init__(self, config: CosyVoiceV1FlowConfig, estimator: nn.Module):
        super().__init__()
        self.sigma_min = config.sigma_min
        self.t_scheduler = config.t_scheduler
        self.training_cfg_rate = config.training_cfg_rate
        self.inference_cfg_rate = config.inference_cfg_rate
        self.estimator = estimator

    def compute_loss(self, x1, mask, mu, spks, cond) -> torch.Tensor:
        """
        Args:
            x1 (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Target mel spectrogram.
            mask (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
                Mask over the mel sequence dimension.
            mu (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Length-regulated encoder output.
            spks (`torch.FloatTensor` of shape `(batch_size, spk_embed_dim)`):
                Speaker embedding.
            cond (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Prompt mel spectrogram.

        Returns:
            `torch.FloatTensor` of shape `(1,)`: The flow-matching mean-squared-error loss.
        """
        batch_size = mu.size(0)
        t = torch.rand(batch_size, 1, 1, device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        target = x1 - (1 - self.sigma_min) * z

        if self.training_cfg_rate > 0:
            cfg_mask = (torch.rand(batch_size, device=x1.device) > self.training_cfg_rate).view(-1, 1, 1)
            mu = mu * cfg_mask
            spks = spks * cfg_mask.squeeze(-1)
            cond = cond * cfg_mask

        pred = self.estimator(y, mask, mu, t.squeeze(-1).squeeze(-1), spks, cond)
        return F.mse_loss(pred * mask, target * mask, reduction="sum") / (mask.sum() * target.size(1))

    @torch.no_grad()
    def forward(self, mu, mask, spks, cond, n_timesteps: int = 10, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            mu (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Length-regulated encoder output.
            mask (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
                Mask over the mel sequence dimension.
            spks (`torch.FloatTensor` of shape `(batch_size, spk_embed_dim)`):
                Speaker embedding.
            cond (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Prompt mel spectrogram, zeroed outside the reference span.
            n_timesteps (`int`, *optional*, defaults to 10):
                Number of Euler integration steps.
            temperature (`float`, *optional*, defaults to 1.0):
                Temperature scaling the initial Gaussian noise sample.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`: The sampled mel spectrogram.
        """
        x = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        zeros_mu = torch.zeros_like(mu)
        zeros_spks = torch.zeros_like(spks)
        zeros_cond = torch.zeros_like(cond)
        for i in range(n_timesteps):
            t = t_span[i].expand(mu.size(0))
            dt = t_span[i + 1] - t_span[i]
            velocity_cond = self.estimator(x, mask, mu, t, spks, cond)
            velocity_uncond = self.estimator(x, mask, zeros_mu, t, zeros_spks, zeros_cond)
            velocity = (1 + self.inference_cfg_rate) * velocity_cond - self.inference_cfg_rate * velocity_uncond
            x = x + dt * velocity
        return x


class CosyVoiceV1InterpolateRegulator(nn.Module):
    """Length-regulates a speech-token encoder output sequence to the target mel sequence length by linear
    interpolation, followed by a small convolutional refinement stack."""

    def __init__(self, channels: int, out_channels: int, num_stages: int = 4):
        super().__init__()
        layers = []
        for _ in range(num_stages):
            layers += [nn.Conv1d(channels, channels, 3, padding=1), nn.GroupNorm(1, channels), nn.Mish()]
        layers.append(nn.Conv1d(channels, out_channels, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor, target_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (~_make_pad_mask(target_lengths, int(target_lengths.max().item()))).to(hidden_states).unsqueeze(-1)
        hidden_states = F.interpolate(hidden_states.transpose(1, 2), size=int(target_lengths.max().item()), mode="linear")
        out = self.model(hidden_states).transpose(1, 2)
        return out * mask, target_lengths


class CosyVoiceV1FlowMatchingModel(CosyVoiceV1PreTrainedModel):
    r"""
    The CosyVoice v1 conditional-flow-matching decoder. Encodes a discrete speech token sequence with a
    bidirectional Conformer encoder, length-regulates it to the target mel length, and samples a mel spectrogram
    with a conditional-flow-matching Euler solver conditioned on a speaker embedding and an optional prompt mel.
    """

    config_class = CosyVoiceV1FlowConfig

    def __init__(self, config: CosyVoiceV1FlowConfig):
        super().__init__(config)
        self.input_embedding = nn.Embedding(config.vocab_size, config.input_size)
        self.spk_embed_affine_layer = nn.Linear(config.spk_embed_dim, config.output_size)
        self.encoder = CosyVoiceV1RelPositionEncoder(
            config.input_size,
            _conformer_config(
                config.encoder_hidden_size,
                config.encoder_num_attention_heads,
                config.encoder_intermediate_size,
                0.1,
                0.1,
            ),
            config.encoder_num_hidden_layers,
        )
        self.encoder_proj = nn.Linear(config.encoder_hidden_size, config.output_size)
        self.length_regulator = CosyVoiceV1InterpolateRegulator(config.output_size, config.output_size)
        self.decoder = CosyVoiceV1ConditionalCFM(config, CosyVoiceV1ConditionalDecoder(config))
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
        """
        Args:
            speech_token (`torch.LongTensor` of shape `(batch_size, speech_sequence_length)`):
                Discrete speech token ids.
            speech_token_len (`torch.LongTensor` of shape `(batch_size,)`):
                Number of valid speech tokens per example.
            embedding (`torch.FloatTensor` of shape `(batch_size, spk_embed_dim)`):
                Speaker x-vector embedding.
            speech_feat (`torch.FloatTensor` of shape `(batch_size, mel_sequence_length, mel_dim)`, *optional*):
                Target mel spectrogram. When given, a flow-matching loss is computed instead of sampling.
            speech_feat_len (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Number of valid mel frames per example. Required when `speech_feat` is given.
            n_timesteps (`int`, *optional*, defaults to 10):
                Number of Euler integration steps used when sampling (`speech_feat` is `None`).

        Returns:
            [`~transformers.utils.ModelOutput`] with `mel` (sampled or `None`) and `loss` (computed or `None`).
        """
        spk_emb = F.normalize(embedding, dim=1)
        spk_emb = self.spk_embed_affine_layer(spk_emb)

        mask = (~_make_pad_mask(speech_token_len, speech_token.size(1))).to(spk_emb.dtype).unsqueeze(-1)
        token_emb = self.input_embedding(speech_token.clamp(min=0)) * mask
        hidden_states = self.encoder(token_emb, attention_mask=mask.squeeze(-1).bool())
        hidden_states = hidden_states.last_hidden_state
        hidden_states = self.encoder_proj(hidden_states)

        loss = None
        mel = None
        if speech_feat is not None:
            hidden_states, _ = self.length_regulator(hidden_states, speech_feat_len)
            cond = torch.zeros_like(speech_feat).transpose(1, 2)
            out_mask = (~_make_pad_mask(speech_feat_len, speech_feat.size(1))).to(hidden_states).unsqueeze(1)
            loss = self.decoder.compute_loss(
                speech_feat.transpose(1, 2), out_mask, hidden_states.transpose(1, 2), spk_emb, cond
            )
        else:
            hidden_states, target_len = self.length_regulator(hidden_states, speech_token_len)
            out_mask = (~_make_pad_mask(target_len, hidden_states.size(1))).to(hidden_states).unsqueeze(1)
            cond = torch.zeros(hidden_states.size(0), self.config.output_size, hidden_states.size(1), device=hidden_states.device, dtype=hidden_states.dtype)
            mel = self.decoder(hidden_states.transpose(1, 2), out_mask, spk_emb, cond, n_timesteps=n_timesteps)
        return ModelOutput(mel=mel, loss=loss)


class CosyVoiceV1SineGen(nn.Module):
    """Generates a harmonic sine excitation signal from a frame-rate F0 curve, upsampled to the sample rate."""

    def __init__(self, sampling_rate: int, harmonic_num: int, sine_amp: float, noise_std: float, voiced_threshold: float):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = f0.transpose(1, 2)
        harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype)
        f_mat = f0 * harmonics.view(1, -1, 1) / self.sampling_rate
        theta = 2 * math.pi * (torch.cumsum(f_mat, dim=-1) % 1)
        sine_waves = self.sine_amp * torch.sin(theta)
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        sine_waves = sine_waves * uv + noise_amp * torch.randn_like(sine_waves)
        return sine_waves.transpose(1, 2), uv.transpose(1, 2)


class CosyVoiceV1SourceModule(nn.Module):
    def __init__(self, config: CosyVoiceV1HiftConfig, upsample_scale: int):
        super().__init__()
        self.sine_gen = CosyVoiceV1SineGen(config.sampling_rate, config.nb_harmonics, config.nsf_alpha, config.nsf_sigma, config.nsf_voiced_threshold)
        self.l_linear = nn.Linear(config.nb_harmonics + 1, 1)
        self.tanh = nn.Tanh()
        self.sine_amp = config.nsf_alpha

    def forward(self, f0_upsampled: torch.Tensor) -> torch.Tensor:
        sine_waves, uv = self.sine_gen(f0_upsampled)
        return self.tanh(self.l_linear(sine_waves))


class CosyVoiceV1F0Predictor(nn.Module):
    def __init__(self, config: CosyVoiceV1HiftConfig):
        super().__init__()
        layers = []
        channels = config.in_channels
        for _ in range(config.f0_predictor_num_layers):
            layers += [
                nn.utils.parametrizations.weight_norm(nn.Conv1d(channels, config.base_channels, 3, padding=1)),
                nn.ELU(),
            ]
            channels = config.base_channels
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(config.base_channels, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.condnet(mel).transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class CosyVoiceV1Snake(nn.Module):
    """Learnable Snake activation `x + (1/alpha) * sin(alpha * x)^2`, one `alpha` per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1)
        return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class CosyVoiceV1ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [nn.utils.parametrizations.weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size - 1) * d // 2)) for d in dilations]
        )
        self.convs2 = nn.ModuleList(
            [nn.utils.parametrizations.weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)) for _ in dilations]
        )
        self.activations1 = nn.ModuleList([CosyVoiceV1Snake(channels) for _ in dilations])
        self.activations2 = nn.ModuleList([CosyVoiceV1Snake(channels) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, self.activations1, self.activations2):
            xt = conv1(act1(x))
            xt = conv2(act2(xt))
            x = x + xt
        return x


class CosyVoiceV1HiFTGenerator(CosyVoiceV1PreTrainedModel):
    r"""
    The CosyVoice v1 vocoder: a HiFTNet (neural-source-filter + ISTFT) generator that renders a mel spectrogram
    into a waveform.
    """

    config_class = CosyVoiceV1HiftConfig
    main_input_name = "mel"

    def __init__(self, config: CosyVoiceV1HiftConfig):
        super().__init__(config)
        self.num_upsamples = len(config.upsample_rates)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.lrelu_slope = config.lrelu_slope
        self.audio_limit = config.audio_limit
        self.istft_n_fft = config.istft_n_fft
        self.istft_hop_len = config.istft_hop_len

        upsample_scale = 1
        for rate in config.upsample_rates:
            upsample_scale *= rate
        upsample_scale *= config.istft_hop_len
        self.f0_predictor = CosyVoiceV1F0Predictor(config)
        self.m_source = CosyVoiceV1SourceModule(config, upsample_scale)
        self.f0_upsample = nn.Upsample(scale_factor=upsample_scale)

        self.conv_pre = nn.utils.parametrizations.weight_norm(nn.Conv1d(config.in_channels, config.base_channels, 7, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(config.base_channels // (2**i), config.base_channels // (2 ** (i + 1)), k, u, padding=(k - u) // 2)
                )
            )

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + list(config.upsample_rates[::-1][:-1])
        cum_rates = torch.cumprod(torch.tensor(downsample_rates), dim=0).tolist()
        for i, (u, k, d) in enumerate(zip(cum_rates[::-1], config.source_resblock_kernel_sizes, config.source_resblock_dilation_sizes)):
            out_ch = config.base_channels // (2 ** (i + 1))
            if u == 1:
                self.source_downs.append(nn.Conv1d(config.istft_n_fft + 2, out_ch, 1))
            else:
                self.source_downs.append(nn.Conv1d(config.istft_n_fft + 2, out_ch, int(u) * 2, int(u), padding=int(u) // 2))
            self.source_resblocks.append(CosyVoiceV1ResBlock(out_ch, k, d))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.base_channels // (2 ** (i + 1))
            for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(CosyVoiceV1ResBlock(ch, k, d))
        self.conv_post = nn.utils.parametrizations.weight_norm(nn.Conv1d(ch, config.istft_n_fft + 2, 7, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.register_buffer("stft_window", torch.hann_window(config.istft_n_fft), persistent=False)
        self.post_init()

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = torch.clip(magnitude, max=1e2)
        complex_spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        return torch.istft(complex_spec, self.istft_n_fft, self.istft_hop_len, self.istft_n_fft, window=self.stft_window)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel (`torch.FloatTensor` of shape `(batch_size, mel_dim, sequence_length)`):
                Mel spectrogram to render.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_samples)`: The synthesized waveform.
        """
        f0 = self.f0_predictor(mel)
        f0_upsampled = self.f0_upsample(f0[:, None]).transpose(1, 2)
        source = self.m_source(f0_upsampled).transpose(1, 2)
        source_real, source_imag = self._stft(source.squeeze(1))
        source_stft = torch.cat([source_real, source_imag], dim=1)

        hidden_states = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            hidden_states = F.leaky_relu(hidden_states, self.lrelu_slope)
            hidden_states = self.ups[i](hidden_states)
            if i == self.num_upsamples - 1:
                hidden_states = self.reflection_pad(hidden_states)
            source_i = self.source_resblocks[i](self.source_downs[i](source_stft))
            hidden_states = hidden_states + source_i
            summed = None
            for j in range(self.num_kernels):
                block_out = self.resblocks[i * self.num_kernels + j](hidden_states)
                summed = block_out if summed is None else summed + block_out
            hidden_states = summed / self.num_kernels

        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        magnitude = torch.exp(hidden_states[:, : self.istft_n_fft // 2 + 1])
        phase = torch.sin(hidden_states[:, self.istft_n_fft // 2 + 1 :])
        waveform = self._istft(magnitude, phase)
        return torch.clamp(waveform, -self.audio_limit, self.audio_limit)

    def _stft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(x, self.istft_n_fft, self.istft_hop_len, self.istft_n_fft, window=self.stft_window, return_complex=True)
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]


class CosyVoiceV1Model(CosyVoiceV1PreTrainedModel):
    r"""
    The full CosyVoice v1 model: an autoregressive speech-token language model, a conditional-flow-matching mel
    decoder, and a HiFTNet vocoder.
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__(config)
        self.llm = CosyVoiceV1LLM(config.llm_config, config.text_encoder_config)
        self.flow = CosyVoiceV1FlowMatchingModel(config.flow_config)
        self.hift = CosyVoiceV1HiFTGenerator(config.hift_config)
        self.post_init()

    def forward(self, **kwargs) -> CosyVoiceV1LLMOutput:
        return self.llm(**kwargs)

    @torch.no_grad()
    def generate_speech(self, text_token, text_token_len, embedding, prompt_speech_token=None, **kwargs) -> torch.Tensor:
        """
        Args:
            text_token (`torch.LongTensor` of shape `(1, text_sequence_length)`):
                Text token ids.
            text_token_len (`torch.LongTensor` of shape `(1,)`):
                Number of valid text tokens.
            embedding (`torch.FloatTensor` of shape `(1, spk_embed_dim)`):
                Speaker x-vector embedding.
            prompt_speech_token (`torch.LongTensor` of shape `(1, prompt_length)`, *optional*):
                Previously generated speech tokens used to prime decoding.

        Returns:
            `torch.FloatTensor` of shape `(1, num_samples)`: The synthesized waveform.
        """
        speech_token = self.llm.generate(text_token, text_token_len, embedding, prompt_speech_token, **kwargs)
        speech_token_len = torch.tensor([speech_token.size(1)], device=speech_token.device)
        flow_out = self.flow(speech_token, speech_token_len, embedding)
        return self.hift(flow_out.mel)


class CosyVoiceV1ForConditionalGeneration(CosyVoiceV1PreTrainedModel):
    r"""
    CosyVoice v1 model with the speech-token language model, the flow-matching decoder, and the vocoder, with a
    `generate` method producing a waveform end-to-end. The trainable `forward` pass is the speech-token language
    model's next-token cross-entropy objective; the flow-matching decoder and vocoder are trained separately via
    [`CosyVoiceV1FlowMatchingModel`] and [`CosyVoiceV1HiFTGenerator`].
    """

    def __init__(self, config: CosyVoiceV1Config):
        super().__init__(config)
        self.model = CosyVoiceV1Model(config)
        self.post_init()

    def forward(self, **kwargs) -> CosyVoiceV1LLMOutput:
        return self.model.llm(**kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """See [`~CosyVoiceV1Model.generate_speech`]."""
        return self.model.generate_speech(*args, **kwargs)


__all__ = [
    "CosyVoiceV1ForConditionalGeneration",
    "CosyVoiceV1Model",
    "CosyVoiceV1LLM",
    "CosyVoiceV1LLMOutput",
    "CosyVoiceV1FlowMatchingModel",
    "CosyVoiceV1HiFTGenerator",
    "CosyVoiceV1PreTrainedModel",
    "CosyVoiceV1TextEncoder",
    "CosyVoiceV1RelPositionEncoder",
    "CosyVoiceV1ConditionalDecoder",
    "CosyVoiceV1ConditionalCFM",
    "CosyVoiceV1InterpolateRegulator",
]
