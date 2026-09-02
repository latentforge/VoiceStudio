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
"""PyTorch CosyVoice v2 model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerRelPositionalEmbedding

from ..cosyvoice_v1.modeling_cosyvoice_v1 import (
    CosyVoiceV1ConditionalCFM,
    CosyVoiceV1ConditionalDecoder,
    CosyVoiceV1EncoderLayer,
    CosyVoiceV1HiFTGenerator,
    CosyVoiceV1PreTrainedModel,
    _conformer_config,
    _make_pad_mask,
)
from .configuration_cosyvoice_v2 import CosyVoiceV2Config, CosyVoiceV2FlowConfig, CosyVoiceV2LLMConfig


@dataclass
class CosyVoiceV2LLMOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Speech-token cross-entropy loss, returned when `labels` is given.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, speech_token_size + 3)`):
            Prediction scores over the speech-token vocabulary (plus the start/task and end-of-speech tokens).
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None


class CosyVoiceV2LLM(CosyVoiceV1PreTrainedModel):
    r"""
    The CosyVoice v2 speech-token language model. Repurposes a pretrained [`Qwen2Model`] text backbone to
    autoregressively predict discrete speech tokens: text tokens are embedded with the Qwen2 embedding table, and
    speech tokens are embedded with a separate embedding table that also holds the start/task and end-of-speech
    ids.
    """

    config_class = CosyVoiceV2LLMConfig
    _keys_to_ignore_on_load_unexpected = ["llm.lm_head.weight"]

    def __init__(self, config: CosyVoiceV2LLMConfig):
        super().__init__(config)
        self.speech_token_size = config.speech_token_size
        self.sos_eos = 0
        self.task_id = 1
        self.eos_token_id = config.speech_token_size
        self.fill_token_id = config.speech_token_size + 2

        self.llm = Qwen2Model(config)
        self.llm_embedding = nn.Embedding(2, config.hidden_size)
        self.llm_decoder = nn.Linear(config.hidden_size, config.speech_token_size + 3)
        self.speech_embedding = nn.Embedding(config.speech_token_size + 3, config.hidden_size)
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
        sos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1).expand(text_emb.size(0), -1, -1)
        task_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1).expand(text_emb.size(0), -1, -1)

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
        """
        Autoregressively samples discrete speech tokens conditioned on `text_token`.

        Args:
            text_token (`torch.LongTensor` of shape `(1, text_sequence_length)`):
                Text token ids.
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
        text_emb = self.llm.embed_tokens(text_token)
        sos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
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


class CosyVoiceV2PreLookaheadLayer(nn.Module):
    """Convolves each mel-encoder frame with a small window of future frames before the causal Conformer
    encoder, giving the causal flow decoder limited lookahead without breaking streaming.

    Args:
        channels (`int`):
            Number of channels in the input and output hidden states.
        pre_lookahead_len (`int`):
            Number of future frames convolved into each frame's representation.
    """

    def __init__(self, channels: int, pre_lookahead_len: int):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, pre_lookahead_len + 1)
        self.conv2 = nn.Conv1d(channels, channels, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.pad(hidden_states, (0, self.pre_lookahead_len))
        hidden_states = F.leaky_relu(self.conv1(hidden_states))
        hidden_states = F.pad(hidden_states, (2, 0))
        hidden_states = self.conv2(hidden_states)
        return hidden_states.transpose(1, 2) + residual


class CosyVoiceV2CausalConditionalDecoder(CosyVoiceV1ConditionalDecoder):
    r"""Same U-Net estimator as [`CosyVoiceV1ConditionalDecoder`]."""


class CosyVoiceV2Upsample1D(nn.Module):
    r"""Nearest-neighbor upsampling followed by a causal convolution, matching the original CosyVoice v2
    `Upsample1D`: interpolates the time axis by `stride`, left-pads by `2 * stride`, then applies a single
    `Conv1d` with kernel size `2 * stride + 1` and no built-in padding, so every output frame only depends on
    past and present input frames.

    Args:
        channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        stride (`int`, *optional*, defaults to 2):
            Upsampling factor.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(channels, out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = F.interpolate(hidden_states, scale_factor=float(self.stride), mode="nearest")
        hidden_states = F.pad(hidden_states, (self.stride * 2, 0), value=0.0)
        hidden_states = self.conv(hidden_states)
        return hidden_states, lengths * self.stride


class CosyVoiceV2UpsampleConformerEncoder(nn.Module):
    r"""
    The CosyVoice v2 flow-matching token encoder, matching the original `UpsampleConformerEncoder`: a
    [`CosyVoiceV2PreLookaheadLayer`], a stack of `encoder_num_hidden_layers` [`CosyVoiceV1EncoderLayer`]s at the
    input token rate, a [`CosyVoiceV2Upsample1D`] that doubles the sequence length, and a second, independently
    embedded and positionally encoded stack of 4 more [`CosyVoiceV1EncoderLayer`]s at the upsampled (mel) rate,
    followed by a final `LayerNorm`. Unlike [`CosyVoiceV1RelPositionEncoder`], the final norm is applied only
    once, after the second stack, not after each stack.
    """

    def __init__(self, config: CosyVoiceV2FlowConfig):
        super().__init__()
        conformer_config = _conformer_config(
            config.encoder_hidden_size,
            config.encoder_num_attention_heads,
            config.encoder_intermediate_size,
            0.1,
            0.1,
        )
        self.pre_lookahead_layer = CosyVoiceV2PreLookaheadLayer(config.encoder_hidden_size, config.pre_lookahead_len)
        self.embed = nn.Sequential(
            nn.Linear(config.input_size, config.encoder_hidden_size), nn.LayerNorm(config.encoder_hidden_size)
        )
        self.embed_dropout = nn.Dropout(0.1)
        self.embed_positions = Wav2Vec2ConformerRelPositionalEmbedding(conformer_config)
        self.layers = nn.ModuleList(
            [CosyVoiceV1EncoderLayer(conformer_config) for _ in range(config.encoder_num_hidden_layers)]
        )
        self.up_layer = CosyVoiceV2Upsample1D(config.encoder_hidden_size, config.encoder_hidden_size, stride=2)
        self.up_embed = nn.Sequential(
            nn.Linear(config.input_size, config.encoder_hidden_size), nn.LayerNorm(config.encoder_hidden_size)
        )
        self.up_embed_dropout = nn.Dropout(0.1)
        self.up_embed_positions = Wav2Vec2ConformerRelPositionalEmbedding(conformer_config)
        self.up_layers = nn.ModuleList([CosyVoiceV1EncoderLayer(conformer_config) for _ in range(4)])
        self.after_norm = nn.LayerNorm(config.encoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor, lengths: torch.LongTensor) -> tuple[torch.Tensor, torch.LongTensor]:
        mask = ~_make_pad_mask(lengths, hidden_states.size(1))
        hidden_states = self.embed(hidden_states)
        hidden_states = hidden_states.masked_fill(~mask.unsqueeze(-1), 0.0)
        hidden_states = self.embed_dropout(hidden_states)
        pos_emb = self.embed_positions(hidden_states)
        hidden_states = self.pre_lookahead_layer(hidden_states)

        attention_bias = mask[:, None, None, :].to(hidden_states.dtype)
        attention_bias = (1.0 - attention_bias) * torch.finfo(hidden_states.dtype).min
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_bias, relative_position_embeddings=pos_emb)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states, lengths = self.up_layer(hidden_states, lengths)
        hidden_states = hidden_states.transpose(1, 2)
        mask = ~_make_pad_mask(lengths, hidden_states.size(1))
        hidden_states = self.up_embed(hidden_states)
        hidden_states = hidden_states.masked_fill(~mask.unsqueeze(-1), 0.0)
        hidden_states = self.up_embed_dropout(hidden_states)
        pos_emb = self.up_embed_positions(hidden_states)

        attention_bias = mask[:, None, None, :].to(hidden_states.dtype)
        attention_bias = (1.0 - attention_bias) * torch.finfo(hidden_states.dtype).min
        for layer in self.up_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_bias, relative_position_embeddings=pos_emb)

        hidden_states = self.after_norm(hidden_states)
        return hidden_states, lengths


class CosyVoiceV2FlowMatchingModel(CosyVoiceV1PreTrainedModel):
    r"""
    The CosyVoice v2 conditional-flow-matching decoder. Differs from [`CosyVoiceV1FlowMatchingModel`] by using a
    [`CosyVoiceV2UpsampleConformerEncoder`], which folds the pre-lookahead convolution and a fixed 2x
    nearest-neighbor upsampling stage into the token encoder itself, instead of a plain bidirectional Conformer
    followed by an interpolate-to-target-length regulator.
    """

    config_class = CosyVoiceV2FlowConfig

    def __init__(self, config: CosyVoiceV2FlowConfig):
        super().__init__(config)
        self.input_embedding = nn.Embedding(config.vocab_size, config.input_size)
        self.spk_embed_affine_layer = nn.Linear(config.spk_embed_dim, config.output_size)
        self.encoder = CosyVoiceV2UpsampleConformerEncoder(config)
        self.encoder_proj = nn.Linear(config.encoder_hidden_size, config.output_size)
        self.decoder = CosyVoiceV1ConditionalCFM(config, CosyVoiceV2CausalConditionalDecoder(config))
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
                Number of valid mel frames per example. Unused: the encoder derives the output length from
                `speech_token_len` through its fixed 2x upsampling, matching the original training assumption
                that dataset mel lengths equal `speech_token_len * 2`.
            n_timesteps (`int`, *optional*, defaults to 10):
                Number of Euler integration steps used when sampling (`speech_feat` is `None`).

        Returns:
            [`~transformers.utils.ModelOutput`] with `mel` (sampled or `None`) and `loss` (computed or `None`).
        """
        spk_emb = F.normalize(embedding, dim=1)
        spk_emb = self.spk_embed_affine_layer(spk_emb)

        mask = (~_make_pad_mask(speech_token_len, speech_token.size(1))).to(spk_emb.dtype).unsqueeze(-1)
        token_emb = self.input_embedding(speech_token.clamp(min=0)) * mask
        hidden_states, out_lengths = self.encoder(token_emb, speech_token_len)
        hidden_states = self.encoder_proj(hidden_states)

        loss = None
        mel = None
        if speech_feat is not None:
            cond = torch.zeros_like(speech_feat).transpose(1, 2)
            out_mask = (~_make_pad_mask(out_lengths, speech_feat.size(1))).to(hidden_states).unsqueeze(1)
            loss = self.decoder.compute_loss(
                speech_feat.transpose(1, 2), out_mask, hidden_states.transpose(1, 2), spk_emb, cond
            )
        else:
            out_mask = (~_make_pad_mask(out_lengths, hidden_states.size(1))).to(hidden_states).unsqueeze(1)
            cond = torch.zeros(hidden_states.size(0), self.config.output_size, hidden_states.size(1), device=hidden_states.device, dtype=hidden_states.dtype)
            mel = self.decoder(hidden_states.transpose(1, 2), out_mask, spk_emb, cond, n_timesteps=n_timesteps)
        return ModelOutput(mel=mel, loss=loss)


class CosyVoiceV2Model(CosyVoiceV1PreTrainedModel):
    r"""
    The full CosyVoice v2 model: a Qwen2-backbone speech-token language model, a causal conditional-flow-matching
    mel decoder, and the same HiFTNet vocoder as CosyVoice v1.
    """

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        self.llm = CosyVoiceV2LLM(config.llm_config)
        self.flow = CosyVoiceV2FlowMatchingModel(config.flow_config)
        self.hift = CosyVoiceV1HiFTGenerator(config.hift_config)
        self.post_init()

    def forward(self, **kwargs) -> CosyVoiceV2LLMOutput:
        return self.llm(**kwargs)

    @torch.no_grad()
    def generate_speech(self, text_token, embedding, prompt_speech_token=None, **kwargs) -> torch.Tensor:
        """
        Args:
            text_token (`torch.LongTensor` of shape `(1, text_sequence_length)`):
                Text token ids.
            embedding (`torch.FloatTensor` of shape `(1, spk_embed_dim)`):
                Speaker x-vector embedding.
            prompt_speech_token (`torch.LongTensor` of shape `(1, prompt_length)`, *optional*):
                Previously generated speech tokens used to prime decoding.

        Returns:
            `torch.FloatTensor` of shape `(1, num_samples)`: The synthesized waveform.
        """
        speech_token = self.llm.generate(text_token, prompt_speech_token, **kwargs)
        speech_token_len = torch.tensor([speech_token.size(1)], device=speech_token.device)
        flow_out = self.flow(speech_token, speech_token_len, embedding)
        return self.hift(flow_out.mel)


class CosyVoiceV2ForConditionalGeneration(CosyVoiceV1PreTrainedModel):
    r"""
    CosyVoice v2 model with the Qwen2-backbone speech-token language model, the causal flow-matching decoder, and
    the vocoder, with a `generate` method producing a waveform end-to-end. The trainable `forward` pass is the
    speech-token language model's next-token cross-entropy objective.
    """

    config_class = CosyVoiceV2Config

    def __init__(self, config: CosyVoiceV2Config):
        super().__init__(config)
        self.model = CosyVoiceV2Model(config)
        self.post_init()

    def forward(self, **kwargs) -> CosyVoiceV2LLMOutput:
        return self.model.llm(**kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """See [`~CosyVoiceV2Model.generate_speech`]."""
        return self.model.generate_speech(*args, **kwargs)


__all__ = [
    "CosyVoiceV2ForConditionalGeneration",
    "CosyVoiceV2Model",
    "CosyVoiceV2LLM",
    "CosyVoiceV2LLMOutput",
    "CosyVoiceV2FlowMatchingModel",
    "CosyVoiceV2CausalConditionalDecoder",
    "CosyVoiceV2PreLookaheadLayer",
    "CosyVoiceV2Upsample1D",
    "CosyVoiceV2UpsampleConformerEncoder",
]
