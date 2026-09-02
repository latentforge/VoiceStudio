# Copyright 2026 Nari Labs and the LatentForge team. All rights reserved.
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
"""PyTorch Dia2 model."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging

from .configuration_dia2 import Dia2Config, Dia2DepthDecoderConfig
from .generation_dia2 import Dia2GenerationMixin


logger = logging.get_logger(__name__)


class Dia2RMSNorm(LlamaRMSNorm):
    pass


class Dia2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Dia2MLP(LlamaMLP):
    pass


class Dia2Attention(Qwen3Attention):
    def __init__(self, config: Dia2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Dia2 scores attention on the raw query/key dot product, with no 1/sqrt(head_dim) factor.
        self.scaling = 1.0


class Dia2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Dia2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Dia2Attention(config, layer_idx)


class Dia2BackboneEmbeddings(nn.Module):
    r"""
    Embeds one frame of Dia2's `2 + num_codebooks` input channels into a single backbone hidden state.

    The two text channels share one embedding table but own a projection each, and the second channel only
    contributes on frames where it does not hold `config.text_pad_token_id`. Every codebook channel owns a slice
    of one shared audio table, selected by a per-codebook offset, and all channels are summed.

    Args:
        config ([`Dia2Config`]):
            Configuration of the parent model.
    """

    def __init__(self, config: Dia2Config):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.text_pad_token_id = config.text_pad_token_id
        rank = config.text_low_rank_dim or config.hidden_size

        self.embed_text_tokens = nn.Embedding(config.text_vocab_size, rank)
        self.text_stream_proj = nn.Linear(rank, config.hidden_size, bias=False)
        self.second_text_stream_proj = nn.Linear(rank, config.hidden_size, bias=False)
        self.embed_audio_tokens = nn.Embedding(config.num_codebooks * config.vocab_size, config.hidden_size)
        self.audio_tokens_offsets = nn.Buffer(
            torch.arange(config.num_codebooks) * config.vocab_size, persistent=False
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        text_ids, second_text_ids = input_ids[..., 0], input_ids[..., 1]
        audio_ids = input_ids[..., 2:]

        inputs_embeds = self.text_stream_proj(self.embed_text_tokens(text_ids))
        second_embeds = self.second_text_stream_proj(self.embed_text_tokens(second_text_ids))
        second_is_set = (second_text_ids != self.text_pad_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds + torch.where(second_is_set, second_embeds, 0.0)

        audio_embeds = self.embed_audio_tokens(audio_ids + self.audio_tokens_offsets.to(audio_ids.device))
        return inputs_embeds + audio_embeds.sum(dim=-2)


def grouped_projection(
    projections: nn.ModuleList, hidden_states: torch.Tensor, weight_group_ids: list[int]
) -> torch.Tensor:
    r"""
    Applies a different projection to each position of a sequence.

    Args:
        projections (`nn.ModuleList`):
            One [`nn.Linear`] per weight group.
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_dim)`):
            Sequence to project.
        weight_group_ids (`list[int]`):
            Weight group of each of the `sequence_length` positions.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, sequence_length, output_dim)`: The projected sequence.
    """
    if len(set(weight_group_ids)) == 1:
        return projections[weight_group_ids[0]](hidden_states)

    positions_per_group: dict[int, list[int]] = {}
    for position, group in enumerate(weight_group_ids):
        positions_per_group.setdefault(group, []).append(position)

    projected = None
    for group, positions in positions_per_group.items():
        index = torch.tensor(positions, dtype=torch.long, device=hidden_states.device)
        chunk = projections[group](hidden_states.index_select(1, index))
        if projected is None:
            projected = chunk.new_zeros(*hidden_states.shape[:2], chunk.shape[-1])
        projected = projected.index_copy(1, index, chunk)
    return projected


class Dia2DepthAttention(nn.Module):
    r"""
    Self-attention over the codebooks of a single frame, whose projections are selected per position.

    `config.weights_schedule[i]` picks which of `config.num_weight_groups` projection sets position `i` uses, so
    early codebooks can be given their own capacity while later ones share it.

    Args:
        config ([`Dia2DepthDecoderConfig`]):
            Configuration of the depth decoder.
        layer_idx (`int`):
            Index of the layer this attention belongs to, used to address the key/value cache.
    """

    def __init__(self, config: Dia2DepthDecoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        # Same convention as [`Dia2Attention`]: no 1/sqrt(head_dim) factor on the attention logits.
        self.scaling = 1.0

        num_groups = config.num_weight_groups
        query_dim = config.num_attention_heads * config.head_dim
        key_value_dim = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.ModuleList(
            [nn.Linear(config.hidden_size, query_dim, bias=config.attention_bias) for _ in range(num_groups)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(config.hidden_size, key_value_dim, bias=config.attention_bias) for _ in range(num_groups)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(config.hidden_size, key_value_dim, bias=config.attention_bias) for _ in range(num_groups)]
        )
        self.o_proj = nn.ModuleList(
            [nn.Linear(query_dim, config.hidden_size, bias=config.attention_bias) for _ in range(num_groups)]
        )
        self.q_norm = Dia2RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Dia2RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight_group_ids: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = grouped_projection(self.q_proj, hidden_states, weight_group_ids).view(hidden_shape)
        key_states = grouped_projection(self.k_proj, hidden_states, weight_group_ids).view(hidden_shape)
        value_states = grouped_projection(self.v_proj, hidden_states, weight_group_ids).view(hidden_shape)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return grouped_projection(self.o_proj, attn_output, weight_group_ids), attn_weights


class Dia2DepthDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Dia2DepthDecoderConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Dia2DepthAttention(config, layer_idx)
        self.mlp = Dia2MLP(config)
        self.input_layernorm = Dia2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Dia2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight_group_ids: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            weight_group_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Dia2CodebooksHead(nn.Module):
    r"""
    Position-specific language modeling head: depth decoder position `i` predicts codebook `i + 1` through its
    own output projection.

    Args:
        hidden_size (`int`):
            Dimensionality of the depth decoder's hidden states.
        num_codebooks (`int`):
            Number of audio codebooks per frame.
        vocab_size (`int`):
            Size of a single codebook's vocabulary.
    """

    def __init__(self, hidden_size: int, num_codebooks: int, vocab_size: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.weight = nn.Parameter(torch.empty(num_codebooks - 1, hidden_size, vocab_size))

    def forward(self, hidden_states: torch.Tensor, codebook_indices: list[int]) -> torch.Tensor:
        logits = [
            nn.functional.linear(hidden_states[:, position, :], self.weight[codebook].T)
            for position, codebook in enumerate(codebook_indices)
        ]
        return torch.stack(logits, dim=1)


@auto_docstring
class Dia2PreTrainedModel(PreTrainedModel):
    config: Dia2Config
    base_model_prefix = "backbone_model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dia2DecoderLayer", "Dia2DepthDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Dia2DecoderLayer,
        "attentions": Dia2Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Dia2CodebooksHead):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Dia2BackboneEmbeddings):
            init.copy_(
                module.audio_tokens_offsets, torch.arange(self.config.num_codebooks) * self.config.vocab_size
            )
        elif isinstance(module, Dia2DepthDecoderModel):
            init.copy_(
                module.audio_tokens_offsets,
                torch.arange(module.config.num_codebooks - 1) * module.config.vocab_size,
            )


@auto_docstring(
    custom_intro="""
    The Dia2 backbone: a decoder-only transformer over frames whose channels are the two text streams and the
    audio codebooks of the delayed codebook grid.
    """
)
class Dia2BackboneModel(LlamaModel, Dia2PreTrainedModel):
    config: Dia2Config
    _can_record_outputs = {
        "hidden_states": Dia2DecoderLayer,
        "attentions": Dia2Attention,
    }

    def __init__(self, config: Dia2Config):
        super().__init__(config)
        self.embed_tokens = Dia2BackboneEmbeddings(config)
        self.layers = nn.ModuleList(
            [Dia2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Dia2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Dia2RotaryEmbedding(config=config)

        self.post_init()

    def forward(self, **super_kwargs) -> BaseModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, 2 + num_codebooks)`):
            One frame per sequence position. Channel 0 is the main text stream, channel 1 the second text
            stream, and channels 2 onwards the delayed audio codebooks. Can be obtained from [`Dia2Processor`]
            and [`~generation_dia2.apply_delay_pattern`].
        """
        return super().forward(**super_kwargs)


@auto_docstring(
    custom_intro="""
    The Dia2 depth decoder: a decoder-only transformer over the codebooks of a single frame, conditioned on that
    frame's backbone hidden state.
    """
)
class Dia2DepthDecoderModel(Dia2PreTrainedModel):
    config: Dia2DepthDecoderConfig
    _can_record_outputs = {
        "hidden_states": Dia2DepthDecoderLayer,
        "attentions": Dia2DepthAttention,
    }

    def __init__(self, config: Dia2DepthDecoderConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding((config.num_codebooks - 1) * config.vocab_size, config.hidden_size)
        self.audio_tokens_offsets = nn.Buffer(
            torch.arange(config.num_codebooks - 1) * config.vocab_size, persistent=False
        )
        self.inputs_embeds_projector = nn.ModuleList(
            [
                nn.Linear(config.backbone_hidden_size, config.hidden_size, bias=False)
                for _ in range(config.num_weight_groups)
            ]
        )
        self.embed_text_tokens = None
        self.text_stream_proj = None
        self.second_text_stream_proj = None
        if config.use_text_embedding:
            self.embed_text_tokens = nn.Embedding(config.text_vocab_size, config.hidden_size)
            self.text_stream_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_text_stream_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.layers = nn.ModuleList(
            [Dia2DepthDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Dia2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Dia2RotaryEmbedding(config=config) if config.use_rope else None
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        backbone_hidden_states: torch.FloatTensor,
        text_input_ids: torch.LongTensor | None = None,
        second_text_input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Codebook tokens already known for this frame. Position `i` holds codebook `i`, so the hidden state
            at position `i` predicts codebook `i + 1`.
        backbone_hidden_states (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`):
            Last hidden state the [`Dia2BackboneModel`] produced for the frame preceding this one.
        text_input_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Main text stream token of this frame. Only used when `config.use_text_embedding` is `True`.
        second_text_input_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Second text stream token of this frame. Only used when `config.use_text_embedding` is `True`.

        Returns:
            [`~modeling_outputs.BaseModelOutputWithPast`]: Hidden states of the requested codebook positions.
        """
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        sequence_length = input_ids.shape[1]
        codebook_indices = list(range(past_seen_tokens, past_seen_tokens + sequence_length))
        weight_group_ids = [self.config.weights_schedule[index] for index in codebook_indices]

        offsets = self.audio_tokens_offsets.to(input_ids.device)[codebook_indices]
        inputs_embeds = self.embed_tokens(input_ids + offsets)

        if self.embed_text_tokens is not None and codebook_indices[0] == 0:
            if text_input_ids is None or second_text_input_ids is None:
                raise ValueError(
                    "`text_input_ids` and `second_text_input_ids` are required when `config.use_text_embedding` "
                    "is set and codebook 0 is decoded."
                )
            text_embeds = self.text_stream_proj(self.embed_text_tokens(text_input_ids))
            second_embeds = self.second_text_stream_proj(self.embed_text_tokens(second_text_input_ids))
            second_is_set = (second_text_input_ids != self.config.text_pad_token_id).unsqueeze(-1)
            inputs_embeds[:, 0] = inputs_embeds[:, 0] + text_embeds + torch.where(second_is_set, second_embeds, 0.0)

        conditioning = backbone_hidden_states.unsqueeze(1).expand(-1, sequence_length, -1)
        hidden_states = inputs_embeds + grouped_projection(
            self.inputs_embeds_projector, conditioning, weight_group_ids
        )

        position_ids = torch.tensor([codebook_indices], device=input_ids.device)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = (
            self.rotary_emb(hidden_states, position_ids=position_ids) if self.rotary_emb is not None else None
        )

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                weight_group_ids,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring(
    custom_intro="""
    The Dia2 depth decoder with a [`Dia2CodebooksHead`] on top.
    """
)
class Dia2DepthDecoderForCausalLM(Dia2PreTrainedModel):
    config: Dia2DepthDecoderConfig
    _tied_weights_keys = None

    def __init__(self, config: Dia2DepthDecoderConfig):
        super().__init__(config)
        self.model = Dia2DepthDecoderModel(config)
        self.codebooks_head = Dia2CodebooksHead(config.hidden_size, config.num_codebooks, config.vocab_size)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        backbone_hidden_states: torch.FloatTensor,
        text_input_ids: torch.LongTensor | None = None,
        second_text_input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Codebook tokens already known for this frame. Position `i` holds codebook `i`, so the logits at
            position `i` predict codebook `i + 1`.
        backbone_hidden_states (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`):
            Last hidden state the [`Dia2BackboneModel`] produced for the frame preceding this one.
        text_input_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Main text stream token of this frame. Only used when `config.use_text_embedding` is `True`.
        second_text_input_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Second text stream token of this frame. Only used when `config.use_text_embedding` is `True`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Target codebook token of each position, i.e. `labels[:, i]` holds codebook `i + 1`. Indices set to
            `-100` are ignored by the loss.

        Returns:
            [`~modeling_outputs.CausalLMOutputWithPast`]: Codebook logits and, when `labels` is given, the
            cross-entropy loss over them.
        """
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        codebook_indices = list(range(past_seen_tokens, past_seen_tokens + input_ids.shape[1]))

        outputs = self.model(
            input_ids=input_ids,
            backbone_hidden_states=backbone_hidden_states,
            text_input_ids=text_input_ids,
            second_text_input_ids=second_text_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        logits = self.codebooks_head(outputs.last_hidden_state, codebook_indices).contiguous()

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=None, vocab_size=self.config.vocab_size, shift_labels=labels, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Base class for [`Dia2ForConditionalGeneration`] outputs.
    """
)
@dataclass
class Dia2OutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Sum of `backbone_loss`, `action_loss` and `depth_decoder_loss`.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
        Prediction scores of the first codebook of the next frame.
    action_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_vocab_size)`):
        Prediction scores of the word-advance action of the next frame.
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed):
        Key/value states of the backbone, usable to speed up sequential decoding.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
        Backbone hidden states at the output of each layer plus the initial embedding outputs.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
        Backbone attention weights of each layer.
    backbone_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Cross-entropy loss of the first codebook.
    action_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `action_labels` is provided):
        Cross-entropy loss of the word-advance action.
    depth_decoder_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Cross-entropy loss of the remaining codebooks, as computed by the depth decoder.
    depth_decoder_logits (`torch.FloatTensor` of shape `(num_frames, num_codebooks - 1, vocab_size)`, *optional*):
        Prediction scores of the remaining codebooks, over the frames the depth decoder was run on.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    action_logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    backbone_loss: torch.FloatTensor | None = None
    action_loss: torch.FloatTensor | None = None
    depth_decoder_loss: torch.FloatTensor | None = None
    depth_decoder_logits: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Dia2 pairs a decoder-only backbone that predicts a word-advance action and the first Mimi codebook of every
    frame with a depth decoder that predicts the frame's remaining codebooks.
    """
)
class Dia2ForConditionalGeneration(Dia2PreTrainedModel, Dia2GenerationMixin):
    _tied_weights_keys = None

    def __init__(self, config: Dia2Config):
        super().__init__(config)
        self.backbone_model = Dia2BackboneModel._from_config(config)
        self.depth_decoder = Dia2DepthDecoderForCausalLM._from_config(config.depth_decoder_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.action_head = nn.Linear(config.hidden_size, config.action_vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone_model.embed_tokens

    def set_input_embeddings(self, value):
        self.backbone_model.embed_tokens = value

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
        action_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Dia2OutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, 2 + num_codebooks)`):
            One frame per sequence position. Channel 0 is the main text stream, channel 1 the second text
            stream, and channels 2 onwards the delayed audio codebooks.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
            Target codebook tokens on the same delayed grid as `input_ids`, aligned frame for frame. They are
            shifted internally, so frame `t` is trained to predict frame `t + 1`. Indices set to `-100` are
            ignored by the loss; a frame whose codebooks `1 ..` are all `-100` is skipped by the depth decoder.
        action_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Target word-advance action, aligned with `input_ids` frame for frame and shifted internally.
            Indices set to `-100` are ignored by the loss.
        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            Number of trailing frames to compute logits for. `0` computes all of them.

        Returns:
            [`Dia2OutputWithPast`]: Codebook and action logits and, when labels are provided, their losses.

        Example:

        ```python
        >>> import torch
        >>> from voicestudio.models.dia2 import Dia2Config, Dia2ForConditionalGeneration

        >>> model = Dia2ForConditionalGeneration(Dia2Config())
        >>> frames = torch.zeros(1, 8, model.config.num_channels, dtype=torch.long)
        >>> labels = torch.zeros(1, 8, model.config.num_codebooks, dtype=torch.long)
        >>> action_labels = torch.zeros(1, 8, dtype=torch.long)
        >>> outputs = model(input_ids=frames, labels=labels, action_labels=action_labels)
        >>> outputs.loss.backward()
        ```"""
        outputs = self.backbone_model(
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
        action_logits = self.action_head(hidden_states[:, slice_indices, :])

        loss = None
        backbone_loss = None
        action_loss = None
        depth_decoder_loss = None
        depth_decoder_logits = None

        if action_labels is not None:
            action_loss = self.loss_function(
                logits=action_logits, labels=action_labels, vocab_size=self.config.action_vocab_size, **kwargs
            )
            loss = action_loss

        if labels is not None:
            backbone_loss = self.loss_function(
                logits=logits, labels=labels[..., 0], vocab_size=self.config.vocab_size, **kwargs
            )
            loss = backbone_loss if loss is None else loss + backbone_loss

            depth_decoder_outputs = self._forward_depth_decoder(input_ids, hidden_states, labels, **kwargs)
            if depth_decoder_outputs is not None:
                depth_decoder_loss = depth_decoder_outputs.loss
                depth_decoder_logits = depth_decoder_outputs.logits
                loss = loss + depth_decoder_loss

        return Dia2OutputWithPast(
            loss=loss,
            logits=logits,
            action_logits=action_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            backbone_loss=backbone_loss,
            action_loss=action_loss,
            depth_decoder_loss=depth_decoder_loss,
            depth_decoder_logits=depth_decoder_logits,
        )

    def _forward_depth_decoder(
        self,
        input_ids: torch.LongTensor | None,
        hidden_states: torch.FloatTensor,
        labels: torch.LongTensor,
        **kwargs,
    ) -> CausalLMOutputWithPast | None:
        """
        Teacher-forces the depth decoder over every frame whose codebooks `1 ..` are not entirely masked out.

        Frame `t`'s backbone hidden state conditions the prediction of frame `t + 1`'s codebooks `1 ..` from that
        same frame's codebooks `0 ..`, which is the pairing the decoding loop produces at generation time.
        """
        num_codebooks = self.config.num_codebooks
        target_labels = labels[:, 1:]
        conditioning = hidden_states[:, :-1]
        if target_labels.shape[1] == 0:
            return None

        train_mask = ~(target_labels[..., 1:] == -100).all(dim=-1)
        if not train_mask.any():
            return None

        depth_input_ids = target_labels[train_mask][..., : num_codebooks - 1].clamp(min=0).contiguous()
        depth_labels = target_labels[train_mask][..., 1:].contiguous()

        text_input_ids = None
        second_text_input_ids = None
        if self.depth_decoder.config.use_text_embedding and input_ids is not None:
            text_input_ids = input_ids[:, 1:, 0][train_mask]
            second_text_input_ids = input_ids[:, 1:, 1][train_mask]

        return self.depth_decoder(
            input_ids=depth_input_ids,
            backbone_hidden_states=conditioning[train_mask],
            text_input_ids=text_input_ids,
            second_text_input_ids=second_text_input_ids,
            labels=depth_labels,
            use_cache=False,
            **kwargs,
        )


__all__ = [
    "Dia2BackboneModel",
    "Dia2DepthDecoderForCausalLM",
    "Dia2DepthDecoderModel",
    "Dia2ForConditionalGeneration",
    "Dia2PreTrainedModel",
]
