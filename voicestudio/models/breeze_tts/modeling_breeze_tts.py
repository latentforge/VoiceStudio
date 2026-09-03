# Copyright 2026 RESONIA, INC., Sesame, The HuggingFace Inc. team and the LatentForge team.
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
"""PyTorch Breeze TTS 2 model."""

from dataclasses import dataclass

import torch
from torch import nn

from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.auto import AutoModel
from transformers.models.csm.modeling_csm import (
    CsmBackboneModelEmbeddings,
    CsmDepthDecoderForCausalLM,
    CsmDepthDecoderModel,
    CsmForConditionalGeneration,
    CsmOutputWithPast,
    CsmPreTrainedModel,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging

from .configuration_breeze_tts import BreezeTTSConfig, BreezeTTSDepthDecoderConfig
from .generation_breeze_tts import BreezeTTSGenerationMixin


logger = logging.get_logger(__name__)


# Encoder classes `AutoModel` cannot reach: `AutoModel.register` skips config classes that live in
# `transformers` itself, so a text encoder whose `model_type` has no `MODEL_MAPPING_NAMES` entry is looked up
# here instead.
TEXT_ENCODER_CLASSES = {"t5gemma2_text": T5Gemma2TextEncoder, "t5_gemma_module": T5GemmaEncoder}


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Breeze TTS 2 autoregressive outputs.
    """
)
class BreezeTTSOutputWithPast(CsmOutputWithPast):
    pass


class BreezeTTSPreTrainedModel(CsmPreTrainedModel):
    config: BreezeTTSConfig
    _no_split_modules = ["Qwen3DecoderLayer", "CsmDecoderLayer", "T5Gemma2EncoderLayer"]


@auto_docstring
class BreezeTTSDepthDecoderModel(CsmDepthDecoderModel):
    config: BreezeTTSDepthDecoderConfig

    def __init__(self, config: BreezeTTSDepthDecoderConfig):
        super().__init__(config)
        self.audio_embed_size = config.audio_embed_size
        if self.audio_embed_size != config.backbone_hidden_size:
            self.embed_tokens = nn.Embedding(config.num_codebooks * config.vocab_size, self.audio_embed_size)
            self.inputs_embeds_projector = nn.Linear(self.audio_embed_size, config.hidden_size, bias=False)
            self.backbone_hidden_state_projector = nn.Linear(
                config.backbone_hidden_size, self.audio_embed_size, bias=False
            )
        else:
            self.backbone_hidden_state_projector = None
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        backbone_last_hidden_state: torch.FloatTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        backbone_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`, *optional*):
            The last hidden state of the backbone model, spliced in at depth position 0. Required whenever the
            first codebook token, the one the backbone predicted, is part of `input_ids`.
        """
        if backbone_last_hidden_state is not None and self.backbone_hidden_state_projector is not None:
            backbone_last_hidden_state = self.backbone_hidden_state_projector(backbone_last_hidden_state)
        return super().forward(
            input_ids=input_ids, backbone_last_hidden_state=backbone_last_hidden_state, **kwargs
        )


@auto_docstring(
    custom_intro="""
    The Breeze TTS 2 depth decoder with a position-specific language modeling head on top, one linear layer per
    codebook it predicts.
    """
)
class BreezeTTSDepthDecoderForCausalLM(CsmDepthDecoderForCausalLM):
    config: BreezeTTSDepthDecoderConfig

    def __init__(self, config: BreezeTTSDepthDecoderConfig):
        super().__init__(config)
        self.model = BreezeTTSDepthDecoderModel(config)
        self.post_init()


class BreezeTTSBackboneModelEmbeddings(CsmBackboneModelEmbeddings):
    def __init__(self, config: BreezeTTSConfig):
        nn.Module.__init__(self)
        self.num_codebooks = config.num_codebooks
        self.vocab_size = config.vocab_size
        self.audio_embed_size = config.audio_embed_size
        self.embed_audio_tokens = nn.Embedding(config.num_codebooks * config.vocab_size, self.audio_embed_size)
        if self.audio_embed_size != config.hidden_size:
            self.audio_embeds_projector = nn.Linear(self.audio_embed_size, config.hidden_size, bias=False)
        else:
            self.audio_embeds_projector = None
        self.audio_tokens_offsets = nn.Buffer(
            torch.arange(config.num_codebooks) * config.vocab_size, persistent=False
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        inputs_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        if self.audio_embeds_projector is not None:
            inputs_embeds = self.audio_embeds_projector(inputs_embeds)
        return inputs_embeds.sum(dim=2)


@auto_docstring(
    custom_intro="""
    The Breeze TTS 2 backbone: the decoder layer stack named by `config.backbone_model_type`, reading summed
    multi-codebook audio frame embeddings instead of single-token text embeddings.
    """
)
class BreezeTTSBackboneModel(Qwen3Model):
    config: BreezeTTSConfig

    def __init__(self, config: BreezeTTSConfig):
        super().__init__(config.backbone_config)
        self.embed_tokens = BreezeTTSBackboneModelEmbeddings(config)
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        # `self.config` is the decoder layer stack's config, which carries no codebook dimensions, so the
        # frame embedding offsets are rebuilt from the ones the embedding module kept
        if isinstance(module, BreezeTTSBackboneModelEmbeddings):
            init.copy_(module.audio_tokens_offsets, torch.arange(module.num_codebooks) * module.vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        text_encoder_layer_hidden_states: list[torch.FloatTensor] | None = None,
        text_ids_mask: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`):
            Codebook tokens of one audio frame per sequence position.
        text_encoder_layer_hidden_states (`list[torch.FloatTensor]`, *optional*):
            One projected text encoder hidden state per backbone layer, of shape `(num_text_tokens, hidden_size //
            2)`. Layer `i > 0` overwrites the second half of the hidden state at the `text_ids_mask` positions with
            entry `i`, which is how the `"breeze_dimfusion"` text encoder projection conditions the backbone.
        text_ids_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Positions holding a text token, i.e. the positions `text_encoder_layer_hidden_states` is written to.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if text_encoder_layer_hidden_states is not None and i > 0:
                hidden_states = self._fuse_text_encoder_layer(
                    hidden_states, text_encoder_layer_hidden_states[i], text_ids_mask
                )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _fuse_text_encoder_layer(
        self,
        hidden_states: torch.Tensor,
        text_layer_hidden_states: torch.Tensor,
        text_ids_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Overwrites the second half of `hidden_states` at the `text_ids_mask` positions.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Hidden states entering the next backbone layer.
            text_layer_hidden_states (`torch.Tensor` of shape `(num_text_tokens, hidden_size // 2)`):
                Projected text encoder hidden states of that layer, in `text_ids_mask` row-major order.
            text_ids_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
                Positions holding a text token.

        Returns:
            `torch.Tensor`: the fused hidden states.
        """
        first_half, second_half = hidden_states.split(hidden_states.shape[-1] // 2, dim=-1)
        if self.training:
            second_half = second_half.clone()
        second_half[text_ids_mask] = text_layer_hidden_states
        return torch.cat([first_half, second_half], dim=-1)


@auto_docstring(
    custom_intro="""
    Breeze TTS 2: a backbone predicting the first codebook of every audio frame, a depth decoder predicting the
    remaining codebooks of that frame, and a text encoder embedding the text spans of the prompt.
    """
)
class BreezeTTSForConditionalGeneration(
    BreezeTTSGenerationMixin, CsmForConditionalGeneration, BreezeTTSPreTrainedModel
):
    config: BreezeTTSConfig
    _tied_weights_keys = {
        "backbone_model.embed_tokens.embed_audio_tokens.weight": "depth_decoder.model.embed_tokens.weight"
    }

    def __init__(self, config: BreezeTTSConfig):
        BreezeTTSPreTrainedModel.__init__(self, config)
        self.vocab_size = config.vocab_size
        # The backbone head carries one class on top of a codebook vocabulary, used as the end-of-audio class.
        self.backbone_eos_token_id = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size + 1, bias=False)
        self.embed_text_tokens = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.backbone_model = BreezeTTSBackboneModel(config)
        self.depth_decoder = BreezeTTSDepthDecoderForCausalLM._from_config(config.depth_decoder_config)
        self.codec_model = AutoModel.from_config(config.codec_config)
        self.codec_model.requires_grad_(False)

        self.text_encoder = None
        self.text_encoder_proj = None
        self.text_encoder_layer_projs = None
        self.text_encoder_trainable = False
        if config.text_encoder_config is not None:
            self.text_encoder = self._build_text_encoder(config.text_encoder_config)
            self.text_encoder_proj = self._build_text_encoder_proj(config)
            self.text_encoder_trainable = bool(getattr(config.text_encoder_config, "requires_grad", False))
            if not self.text_encoder_trainable:
                nonzero_dropout_fields = self._get_nonzero_dropout_fields(config.text_encoder_config)
                if nonzero_dropout_fields:
                    formatted_fields = ", ".join(f"{name}={value}" for name, value in nonzero_dropout_fields)
                    raise ValueError(
                        "A frozen text encoder must be deterministic, but its config has non-zero dropout "
                        f"fields: {formatted_fields}."
                    )
            self.text_encoder.requires_grad_(self.text_encoder_trainable)

        # The backbone and the depth decoder are scored over different token counts, so a shared
        # `num_items_in_batch` would normalize one of the two losses by the wrong denominator.
        self.accepts_loss_kwargs = False

        self.post_init()
        self.train(self.training)

    @staticmethod
    def _build_text_encoder(text_encoder_config) -> nn.Module:
        """
        Builds the text encoder embedding the text spans of a prompt.

        Args:
            text_encoder_config ([`PreTrainedConfig`]):
                Configuration of the text encoder.

        Returns:
            `nn.Module`: the text encoder.
        """
        encoder_class = TEXT_ENCODER_CLASSES.get(text_encoder_config.model_type)
        if encoder_class is not None:
            return encoder_class._from_config(text_encoder_config)
        return AutoModel.from_config(text_encoder_config)

    def _build_text_encoder_proj(self, config: BreezeTTSConfig) -> nn.Module:
        """
        Builds the module projecting text encoder hidden states to `config.hidden_size`.

        Args:
            config ([`BreezeTTSConfig`]):
                Configuration naming the projection through `text_encoder_proj_type`.

        Returns:
            `nn.Module`: the projection module.

        Raises:
            ValueError: if `text_encoder_proj_type` names no known projection.
        """
        text_encoder_hidden_size = config.text_encoder_config.hidden_size
        proj_type = config.text_encoder_proj_type
        if proj_type == "linear":
            return nn.Linear(text_encoder_hidden_size, config.hidden_size, bias=False)
        if proj_type == "mlp":
            return nn.Sequential(
                nn.Linear(text_encoder_hidden_size, config.hidden_size * 2, bias=False),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False),
            )
        if proj_type == "breeze_dimfusion":
            proj_out_hidden_size = config.hidden_size
            if config.text_encoder_dimfusion_fuse_first_layer:
                proj_out_hidden_size //= 2
            self.text_encoder_layer_projs = nn.ModuleList(
                [
                    nn.Linear(text_encoder_hidden_size, config.hidden_size // 2, bias=False)
                    for _ in range(config.num_hidden_layers)
                ]
            )
            return nn.Linear(
                text_encoder_hidden_size * len(config.text_encoder_feature_layer_idx),
                proj_out_hidden_size,
                bias=True,
            )
        raise ValueError(f"Unsupported text_encoder_proj_type: {proj_type}")

    @staticmethod
    def _get_nonzero_dropout_fields(config) -> list[tuple[str, float]]:
        """
        Collects the dropout fields of a config that are set to a non-zero rate.

        Args:
            config ([`PreTrainedConfig`]):
                Configuration to inspect.

        Returns:
            `list[tuple[str, float]]`: the `(name, value)` pairs of every non-zero dropout field.
        """
        dropout_field_names = (
            "dropout",
            "dropout_rate",
            "attention_dropout",
            "hidden_dropout",
            "hidden_dropout_prob",
            "activation_dropout",
            "classifier_dropout",
            "classifier_dropout_rate",
            "embd_pdrop",
            "resid_pdrop",
        )
        nonzero_fields = []
        for field_name in dropout_field_names:
            value = getattr(config, field_name, None)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value != 0.0:
                nonzero_fields.append((field_name, value))
        return nonzero_fields

    def train(self, mode: bool = True):
        super().train(mode)
        # Loading a checkpoint rebuilds every float parameter with `requires_grad=True`, so the modules that
        # never train are frozen on every mode switch rather than at construction only.
        self.codec_model.eval()
        self.codec_model.requires_grad_(False)
        if self.text_encoder is not None and not self.text_encoder_trainable:
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
        return self

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return super(CsmForConditionalGeneration, self).prepare_inputs_for_generation(*args, **kwargs)

    def _encode_text_segments(
        self, segments: list[torch.LongTensor], output_hidden_states: bool = False
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]] | None]:
        """
        Runs the text encoder over variable-length token id segments, one segment per padded batch row.

        Batching segments separately rather than as one padded prompt is what keeps the bidirectional text
        encoder from attending across segment and sample boundaries.

        Args:
            segments (`list[torch.LongTensor]`):
                Token ids of each text segment, each of shape `(segment_length,)`.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether the per-layer hidden states of every segment are returned as well.

        Returns:
            `tuple[list[torch.Tensor], list[list[torch.Tensor]] | None]`: the feature of each segment, of shape
            `(segment_length, hidden_size * len(text_encoder_feature_layer_idx))`, and the per-layer hidden states
            of each segment when requested.
        """
        if not segments:
            return [], None

        device = segments[0].device
        lengths = [segment.shape[0] for segment in segments]
        feature_layer_idx = tuple(self.config.text_encoder_feature_layer_idx)
        max_length_ratio = self.config.text_encoder_bucket_max_length_ratio

        buckets = []
        current_bucket = []
        current_min_len = None
        for idx in sorted(range(len(segments)), key=lambda i: lengths[i]):
            if not current_bucket:
                current_bucket, current_min_len = [idx], lengths[idx]
            elif lengths[idx] / max(current_min_len, 1) <= max_length_ratio:
                current_bucket.append(idx)
            else:
                buckets.append(current_bucket)
                current_bucket, current_min_len = [idx], lengths[idx]
        if current_bucket:
            buckets.append(current_bucket)

        hidden_states = [None] * len(segments)
        layer_hidden_states = [None] * len(segments) if output_hidden_states else None

        for bucket_indices in buckets:
            max_len = max(lengths[i] for i in bucket_indices)
            padded_ids = torch.zeros(len(bucket_indices), max_len, dtype=segments[0].dtype, device=device)
            attention_mask = torch.zeros(len(bucket_indices), max_len, dtype=torch.long, device=device)
            position_ids = torch.zeros(len(bucket_indices), max_len, dtype=torch.long, device=device)
            for bucket_pos, idx in enumerate(bucket_indices):
                length = lengths[idx]
                padded_ids[bucket_pos, :length] = segments[idx]
                attention_mask[bucket_pos, :length] = 1
                position_ids[bucket_pos, :length] = torch.arange(length, device=device)

            outputs = self.text_encoder(
                input_ids=padded_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=output_hidden_states,
            )

            if feature_layer_idx == (-1,):
                features = outputs.last_hidden_state
            else:
                if outputs.hidden_states is None:
                    raise ValueError(
                        "output_hidden_states must be enabled when selecting non-final text encoder layers"
                    )
                features = torch.concat([outputs.hidden_states[i] for i in feature_layer_idx], dim=-1)

            for bucket_pos, idx in enumerate(bucket_indices):
                hidden_states[idx] = features[bucket_pos, : lengths[idx]]
                if layer_hidden_states is not None:
                    layer_hidden_states[idx] = [
                        layer[bucket_pos, : lengths[idx]] for layer in outputs.hidden_states
                    ]

        return hidden_states, layer_hidden_states

    def _project_text_segments(
        self,
        segment_hidden_states: list[torch.Tensor],
        segment_layer_hidden_states: list[list[torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Projects text encoder features to `config.hidden_size` and, for `"breeze_dimfusion"`, builds the per
        backbone layer conditioning signal.

        Args:
            segment_hidden_states (`list[torch.Tensor]`):
                Feature of each text segment, of shape `(segment_length, text_encoder_feature_size)`.
            segment_layer_hidden_states (`list[list[torch.Tensor]]`, *optional*):
                Per-layer hidden states of each text segment.

        Returns:
            `tuple[torch.Tensor, list[torch.Tensor] | None]`: the concatenated projected text embeddings of shape
            `(num_text_tokens, hidden_size)`, and one projected hidden state per backbone layer.
        """
        text_embeds = torch.cat([self.text_encoder_proj(segment) for segment in segment_hidden_states], dim=0)

        layer_hidden_states = None
        if self.text_encoder_layer_projs is not None:
            num_layers = len(self.text_encoder_layer_projs)
            start_idx = self.config.text_encoder_dimfusion_layer_start_idx
            end_idx = self.config.text_encoder_dimfusion_layer_end_idx
            layer_hidden_states = []
            for layer_idx, layer_proj in enumerate(self.text_encoder_layer_projs):
                layer_parts = []
                for segment_layers in segment_layer_hidden_states:
                    selected = segment_layers[start_idx:end_idx]
                    if len(selected) < num_layers:
                        selected = selected + [selected[-1]] * (num_layers - len(selected))
                    elif len(selected) > num_layers:
                        selected = selected[-num_layers:]
                    layer_parts.append(layer_proj(selected[layer_idx]))
                layer_hidden_states.append(torch.cat(layer_parts, dim=0))

        if self.config.text_encoder_dimfusion_fuse_first_layer:
            text_embeds = torch.cat([text_embeds, layer_hidden_states[0]], dim=-1)

        return text_embeds, layer_hidden_states

    def _embed_text_ids(
        self,
        input_ids: torch.LongTensor,
        text_ids_mask: torch.BoolTensor,
        text_ids_len: torch.LongTensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Embeds the text positions of `input_ids` with the text encoder and its projection.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Prompt token ids.
            text_ids_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
                Positions holding a text token.
            text_ids_len (`torch.LongTensor` of shape `(num_segments,)`):
                Length of every text segment, concatenated over the batch in row-major order.

        Returns:
            `tuple[torch.Tensor, list[torch.Tensor] | None]`: the input embeddings, zero everywhere but the text
            positions, and the per backbone layer conditioning signal of the `"breeze_dimfusion"` projection.

        Raises:
            ValueError: if `text_ids_len` does not tile the text positions of `text_ids_mask` exactly.
        """
        segment_lengths = [int(length) for length in text_ids_len.reshape(-1).tolist() if length > 0]
        if sum(segment_lengths) != int(text_ids_mask.sum()):
            raise ValueError(
                f"text_ids_len sums to {sum(segment_lengths)} but text_ids_mask marks "
                f"{int(text_ids_mask.sum())} text positions."
            )
        segments = list(input_ids[text_ids_mask].split(segment_lengths, dim=0))

        needs_layer_hidden_states = (
            self.text_encoder_layer_projs is not None
            or tuple(self.config.text_encoder_feature_layer_idx) != (-1,)
        )
        segment_hidden_states, segment_layer_hidden_states = self._encode_text_segments(
            segments, output_hidden_states=needs_layer_hidden_states
        )
        text_embeds, layer_hidden_states = self._project_text_segments(
            segment_hidden_states, segment_layer_hidden_states
        )

        inputs_embeds = torch.zeros(
            (*input_ids.shape, self.config.hidden_size), dtype=text_embeds.dtype, device=text_embeds.device
        )
        inputs_embeds[text_ids_mask] = text_embeds
        return inputs_embeds, layer_hidden_states

    def _merge_input_ids_with_input_values(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        text_ids_mask: torch.BoolTensor | None = None,
        text_ids_len: torch.LongTensor | None = None,
    ) -> dict:
        """
        Builds `inputs_embeds` from the prompt token ids and the codebook frames they reserve room for, and
        expands `labels` over the codebook dimension.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Prompt token ids, with one `config.audio_token_id` per audio frame.
            input_values (`torch.Tensor` of shape `(batch_size, num_frames, num_codebooks)`, *optional*):
                Codebook ids of the audio frames, already produced by the audio tokenizer.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Per-position training targets, as described on [`~BreezeTTSForConditionalGeneration.forward`].
            text_ids_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Positions holding a text token.
            text_ids_len (`torch.LongTensor` of shape `(num_segments,)`, *optional*):
                Length of every text segment, concatenated over the batch in row-major order.

        Returns:
            `dict`: the `inputs_embeds`, the codebook-expanded `labels` and the per backbone layer text encoder
            conditioning signal.
        """
        text_encoder_layer_hidden_states = None
        if self.text_encoder is not None:
            if text_ids_mask is None or text_ids_len is None:
                raise ValueError(
                    "`text_ids_mask` and `text_ids_len` are required to embed `input_ids` with the text encoder."
                )
            inputs_embeds, text_encoder_layer_hidden_states = self._embed_text_ids(
                input_ids, text_ids_mask, text_ids_len
            )
        else:
            inputs_embeds = self.embed_text_tokens(input_ids)

        if input_values is not None:
            audio_token_mask = input_ids == self.config.audio_token_id
            if not audio_token_mask.any():
                raise ValueError(
                    f"`input_values` holds {input_values.shape[1]} audio frames but `input_ids` reserves no "
                    f"position for them: no id equals `config.audio_token_id` ({self.config.audio_token_id})."
                )

            audio_embeds = self.backbone_model.embed_tokens(input_values)
            inputs_embeds[audio_token_mask] = audio_embeds.reshape(-1, audio_embeds.shape[-1]).to(
                inputs_embeds.dtype
            )

            audio_eos_frame_ids = torch.full(
                (1, 1, self.config.num_codebooks),
                self.config.codebook_eos_token_id,
                device=input_ids.device,
                dtype=torch.long,
            )
            audio_eos_embeds = self.backbone_model.embed_tokens(audio_eos_frame_ids).squeeze(1)
            audio_eos_token_mask = input_ids == self.config.audio_eos_token_id
            inputs_embeds[audio_eos_token_mask] = audio_eos_embeds.repeat(int(audio_eos_token_mask.sum()), 1).to(
                inputs_embeds.dtype
            )

            if labels is not None:
                labels_expanded = labels.unsqueeze(-1).repeat(1, 1, self.config.num_codebooks)
                labels_expanded[audio_token_mask] = input_values.reshape(-1, input_values.shape[-1]).to(
                    labels_expanded.dtype
                )
                # The end-of-audio frame is scored on the backbone head only, through its extra class.
                labels_expanded[audio_eos_token_mask] = -100
                eos_positions = audio_eos_token_mask.nonzero(as_tuple=True)
                labels_expanded[eos_positions[0], eos_positions[1], 0] = self.backbone_eos_token_id

                depth_decoder_ignore_mask = (labels == -101) & ~audio_eos_token_mask
                depth_decoder_ignore_idxs = depth_decoder_ignore_mask.nonzero(as_tuple=True)
                labels_expanded[depth_decoder_ignore_idxs[0], depth_decoder_ignore_idxs[1], 1:] = -100
                labels = labels_expanded

        return {
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "text_encoder_layer_hidden_states": text_encoder_layer_hidden_states,
        }

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        text_ids_mask: torch.BoolTensor | None = None,
        text_ids_len: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BreezeTTSOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)` or `(batch_size, sequence_length)`):
            1. `(batch_size, sequence_length)`: the prompt token ids built by [`BreezeTTSProcessor`]. Every audio
            frame of the prompt reserves one `config.audio_token_id` position, whose codebook ids are passed
            through `input_values`.

            2. `(batch_size, sequence_length, num_codebooks)`: codebook tokens generated during autoregressive
            decoding. Such input is not meant to be used by end users.
        input_values (`torch.Tensor` of shape `(batch_size, num_frames, num_codebooks)`, *optional*):
            Codebook ids of the prompt's audio frames, produced by the audio tokenizer.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-position training targets, in `[config.audio_token_id, -100, -101]`. Requires `input_values`.
            - `config.audio_token_id` marks a frame both the backbone and the depth decoder are scored on
            - `-100` marks a position neither is scored on
            - `-101` marks a frame only the backbone is scored on, through its first codebook
        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            Kept for compatibility. Does not support another value than:
            1. `0`, which is equivalent to keeping all logits, used in the training regime
            2. `1`, which is equivalent to keeping only the last logit, used in the generation regime
        text_ids_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Positions of `input_ids` holding a text token, i.e. the positions the text encoder embeds.
        text_ids_len (`torch.LongTensor` of shape `(num_segments,)`, *optional*):
            Length of every text segment of the batch, concatenated in row-major order. Segments are encoded
            independently of each other.
        """
        text_encoder_layer_hidden_states = None
        if input_ids is not None and input_ids.ndim == 2:
            merged_inputs = self._merge_input_ids_with_input_values(
                input_ids,
                input_values,
                labels,
                text_ids_mask=text_ids_mask,
                text_ids_len=text_ids_len,
            )
            inputs_embeds = merged_inputs["inputs_embeds"]
            labels = merged_inputs["labels"]
            text_encoder_layer_hidden_states = merged_inputs["text_encoder_layer_hidden_states"]
            input_ids = None

        backbone_outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            text_encoder_layer_hidden_states=text_encoder_layer_hidden_states,
            text_ids_mask=text_ids_mask,
            **kwargs,
        )

        backbone_hidden_states = backbone_outputs[0]
        # The backbone reports no per-layer hidden states; decoding reads the state the depth decoder is
        # conditioned on from `hidden_states[-1]`, so the last one always has to be there.
        backbone_all_hidden_states = backbone_outputs.hidden_states or (backbone_hidden_states,)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        backbone_logits = self.lm_head(backbone_hidden_states[:, slice_indices, :])

        loss = None
        backbone_loss = None
        depth_decoder_loss = None
        depth_decoder_outputs = None
        if labels is not None:
            loss_kwargs = {k: v for k, v in kwargs.items() if k != "num_items_in_batch"}

            # the backbone is scored on the first codebook of every frame, plus its extra end-of-audio class
            backbone_labels = labels[:, :, 0]
            backbone_loss = self.loss_function(
                logits=backbone_logits,
                labels=backbone_labels,
                vocab_size=self.lm_head.out_features,
                **loss_kwargs,
            )

            # the depth decoder is scored on the frames whose labels are not uniformly `-100` past codebook 0
            train_mask = ~(labels[:, :, 1:] == -100).all(dim=-1)
            depth_decoder_input_ids = labels[train_mask][..., : self.config.num_codebooks - 1]
            # position 0 is a placeholder, replaced by the backbone hidden state of the previous position
            depth_decoder_input_ids = nn.functional.pad(depth_decoder_input_ids, (1, 0), value=0)

            train_idxs = train_mask.nonzero(as_tuple=True)
            backbone_last_hidden_states = backbone_hidden_states[train_idxs[0], train_idxs[1] - 1, :]

            depth_decoder_outputs = self.depth_decoder(
                input_ids=depth_decoder_input_ids,
                backbone_last_hidden_state=backbone_last_hidden_states,
                use_cache=use_cache,
                return_dict=True,
                labels=labels[train_mask],
                **loss_kwargs,
            )
            depth_decoder_loss = depth_decoder_outputs.loss
            depth_header_loss_weight = self.config.depth_header_loss_weight if self.training else 1.0
            loss = backbone_loss + depth_header_loss_weight * depth_decoder_loss

        return BreezeTTSOutputWithPast(
            loss=loss,
            backbone_loss=backbone_loss,
            depth_decoder_loss=depth_decoder_loss,
            logits=backbone_logits,
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_all_hidden_states,
            attentions=backbone_outputs.attentions,
            depth_decoder_logits=depth_decoder_outputs.logits if depth_decoder_outputs is not None else None,
            depth_decoder_past_key_values=depth_decoder_outputs.past_key_values
            if depth_decoder_outputs is not None
            else None,
            depth_decoder_hidden_states=depth_decoder_outputs.hidden_states
            if depth_decoder_outputs is not None
            else None,
            depth_decoder_attentions=depth_decoder_outputs.attentions
            if depth_decoder_outputs is not None
            else None,
        )


__all__ = [
    "BreezeTTSBackboneModel",
    "BreezeTTSDepthDecoderForCausalLM",
    "BreezeTTSDepthDecoderModel",
    "BreezeTTSForConditionalGeneration",
    "BreezeTTSOutputWithPast",
    "BreezeTTSPreTrainedModel",
]
