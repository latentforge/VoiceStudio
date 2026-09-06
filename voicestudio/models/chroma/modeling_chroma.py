# Copyright 2025 The FlashLabs team. All rights reserved.
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
"""PyTorch Chroma model."""

from dataclasses import dataclass, fields
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.csm.modeling_csm import CsmCodebooksHead
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaModel
from transformers.models.mimi.modeling_mimi import MimiModel
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers.utils import ModelOutput, auto_docstring, logging

from .configuration_chroma import ChromaBackboneConfig, ChromaConfig, ChromaDecoderConfig
from .generation_chroma import ChromaGenerationMixin


logger = logging.get_logger(__name__)

PASSTHROUGH_KEYS = [
    "thinker_input_ids",
    "thinker_attention_mask",
    "thinker_cache_position",
    "thinker_past_key_values",
    "thinker_input_features",
    "thinker_feature_attention_mask",
    "thinker_eos",
    "thinker_hidden_states",
    "thinker_logits",
    "thinker_flag",
    "prefilled",
    "attention_mask",
]

ONE_TIME_KEYS = [
    "input_values",
    "thinker_input_features",
    "thinker_feature_attention_mask",
]


@dataclass
class ChromaOutputWithPast(ModelOutput):
    r"""
    Base class for Chroma outputs, carrying the backbone state that drives generation together with the reasoner
    state that has to survive to the next generation step.

    Args:
        loss (`torch.FloatTensor`, *optional*):
            Weighted sum of `backbone_loss` and `decoder_loss`, returned when `labels` is provided.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Backbone hidden states, one per layer plus the embedding output.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Codebook 0 logits produced by the backbone.
        past_key_values (`Cache`, *optional*):
            Backbone cache.
        cache_position (`torch.LongTensor`, *optional*):
            Backbone cache positions of the tokens consumed by this step.
        attention_mask (`torch.LongTensor`, *optional*):
            Backbone attention mask after the interleaved reasoner tokens of this step were appended.
        thinker_loss (`torch.FloatTensor`, *optional*):
            Unused, the reasoner is frozen and contributes no term to the training loss.
        thinker_logits (`torch.FloatTensor`, *optional*):
            Reasoner logits of the last step, used to sample the next interleaved text token.
        thinker_past_key_values (`Cache`, *optional*):
            Reasoner cache.
        thinker_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Reasoner hidden states of the last step.
        thinker_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Reasoner attentions of the last step.
        thinker_input_ids (`torch.LongTensor`, *optional*):
            Token ids the reasoner consumes on the next step, `None` once every sequence reached its turn end.
        thinker_attention_mask (`torch.LongTensor`, *optional*):
            Reasoner attention mask.
        thinker_input_features (`torch.FloatTensor`, *optional*):
            Reasoner audio features of the user turn, consumed once during prefill.
        thinker_feature_attention_mask (`torch.LongTensor`, *optional*):
            Attention mask of `thinker_input_features`.
        thinker_cache_position (`torch.LongTensor`, *optional*):
            Reasoner cache positions.
        thinker_flag (`bool`, *optional*):
            Whether the next step interleaves a reasoner token pair into the backbone sequence, which alternates to
            realize the 1:2 text to audio schedule.
        thinker_eos (`torch.BoolTensor`, *optional*):
            Per sequence flag, set once the reasoner sampled `im_end_token_id` and never cleared.
        backbone_loss (`torch.FloatTensor`, *optional*):
            Cross entropy of codebook 0 over the audio frames, returned when `labels` is provided.
        backbone_logits (`torch.FloatTensor`, *optional*):
            Alias of `logits`.
        backbone_past_key_values (`Cache`, *optional*):
            Alias of `past_key_values`.
        backbone_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Alias of `hidden_states`.
        backbone_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Backbone attentions.
        decoder_loss (`torch.FloatTensor`, *optional*):
            Cross entropy of codebooks 1 to `audio_num_codebooks - 1`, returned when `labels` is provided.
        decoder_logits (`torch.FloatTensor`, *optional*):
            Residual codebook logits of the frames the decoder was trained on.
        decoder_past_key_values (`Cache`, *optional*):
            Decoder cache.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Decoder hidden states.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Decoder attentions.
    """

    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    cache_position: Optional[torch.LongTensor] = None
    attention_mask: Optional[torch.LongTensor] = None

    thinker_loss: Optional[torch.FloatTensor] = None
    thinker_logits: Optional[torch.FloatTensor] = None
    thinker_past_key_values: Optional[Cache] = None
    thinker_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    thinker_attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    thinker_input_ids: Optional[torch.LongTensor] = None
    thinker_attention_mask: Optional[torch.LongTensor] = None
    thinker_input_features: Optional[torch.FloatTensor] = None
    thinker_feature_attention_mask: Optional[torch.LongTensor] = None
    thinker_cache_position: Optional[torch.LongTensor] = None
    thinker_flag: Optional[bool] = None
    thinker_eos: Optional[torch.BoolTensor] = None

    backbone_loss: Optional[torch.FloatTensor] = None
    backbone_logits: Optional[torch.FloatTensor] = None
    backbone_past_key_values: Optional[Cache] = None
    backbone_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    backbone_attentions: Optional[tuple[torch.FloatTensor, ...]] = None

    decoder_loss: Optional[torch.FloatTensor] = None
    decoder_logits: Optional[torch.FloatTensor] = None
    decoder_past_key_values: Optional[Cache] = None
    decoder_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class ChromaLlamaModel(LlamaModel):
    r"""
    Llama decoder stack shared by the Chroma backbone and the Chroma decoder.

    Args:
        config ([`ChromaBackboneConfig`] or [`ChromaDecoderConfig`]):
            Configuration of the stack. Both derive from [`LlamaConfig`].
    """

    def __init__(self, config: Union[ChromaBackboneConfig, ChromaDecoderConfig]):
        super().__init__(config)
        self.embed_tokens = nn.Identity()


@auto_docstring
class ChromaPreTrainedModel(PreTrainedModel):
    config: ChromaConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer", "Qwen2_5OmniDecoderLayer", "Qwen2_5OmniVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None and not getattr(module.bias, "_is_hf_initialized", False):
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ChromaCodebookHead):
            if not getattr(module.weight, "_is_hf_initialized", False):
                module.weight.data.normal_(mean=0.0, std=std)


class ChromaAudioEmbedding(nn.Module):
    r"""
    Codebook token embedding table shared by the backbone and the decoder. Codebook `j` occupies the id range
    `[j * audio_vocab_size, (j + 1) * audio_vocab_size)` of a single flat table.

    Args:
        audio_num_codebooks (`int`):
            Number of codebooks the table covers.
        audio_vocab_size (`int`):
            Size of a single codebook vocabulary.
        hidden_size (`int`):
            Width of the embedding.
    """

    def __init__(self, audio_num_codebooks: int, audio_vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_audio_tokens = nn.Embedding(
            num_embeddings=audio_num_codebooks * audio_vocab_size,
            embedding_dim=hidden_size,
        )
        self.audio_vocab_size = audio_vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(..., num_codebooks)`):
                Codebook ids of one audio frame, codebook `j` at position `j`.

        Returns:
            `torch.Tensor` of shape `(..., num_codebooks, hidden_size)`: The per codebook embeddings.
        """
        num_codebooks = input_ids.shape[-1]
        audio_frames = input_ids + (self.audio_vocab_size * torch.arange(num_codebooks, device=input_ids.device))
        embeddings = self.embed_audio_tokens(audio_frames.view(-1)).reshape(
            audio_frames.shape + (self.embed_audio_tokens.embedding_dim,)
        )
        return embeddings


@auto_docstring(
    custom_intro="""
    The Chroma backbone, a Llama decoder stack over the interleaved reasoner/audio embedding sequence with a
    linear head on top that predicts codebook 0 of the next audio frame.
    """
)
class ChromaBackboneForCausalLM(ChromaPreTrainedModel):
    config: ChromaBackboneConfig

    def __init__(self, config: ChromaBackboneConfig):
        super().__init__(config)
        self.model = ChromaLlamaModel(config)
        self.codebook0_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.audio_embedding = ChromaAudioEmbedding(
            config.audio_num_codebooks,
            config.vocab_size,
            config.hidden_size,
        )

        self.post_init()

    def emb_audio_frames(self, audio_frames: torch.Tensor, add_frame: bool = True) -> torch.Tensor:
        """
        Embeds whole audio frames with the shared codebook table.

        Args:
            audio_frames (`torch.LongTensor` of shape `(..., num_codebooks)`):
                Codebook ids of the frames. Positions holding the ignore index `-100` are embedded as id 0 and are
                expected to be masked out by the caller.
            add_frame (`bool`, *optional*, defaults to `True`):
                Whether the per codebook embeddings of a frame are summed into a single frame embedding.

        Returns:
            `torch.Tensor`: The frame embeddings of shape `(..., hidden_size)`, or the per codebook embeddings of
            shape `(..., num_codebooks, hidden_size)` when `add_frame` is `False`.
        """
        if audio_frames.dim() < 2:
            raise ValueError("audio_frames must have a trailing codebook dimension")
        audio_frames = audio_frames.contiguous()
        audio_frames = audio_frames.masked_fill(audio_frames == -100, 0)
        audio_embeddings = self.audio_embedding(audio_frames)

        if add_frame:
            audio_embeddings = audio_embeddings.sum(dim=-2)
        return audio_embeddings

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """
        Computes the codebook 0 cross entropy of the backbone.

        Args:
            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
                Codebook 0 logits.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Codebook 0 ids aligned with `logits`, shifted internally so that position `t` predicts `t + 1`.
            ignore_index (`int`, *optional*, defaults to -100):
                Label value excluded from the mean.

        Returns:
            `torch.FloatTensor`: The scalar mean cross entropy.
        """
        logits = logits.float()

        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)

        logits = logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.to(logits.device)

        return F.cross_entropy(logits, shift_labels, ignore_index=ignore_index)

    def forward(
        self,
        input_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Interleaved reasoner and audio frame embeddings.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Codebook 0 ids, `-100` on the positions excluded from the loss.
            use_cache (`bool`, *optional*):
                Whether the backbone cache is returned.
            output_hidden_states (`bool`, *optional*, defaults to `True`):
                Whether all layer hidden states are returned. The last one conditions [`ChromaDecoderForCausalLM`].
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether attention weights are returned.
            cache_position (`torch.LongTensor`, *optional*):
                Position of `input_embeddings` in the cached sequence.
            attention_mask (`torch.LongTensor` of shape `(batch_size, cached_length)`, *optional*):
                Mask over the cached sequence and `input_embeddings`.

        Returns:
            [`~modeling_outputs.CausalLMOutputWithPast`]: The codebook 0 logits and, when `labels` is provided, the
            codebook 0 cross entropy.

        Raises:
            ValueError: If `input_embeddings` is missing or does not have `config.hidden_size` channels.
        """
        if input_embeddings is None:
            raise ValueError("input_embeddings is required")

        if input_embeddings.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"input_embeddings must have {self.config.hidden_size} channels, got {input_embeddings.shape[-1]}"
            )

        output: BaseModelOutputWithPast = self.model(
            inputs_embeds=input_embeddings,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = self.codebook0_head(output.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class ChromaCodebookHead(CsmCodebooksHead):
    r"""
    Position specific language modeling head of [`ChromaDecoderForCausalLM`]: one `(hidden_size, vocab_size)`
    projection per residual codebook, held in a single `(audio_num_codebooks - 1, hidden_size, vocab_size)`
    parameter.

    Args:
        hidden_size (`int`):
            Width of the decoder hidden states.
        num_codebooks (`int`):
            Total number of codebooks per frame. The head covers the `num_codebooks - 1` residual ones.
        vocab_size (`int`):
            Size of a single codebook vocabulary.
    """

    def forward(self, hidden_states: torch.Tensor, codebook_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, num_positions, hidden_size)`):
                Decoder hidden states of the residual codebook positions, position 0 of the frame excluded.
            codebook_indices (`torch.LongTensor` of shape `(num_positions,)`, *optional*):
                One based codebook index of each position. Defaults to `1 .. num_positions`.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_positions, vocab_size)`: The per codebook logits.
        """
        if codebook_indices is None:
            codebook_indices = torch.arange(1, hidden_states.shape[1] + 1, device=hidden_states.device)
        return super().forward(hidden_states, codebook_indices)

    def get_logits(self, hidden_states: torch.Tensor, codebook_id: int) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
                Decoder hidden state of a single residual codebook position.
            codebook_id (`int`):
                One based index of the codebook to project onto.

        Returns:
            `torch.Tensor` of shape `(batch_size, vocab_size)`: The logits of that codebook.

        Raises:
            ValueError: If `codebook_id` is outside `1 .. num_codebooks - 1`.
        """
        num_residual_codebooks = self.weight.shape[0]
        if codebook_id < 1 or codebook_id > num_residual_codebooks:
            raise ValueError(f"codebook_id must be between 1 and {num_residual_codebooks}, but got {codebook_id}")
        return torch.mm(hidden_states, self.weight[codebook_id - 1, :, :])


@auto_docstring(
    custom_intro="""
    The Chroma decoder, a small Llama decoder stack that runs once per audio frame and autoregressively predicts
    the residual codebooks of that frame from the backbone hidden state and codebook 0.
    """
)
class ChromaDecoderForCausalLM(ChromaPreTrainedModel, GenerationMixin):
    config: ChromaDecoderConfig

    def __init__(self, config: ChromaDecoderConfig):
        super().__init__(config)

        self.projection = nn.Linear(config.audio_embedding_dim, config.hidden_size, bias=False)
        self.model = ChromaLlamaModel(config)
        self.codebook_head = ChromaCodebookHead(
            hidden_size=config.hidden_size,
            num_codebooks=config.audio_num_codebooks,
            vocab_size=config.vocab_size,
        )
        self.audio_embedding = ChromaAudioEmbedding(
            config.audio_num_codebooks,
            config.vocab_size,
            config.audio_embedding_dim,
        )

        self.post_init()

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """
        Computes the residual codebook cross entropy of the decoder.

        Args:
            logits (`torch.FloatTensor` of shape `(num_frames, audio_num_codebooks - 1, vocab_size)`):
                Residual codebook logits. Position `j` already predicts codebook `j + 1`, so no shift is applied.
            labels (`torch.LongTensor` of shape `(num_frames, audio_num_codebooks - 1)`):
                Ids of codebooks 1 to `audio_num_codebooks - 1`.
            ignore_index (`int`, *optional*, defaults to -100):
                Label value excluded from the mean.

        Returns:
            `torch.FloatTensor`: The scalar mean cross entropy.
        """
        vocab_size = logits.size(-1)
        logits_flat = logits.contiguous().view(-1, vocab_size)
        labels_flat = labels.contiguous().view(-1)

        return F.cross_entropy(logits_flat.float(), labels_flat, ignore_index=ignore_index)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(num_frames, num_positions)`, *optional*):
                Ids of codebooks 0 to `num_positions - 1` of the frame, teacher forced. Position `j` of the
                embedded sequence is offset into codebook `j` of the shared table.
            labels (`torch.LongTensor` of shape `(num_frames, audio_num_codebooks - 1)` or
                `(batch_size, sequence_length, audio_num_codebooks - 1)`, *optional*):
                Ids of codebooks 1 to `audio_num_codebooks - 1`, `-100` on the positions excluded from the loss.
            backbone_last_hidden_state (`torch.FloatTensor` of shape `(num_frames, audio_embedding_dim)`,
                *optional*):
                Backbone hidden state that occupies position 0 of the frame. Required whenever `input_ids` starts
                at codebook 0.
            past_key_values (`Cache`, *optional*):
                Intra-frame cache, holding one entry per codebook already decoded in this frame.
            inputs_embeds (`torch.FloatTensor` of shape `(..., num_positions, audio_embedding_dim)`, *optional*):
                Pre-embedded frame positions, position 0 holding the backbone hidden state. Mutually exclusive
                with `input_ids`.
            use_cache (`bool`, *optional*):
                Whether the intra-frame cache is returned.
            output_attentions (`bool`, *optional*):
                Whether attention weights are returned.
            output_hidden_states (`bool`, *optional*):
                Whether all layer hidden states are returned.
            cache_position (`torch.LongTensor`, *optional*):
                Position of the inputs in the cached frame.

        Returns:
            [`~modeling_outputs.CausalLMOutputWithPast`]: The residual codebook logits, padded back to the input
            sequence length, and, when `labels` is provided, the residual codebook cross entropy.

        Raises:
            ValueError: If neither or both of `input_ids` and `inputs_embeds` are given, if the intra-frame cache
                is longer than the frame, or if `labels` does not cover exactly `audio_num_codebooks - 1` codebooks.
        """
        if inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds or input_ids is required")

        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("inputs_embeds and input_ids cannot be used at the same time")

        loss = None

        # `generate` hands in an empty cache on the prefill step, so the cache length rather than its presence
        # tells the codebooks already decoded in this frame; position 0 holds the backbone hidden state
        past_seen_positions = past_key_values.get_seq_length() if past_key_values is not None else 0
        past_codebook_num = max(past_seen_positions - 1, 0)

        if past_codebook_num > self.config.audio_num_codebooks - 1:
            raise ValueError(
                f"past_codebook_num is greater than audio_num_codebooks - 1, "
                f"{past_codebook_num} > {self.config.audio_num_codebooks - 1}"
            )

        if inputs_embeds is None:
            offset = (
                torch.arange(input_ids.shape[-1], device=input_ids.device) + past_codebook_num
            ) * self.config.vocab_size
            audio_ids_embed = self.audio_embedding.embed_audio_tokens(
                input_ids.masked_fill(input_ids == -100, 0) + offset
            )
            inputs_embeds = (
                torch.cat([backbone_last_hidden_state.unsqueeze(1), audio_ids_embed], dim=1)
                if backbone_last_hidden_state is not None
                else audio_ids_embed
            )

        orig_shape = inputs_embeds.shape

        if inputs_embeds.dim() == 4:
            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            labels = labels.reshape(-1, labels.shape[-1]) if labels is not None else None

        has_eos = inputs_embeds.shape[1] == self.config.audio_num_codebooks + 1
        inputs_embeds = inputs_embeds[:, : self.config.audio_num_codebooks, :]

        inputs_embeds = self.projection(inputs_embeds)
        output: BaseModelOutputWithPast = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        if past_seen_positions > 0:
            logits = self.codebook_head.get_logits(
                output.last_hidden_state.squeeze(1),
                past_codebook_num + 1,
            ).unsqueeze(1)
        else:
            logits = self.codebook_head(output.last_hidden_state[:, 1:, :])

        if labels is not None:
            expected = self.config.audio_num_codebooks - 1
            if labels.shape[1] != expected:
                raise ValueError(f"labels must cover {expected} codebooks, but got {labels.shape[1]}")
            if logits.shape[1] != expected:
                raise ValueError(f"logits must cover {expected} codebooks, but got {logits.shape[1]}")
            loss = self.loss_fn(logits, labels)

        pad_left = 1 if backbone_last_hidden_state is not None or has_eos or input_ids is None else 0
        pad_right = 1 if has_eos else 0
        logits = F.pad(logits, (0, 0, pad_left, pad_right), value=0)
        logits = logits.reshape(*orig_shape[:-1], logits.shape[-1])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        next_sequence_length: Optional[int] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        is_first_iteration: Optional[bool] = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Args:
            input_ids (`torch.LongTensor`):
                Codebook ids decoded so far in this frame.
            next_sequence_length (`int`, *optional*):
                Number of positions the next forward consumes.
            past_key_values (`Cache`, *optional*):
                Intra-frame cache.
            attention_mask (`torch.LongTensor`, *optional*):
                Unused, the intra-frame sequence is never padded.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Pre-embedded frame positions.
            is_first_iteration (`bool`, *optional*, defaults to `False`):
                Whether this is the prefill step, the only one that consumes `backbone_last_hidden_state`.

        Returns:
            `dict[str, Any]`: The keyword arguments of the next [`~ChromaDecoderForCausalLM.forward`] call.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration:
            model_inputs.pop("backbone_last_hidden_state", None)
        model_inputs.pop("position_ids", None)

        return model_inputs


@auto_docstring(
    custom_intro="""
    The Chroma model, a frozen Qwen2.5-Omni thinker reasoner whose token embeddings and hidden states are
    interleaved with audio frame embeddings at a 1:2 ratio and consumed by a Llama backbone that predicts Mimi
    codebook 0, a Llama decoder that predicts the remaining codebooks of each frame, and a frozen Mimi codec that
    encodes the reference audio and reconstructs the waveform.
    """
)
class ChromaForConditionalGeneration(ChromaPreTrainedModel, ChromaGenerationMixin):
    base_model_prefix = "chroma"
    _supports_cache_class = True

    _tied_weights_keys = {
        "backbone.audio_embedding.embed_audio_tokens.weight": "decoder.audio_embedding.embed_audio_tokens.weight",
    }

    def __init__(self, config: ChromaConfig):
        super().__init__(config)
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.backbone = ChromaBackboneForCausalLM._from_config(config.backbone_config)
        self.decoder = ChromaDecoderForCausalLM._from_config(config.decoder_config)
        self.codec_model = MimiModel._from_config(config.codec_config)

        if self.backbone.config.audio_num_codebooks != config.audio_num_codebooks:
            raise ValueError(
                f"backbone_config.audio_num_codebooks {self.backbone.config.audio_num_codebooks} != "
                f"config.audio_num_codebooks {config.audio_num_codebooks}"
            )
        if self.decoder.config.audio_num_codebooks != config.audio_num_codebooks:
            raise ValueError(
                f"decoder_config.audio_num_codebooks {self.decoder.config.audio_num_codebooks} != "
                f"config.audio_num_codebooks {config.audio_num_codebooks}"
            )

        self.post_init()

        self.freeze_reasoner()
        self.freeze_codec()

        self._prompt_embeddings_initialized = False

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Args:
            args:
                Forwarded to [`~PreTrainedModel.from_pretrained`].
            kwargs:
                Forwarded to [`~PreTrainedModel.from_pretrained`].

        Returns:
            [`ChromaForConditionalGeneration`]: The loaded model, with the reasoner and the codec frozen again
            after loading replaced the parameters created by `__init__`.
        """
        outputs = super().from_pretrained(*args, **kwargs)
        model = outputs[0] if isinstance(outputs, tuple) else outputs
        model.freeze_reasoner()
        model.freeze_codec()
        return outputs

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns:
            `nn.Module`: The reasoner token embedding, which also embeds the text half of the backbone sequence.
        """
        return self.thinker.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        """
        Args:
            value (`nn.Module`):
                The new reasoner token embedding.
        """
        self.thinker.set_input_embeddings(value)

    def freeze_reasoner(self):
        """
        Freezes the reasoner, which provides fixed text embeddings and hidden states and is never optimized.
        """
        for param in self.thinker.parameters():
            param.requires_grad = False
        self.thinker._requires_grad = False

    def freeze_codec(self):
        """
        Freezes the Mimi codec, which only supplies the discrete targets and reconstructs the waveform.
        """
        for param in self.codec_model.parameters():
            param.requires_grad = False
        self.codec_model._requires_grad = False

    def freeze_backbone(self):
        """
        Freezes the backbone, matching the second training stage in which only the decoder is optimized.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone._requires_grad = False

    def _embed_text_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings()(ids.to(self.device))

    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_values_cutoffs: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        thinker_input_ids: Optional[torch.LongTensor] = None,
        thinker_attention_mask: Optional[torch.LongTensor] = None,
        thinker_cache_position: Optional[torch.LongTensor] = None,
        thinker_past_key_values: Optional[Cache] = None,
        thinker_hidden_states: Optional[torch.FloatTensor] = None,
        thinker_input_features: Optional[torch.FloatTensor] = None,
        thinker_feature_attention_mask: Optional[torch.LongTensor] = None,
        thinker_logits: Optional[torch.FloatTensor] = None,
        thinker_flag: bool = True,
        thinker_eos: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Builds the backbone inputs of one generation step and advances the reasoner when the 1:2 schedule calls
        for it.

        On the prefill step `input_values` carries the reference waveform and the backbone prompt is built from
        it; afterwards `input_ids` carries the codebook ids of the previous frame. A reasoner hidden state and the
        embedding of its next token are appended whenever `thinker_flag` is set, which alternates so that one text
        pair is injected for every two audio frames.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)` or
                `(batch_size, audio_num_codebooks)`, *optional*):
                Prompt text ids on the prefill step, codebook ids of the previous frame afterwards.
            input_values (`torch.FloatTensor` of shape `(batch_size, 1, audio_length)`, *optional*):
                Reference waveform, consumed on the prefill step only.
            input_values_cutoffs (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Number of valid samples of each reference waveform.
            past_key_values (`Cache`, *optional*):
                Backbone cache.
            attention_mask (`torch.LongTensor`, *optional*):
                Backbone attention mask over the cached sequence.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Unused, the backbone inputs are always rebuilt here.
            cache_position (`torch.LongTensor`, *optional*):
                Unused, recomputed from the cache length.
            thinker_input_ids (`torch.LongTensor`, *optional*):
                Reasoner tokens to consume, `None` once every sequence reached its turn end.
            thinker_attention_mask (`torch.LongTensor`, *optional*):
                Reasoner attention mask.
            thinker_cache_position (`torch.LongTensor`, *optional*):
                Reasoner cache positions.
            thinker_past_key_values (`Cache`, *optional*):
                Reasoner cache.
            thinker_hidden_states (`torch.FloatTensor`, *optional*):
                Reasoner hidden states of the previous step.
            thinker_input_features (`torch.FloatTensor`, *optional*):
                Reasoner audio features of the user turn.
            thinker_feature_attention_mask (`torch.LongTensor`, *optional*):
                Attention mask of `thinker_input_features`.
            thinker_logits (`torch.FloatTensor`, *optional*):
                Reasoner logits of the previous step.
            thinker_flag (`bool`, *optional*, defaults to `True`):
                Whether a reasoner token pair is interleaved into the backbone sequence on this step.
            thinker_eos (`torch.BoolTensor` of shape `(batch_size,)`, *optional*):
                Per sequence flag, set once the reasoner sampled `im_end_token_id`.

        Returns:
            `dict[str, Any]`: The keyword arguments of the next [`~ChromaForConditionalGeneration.forward`] call.

        Raises:
            ValueError: If the attention mask length does not match the cached sequence plus the new positions.
        """
        if input_values is not None:
            inputs_embeds, attention_mask = self._build_prompt_embeds(
                input_ids, attention_mask, input_values, input_values_cutoffs
            )
        else:
            inputs_embeds = self.backbone.emb_audio_frames(input_ids.to(self.device))

        if thinker_eos is None:
            if thinker_input_ids is not None:
                thinker_eos = torch.zeros(
                    thinker_input_ids.shape[0], dtype=torch.bool, device=thinker_input_ids.device
                )
            else:
                thinker_eos = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)

        if thinker_input_ids is not None and thinker_flag:
            (
                thinker_input_ids,
                thinker_attention_mask,
                thinker_cache_position,
                thinker_past_key_values,
            ) = self._update_thinker_model_kwargs(
                thinker_input_ids, thinker_attention_mask, thinker_cache_position, thinker_past_key_values
            )

            thinker_outputs = self.thinker(
                input_ids=thinker_input_ids,
                input_features=thinker_input_features,
                attention_mask=thinker_attention_mask,
                feature_attention_mask=thinker_feature_attention_mask,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                past_key_values=thinker_past_key_values,
                cache_position=thinker_cache_position,
                use_audio_in_video=False,
            )

            thinker_hidden_states = thinker_outputs.hidden_states[-1]
            thinker_past_key_values = thinker_outputs.past_key_values
            thinker_logits = thinker_outputs.logits

            thinker_next_ids = thinker_logits[:, -1:, :].argmax(dim=-1)
            next_token_emb = self._embed_text_tokens(thinker_next_ids)

            next_token_eos = thinker_next_ids.squeeze(-1) == self.config.im_end_token_id
            new_thinker_eos = thinker_eos | next_token_eos

            thinker_input_embeddings = torch.cat([thinker_hidden_states[:, -1:, :], next_token_emb], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, thinker_input_embeddings], dim=1)

            # the injected pair is attended to even on the step that produced the turn end token; only pairs
            # appended after that step are masked out
            thinker_attention_values = (~thinker_eos).long().unsqueeze(1)
            attention_mask = torch.cat([attention_mask, thinker_attention_values, thinker_attention_values], dim=1)

            thinker_eos = new_thinker_eos
            thinker_input_ids = thinker_next_ids if not thinker_eos.all() else None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        expected_attention_mask_length = past_seen_tokens + inputs_embeds.shape[1]
        if attention_mask.shape[1] != expected_attention_mask_length:
            raise ValueError(
                f"attention_mask has length {attention_mask.shape[1]}, expected {expected_attention_mask_length}"
            )

        return {
            "input_ids": None,
            "input_embeddings": inputs_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": True,
            "output_hidden_states": True,
            "thinker_past_key_values": thinker_past_key_values,
            "thinker_hidden_states": thinker_hidden_states,
            "thinker_logits": thinker_logits,
            "thinker_input_ids": thinker_input_ids,
            "thinker_attention_mask": thinker_attention_mask,
            "thinker_input_features": thinker_input_features,
            "thinker_feature_attention_mask": thinker_feature_attention_mask,
            "thinker_cache_position": thinker_cache_position,
            "thinker_flag": not thinker_flag if thinker_input_ids is not None else False,
            "thinker_eos": thinker_eos,
        }

    @torch.no_grad()
    def _register_prompt_embeddings(self):
        text_start_ids = torch.tensor([self.config.text_start_token_id], dtype=torch.long, device=self.device)
        text_start_emb = self._embed_text_tokens(text_start_ids).unsqueeze(0)
        self.register_buffer("text_start_emb", text_start_emb, persistent=False)

        text_end_ids = torch.tensor([self.config.text_end_token_id], dtype=torch.long, device=self.device)
        text_end_emb = self._embed_text_tokens(text_end_ids).unsqueeze(0)
        self.register_buffer("text_end_emb", text_end_emb, persistent=False)

        eos_token_audio = torch.zeros(
            (1, 1, self.config.backbone_config.hidden_size), dtype=text_start_emb.dtype, device=self.device
        )
        self.register_buffer("eos_token_audio", eos_token_audio, persistent=False)

        attention_mask = torch.ones(1, 1, dtype=torch.long, device=self.device)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

        arr = torch.arange(self.config.backbone_config.max_position_embeddings, device=self.device)
        self.register_buffer("arr", arr, persistent=False)

        self._prompt_embeddings_initialized = True

    def _build_prompt_embeds(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_values_cutoffs: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the voice cloning prompt of the backbone from the reference transcript and waveform.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Reference transcript ids.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask of the reference transcript.
            input_values (`torch.FloatTensor` of shape `(batch_size, 1, audio_length)`, *optional*):
                Reference waveform at the codec sample rate.
            input_values_cutoffs (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Number of valid samples of each reference waveform.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: The prompt embeddings of shape
            `(batch_size, prompt_length, hidden_size)` and their attention mask.

        Raises:
            ValueError: If the batch sizes of the reference transcript, waveform and masks disagree.
        """
        if not self._prompt_embeddings_initialized:
            self._register_prompt_embeddings()

        batch_size = input_ids.shape[0]
        for name, tensor in (
            ("input_values", input_values),
            ("input_values_cutoffs", input_values_cutoffs),
            ("attention_mask", attention_mask),
        ):
            if tensor.shape[0] != batch_size:
                raise ValueError(f"{name} has batch size {tensor.shape[0]}, expected {batch_size}")

        with torch.no_grad():
            audio_codes = self.codec_model.encode(input_values).audio_codes
        audio_codes = audio_codes[:, : self.config.audio_num_codebooks, :]

        prompt_audio_emb = self.backbone.emb_audio_frames(audio_codes.permute(0, 2, 1).to(self.device))
        prompt_audio_attention_mask = torch.ones((batch_size, prompt_audio_emb.shape[1]), device=self.device)

        audio_codes_cutoffs = torch.ceil(input_values_cutoffs / self.config.audio_frame_freq).long().unsqueeze(1)
        arr = self.arr[: prompt_audio_emb.shape[1]].unsqueeze(0).expand(batch_size, -1)
        prompt_audio_attention_mask[arr >= audio_codes_cutoffs] = 0

        prompt_text_emb = self._embed_text_tokens(input_ids.to(self.device))
        prompt_text_attention_mask = attention_mask.clone()

        input_embeddings = torch.cat(
            [
                self.text_start_emb.expand(batch_size, 1, -1),
                prompt_text_emb,
                self.text_end_emb.expand(batch_size, 1, -1),
                prompt_audio_emb,
                self.eos_token_audio.expand(batch_size, 1, -1),
            ],
            dim=1,
        )

        attention_mask = torch.cat(
            [
                self.attention_mask.expand(batch_size, 1),
                prompt_text_attention_mask,
                self.attention_mask.expand(batch_size, 1),
                prompt_audio_attention_mask,
                self.attention_mask.expand(batch_size, 1),
            ],
            dim=1,
        )

        return input_embeddings, attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> ChromaOutputWithPast:
        """
        Runs the backbone over the interleaved sequence and, during training, the decoder over the audio frames.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Unused, the backbone consumes `input_embeddings`.
            input_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Interleaved reasoner and audio frame embeddings, built by
                [`~ChromaForConditionalGeneration.prepare_inputs_for_generation`] during generation and by the
                training pipeline otherwise.
            attention_mask (`torch.Tensor` of shape `(batch_size, cached_length)`, *optional*):
                Backbone attention mask.
            past_key_values (`Cache`, *optional*):
                Backbone cache.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length, audio_num_codebooks)` or
                `(batch_size, sequence_length)`, *optional*):
                Codebook ids of the frame at each position, `-100` on the positions excluded from the loss.
                `labels[..., 0]` supervises the backbone and `labels[..., 1:]` supervises the decoder; a frame
                whose residual codebooks are all `-100` is skipped by the decoder. A 2D `labels` supervises the
                backbone alone.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether the backbone cache is returned.
            output_attentions (`bool`, *optional*):
                Whether attention weights are returned.
            output_hidden_states (`bool`, *optional*, defaults to `True`):
                Whether all backbone layer hidden states are returned. Forced on when `labels` is provided, since
                the decoder is conditioned on the last one.
            cache_position (`torch.LongTensor`, *optional*):
                Position of `input_embeddings` in the cached sequence.

        Returns:
            [`ChromaOutputWithPast`]: The codebook 0 logits and backbone state, plus, when `labels` is provided,
            `backbone_loss`, `decoder_loss` and their weighted sum in `loss`.
        """
        if labels is not None:
            output_hidden_states = True

        backbone_labels = labels[..., 0] if labels is not None and labels.dim() == 3 else labels

        backbone_outputs: CausalLMOutputWithPast = self.backbone(
            input_embeddings=input_embeddings,
            labels=backbone_labels,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        loss = backbone_outputs.loss
        decoder_outputs = None
        decoder_loss = None

        if labels is not None and labels.dim() == 3:
            decoder_outputs = self._forward_decoder(backbone_outputs.hidden_states[-1], labels)
            if decoder_outputs is not None:
                decoder_loss = decoder_outputs.loss
                weight = self.config.decoder_loss_weight
                loss = (1.0 - weight) * backbone_outputs.loss + weight * decoder_loss

        return self._build_outputs(
            loss=loss,
            logits=backbone_outputs.logits,
            hidden_states=backbone_outputs.hidden_states,
            past_key_values=backbone_outputs.past_key_values,
            attention_mask=attention_mask,
            backbone_loss=backbone_outputs.loss,
            backbone_logits=backbone_outputs.logits,
            backbone_hidden_states=backbone_outputs.hidden_states,
            backbone_attentions=backbone_outputs.attentions,
            backbone_past_key_values=backbone_outputs.past_key_values,
            decoder_loss=decoder_loss,
            decoder_logits=decoder_outputs.logits if decoder_outputs is not None else None,
            decoder_hidden_states=decoder_outputs.hidden_states if decoder_outputs is not None else None,
            decoder_attentions=decoder_outputs.attentions if decoder_outputs is not None else None,
            **kwargs,
        )

    def _forward_decoder(
        self, backbone_hidden_states: torch.Tensor, labels: torch.LongTensor
    ) -> Optional[CausalLMOutputWithPast]:
        """
        Runs the decoder over the audio frames of a training batch.

        Frames whose residual codebooks are all `-100` are dropped, the surviving ones are flattened into a single
        intra-frame batch, and each is conditioned on the backbone hidden state of the position that predicted it.

        Args:
            backbone_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Last backbone hidden state.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length, audio_num_codebooks)`):
                Codebook ids of the frame at each position.

        Returns:
            [`~modeling_outputs.CausalLMOutputWithPast`] or `None`: The decoder outputs, or `None` when the batch
            holds no supervised audio frame.
        """
        num_codebooks = self.config.audio_num_codebooks
        train_mask = ~(labels[..., 1:] == -100).all(dim=-1)
        if not train_mask.any():
            return None

        frame_labels = labels[train_mask]
        batch_idxs, position_idxs = train_mask.nonzero(as_tuple=True)
        backbone_last_hidden_state = backbone_hidden_states[batch_idxs, position_idxs - 1, :]

        return self.decoder(
            input_ids=frame_labels[:, : num_codebooks - 1],
            backbone_last_hidden_state=backbone_last_hidden_state,
            labels=frame_labels[:, 1:],
            use_cache=False,
        )

    def _build_outputs(self, **kwargs) -> ChromaOutputWithPast:
        fields_names = [f.name for f in fields(ChromaOutputWithPast)]
        return ChromaOutputWithPast(**{k: v for k, v in kwargs.items() if k in fields_names})

    def _update_model_kwargs_for_generation(
        self,
        outputs: ChromaOutputWithPast,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        for key in PASSTHROUGH_KEYS:
            model_kwargs[key] = getattr(outputs, key, None)

        for key in ONE_TIME_KEYS:
            model_kwargs[key] = None

        # the backbone grows by one frame per step regardless of how many reasoner positions were interleaved
        return super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, 1)

    def _update_thinker_model_kwargs(
        self,
        thinker_input_ids: torch.Tensor,
        thinker_attention_mask: Optional[torch.Tensor] = None,
        thinker_cache_position: Optional[torch.Tensor] = None,
        thinker_past_key_values: Optional[Cache] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Cache]]:
        """
        Args:
            thinker_input_ids (`torch.LongTensor` of shape `(batch_size, num_new_tokens)`):
                Reasoner tokens consumed on this step.
            thinker_attention_mask (`torch.LongTensor`, *optional*):
                Reasoner attention mask of the previous step.
            thinker_cache_position (`torch.LongTensor`, *optional*):
                Reasoner cache positions of the previous step.
            thinker_past_key_values (`Cache`, *optional*):
                Reasoner cache.

        Returns:
            `tuple`: The reasoner input ids, attention mask, cache positions and cache, advanced by
            `num_new_tokens`.
        """
        past_seen_tokens = thinker_past_key_values.get_seq_length() if thinker_past_key_values is not None else 0
        num_new_tokens = thinker_input_ids.shape[1]

        if thinker_cache_position is None:
            thinker_cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + num_new_tokens, device=thinker_input_ids.device
            )
        else:
            thinker_cache_position = thinker_cache_position[-num_new_tokens:] + num_new_tokens

        if thinker_attention_mask is None:
            thinker_attention_mask = torch.ones(
                (thinker_input_ids.shape[0], num_new_tokens), device=thinker_input_ids.device
            )
        elif thinker_past_key_values is not None:
            thinker_attention_mask = torch.cat(
                [
                    thinker_attention_mask,
                    thinker_attention_mask.new_ones((thinker_attention_mask.shape[0], num_new_tokens)),
                ],
                dim=-1,
            )

        return thinker_input_ids, thinker_attention_mask, thinker_cache_position, thinker_past_key_values


__all__ = [
    "ChromaBackboneForCausalLM",
    "ChromaDecoderForCausalLM",
    "ChromaForConditionalGeneration",
    "ChromaLlamaModel",
    "ChromaPreTrainedModel",
]
