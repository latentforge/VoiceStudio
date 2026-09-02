# Copyright 2026 Xiaomi Corporation and the LatentForge team. All rights reserved.
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
"""PyTorch OmniVoice model."""

import torch
import torch.nn as nn

from transformers.conversion_mapping import WeightRenaming, register_checkpoint_conversion_mapping
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    AttentionMaskInterface,
    create_bidirectional_mask,
    packed_sequence_mask_function,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, MaskedLMOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.higgs_audio_v2.modeling_higgs_audio_v2 import (
    HiggsAudioV2Embeddings,
    HiggsAudioV2PreTrainedModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging

from .configuration_omnivoice import OmniVoiceConfig
from .generation_omnivoice import OmniVoiceGenerationMixin


logger = logging.get_logger(__name__)


_AUTOCAST_FLEX_ATTENTION = "omnivoice_flex_attention"


def _autocast_flex_attention(module, query, key, value, *args, **kwargs):
    """
    Runs `flex_attention` with the autocast treatment SDPA already gets.

    Under mixed precision the fp32 `q_norm`/`k_norm` weights of a Qwen3 backbone promote q/k, and the fp32 rotary
    constants promote v, back to fp32. SDPA is on autocast's cast list and is downcast at the kernel boundary,
    while `flex_attention` is not, so all attention math would otherwise run in fp32: the fp32 backward template is
    roughly 12x slower at `head_dim=128`, and the flex and sdpa paths stop agreeing numerically. Softmax
    accumulation inside the kernel stays fp32 either way.
    """
    if torch.is_autocast_enabled(query.device.type) and query.dtype == torch.float32:
        dtype = torch.get_autocast_dtype(query.device.type)
        query, key, value = (tensor.to(dtype) for tensor in (query, key, value))
    return ALL_ATTENTION_FUNCTIONS["flex_attention"](module, query, key, value, *args, **kwargs)


AttentionInterface.register(_AUTOCAST_FLEX_ATTENTION, _autocast_flex_attention)
AttentionMaskInterface.register(_AUTOCAST_FLEX_ATTENTION, ALL_MASK_ATTENTION_FUNCTIONS["flex_attention"])


# The `k2-fsa/OmniVoice` checkpoint stores the backbone under a bare `llm.` prefix and the fused audio embedding,
# its codebook offsets and the fused output head at the top level.
register_checkpoint_conversion_mapping(
    "OmniVoiceForConditionalGeneration",
    [
        WeightRenaming(source_patterns=r"^llm\.", target_patterns=r"model.llm."),
        WeightRenaming(
            source_patterns=r"^audio_embeddings\.weight$",
            target_patterns=r"model.audio_embeddings.embed_audio_tokens.weight",
        ),
        WeightRenaming(
            source_patterns=r"^codebook_layer_offsets$",
            target_patterns=r"model.audio_embeddings.audio_tokens_offsets",
        ),
    ],
    overwrite=True,
)


class OmniVoiceAudioEmbeddings(HiggsAudioV2Embeddings):
    def __init__(self, config: OmniVoiceConfig):
        super().__init__(config)
        # The released checkpoint ships the offsets, so they have to round-trip through `state_dict`.
        self.audio_tokens_offsets = nn.Buffer(self.audio_tokens_offsets.clone(), persistent=True)

    def forward(self, input_ids: torch.LongTensor, audio_mask: torch.BoolTensor) -> torch.Tensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                Interleaved text and audio ids. Text positions carry text-vocabulary ids, which exceed the fused
                audio table, so they are zeroed before the lookup and discarded by the caller.
            audio_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
                Marks the positions of `input_ids` that hold audio frames.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: The per-frame sum of the
            codebook embeddings.
        """
        audio_ids = (input_ids * audio_mask.unsqueeze(1)).transpose(1, 2)
        return super().forward(audio_ids)


@auto_docstring
class OmniVoicePreTrainedModel(HiggsAudioV2PreTrainedModel):
    config: OmniVoiceConfig
    base_model_prefix = "model"
    _no_split_modules = []
    _can_record_outputs = {}


@auto_docstring(
    custom_intro="""
    The bare OmniVoice model, a language model backbone fed a single stream of text embeddings and fused
    multi-codebook audio embeddings.
    """
)
class OmniVoiceModel(OmniVoicePreTrainedModel):
    def __init__(self, config: OmniVoiceConfig):
        super().__init__(config)
        self.llm = AutoModel.from_config(config.llm_config)
        if self.llm.config._attn_implementation == "flex_attention":
            self.llm.set_attn_implementation(_AUTOCAST_FLEX_ATTENTION)
        self.audio_embeddings = OmniVoiceAudioEmbeddings(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.BoolTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        document_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`):
            Interleaved text and audio ids. Text positions repeat the same text token id across every codebook
            row; audio positions hold one code per codebook row.
        audio_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Marks the positions of `input_ids` that hold audio frames rather than text tokens.
        document_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Sample index of every position when several samples are packed into one sequence. Positions only
            attend to positions carrying the same index.

        Returns:
            [`~modeling_outputs.BaseModelOutputWithPast`]: The backbone hidden states, with audio frames already
            embedded.
        """
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])
        audio_embeds = self.audio_embeddings(input_ids, audio_mask)
        inputs_embeds = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

        # OmniVoice predicts a masked canvas rather than a next token, so every position attends in both
        # directions. `allow_is_bidirectional_skip=False` forces the mask to be materialized: an unset mask makes
        # the backbone fall back to building a causal one.
        attention_mask = create_bidirectional_mask(
            config=self.llm.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            and_mask_function=(
                packed_sequence_mask_function(document_ids) if document_ids is not None else None
            ),
            allow_is_bidirectional_skip=False,
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,
        )


@auto_docstring(
    custom_intro="""
    OmniVoice with a fused multi-codebook output head, generating audio frames by iterative unmasking.
    """
)
class OmniVoiceForConditionalGeneration(OmniVoicePreTrainedModel, OmniVoiceGenerationMixin):
    base_model_prefix = "model"

    def __init__(self, config: OmniVoiceConfig):
        super().__init__(config)
        self.model = OmniVoiceModel(config)
        self.audio_heads = nn.Linear(
            config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.BoolTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        document_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`):
            Interleaved text and audio ids, as built by [`OmniVoiceProcessor`].
        audio_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Marks the positions of `input_ids` that hold audio frames rather than text tokens.
        document_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Sample index of every position when several samples are packed into one sequence.
        labels (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`, *optional*):
            Target codes for the masked audio frames. Indices should be in `[0, ..., config.audio_vocab_size - 1]`;
            positions set to `-100` are ignored. Can be obtained with `output_labels=True` when calling
            [`OmniVoiceProcessor`].

        Returns:
            [`~modeling_outputs.MaskedLMOutput`]: The per-codebook logits, of shape
            `(batch_size, num_codebooks, sequence_length, audio_vocab_size)`, and the codebook-weighted
            cross-entropy loss when `labels` is provided.

        Example:

        ```python
        >>> from voicestudio.models.ommivoice import OmniVoiceForConditionalGeneration, OmniVoiceProcessor

        >>> model_id = "k2-fsa/OmniVoice"
        >>> processor = OmniVoiceProcessor.from_pretrained(model_id)
        >>> model = OmniVoiceForConditionalGeneration.from_pretrained(model_id)

        >>> inputs = processor(text="The sun rises in the east.", instruct="female, british accent")
        >>> audio_codes = model.generate(**inputs)
        >>> waveform = processor.batch_decode(audio_codes)[0]
        ```
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        batch_size, seq_length = hidden_states.shape[:2]
        logits = self.audio_heads(hidden_states).view(
            batch_size, seq_length, self.config.num_audio_codebook, self.config.audio_vocab_size
        )
        logits = logits.permute(0, 2, 1, 3)

        loss = None
        if labels is not None:
            per_token_loss = nn.functional.cross_entropy(
                logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=-100
            )
            valid_mask = (labels != -100).to(per_token_loss.dtype)
            codebook_losses = (per_token_loss * valid_mask).sum(dim=(0, 2)) / valid_mask.sum(dim=(0, 2)).clamp(
                min=1.0
            )
            weights = torch.tensor(
                self.config.audio_codebook_weights, device=logits.device, dtype=codebook_losses.dtype
            )
            loss = (codebook_losses * weights / weights.sum()).sum()

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "OmniVoiceForConditionalGeneration",
    "OmniVoiceModel",
    "OmniVoicePreTrainedModel",
]
