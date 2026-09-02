# Copyright 2026 Boson AI and the LatentForge team. All rights reserved.
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
"""PyTorch Higgs TTS 3 model."""

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.conversion_mapping import WeightRenaming, register_checkpoint_conversion_mapping
from transformers.models.higgs_audio_v2.modeling_higgs_audio_v2 import (
    HiggsAudioV2Embeddings,
    HiggsAudioV2PreTrainedModel,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3Model
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging

from .configuration_higgs_tts3 import HiggsTTS3Config


logger = logging.get_logger(__name__)


# The upstream `bosonai/higgs-tts-3-4b` checkpoint stores the Qwen3 text backbone under a bare
# `body.` prefix and keeps the text/audio embedding and (tied) output-head weights under a `tied.`
# namespace shared with the full neural audio codec (semantic model, acoustic encoder/decoder,
# quantizer). Only the text backbone and the two embedding tables are relevant to this model; the
# codec weights are intentionally left unmapped (they surface as harmless UNEXPECTED keys) since
# they belong to a separate audio tokenizer model, not to `HiggsTTS3ForConditionalGeneration`.
register_checkpoint_conversion_mapping(
    "HiggsTTS3ForConditionalGeneration",
    [
        WeightRenaming(source_patterns=r"^body\.layers\.", target_patterns=r"model.text_model.layers."),
        WeightRenaming(source_patterns=r"^body\.norm\.weight$", target_patterns=r"model.text_model.norm.weight"),
        WeightRenaming(
            source_patterns=r"^tied\.embedding\.text_embedding\.weight$",
            target_patterns=r"model.text_model.embed_tokens.weight",
        ),
        WeightRenaming(
            source_patterns=r"^tied\.embedding\.modality_embeddings\.0\.embedding\.weight$",
            target_patterns=r"model.embed_audio_tokens.embed_audio_tokens.weight",
        ),
        WeightRenaming(source_patterns=r"^tied\.head\.text_head\.weight$", target_patterns=r"text_lm_head.weight"),
        WeightRenaming(source_patterns=r"^tied\.head\.modality_heads\.0\.weight$", target_patterns=r"audio_head.weight"),
    ],
    overwrite=True,
)


@auto_docstring
class HiggsTTS3PreTrainedModel(HiggsAudioV2PreTrainedModel):
    config: HiggsTTS3Config
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3DecoderLayer"]
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


@auto_docstring
class HiggsTTS3Model(HiggsTTS3PreTrainedModel):
    def __init__(self, config: HiggsTTS3Config):
        super().__init__(config)
        self.text_model = Qwen3Model(config.text_config)
        self.embed_audio_tokens = HiggsAudioV2Embeddings(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.embed_tokens = value

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.BoolTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        audio_input_ids (`torch.LongTensor` of shape `(batch_size, num_audio_frames, num_codebooks)`, *optional*):
            Indices of audio codebook tokens.

            Indices can be obtained using [`HiggsAudioV2TokenizerModel.encode`].
        audio_input_ids_mask (`torch.BoolTensor` of shape `(batch_size, num_audio_frames)`, *optional*):
            Indicates which audio frames in `audio_input_ids` are valid.

        Returns:
            [`~models.modeling_outputs.BaseModelOutputWithPast`]:
                Usual decoder outputs with the placeholder positions already substituted by their corresponding
                audio embeddings.
        """
        if (input_ids is None) and (inputs_embeds is None) and (audio_input_ids is None):
            raise ValueError("You must specify at least one of input_ids, inputs_embeds, or audio_input_ids")

        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError("Only one of input_ids or inputs_embeds can be provided")

        audio_token_mask = self.get_placeholder_mask(input_ids, inputs_embeds, audio_input_ids_mask)

        if input_ids is not None:
            safe_input_ids = input_ids.masked_fill(input_ids == self.config.audio_token_id, 0)
            inputs_embeds = self.text_model.embed_tokens(safe_input_ids)

        if audio_input_ids is not None:
            audio_embeds = self.embed_audio_tokens(audio_input_ids)

        if inputs_embeds is not None and audio_input_ids is not None:
            audio_embeds = (
                audio_embeds[audio_input_ids_mask.to(audio_embeds.device)]
                if audio_input_ids_mask is not None
                else audio_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask[..., None], audio_embeds.to(inputs_embeds.device)
            )
        elif audio_input_ids is not None:
            inputs_embeds = audio_embeds

        return self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, audio_input_ids_mask: torch.LongTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of audio_input_ids. If the lengths are different, an error is raised.

        If input_ids and inputs_embeds are None, we return None.
        Indeed this means we cannot determine the placeholder mask, the model is to be used in a audio-only mode, hence we return None.
        """
        if input_ids is None and inputs_embeds is None:
            return None

        elif input_ids is None:
            return None

        return input_ids == self.config.audio_token_id


@auto_docstring(
    custom_intro="""
    The Higgs TTS 3 model, a Qwen3-backbone auto-regressive transformer paired with a fused multi-codebook
    audio embedding and output head.
    """
)
class HiggsTTS3ForConditionalGeneration(HiggsTTS3PreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = {"audio_head.weight": "model.embed_audio_tokens.embed_audio_tokens.weight"}

    def __init__(self, config: HiggsTTS3Config, use_text_head: bool = True):
        r"""
        use_text_head (`bool`, *optional*, defaults to True):
            Whether to instantiate a text language model head. Such head is not required for generation,
            but is used to compute the text loss when `labels` are passed to `forward`. Set to False only
            to save the (tied-vocab-size) parameters when text-side training is not needed.
        """
        super().__init__(config)
        self.model = HiggsTTS3Model(config)
        self.audio_head = nn.Linear(config.hidden_size, config.num_codebooks * config.codebook_size, bias=False)
        self.text_lm_head = (
            nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False) if use_text_head else None
        )

        self.post_init()

    def tie_weights(self, *args, **kwargs):
        super().tie_weights(*args, **kwargs)
        if self.config.audio_encoder_config.tie_word_embeddings:
            self.audio_head.weight = self.model.embed_audio_tokens.embed_audio_tokens.weight

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        audio_input_ids (`torch.LongTensor` of shape `(batch_size, num_audio_frames, num_codebooks)`, *optional*):
            Indices of audio codebook tokens.

            Indices can be obtained using [`HiggsAudioV2TokenizerModel.encode`].
        audio_input_ids_mask (`torch.BoolTensor` of shape `(batch_size, num_audio_frames)`, *optional*):
            Indicates which audio frames in `audio_input_ids` are valid.
        audio_labels (`torch.LongTensor` of shape `(batch_size, num_audio_frames, num_codebooks)`, *optional*):
            Labels for the audio codebook tokens for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.codebook_size]. Token with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.codebook_size]`.
            Can be obtained using `output_labels=True` when calling [`HiggsTTS3Processor`].

        Returns:
            [`~models.modeling_outputs.CausalLMOutputWithPast`]:
                A [`~models.modeling_outputs.CausalLMOutputWithPast`] containing the logits, loss (if labels are provided),
                and other outputs from the model.

        Example:

        ```python
        >>> from voicestudio.models.higgs_tts3 import HiggsTTS3ForConditionalGeneration, HiggsTTS3Processor
        >>> model_id = "bosonai/higgs-tts-3-4b"
        >>> processor = HiggsTTS3Processor.from_pretrained(model_id)
        >>> model = HiggsTTS3ForConditionalGeneration.from_pretrained(model_id)
        >>> inputs = processor(text="The sun rises in the east.", return_tensors="pt")
        >>> outputs = model(**inputs)
        ```
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.audio_head(hidden_states[:, slice_indices, :])

        loss = None
        if audio_labels is not None:
            audio_logits = logits.reshape(*logits.shape[:2], self.config.num_codebooks, self.config.codebook_size)
            if input_ids is not None:
                label_batch, label_seq = input_ids.shape[:2]
                audio_token_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds, audio_input_ids_mask)
            else:
                # No `input_ids` (e.g. `inputs_embeds`-only training): the whole sequence is audio.
                label_batch, label_seq = inputs_embeds.shape[:2]
                audio_token_mask = torch.ones(
                    (label_batch, label_seq), dtype=torch.bool, device=inputs_embeds.device
                )
            audio_labels_expanded = audio_labels.new_full((label_batch, label_seq, self.config.num_codebooks), -100)
            valid_audio_labels = (
                audio_labels[audio_input_ids_mask] if audio_input_ids_mask is not None else audio_labels
            )
            audio_labels_expanded[audio_token_mask] = valid_audio_labels.reshape(-1, self.config.num_codebooks)

            codebook_losses = []
            for codebook_idx in range(self.config.num_codebooks):
                codebook_logits = audio_logits[:, :, codebook_idx, :]
                codebook_labels = audio_labels_expanded[:, :, codebook_idx]
                codebook_losses.append(
                    self.loss_function(codebook_logits, codebook_labels, self.config.codebook_size, **kwargs)
                )

            loss = sum(codebook_losses)

        if labels is not None:
            if self.text_lm_head is not None:
                text_logits = self.text_lm_head(hidden_states[:, slice_indices, :])
                text_loss = self.loss_function(text_logits, labels, self.config.text_config.vocab_size, **kwargs)
                loss = text_loss if loss is None else loss + text_loss
            else:
                logger.warning_once(
                    f"`labels` provided to {self.__class__.__name__} but `text_lm_head` is disabled. "
                    f"Text labels ignored. Set `use_text_head=True` in model init to enable text loss."
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.LongTensor | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_k: int | None = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Autoregressively samples audio codebook tokens in Higgs' delay pattern, until codebook 0 emits
        `config.audio_stream_eos_id` and the remaining codebooks have caught up, or `max_new_tokens` steps
        are reached.

        Args:
            input_ids (`torch.LongTensor` of shape `(1, sequence_length)`):
                Prompt token ids, with `config.audio_token_id` placeholders for `audio_input_ids` frames.
            attention_mask (`torch.LongTensor` of shape `(1, sequence_length)`, *optional*):
                Mask to avoid attending to padding tokens.
            audio_input_ids (`torch.LongTensor` of shape `(1, num_audio_frames, num_codebooks)`, *optional*):
                Reference-audio codes to condition generation on, as built by [`HiggsTTS3Processor`].
            audio_input_ids_mask (`torch.BoolTensor` of shape `(1, num_audio_frames)`, *optional*):
                Indicates which frames in `audio_input_ids` are valid.
            max_new_tokens (`int`, *optional*, defaults to 2048):
                Maximum number of delayed codebook rows to sample.
            temperature (`float`, *optional*, defaults to 1.0):
                Sampling temperature. `0` selects the highest-probability code at every step.
            top_k (`int`, *optional*):
                Number of highest-probability codes to sample from, per codebook.

        Returns:
            `torch.LongTensor` of shape `(1, num_generated_frames, num_codebooks)`: The generated, delayed audio
            codes, ready for [`HiggsTTS3Processor.decode`].
        """
        if input_ids.shape[0] != 1:
            raise ValueError("HiggsTTS3ForConditionalGeneration.generate only supports batch_size=1.")

        num_codebooks = self.config.num_codebooks
        bos_id = self.config.audio_stream_bos_id
        eos_id = self.config.audio_stream_eos_id

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden_last = outputs.last_hidden_state[:, -1:, :]
        position = input_ids.shape[1]

        rows = []
        delay_step = 0
        eos_countdown = None
        for _ in range(max_new_tokens):
            logits = self.audio_head(hidden_last).reshape(num_codebooks, self.config.codebook_size).float()
            if temperature <= 1e-5:
                codes = logits.argmax(dim=-1)
            else:
                probs = (logits / temperature).softmax(dim=-1)
                if top_k is not None:
                    top_probs, top_indices = probs.topk(min(top_k, probs.shape[-1]), dim=-1)
                    codes = top_indices.gather(-1, torch.multinomial(top_probs, num_samples=1)).squeeze(-1)
                else:
                    codes = torch.multinomial(probs, num_samples=1).squeeze(-1)

            done = False
            if delay_step < num_codebooks:
                # Codebook c only starts carrying real codes at step c, so every codebook above the one
                # this step opens is pinned to the beginning-of-codebook id instead of its sampled value.
                codes[delay_step + 1 :] = bos_id
                delay_step += 1
            elif eos_countdown is not None:
                eos_countdown -= 1
                done = eos_countdown <= 0
            elif codes[0] == eos_id:
                # Codebook 0 runs `num_codebooks - 1` frames ahead of the last codebook, and the final
                # row is emitted below, so `num_codebooks - 2` more rows are needed to flush the tail.
                if num_codebooks > 2:
                    eos_countdown = num_codebooks - 2
                else:
                    done = True
            rows.append(codes)

            if done:
                break

            cache_position = torch.tensor([position], device=input_ids.device)
            step_embeds = self.model.embed_audio_tokens(codes.unsqueeze(0).unsqueeze(0))
            outputs = self.model.text_model(
                inputs_embeds=step_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
            past_key_values = outputs.past_key_values
            hidden_last = outputs.last_hidden_state[:, -1:, :]
            position += 1

        return torch.stack(rows, dim=0).unsqueeze(0)


__all__ = ["HiggsTTS3ForConditionalGeneration", "HiggsTTS3Model", "HiggsTTS3PreTrainedModel"]
