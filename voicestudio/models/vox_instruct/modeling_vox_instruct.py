# coding=utf-8
# Copyright 2024 The VoxInstruct Authors and the HuggingFace Inc. team. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""PyTorch VoxInstruct model."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.encodec.modeling_encodec import EncodecModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from transformers.models.mt5.modeling_mt5 import MT5EncoderModel
from transformers.utils import auto_docstring, logging

from .configuration_vox_instruct import VoxInstructARConfig, VoxInstructConfig, VoxInstructNARConfig
from .generation_vox_instruct import VoxInstructGenerationMixin
from .tokenization_vox_instruct import VoxInstructSemanticTokenizerModel


logger = logging.get_logger(__name__)


@dataclass
class VoxInstructARCausalLMOutputWithPast(ModelOutput):
    r"""
    Output of the VoxInstruct autoregressive stage.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Cross entropy of the next flat token, averaged over the unpadded target positions.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Prediction scores aligned with `input_ids`, with the text prefix positions already removed.
        past_key_values (`Cache`, *optional*):
            Key value cache of the decoder, holding the text prefix as well as the token positions.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, max_text_len, hidden_size)`, *optional*):
            Projected text encoder output prepended to the decoder inputs. Pass it back on the next call to skip the
            text encoder.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    text_embeds: torch.FloatTensor | None = None


@dataclass
class VoxInstructNAROutput(ModelOutput):
    r"""
    Output of the VoxInstruct non-autoregressive stage.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Cross entropy of the masked positions of the drawn residual codebook.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, acoustic_vocab_size + 3)`):
            Prediction scores of the residual codebook selected by `codebook_index`.
        codebook_index (`torch.LongTensor` of shape `(batch_size,)`):
            Index of the residual codebook the logits belong to.
        loss_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*, returned when `labels` is
        provided):
            Positions the loss was averaged over, that is the masked positions inside the unpadded span.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, max_text_len, hidden_size)`, *optional*):
            Projected text encoder output prepended to the backbone inputs.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    codebook_index: torch.LongTensor | None = None
    loss_mask: torch.BoolTensor | None = None
    text_embeds: torch.FloatTensor | None = None


@dataclass
class VoxInstructOutput(ModelOutput):
    r"""
    Output of the full VoxInstruct model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when a stage receives labels):
            Sum of the stage losses that were computed. The two stages are trained separately upstream, so exactly one
            of them is normally present and `loss` then equals that stage's loss.
        ar_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss of the autoregressive stage.
        nar_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss of the non-autoregressive stage.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`, *optional*):
            Prediction scores of the autoregressive stage.
        nar_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, acoustic_vocab_size + 3)`, *optional*):
            Prediction scores of the non-autoregressive stage.
        past_key_values (`Cache`, *optional*):
            Key value cache of the autoregressive stage.
    """

    loss: torch.FloatTensor | None = None
    ar_loss: torch.FloatTensor | None = None
    nar_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    nar_logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None


class VoxInstructLoraLinear(nn.Module):
    r"""
    Wraps a linear layer with a low rank adapter, so that only the adapter is trained while the wrapped weight stays
    frozen.

    Args:
        base_layer (`nn.Linear`):
            The layer to adapt. Its weight is kept as is.
        rank (`int`):
            Inner dimension of the adapter.
        alpha (`int`):
            Scaling numerator. The adapter output is scaled by `alpha / rank`.
        dropout (`float`):
            Dropout applied to the adapter input.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base_layer = base_layer
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        self.scaling = alpha / rank

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        adapted = self.lora_B(self.lora_A(self.lora_dropout(hidden_states)))
        return self.base_layer(hidden_states) + adapted * self.scaling


class VoxInstructTextEncoder(MT5EncoderModel):
    r"""
    Constructs the VoxInstruct instruction encoder, an mT5 encoder whose body is frozen and whose query and value
    projections carry, when `config.use_lora` is set, the low rank adapters that hold the whole trainable part of the
    instruction path.

    Args:
        config ([`VoxInstructARConfig`]):
            Configuration of the stage the encoder belongs to. Its `text_encoder_config` defines the mT5 architecture.
    """

    def __init__(self, config: VoxInstructARConfig):
        super().__init__(config.text_encoder_config)
        if config.use_lora:
            for block in self.encoder.block:
                attention = block.layer[0].SelfAttention
                for projection in ("q", "v"):
                    adapted = VoxInstructLoraLinear(
                        getattr(attention, projection), config.lora_rank, config.lora_alpha, config.lora_dropout
                    )
                    setattr(attention, projection, adapted)
        self.freeze()

    def freeze(self):
        """Freezes the encoder body, leaving only the low rank adapters trainable."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad = "lora_" in name


class VoxInstructNARBackbone(LlamaModel):
    r"""
    Llama backbone of the non-autoregressive stage. Every position attends to every other one, since the stage reads a
    whole masked codebook grid at once instead of decoding it left to right.
    """

    config: VoxInstructNARConfig

    def __init__(self, config: VoxInstructNARConfig):
        super().__init__(config)
        # Flash attention reads causality off the attention module rather than off the mask it is handed.
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def forward(self, attention_mask=None, inputs_embeds=None, **kwargs):
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            allow_is_bidirectional_skip=False,
        )
        return super().forward(attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


class VoxInstructTokenEmbedding(nn.Embedding):
    r"""
    Embedding of the flat VoxInstruct token space, truncated normal initialized with a zeroed padding row.
    """


class VoxInstructPreTrainedModel(PreTrainedModel):
    config: VoxInstructConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer", "MT5Block", "HubertEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        r"""
        Args:
            args:
                Forwarded to [`~PreTrainedModel.from_pretrained`].
            kwargs:
                Forwarded to [`~PreTrainedModel.from_pretrained`].

        Returns:
            The loaded model, frozen again after loading replaced the parameters created by `__init__`.
        """
        outputs = super().from_pretrained(*args, **kwargs)
        model = outputs[0] if isinstance(outputs, tuple) else outputs
        model.freeze_encoders()
        return outputs

    def freeze_encoders(self):
        """Freezes the instruction encoder body, leaving only its low rank adapters trainable."""
        self.text_encoder.freeze()

    def encode_text(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        r"""
        Encodes the instruction and projects it onto the decoder hidden size.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, max_text_len)`):
                Instruction token ids, right padded to `config.max_text_len`.
            attention_mask (`torch.Tensor` of shape `(batch_size, max_text_len)`):
                Mask over the instruction tokens.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, max_text_len, hidden_size)`: The projected encoding, zeroed on
            the padded positions.
        """
        hidden_states = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden_states = self.prompt_fc(hidden_states)
        return torch.where((attention_mask > 0).unsqueeze(-1), hidden_states, torch.zeros_like(hidden_states))

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, VoxInstructLoraLinear):
            nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B.weight)
        elif isinstance(module, VoxInstructTokenEmbedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.weight[module.padding_idx])


@auto_docstring(
    custom_intro="""
    The autoregressive VoxInstruct stage. It prepends the encoded instruction to the flat token sequence and predicts,
    left to right, the language identity, the semantic tokens and the first EnCodec codebook.
    """
)
class VoxInstructARForCausalLM(VoxInstructPreTrainedModel):
    config: VoxInstructARConfig
    _tied_weights_keys = None

    def __init__(self, config: VoxInstructARConfig):
        super().__init__(config)
        self.text_encoder = VoxInstructTextEncoder(config)
        self.prompt_fc = nn.Linear(config.text_encoder_config.d_model, config.hidden_size)
        self.embed_tokens = VoxInstructTokenEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_segments = nn.Embedding(config.num_segment_ids, config.hidden_size)
        self.model = LlamaForCausalLM(config)
        self.post_init()
        self.freeze_encoders()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def _drop_conditioning(
        self, text_embeds: torch.Tensor, input_ids: torch.LongTensor, segment_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """Applies the two training time conditioning dropouts that train the guidance branches."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        drop_text = torch.rand((batch_size,), device=device) < self.config.text_free_guidance_ratio
        text_embeds = torch.where(
            drop_text[:, None, None].expand_as(text_embeds), torch.zeros_like(text_embeds), text_embeds
        )

        drop_semantic = torch.rand((batch_size,), device=device) < self.config.semantic_free_guidance_ratio
        drop_semantic = (segment_ids == 1) & drop_semantic[:, None].expand_as(segment_ids)
        input_ids = torch.where(drop_semantic, torch.zeros_like(input_ids), input_ids)
        return text_embeds, input_ids

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        text_input_ids: torch.LongTensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        text_embeds: torch.FloatTensor | None = None,
        mask_semantic: bool = False,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> VoxInstructARCausalLMOutputWithPast:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Flat token ids laid out as `<bos> <language> <semantic...> <eos> <acoustic...> <eos + 1>`.
            segment_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                `1` on the span up to and including the semantic end of sequence token, `2` on the acoustic span.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask over `input_ids`. The text prefix is always attended.
            text_input_ids (`torch.LongTensor` of shape `(batch_size, max_text_len)`, *optional*):
                Instruction token ids. Not needed when `text_embeds` is given.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, max_text_len)`, *optional*):
                Mask over `text_input_ids`.
            text_embeds (`torch.FloatTensor` of shape `(batch_size, max_text_len, hidden_size)`, *optional*):
                Already encoded instruction, which skips the text encoder. Pass zeros for the unconditional branch of
                classifier free guidance.
            mask_semantic (`bool`, *optional*, defaults to `False`):
                Whether to replace the semantic span of `input_ids` with the padding token, which gives the branch of
                classifier free guidance that drops the semantic conditioning.
            past_key_values (`Cache`, *optional*):
                Cache holding the text prefix and the tokens decoded so far. When it is non empty only the last
                position of `input_ids` is forwarded.
            use_cache (`bool`, *optional*):
                Whether to return the updated cache.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Targets for the next token loss, with `-100` on the positions to ignore. They are shifted inside the
                model, so `labels` is normally `input_ids` with the padding replaced by `-100`.

        Returns:
            [`VoxInstructARCausalLMOutputWithPast`]
        """
        if text_embeds is None:
            text_embeds = self.encode_text(text_input_ids, text_attention_mask)

        if labels is not None:
            text_embeds, input_ids = self._drop_conditioning(text_embeds, input_ids, segment_ids)
        elif mask_semantic:
            input_ids = torch.where(segment_ids == 1, torch.zeros_like(input_ids), input_ids)

        cached_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cached_length > 0:
            inputs_embeds = self.embed_tokens(input_ids[:, -1:]) + self.embed_segments(segment_ids[:, -1:])
            prefix_length = 0
        else:
            token_embeds = self.embed_tokens(input_ids) + self.embed_segments(segment_ids)
            prefix_length = text_embeds.shape[1]
            inputs_embeds = torch.cat(
                [text_embeds + self.embed_segments.weight[0], token_embeds],
                dim=1,
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask.new_ones(text_embeds.shape[:2]), attention_mask], dim=1
                )

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask if cached_length == 0 else None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = outputs.logits[:, prefix_length:].contiguous()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return VoxInstructARCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            text_embeds=text_embeds,
        )


@auto_docstring(
    custom_intro="""
    The non-autoregressive VoxInstruct stage. It reads the whole codebook grid at once and fills in the masked
    positions of one residual codebook, in the manner of SoundStorm.
    """
)
class VoxInstructNARModel(VoxInstructPreTrainedModel):
    config: VoxInstructNARConfig

    def __init__(self, config: VoxInstructNARConfig):
        super().__init__(config)
        self.text_encoder = VoxInstructTextEncoder(config)
        self.prompt_fc = nn.Linear(config.text_encoder_config.d_model, config.hidden_size)
        self.all_embed_tokens = nn.ModuleList(
            [
                VoxInstructTokenEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
                for _ in range(config.num_codebooks)
            ]
        )
        self.embed_segments = nn.Embedding(config.num_segment_ids, config.hidden_size)
        self.embed_res = nn.Embedding(config.num_codebooks - 1, config.hidden_size)
        self.model = VoxInstructNARBackbone(config)
        self.out_proj = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.acoustic_vocab_size + 3, bias=False)
                for _ in range(config.num_codebooks - 1)
            ]
        )
        self.post_init()
        self.freeze_encoders()

    def get_input_embeddings(self):
        return self.all_embed_tokens[0]

    def set_input_embeddings(self, value):
        self.all_embed_tokens[0] = value

    def random_masking(
        self, semantic_lengths: torch.LongTensor, acoustic_lengths: torch.LongTensor, max_length: int
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        r"""
        Draws the acoustic prompt and the positions of the target codebook that are hidden, following the schedule of
        SoundStorm.

        A prompt of `u * acoustic_length` frames stays visible, with `u` uniform or, with probability
        `config.acoustic_free_guidance_ratio`, zero. Of the frames past it, a `cos(pi / 2 * v)` fraction with `v`
        uniform is hidden, and those are the ones the loss is taken over.

        Args:
            semantic_lengths (`torch.LongTensor` of shape `(batch_size,)`):
                Number of positions in the semantic span, including the leading tokens.
            acoustic_lengths (`torch.LongTensor` of shape `(batch_size,)`):
                Number of positions in the acoustic span.
            max_length (`int`):
                Padded sequence length.

        Returns:
            `tuple[torch.BoolTensor, torch.BoolTensor]`: a mask that is `True` on the positions of the target codebook
            that stay visible, and a mask that is `True` on the acoustic prompt positions visible in every codebook.
        """
        device = semantic_lengths.device
        batch_size = semantic_lengths.shape[0]
        sequence_lengths = semantic_lengths + acoustic_lengths
        positions = torch.arange(max_length, device=device).unsqueeze(0)

        if torch.rand(()).item() < self.config.acoustic_free_guidance_ratio:
            prompt_ratio = torch.zeros((batch_size,), device=device)
        else:
            prompt_ratio = torch.rand((batch_size,), device=device)
        prompt_lengths = semantic_lengths + (prompt_ratio * acoustic_lengths).floor().long()
        prompt_mask = positions < prompt_lengths.unsqueeze(1)

        if self.config.mask_strategy == "cosine":
            mask_ratio = torch.cos(math.pi / 2 * torch.rand((batch_size,), device=device))
        else:
            mask_ratio = torch.ones((batch_size,), device=device)
        mask_lengths = ((sequence_lengths - prompt_lengths) * mask_ratio).floor().clamp(min=1).long()

        # Shuffling noise biased so that the positions past the prompt sort first, which is what draws the hidden
        # positions from them alone.
        predictable = prompt_lengths.unsqueeze(1) <= positions
        predictable &= positions < sequence_lengths.unsqueeze(1)
        noise = torch.rand(batch_size, max_length, device=device) - predictable.float()
        restore = torch.argsort(torch.argsort(noise, dim=1), dim=1)
        visible = (positions >= mask_lengths.unsqueeze(1)).gather(1, restore)
        return visible, prompt_mask

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        text_input_ids: torch.LongTensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        text_embeds: torch.FloatTensor | None = None,
        codebook_index: torch.LongTensor | int | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> VoxInstructNAROutput:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`):
                Flat token ids per codebook. The semantic span repeats the same token across codebooks. Positions that
                are not yet known carry the padding token. When `labels` is given the grid is masked inside the model
                instead, so the unmasked grid is passed.
            segment_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                `1` on the semantic span, `2` on the acoustic span.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask over the sequence positions.
            text_input_ids (`torch.LongTensor` of shape `(batch_size, max_text_len)`, *optional*):
                Instruction token ids. Not needed when `text_embeds` is given.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, max_text_len)`, *optional*):
                Mask over `text_input_ids`.
            text_embeds (`torch.FloatTensor` of shape `(batch_size, max_text_len, hidden_size)`, *optional*):
                Already encoded instruction, which skips the text encoder.
            codebook_index (`torch.LongTensor` of shape `(batch_size,)` or `int`, *optional*):
                Residual codebook to predict, in `[1, num_codebooks)`. Drawn uniformly when `labels` is given.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
                Unmasked token grid the drawn codebook is scored against, normally equal to `input_ids`.

        Returns:
            [`VoxInstructNAROutput`]
        """
        codes = input_ids.transpose(1, 2)
        batch_size, num_codebooks, sequence_length = codes.shape
        if num_codebooks != self.config.num_codebooks:
            raise ValueError(f"Expected {self.config.num_codebooks} codebooks, got {num_codebooks}.")
        device = codes.device

        if text_embeds is None:
            text_embeds = self.encode_text(text_input_ids, text_attention_mask)

        loss_mask = None
        if labels is not None:
            drop_text = torch.rand((batch_size,), device=device) < self.config.text_free_guidance_ratio
            text_embeds = torch.where(
                drop_text[:, None, None].expand_as(text_embeds), torch.zeros_like(text_embeds), text_embeds
            )
            codebook_index = torch.randint(1, num_codebooks, (batch_size,), device=device)
            semantic_lengths = (segment_ids == 1).sum(dim=1)
            acoustic_lengths = (segment_ids == 2).sum(dim=1)
            visible, prompt_mask = self.random_masking(semantic_lengths, acoustic_lengths, sequence_length)

            below = codebook_index.unsqueeze(1) > torch.arange(num_codebooks, device=device).unsqueeze(0)
            codes = torch.where(below.unsqueeze(2) | prompt_mask.unsqueeze(1), codes, torch.zeros_like(codes))
            loss_mask = ~visible
            if attention_mask is not None:
                loss_mask = loss_mask & attention_mask.bool()
        elif codebook_index is None:
            raise ValueError("`codebook_index` is required when no `labels` are given.")

        if not torch.is_tensor(codebook_index):
            codebook_index = torch.full((batch_size,), int(codebook_index), device=device, dtype=torch.long)

        inputs_embeds = self.embed_res(codebook_index - 1).unsqueeze(1)
        for index in range(num_codebooks):
            inputs_embeds = self.all_embed_tokens[index](codes[:, index, :]) + inputs_embeds

        inputs_embeds = inputs_embeds + self.embed_segments(segment_ids)
        inputs_embeds = torch.cat([text_embeds + self.embed_segments.weight[0], inputs_embeds], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask.new_ones(text_embeds.shape[:2]), attention_mask], dim=1)

        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            **kwargs,
        ).last_hidden_state[:, text_embeds.shape[1] :]

        logits = torch.stack(
            [self.out_proj[index - 1](hidden_states[sample]) for sample, index in enumerate(codebook_index.tolist())]
        )

        loss = None
        if labels is not None:
            targets = labels.transpose(1, 2).gather(1, codebook_index[:, None, None].expand(-1, 1, sequence_length))
            targets = (targets.squeeze(1) - self.config.acoustic_token_offset).clamp(min=0)
            targets = targets.masked_fill(~loss_mask, -100)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )

        return VoxInstructNAROutput(
            loss=loss,
            logits=logits,
            codebook_index=codebook_index,
            loss_mask=loss_mask,
            text_embeds=text_embeds,
        )


@auto_docstring(
    custom_intro="""
    VoxInstruct, an instruction to speech model. A frozen mT5 encoder turns a free form instruction into a prefix, an
    autoregressive stage predicts HuBERT semantic tokens followed by the first EnCodec codebook, a non-autoregressive
    stage fills in the remaining seven codebooks, and the EnCodec decoder turns them back into a waveform.
    """
)
class VoxInstructForConditionalGeneration(VoxInstructPreTrainedModel, VoxInstructGenerationMixin):
    config: VoxInstructConfig
    _tied_weights_keys = None

    def __init__(self, config: VoxInstructConfig):
        super().__init__(config)
        self.ar = VoxInstructARForCausalLM(config.ar_config)
        self.nar = VoxInstructNARModel(config.nar_config)
        self.audio_encoder = EncodecModel(config.audio_encoder_config)
        self.semantic_encoder = VoxInstructSemanticTokenizerModel(config)
        self.post_init()
        self.freeze_encoders()

    def get_input_embeddings(self):
        return self.ar.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.ar.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.ar.get_output_embeddings()

    def freeze_encoders(self):
        """Freezes what upstream never trains: both instruction encoder bodies, the codec and the tokenizer."""
        self.ar.freeze_encoders()
        self.nar.freeze_encoders()
        for module in (self.audio_encoder, self.semantic_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = False
            module._requires_grad = False

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        segment_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_input_ids: torch.LongTensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        nar_input_ids: torch.LongTensor | None = None,
        nar_labels: torch.LongTensor | None = None,
        codebook_index: torch.LongTensor | int | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> VoxInstructOutput:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Flat token sequence of the autoregressive stage.
            segment_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                `1` on the semantic span, `2` on the acoustic span.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask over the sequence positions.
            text_input_ids (`torch.LongTensor` of shape `(batch_size, max_text_len)`, *optional*):
                Instruction token ids.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, max_text_len)`, *optional*):
                Mask over `text_input_ids`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Targets of the autoregressive stage, with `-100` on the positions to ignore.
            nar_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
                Token grid of the non-autoregressive stage.
            nar_labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
                Targets of the non-autoregressive stage.
            codebook_index (`torch.LongTensor` of shape `(batch_size,)` or `int`, *optional*):
                Residual codebook the non-autoregressive stage predicts when it is not training.
            past_key_values (`Cache`, *optional*):
                Cache of the autoregressive stage.
            use_cache (`bool`, *optional*):
                Whether to return the updated cache.

        Returns:
            [`VoxInstructOutput`]

        Example:

        ```python
        >>> from voicestudio.models.vox_instruct import VoxInstructForConditionalGeneration, VoxInstructProcessor

        >>> processor = VoxInstructProcessor.from_pretrained("voxinstruct-converted")
        >>> model = VoxInstructForConditionalGeneration.from_pretrained("voxinstruct-converted")

        >>> batch = processor(
        ...     text=['A young man says cheerfully: "Good morning."'],
        ...     language="en",
        ...     semantic_ids=[[3, 4, 5]],
        ...     acoustic_ids=[[[0] * 8] * 6],
        ... )
        >>> outputs = model(**batch)
        >>> outputs.loss.backward()
        ```"""
        ar_outputs = None
        nar_outputs = None

        if input_ids is not None:
            ar_outputs = self.ar(
                input_ids=input_ids,
                segment_ids=segment_ids,
                attention_mask=attention_mask,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                labels=labels,
                **kwargs,
            )

        if nar_input_ids is not None:
            nar_outputs = self.nar(
                input_ids=nar_input_ids,
                segment_ids=segment_ids,
                attention_mask=attention_mask,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                codebook_index=codebook_index,
                labels=nar_labels,
                **kwargs,
            )

        ar_loss = ar_outputs.loss if ar_outputs is not None else None
        nar_loss = nar_outputs.loss if nar_outputs is not None else None
        losses = [term for term in (ar_loss, nar_loss) if term is not None]

        return VoxInstructOutput(
            loss=sum(losses) if losses else None,
            ar_loss=ar_loss,
            nar_loss=nar_loss,
            logits=ar_outputs.logits if ar_outputs is not None else None,
            nar_logits=nar_outputs.logits if nar_outputs is not None else None,
            past_key_values=ar_outputs.past_key_values if ar_outputs is not None else None,
        )


__all__ = [
    "VoxInstructPreTrainedModel",
    "VoxInstructTextEncoder",
    "VoxInstructARForCausalLM",
    "VoxInstructNARModel",
    "VoxInstructForConditionalGeneration",
]
