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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn

from transformers.generation import GenerateDecoderOnlyOutput, GenerationConfig, GenerationMixin
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.models.csm.generation_csm import CsmGenerationMixin
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer


logger = logging.get_logger(__name__)


class GeneratedTokenRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    Applies a repetition penalty to the first codebook of the frames generated so far.

    Unlike [`RepetitionPenaltyLogitsProcessor`], this one reads the first codebook of the 3D frames Breeze TTS 2
    decodes and leaves the 2D text prompt of the prefill step untouched, since those ids index the text vocabulary
    rather than a codebook.

    Args:
        penalty (`float`):
            Divisor applied to the scores of already generated codebook ids, or multiplier when the score is
            negative. A value above 1 discourages repetition, a value below 1 encourages it.
    """

    def __init__(self, penalty: float) -> None:
        if penalty <= 0:
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim == 2:
            return scores
        if input_ids.ndim != 3:
            raise ValueError(f"Expected 2D prompt ids or 3D codec frames, got {input_ids.ndim}D")

        generated_ids = input_ids[..., 0]
        if self.penalty == 1.0 or generated_ids.numel() == 0:
            return scores

        valid = (generated_ids >= 0) & (generated_ids < scores.shape[-1])
        if not torch.any(valid):
            return scores

        # Out-of-range ids gather and scatter through id 0, whose score is restored right after.
        safe_ids = generated_ids.masked_fill(~valid, 0)
        selected_scores = torch.gather(scores, 1, safe_ids)
        penalized_scores = torch.where(
            selected_scores < 0, selected_scores * self.penalty, selected_scores / self.penalty
        )

        processed = scores.clone()
        processed.scatter_(1, safe_ids, penalized_scores)

        original_zero = scores[:, 0]
        penalized_zero = torch.where(
            original_zero < 0, original_zero * self.penalty, original_zero / self.penalty
        )
        generated_zero = (valid & (generated_ids == 0)).any(dim=1)
        processed[:, 0] = torch.where(generated_zero, penalized_zero, original_zero)
        return processed


@dataclass
class BreezeTTSGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of [`~BreezeTTSForConditionalGeneration.generate`].

    Args:
        audio (`list(torch.FloatTensor)` of length `batch_size`):
            The generated audio.
    """

    audio: list[torch.Tensor] | None = None


class BreezeTTSGenerationMixin(CsmGenerationMixin):
    # Breeze TTS 2 embeds the text spans of a prompt with its own text encoder, which needs `text_ids_mask` and
    # `text_ids_len` alongside the ids; `UnbatchedClassifierFreeGuidanceLogitsProcessor` forwards neither, so the
    # negative branches are carried as model kwargs and run by `_sample` itself. The `cfg_` prefix keeps them out
    # of the `guidance_scale` handling of `GenerationMixin`.
    _cfg_kwargs = {
        "cfg_scale",
        "cfg_negative_prompt_ids",
        "cfg_negative_prompt_attention_mask",
        "cfg_negative_text_ids_mask",
        "cfg_negative_text_ids_len",
        "cfg_negative_input_values",
        "cfg_scale_ref",
        "cfg_scale_ins",
        "cfg_uncond_prompt_ids",
        "cfg_uncond_prompt_attention_mask",
        "cfg_uncond_text_ids_mask",
        "cfg_uncond_text_ids_len",
        "cfg_ref_prompt_ids",
        "cfg_ref_prompt_attention_mask",
        "cfg_ref_text_ids_mask",
        "cfg_ref_text_ids_len",
        "cfg_ins_prompt_ids",
        "cfg_ins_prompt_attention_mask",
        "cfg_ins_text_ids_mask",
        "cfg_ins_text_ids_len",
    }

    def _reserved_codec_token_ids(self) -> list[int]:
        """
        Returns:
            `list[int]`: the ids of `config.vocab_size` that no codebook of the audio tokenizer can emit.
        """
        return list(range(int(self.config.codec_config.codebook_size), int(self.config.vocab_size)))

    def _mask_reserved_codec_logits(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Masks, in place, the scores of the codebook ids the audio tokenizer cannot decode.

        Args:
            scores (`torch.Tensor` of shape `(batch_size, vocab_size)`):
                Scores of one decoding step.

        Returns:
            `torch.Tensor`: the same tensor, with the reserved ids set to `-inf`.
        """
        codebook_size = int(self.config.codec_config.codebook_size)
        token_vocab_size = int(self.config.vocab_size)
        if token_vocab_size > scores.shape[-1]:
            raise ValueError(f"vocab_size={token_vocab_size} exceeds the score width {scores.shape[-1]}")
        if codebook_size < token_vocab_size:
            scores[..., codebook_size:token_vocab_size] = float("-inf")
        return scores

    def _sample_codebook_ids(self, logits: torch.Tensor) -> torch.LongTensor:
        """
        Draws one codebook id per row under the depth decoder's own generation config.

        Args:
            logits (`torch.Tensor` of shape `(batch_size, vocab_size)`):
                Guided logits of one depth position.

        Returns:
            `torch.LongTensor` of shape `(batch_size, 1)`: the sampled ids.
        """
        generation_config = self.depth_decoder.generation_config
        temperature = generation_config.temperature
        top_k = generation_config.top_k
        top_p = generation_config.top_p

        if not generation_config.do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)

        if temperature is not None and temperature != 1.0:
            logits = logits / temperature
        probs = nn.functional.softmax(logits, dim=-1)

        if top_k is not None and top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]))
            probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            sorted_indices_to_remove = torch.cumsum(sorted_probs, dim=-1) > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)

    def _depth_decoder_generate_guided(
        self,
        depth_decoder_input_ids: torch.LongTensor,
        backbone_hidden_states: dict[str, torch.FloatTensor],
        guidance_scales: dict[str, float],
    ) -> torch.LongTensor:
        """
        Runs the depth decoder once per guidance branch at every depth position and samples from the guided logits.

        With a single `"cond"` branch the guided logits are
        `uncond + cfg_scale * (cond - uncond)`; with the `"ref"` and `"ins"` branches they are
        `uncond + cfg_scale_ref * (ref - uncond) + cfg_scale_ins * (ins - uncond)`. Every branch shares the same
        running sequence and differs only in the backbone hidden state spliced in at depth position 0.

        Args:
            depth_decoder_input_ids (`torch.LongTensor` of shape `(batch_size, 2)`):
                The depth position 0 placeholder followed by the codebook 0 id the backbone sampled.
            backbone_hidden_states (`dict[str, torch.FloatTensor]`):
                Backbone hidden state of each branch, keyed `"uncond"` plus either `"cond"` or `"ref"`/`"ins"`.
            guidance_scales (`dict[str, float]`):
                Guidance scale of each branch other than `"uncond"`.

        Returns:
            `torch.LongTensor` of shape `(batch_size, num_codebooks + 1)`: the depth position 0 placeholder,
            the codebook 0 id the backbone sampled, and the `num_codebooks - 1` sampled codebook ids.
        """
        sequences = depth_decoder_input_ids
        for _ in range(self.config.num_codebooks - 1):
            branch_logits = {}
            for branch, hidden_state in backbone_hidden_states.items():
                outputs = self.depth_decoder(
                    input_ids=sequences,
                    backbone_last_hidden_state=hidden_state,
                    use_cache=False,
                    return_dict=True,
                )
                branch_logits[branch] = outputs.logits[:, -1, :].float()

            next_token_logits = branch_logits["uncond"]
            for branch, scale in guidance_scales.items():
                next_token_logits = next_token_logits + scale * (branch_logits[branch] - branch_logits["uncond"])

            self._mask_reserved_codec_logits(next_token_logits)
            sequences = torch.cat([sequences, self._sample_codebook_ids(next_token_logits)], dim=-1)

        return sequences

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]):
        cfg_kwargs = {k: model_kwargs.pop(k) for k in list(model_kwargs) if k in self._cfg_kwargs}
        super()._validate_model_kwargs(model_kwargs)
        model_kwargs.update(cfg_kwargs)

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in self._cfg_kwargs}
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        model_kwargs.update(cfg_kwargs)
        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        """
        This method overrides [`~generation.utils.GenerationMixin._sample`].

        Breeze TTS 2 decodes one audio frame per step:
        1. Infer the backbone to sample codebook 0 of the frame, or its extra end-of-audio class
        2. Run the depth decoder on that codebook 0 id to sample the remaining codebooks of the frame
        3. Feed the whole frame back to the backbone
        4. Repeat until the backbone emits its end-of-audio class or the sequence reaches `max_length`

        Two guidance regimes run extra prompt branches alongside the conditional one, guiding both the backbone
        logits and every depth decoder step: `cfg_scale` with a single negative prompt, and `cfg_scale_ref` plus
        `cfg_scale_ins` with an unconditional, a reference-audio and an instruction prompt.
        """
        pad_token_id = self.config.codebook_pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample

        cfg_scale = model_kwargs.pop("cfg_scale", 1.0)
        negative_prompt_ids = model_kwargs.pop("cfg_negative_prompt_ids", None)
        negative_prompt_attention_mask = model_kwargs.pop("cfg_negative_prompt_attention_mask", None)
        negative_text_ids_mask = model_kwargs.pop("cfg_negative_text_ids_mask", None)
        negative_text_ids_len = model_kwargs.pop("cfg_negative_text_ids_len", None)
        negative_input_values = model_kwargs.pop("cfg_negative_input_values", None)
        use_cfg = cfg_scale != 1.0 and negative_prompt_ids is not None

        cfg_scale_ref = model_kwargs.pop("cfg_scale_ref", None)
        cfg_scale_ins = model_kwargs.pop("cfg_scale_ins", None)
        branch_prompt_ids = {
            branch: model_kwargs.pop(f"cfg_{branch}_prompt_ids", None) for branch in ("uncond", "ref", "ins")
        }
        branch_prompt_kwargs = {
            branch: {
                "attention_mask": model_kwargs.pop(f"cfg_{branch}_prompt_attention_mask", None),
                "text_ids_mask": model_kwargs.pop(f"cfg_{branch}_text_ids_mask", None),
                "text_ids_len": model_kwargs.pop(f"cfg_{branch}_text_ids_len", None),
            }
            for branch in ("uncond", "ref", "ins")
        }
        use_dual_cfg = (
            cfg_scale_ref is not None
            and cfg_scale_ins is not None
            and all(ids is not None for ids in branch_prompt_ids.values())
        )
        if use_dual_cfg:
            use_cfg = False

        # A zero scale is pure unconditional decoding: swapping the negative prompt in for the conditional one
        # gives the same logits as the guidance formula for a single forward pass instead of two.
        if cfg_scale == 0.0 and negative_prompt_ids is not None and not use_dual_cfg:
            input_ids = negative_prompt_ids
            model_kwargs["attention_mask"] = negative_prompt_attention_mask
            model_kwargs["text_ids_mask"] = (
                negative_text_ids_mask
                if negative_text_ids_mask is not None
                else negative_prompt_attention_mask.bool()
            )
            model_kwargs["text_ids_len"] = (
                negative_text_ids_len
                if negative_text_ids_len is not None
                else negative_prompt_attention_mask.sum(dim=1).long()
            )
            use_cfg = False

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["output_hidden_states"] = True

        branch_prompt_ids["negative"] = negative_prompt_ids
        branch_kwargs = {}
        if use_cfg:
            branch_kwargs["negative"] = self._prepare_branch_kwargs(
                model_kwargs,
                negative_prompt_attention_mask,
                negative_text_ids_mask,
                negative_text_ids_len,
                # the negative prompt reserves the same audio frames as the conditional one unless it
                # carries its own
                input_values=negative_input_values
                if negative_input_values is not None
                else model_kwargs.get("input_values"),
            )
        if use_dual_cfg:
            for branch in ("uncond", "ref", "ins"):
                branch_kwargs[branch] = self._prepare_branch_kwargs(
                    model_kwargs,
                    branch_prompt_kwargs[branch]["attention_mask"],
                    branch_prompt_kwargs[branch]["text_ids_mask"],
                    branch_prompt_kwargs[branch]["text_ids_len"],
                    # only the reference branch carries the reference audio frames of the prompt
                    input_values=model_kwargs.get("input_values") if branch == "ref" else None,
                )

        # The 2D text prompt is not part of the returned frames, so it must not count towards `max_length`.
        if input_ids.ndim == 2 and model_kwargs.get("inputs_embeds") is None:
            for criterion in stopping_criteria:
                if isinstance(criterion, MaxLengthCriteria):
                    criterion.max_length -= cur_len

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        branch_outputs = {}
        for branch, branch_model_kwargs in branch_kwargs.items():
            branch_outputs[branch] = self._prefill(
                branch_prompt_ids[branch], generation_config, branch_model_kwargs
            )
            branch_kwargs[branch] = self._update_model_kwargs_for_generation(
                branch_outputs[branch], branch_model_kwargs
            )
        # under dual guidance the fully conditional branch is never run: the guided logits are built from the
        # three branches alone
        outputs = (
            branch_outputs["ref"] if use_dual_cfg else self._prefill(input_ids, generation_config, model_kwargs)
        )
        prefill_consumed = False

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                next_sequence_length = 1 if model_kwargs["use_cache"] else None
                for branch, branch_model_kwargs in branch_kwargs.items():
                    branch_inputs = self.prepare_inputs_for_generation(
                        branch_prompt_ids[branch],
                        next_sequence_length=next_sequence_length,
                        **branch_model_kwargs,
                    )
                    branch_outputs[branch] = model_forward(**branch_inputs, return_dict=True)
                    branch_kwargs[branch] = self._update_model_kwargs_for_generation(
                        branch_outputs[branch], branch_model_kwargs
                    )

                if use_dual_cfg:
                    outputs = branch_outputs["ref"]
                else:
                    model_inputs = self.prepare_inputs_for_generation(
                        input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                    )
                    model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                    outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].clone().float().to(input_ids.device)
            branch_logits = {
                branch: output.logits[:, -1, :].clone().float().to(input_ids.device)
                for branch, output in branch_outputs.items()
            }
            if use_cfg:
                next_token_logits = branch_logits["negative"] + cfg_scale * (
                    next_token_logits - branch_logits["negative"]
                )
            if use_dual_cfg:
                next_token_logits = (
                    branch_logits["uncond"]
                    + cfg_scale_ref * (branch_logits["ref"] - branch_logits["uncond"])
                    + cfg_scale_ins * (branch_logits["ins"] - branch_logits["uncond"])
                )

            next_token_scores = logits_processor(input_ids, next_token_logits)
            self._mask_reserved_codec_logits(next_token_scores)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # the backbone signals the end of the audio stream with the extra class of its head
            backbone_eos_mask = next_tokens == self.backbone_eos_token_id
            backbone_last_hidden_state = outputs.hidden_states[-1][:, -1, :]

            codebook_ids = torch.full(
                (batch_size, self.config.num_codebooks),
                pad_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            active_mask = ~backbone_eos_mask
            if active_mask.any():
                first_codebook_ids = next_tokens.masked_fill(backbone_eos_mask, 0)[active_mask, None]
                # depth position 0 is a placeholder, replaced by the backbone hidden state
                depth_decoder_input_ids = nn.functional.pad(first_codebook_ids, (1, 0), value=0)

                if use_dual_cfg:
                    depth_decoder_sequences = self._depth_decoder_generate_guided(
                        depth_decoder_input_ids,
                        {
                            branch: branch_outputs[branch].hidden_states[-1][:, -1, :][active_mask].clone()
                            for branch in ("uncond", "ref", "ins")
                        },
                        {"ref": cfg_scale_ref, "ins": cfg_scale_ins},
                    )
                elif use_cfg:
                    depth_decoder_sequences = self._depth_decoder_generate_guided(
                        depth_decoder_input_ids,
                        {
                            "uncond": branch_outputs["negative"].hidden_states[-1][:, -1, :][active_mask].clone(),
                            "cond": backbone_last_hidden_state[active_mask].clone(),
                        },
                        {"cond": cfg_scale},
                    )
                else:
                    depth_decoder_sequences = self.depth_decoder.generate(
                        input_ids=depth_decoder_input_ids,
                        backbone_last_hidden_state=backbone_last_hidden_state[active_mask].clone(),
                        suppress_tokens=self._reserved_codec_token_ids(),
                    )
                    if not isinstance(depth_decoder_sequences, torch.Tensor):
                        depth_decoder_sequences = depth_decoder_sequences.sequences

                # position 0 holds the placeholder, position 1 the codebook 0 id the backbone sampled
                codebook_ids[active_mask] = depth_decoder_sequences[:, 1:]

            next_tokens = codebook_ids.unsqueeze(1)
            next_tokens = next_tokens * unfinished_sequences.view(batch_size, 1, 1) + pad_token_id * (
                1 - unfinished_sequences.view(batch_size, 1, 1)
            )

            if input_ids.ndim == 2:
                input_ids = next_tokens
            else:
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
            for branch in branch_kwargs:
                branch_prompt_ids[branch] = self._append_generated_frame(
                    branch_prompt_ids[branch], next_tokens
                )

            if streamer is not None:
                streamer.put(codebook_ids.cpu())

            unfinished_sequences = unfinished_sequences & ~backbone_eos_mask
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    @staticmethod
    def _prepare_branch_kwargs(
        model_kwargs: dict[str, Any],
        attention_mask: torch.Tensor | None,
        text_ids_mask: torch.BoolTensor | None,
        text_ids_len: torch.LongTensor | None,
        input_values: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Builds the model kwargs of one guidance branch, which decodes in lockstep with the conditional prompt but
        keeps its own prompt, mask and cache.

        Args:
            model_kwargs (`dict[str, Any]`):
                Model kwargs of the conditional prompt, whose prompt-specific entries are replaced.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention mask of the branch prompt.
            text_ids_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Text positions of the branch prompt. Defaults to every attended position.
            text_ids_len (`torch.LongTensor`, *optional*):
                Text segment lengths of the branch prompt. Defaults to one segment per prompt.
            input_values (`torch.Tensor`, *optional*):
                Codebook frames the branch prompt reserves room for, if any.

        Returns:
            `dict[str, Any]`: the branch's model kwargs, holding no cache of its own yet.
        """
        prompt_specific = {"attention_mask", "text_ids_mask", "text_ids_len", "past_key_values"}
        branch_kwargs = {k: v for k, v in model_kwargs.items() if k not in prompt_specific}
        branch_kwargs["attention_mask"] = attention_mask
        branch_kwargs["text_ids_mask"] = (
            text_ids_mask if text_ids_mask is not None else attention_mask.bool()
        )
        branch_kwargs["text_ids_len"] = (
            text_ids_len if text_ids_len is not None else attention_mask.sum(dim=1).long()
        )
        if input_values is not None:
            branch_kwargs["input_values"] = input_values
        else:
            branch_kwargs.pop("input_values", None)
        return branch_kwargs

    @staticmethod
    def _append_generated_frame(prompt_ids: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """
        Appends one generated frame to a branch prompt, replacing the 2D text prompt on the first step.

        Args:
            prompt_ids (`torch.Tensor`):
                The branch prompt, 2D before the first frame and 3D after it.
            frame (`torch.Tensor` of shape `(batch_size, 1, num_codebooks)`):
                The frame that was just generated.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_generated_frames, num_codebooks)`: the extended prompt.
        """
        if prompt_ids.ndim == 2:
            return frame
        return torch.cat([prompt_ids, frame], dim=1)

    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        synced_gpus: bool | None = None,
        streamer: Optional["BaseStreamer"] = None,
        output_audio: bool | None = False,
        **kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        r"""
        This method overrides [`~generation.utils.GenerationMixin.generate`] to decode one multi-codebook audio
        frame per step, as described on [`~BreezeTTSGenerationMixin._sample`].

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Prompt token ids built by [`BreezeTTSProcessor`].
            input_values (`torch.Tensor` of shape `(batch_size, num_frames, num_codebooks)`, *optional*):
                Codebook ids of the prompt's audio frames.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                Base parametrization of the generation call. Depth decoder parameters are passed prefixed with
                `depth_decoder_`, guidance parameters prefixed with `cfg_`.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors complementing the ones built from `generation_config`.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria complementing the ones built from `generation_config`. Only
                [`MaxLengthCriteria`] is supported.
            synced_gpus (`bool`, *optional*):
                Whether to keep decoding until `max_length` on every rank.
            streamer (`BaseStreamer`, *optional*):
                Streamer receiving the generated frames.
            output_audio (`bool`, *optional*, defaults to `False`):
                Whether the generated codebook frames are decoded to waveforms with `codec_model`.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and additional model kwargs.

        Returns:
            [`BreezeTTSGenerateOutput`] or `torch.LongTensor` or `list[torch.FloatTensor]`: the generated codebook
            frames, or the decoded waveforms when `output_audio=True`.
        """
        generate_output = GenerationMixin.generate(
            self,
            input_ids=input_ids,
            input_values=input_values,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **kwargs,
        )

        returned_dict = not isinstance(generate_output, torch.Tensor)
        if not output_audio:
            return generate_output

        audio_codes = generate_output.sequences if returned_dict else generate_output
        audio = []
        with torch.no_grad():
            for sample_audio_codes in audio_codes:
                is_pad_frame = (sample_audio_codes == self.config.codebook_pad_token_id).all(dim=-1)
                pad_idxs = is_pad_frame.nonzero()
                cutoff_idx = pad_idxs.min() if pad_idxs.numel() else sample_audio_codes.shape[0]
                sample_audio_codes = sample_audio_codes[:cutoff_idx]
                if cutoff_idx == 0:
                    logger.warning("No codebook tokens were generated, decoding a single silent frame.")
                    sample_audio_codes = torch.ones(
                        1,
                        self.config.num_codebooks,
                        device=audio_codes.device,
                        dtype=audio_codes.dtype,
                    )
                audio.append(
                    self.codec_model.decode(sample_audio_codes.transpose(0, 1).unsqueeze(0)).audio_values[0, 0]
                )

        if returned_dict:
            return BreezeTTSGenerateOutput(audio=audio, **generate_output)
        return audio


__all__ = [
    "BreezeTTSGenerateOutput",
    "BreezeTTSGenerationMixin",
    "GeneratedTokenRepetitionPenaltyLogitsProcessor",
]
