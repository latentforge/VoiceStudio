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
"""Generation mixin for Chroma."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from transformers.generation import GenerationConfig, GenerationMixin, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateNonBeamOutput
from transformers.models.csm.generation_csm import CsmGenerateOutput
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


def multinomial_sample_one_no_sync(probs: torch.Tensor) -> torch.Tensor:
    """
    Samples one index per row without forcing a CUDA synchronization.

    Args:
        probs (`torch.FloatTensor` of shape `(batch_size, vocab_size)`):
            Categorical distribution to sample from.

    Returns:
        `torch.Tensor` of shape `(batch_size, 1)`: The sampled indices.
    """
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    """
    Samples one codebook id per row with top-k filtering and temperature.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, vocab_size)`):
            Unnormalized codebook 0 scores.
        topk (`int`):
            Number of highest scoring ids kept before sampling.
        temperature (`float`):
            Scale applied to the logits before filtering.

    Returns:
        `torch.Tensor` of shape `(batch_size, 1)`: The sampled codebook 0 ids.
    """
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    return multinomial_sample_one_no_sync(probs)


@dataclass
class ChromaGenerateOutput(CsmGenerateOutput):
    """
    Outputs of [`~ChromaForConditionalGeneration.generate`].

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, num_frames, audio_num_codebooks)`):
            The generated codebook ids, one row per audio frame.
        scores (`tuple(torch.FloatTensor)`, *optional*, returned when `output_scores=True`):
            Processed codebook 0 scores of each generation step.
        logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_logits=True`):
            Unprocessed codebook 0 scores of each generation step.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Backbone attentions of each generation step.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Backbone hidden states of each generation step.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True`):
            The backbone cache.
        audio (`list(torch.FloatTensor)` of length `batch_size`, *optional*):
            The waveform the codec reconstructed from `sequences`, returned when `output_audio=True`.
    """


class ChromaGenerationMixin(GenerationMixin):
    # Copied from transformers.models.csm.generation_csm.CsmGenerationMixin._get_stopping_criteria with Csm->Chroma
    def _get_stopping_criteria(self, *args, **kwargs) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(*args, **kwargs)

        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if not isinstance(criterion, MaxLengthCriteria):
                logger.warning(
                    f"Chroma does not support {criterion.__class__.__name__} stopping criteria, it will be ignored."
                )
            else:
                kept_criteria.append(criterion)
        return kept_criteria

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        """
        This method overrides [`~generation.utils.GenerationMixin._prepare_generation_config`]. It ensures that the
        decoder generation config is initialized and that args passed as `decoder_*` are routed to it.

        Args:
            generation_config ([`~generation.GenerationConfig`], *optional*):
                Base parametrization of this `generate` call.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc generation config overrides. Keys prefixed with `decoder_` are stripped of their prefix and
                applied to the decoder generation config instead.

        Returns:
            `tuple[GenerationConfig, dict]`: The prepared generation config and the remaining model kwargs.

        Raises:
            ValueError: If the decoder is asked to emit a number of tokens other than
                `audio_num_codebooks - 1`.
        """
        decoder_kwargs = {k[len("decoder_") :]: v for k, v in kwargs.items() if k.startswith("decoder_")}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("decoder_")}

        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        self.decoder.generation_config.update(**decoder_kwargs)

        num_residual_codebooks = self.decoder.config.audio_num_codebooks - 1
        decoder_min_new_tokens = getattr(self.decoder.generation_config, "min_new_tokens") or num_residual_codebooks
        decoder_max_new_tokens = getattr(self.decoder.generation_config, "max_new_tokens") or num_residual_codebooks

        if {decoder_min_new_tokens, decoder_max_new_tokens} != {num_residual_codebooks}:
            raise ValueError(
                f"decoder generation config's min_new_tokens ({decoder_min_new_tokens}) and max_new_tokens "
                f"({decoder_max_new_tokens}) must both equal audio_num_codebooks - 1 ({num_residual_codebooks})"
            )
        elif self.decoder.generation_config.return_dict_in_generate:
            self.decoder.generation_config.return_dict_in_generate = False

        original_get_generation_mode = generation_config.get_generation_mode

        def patched_get_generation_mode(assistant_model=None):
            generation_mode = original_get_generation_mode(assistant_model)
            if generation_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]:
                raise ValueError(
                    f"Generation mode {generation_mode} is not supported for Chroma. Please set generation "
                    "parameters to use greedy or sampling generation."
                )

            return generation_mode

        generation_config.get_generation_mode = patched_get_generation_mode

        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        generation_config: Optional[GenerationConfig] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        This method overrides [`~generation.utils.GenerationMixin._sample`]. One step samples codebook 0 from the
        backbone, hands it plus the backbone hidden state to the decoder, and uses the resulting full frame as the
        next backbone input.

        Args:
            input_ids (`torch.LongTensor`):
                Prompt text ids on the first step, codebook ids of the previous frame afterwards.
            logits_processor (`LogitsProcessorList`, *optional*):
                Processors applied to the codebook 0 logits before they are stored in `scores`.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Criteria evaluated against the frames generated so far. Only [`MaxLengthCriteria`] is honored.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                Parametrization of this `generate` call.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether the loop runs until `max_length` on every rank.
            streamer ([`~generation.streamers.BaseStreamer`], *optional*):
                Streamer fed with every generated frame.
            model_kwargs (`dict[str, Any]`, *optional*):
                Model state carried across steps, including the backbone and reasoner caches.

        Returns:
            [`~generation.GenerateDecoderOnlyOutput`] or `torch.LongTensor`: The generated frames of shape
            `(batch_size, num_frames, audio_num_codebooks)`.

        Raises:
            ValueError: If the decoder returns a frame that does not cover `audio_num_codebooks` codebooks.
        """
        pad_token_id = self.config.codebook_pad_token_id
        has_eos_stopping_criteria = generation_config._eos_token_tensor is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        top_k = generation_config.top_k
        temperature = generation_config.temperature

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        if input_ids.ndim == 2 and model_kwargs.get("inputs_embeds") is None:
            # the prompt text ids are not part of the returned frames, so they must not count towards max_length
            for criterion in stopping_criteria:
                if isinstance(criterion, MaxLengthCriteria):
                    criterion.max_length -= cur_len

        generated_frames = []

        model_forward = self.__call__
        if self._valid_auto_compile_criteria(model_kwargs, generation_config):
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": True})

            if is_prefill:
                backbone_outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                backbone_outputs = model_forward(**model_inputs, return_dict=True)

            next_token_logits = backbone_outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            backbone_last_hidden_state = backbone_outputs.hidden_states[-1][:, -1, :]

            model_kwargs = self._update_model_kwargs_for_generation(backbone_outputs, model_kwargs)

            if synced_gpus and this_peer_finished:
                continue

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (backbone_outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (backbone_outputs.hidden_states,)

            if do_sample:
                next_tokens = sample_topk(next_token_logits, top_k, temperature)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            frame_codes = self.decoder.generate(
                input_ids=next_tokens,
                backbone_last_hidden_state=backbone_last_hidden_state.clone(),
                max_new_tokens=self.config.decoder_config.audio_num_codebooks - 1,
                min_new_tokens=self.config.decoder_config.audio_num_codebooks - 1,
                do_sample=do_sample,
                use_cache=True,
                temperature=temperature,
                top_k=top_k,
            )

            if frame_codes.shape[-1] != self.config.decoder_config.audio_num_codebooks:
                raise ValueError(
                    f"Generated codebooks shape {frame_codes.shape[-1]} does not match expected "
                    f"audio_num_codebooks {self.config.decoder_config.audio_num_codebooks}"
                )

            next_tokens = frame_codes

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + pad_token_id * (
                    1 - unfinished_sequences.unsqueeze(-1)
                )

            if next_tokens.sum() != 0:
                generated_frames.append(next_tokens.unsqueeze(1))

            input_ids = next_tokens[:, None, :]

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # the eos token is expected to be the same in every codebook of a finished frame
            unfinished_sequences = unfinished_sequences & ~(
                input_ids[:, -1, :-1] == self.config.codebook_eos_token_id
            ).all(-1)
            if generated_frames:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                    torch.cat(generated_frames, dim=1), scores
                )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del backbone_outputs
            del frame_codes

        if streamer is not None:
            streamer.end()

        sequences = torch.cat(generated_frames, dim=1) if len(generated_frames) > 0 else input_ids
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return sequences

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        output_audio: Optional[bool] = False,
        bos_token_id: Optional[int] = 0,
        **kwargs: Any,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor, list[torch.FloatTensor]]:
        r"""
        This method overrides [`~generation.utils.GenerationMixin.generate`] to match the specifics of the Chroma
        model, which requires a custom generation sampling step:

        1. Infer the backbone model to sample the first codebook token
        2. Call generate on the decoder with the first codebook token as `input_ids` to sample the remaining
           codebook tokens
        3. Use these generated codebook tokens as `input_ids` to sample the next first codebook token using the
           backbone model
        4. Repeat until stopping criteria is met

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to
        the model's default generation configuration. You can override any `generation_config` by passing the
        corresponding parameters to generate(), e.g. `.generate(inputs, do_sample=True)`.

        </Tip>

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The reference transcript ids used to build the backbone prompt.
            input_values (`torch.Tensor` of shape `(batch_size, channels, max_concatenated_audio_length)`,
                *optional*):
                The batched reference waveforms, encoded into codebook tokens by the codec model and merged with
                the text prompt.
            input_values_cutoffs (`torch.Tensor` of shape `(batch_size, max_num_audio)`, *optional*):
                Specify the end positions of audio segments within each batch entry, relative to the concatenated
                audio input. If a batch entry has fewer segments than the maximum, it is padded with -1.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config.
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until `max_length`.
            streamer ([`~generation.streamers.BaseStreamer`], *optional*):
                Streamer object that will be used to stream the generated frames.
            output_audio (`bool`, *optional*, defaults to `False`):
                Whether the generated codebook ids are decoded to a waveform by the codec model.
            bos_token_id (`int`, *optional*, defaults to 0):
                Codebook id the backbone sequence is seeded with.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and additional model-specific kwargs forwarded to
                `forward`. Decoder specific kwargs should be prefixed with `decoder_`.

        Returns:
            [`ChromaGenerateOutput`] or `torch.LongTensor` or `list[torch.FloatTensor]`: A
            [`ChromaGenerateOutput`] if `return_dict_in_generate=True`, a `list[torch.FloatTensor]` if
            `output_audio=True`, and the generated codebook ids otherwise.
        """
        generate_output = super().generate(
            input_ids=input_ids,
            input_values=input_values,
            input_values_cutoffs=input_values_cutoffs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            synced_gpus=synced_gpus,
            streamer=streamer,
            bos_token_id=bos_token_id,
            **kwargs,
        )
        generate_returned_dict = not isinstance(generate_output, torch.Tensor)
        audio = None
        if output_audio:
            generated_audio_codes = generate_output.sequences if generate_returned_dict else generate_output

            audio = []
            with torch.no_grad():
                for audio_codes_batch in generated_audio_codes:
                    eos_idxs = (audio_codes_batch == self.config.codebook_eos_token_id).all(dim=-1).nonzero()
                    cutoff_idx = eos_idxs.min() if eos_idxs.numel() != 0 else audio_codes_batch.shape[0]

                    audio_codes_batch = audio_codes_batch[:cutoff_idx]
                    codec_decode_output = self.codec_model.decode(audio_codes_batch.transpose(0, 1).unsqueeze(0))
                    audio.append(codec_decode_output.audio_values)

        if generate_returned_dict:
            return ChromaGenerateOutput(audio=audio, **generate_output)
        elif output_audio:
            return audio
        return generate_output


__all__ = [
    "ChromaGenerateOutput",
    "ChromaGenerationMixin",
]
