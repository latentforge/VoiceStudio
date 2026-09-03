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
"""Generation utilities for VoxInstruct."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput, logging


logger = logging.get_logger(__name__)


@dataclass
class VoxInstructGenerateOutput(ModelOutput):
    r"""
    Output of [`VoxInstructForConditionalGeneration.generate`].

    Args:
        audio_values (`torch.FloatTensor` of shape `(batch_size, 1, num_samples)`, *optional*):
            Decoded waveform at `config.sampling_rate`.
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
            EnCodec codes of the generated speech, with the codebook offsets already removed.
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Flat token sequence the autoregressive stage produced, prompt included.
        segment_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Segment ids of `sequences`.
    """

    audio_values: torch.FloatTensor | None = None
    audio_codes: torch.LongTensor | None = None
    sequences: torch.LongTensor | None = None
    segment_ids: torch.LongTensor | None = None


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -1e5,
    temperature: float = 1.0,
) -> torch.Tensor:
    r"""
    Filters a one dimensional distribution of logits with top-k and nucleus filtering, then applies the temperature.

    Args:
        logits (`torch.Tensor` of shape `(vocab_size,)`):
            Logits of a single position.
        top_k (`int`, *optional*, defaults to 0):
            Keep only the `top_k` highest scoring tokens. Disabled when zero.
        top_p (`float`, *optional*, defaults to 0.0):
            Keep the highest scoring tokens whose cumulative probability reaches `top_p`. Disabled when zero.
        filter_value (`float`, *optional*, defaults to -1e5):
            Score written onto the removed tokens.
        temperature (`float`, *optional*, defaults to 1.0):
            Divides the surviving logits.

    Returns:
        `torch.Tensor` of shape `(vocab_size,)`: The filtered logits.
    """
    if logits.dim() != 1:
        raise ValueError(f"Expected a one dimensional logit vector, got shape {tuple(logits.shape)}.")

    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(remove, filter_value)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(sorted_logits.softmax(dim=-1), dim=-1)
        sorted_remove = cumulative > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = 0
        logits = logits.index_fill(0, sorted_indices[sorted_remove], filter_value)

    return logits / temperature


class VoxInstructGenerationMixin(GenerationMixin):
    r"""
    Decoding loop of VoxInstruct. The autoregressive stage samples the semantic span and then the first EnCodec
    codebook under three independent classifier free guidance branches, and the non-autoregressive stage fills the
    remaining codebooks by confidence ordered iterative decoding.
    """

    def _ar_step(self, model_kwargs, cache_name, text_embeds, mask_semantic=False):
        outputs = self.ar(
            input_ids=model_kwargs["input_ids"],
            segment_ids=model_kwargs["segment_ids"],
            text_input_ids=model_kwargs.get("text_input_ids"),
            text_attention_mask=model_kwargs.get("text_attention_mask"),
            text_embeds=text_embeds,
            mask_semantic=mask_semantic,
            past_key_values=model_kwargs[cache_name],
            use_cache=True,
        )
        model_kwargs[cache_name] = outputs.past_key_values
        return outputs

    @torch.no_grad()
    def prepare_speech_prompt(
        self,
        language_ids: torch.LongTensor,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.Tensor | None = None,
        semantic_input_values: torch.FloatTensor | None = None,
        semantic_prompt_ratio: float = 0.8,
        semantic_prompt_margin: int = 20,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor | None]:
        r"""
        Tokenizes a speech prompt and lays out the sequence the autoregressive stage is primed with.

        Args:
            language_ids (`torch.LongTensor` of shape `(batch_size,)`):
                Language identity index of each sample.
            input_values (`torch.FloatTensor` of shape `(batch_size, 1, num_samples)`, *optional*):
                Speech prompt at `config.sampling_rate`, tokenized by the acoustic tokenizer.
            padding_mask (`torch.Tensor` of shape `(batch_size, 1, num_samples)`, *optional*):
                Mask over `input_values`.
            semantic_input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`, *optional*):
                Speech prompt at `config.semantic_sampling_rate`, tokenized by the semantic tokenizer.
            semantic_prompt_ratio (`float`, *optional*, defaults to 0.8):
                Fraction of the prompt semantic tokens that is kept. Holding some back leaves the stage room to keep
                predicting semantic tokens of its own instead of closing the span at once.
            semantic_prompt_margin (`int`, *optional*, defaults to 20):
                Number of prompt semantic tokens always held back, whichever fraction is asked for.

        Returns:
            `tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor | None]`: the prompt token ids, their segment
            ids, and the offset acoustic prompt codes.
        """
        ar_config = self.config.ar_config
        device = language_ids.device
        batch_size = language_ids.shape[0]

        opening = torch.full((batch_size, 1), ar_config.bos_token_id, device=device, dtype=torch.long)
        input_ids = torch.cat([opening, (language_ids + ar_config.language_token_offset).unsqueeze(1)], dim=1)

        if semantic_input_values is not None:
            semantic_input_values = semantic_input_values.to(self.semantic_encoder.dtype)
            codes = self.semantic_encoder(semantic_input_values, deduplicate=True)
            length = codes.shape[-1]
            keep = max(min(int(length * semantic_prompt_ratio), length - semantic_prompt_margin), 0)
            input_ids = torch.cat([input_ids, codes[:, :keep] + ar_config.semantic_token_offset], dim=1)

        segment_ids = torch.ones_like(input_ids)

        acoustic_prompt_ids = None
        if input_values is not None:
            encoded = self.audio_encoder.encode(
                input_values.to(self.audio_encoder.dtype),
                padding_mask=padding_mask,
                bandwidth=self.config.audio_bandwidth,
            )
            acoustic_prompt_ids = encoded.audio_codes[0] + ar_config.acoustic_token_offset

        return input_ids, segment_ids, acoustic_prompt_ids

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        segment_ids: torch.LongTensor | None = None,
        text_input_ids: torch.LongTensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        acoustic_prompt_ids: torch.LongTensor | None = None,
        language_ids: torch.LongTensor | None = None,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.Tensor | None = None,
        semantic_input_values: torch.FloatTensor | None = None,
        max_new_tokens: int | None = None,
        semantic_top_k: int = 5,
        semantic_top_p: float = 0.95,
        acoustic_top_k: int = 50,
        acoustic_top_p: float = 0.95,
        temperature: float = 0.8,
        guidance_scale_semantic_on_text: float = 1.0,
        guidance_scale_acoustic_on_text: float = 1.0,
        guidance_scale_acoustic_on_semantic: float = 1.0,
        num_nar_iterations: int = 1,
        return_audio: bool = True,
        vocoder: str = "vocos",
        **kwargs,
    ) -> VoxInstructGenerateOutput:
        r"""
        Generates speech for one instruction.

        Args:
            input_ids (`torch.LongTensor` of shape `(1, prompt_length)`, *optional*):
                Prompt sequence, `<bos> <language>` optionally followed by the semantic tokens of a speech prompt.
                Built from `language_ids` and the speech prompt when it is not given.
            segment_ids (`torch.LongTensor` of shape `(1, prompt_length)`, *optional*):
                Segment ids of the prompt, all ones.
            text_input_ids (`torch.LongTensor` of shape `(1, max_text_len)`):
                Instruction token ids.
            text_attention_mask (`torch.Tensor` of shape `(1, max_text_len)`):
                Mask over `text_input_ids`.
            acoustic_prompt_ids (`torch.LongTensor` of shape `(1, num_codebooks, prompt_frames)`, *optional*):
                Offset EnCodec codes of the speech prompt, inserted at the start of the acoustic span.
            language_ids (`torch.LongTensor` of shape `(1,)`, *optional*):
                Language identity index, used to build the prompt when `input_ids` is not given.
            input_values (`torch.FloatTensor` of shape `(1, 1, num_samples)`, *optional*):
                Speech prompt at `config.sampling_rate`.
            padding_mask (`torch.Tensor` of shape `(1, 1, num_samples)`, *optional*):
                Mask over `input_values`.
            semantic_input_values (`torch.FloatTensor` of shape `(1, num_samples)`, *optional*):
                Speech prompt at `config.semantic_sampling_rate`.
            max_new_tokens (`int`, *optional*):
                Maximum number of tokens the autoregressive stage samples. Defaults to
                `config.ar_config.max_position_embeddings - config.ar_config.max_text_len`.
            semantic_top_k (`int`, *optional*, defaults to 5):
                Top-k used while sampling the semantic span.
            semantic_top_p (`float`, *optional*, defaults to 0.95):
                Nucleus threshold used while sampling the semantic span.
            acoustic_top_k (`int`, *optional*, defaults to 50):
                Top-k used while sampling the first codebook.
            acoustic_top_p (`float`, *optional*, defaults to 0.95):
                Nucleus threshold used while sampling the first codebook.
            temperature (`float`, *optional*, defaults to 0.8):
                Sampling temperature of both spans.
            guidance_scale_semantic_on_text (`float`, *optional*, defaults to 1.0):
                Guidance of the semantic span against the branch that drops the instruction. `1.0` disables it and
                spares one forward pass per step.
            guidance_scale_acoustic_on_text (`float`, *optional*, defaults to 1.0):
                Guidance of the first codebook against the branch that drops the instruction.
            guidance_scale_acoustic_on_semantic (`float`, *optional*, defaults to 1.0):
                Guidance of the first codebook against the branch that drops the semantic span.
            num_nar_iterations (`int`, *optional*, defaults to 1):
                Number of confidence ordered passes the non-autoregressive stage runs per residual codebook.
            return_audio (`bool`, *optional*, defaults to `True`):
                Whether to decode the codes into a waveform.
            vocoder (`str`, *optional*, defaults to `"vocos"`):
                Which decoder turns the codes into a waveform, `"vocos"` for the Vocos vocoder over the summed
                codebook embeddings or `"encodec"` for the EnCodec decoder.

        Returns:
            [`VoxInstructGenerateOutput`]

        Raises:
            ValueError: If neither `input_ids` nor `language_ids` is given, if the batch size is not one, or if
                `vocoder` is neither `"vocos"` nor `"encodec"`.
        """
        if vocoder not in ("vocos", "encodec"):
            raise ValueError(f"`vocoder` must be one of 'vocos' or 'encodec', got {vocoder}.")
        if input_ids is None:
            if language_ids is None:
                raise ValueError("Give either `input_ids` or `language_ids` to build the prompt from.")
            input_ids, segment_ids, acoustic_prompt_ids = self.prepare_speech_prompt(
                language_ids=language_ids,
                input_values=input_values,
                padding_mask=padding_mask,
                semantic_input_values=semantic_input_values,
            )
        if input_ids.shape[0] != 1:
            raise ValueError("VoxInstruct decodes one instruction at a time.")

        ar_config = self.config.ar_config
        device = input_ids.device
        num_codebooks = self.config.num_codebooks
        semantic_eos = ar_config.semantic_eos_token_id
        acoustic_eos = ar_config.acoustic_eos_token_id
        if max_new_tokens is None:
            max_new_tokens = ar_config.max_position_embeddings - ar_config.max_text_len

        if acoustic_prompt_ids is None:
            acoustic_prompt_ids = input_ids.new_zeros((1, num_codebooks, 0))
        prompt_frames = acoustic_prompt_ids.shape[-1]

        model_kwargs = {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "base": None,
            "semantic_on_text": None,
            "acoustic_on_text": None,
            "acoustic_on_semantic": None,
        }
        text_embeds = None
        free_text_embeds = torch.zeros(
            (1, ar_config.max_text_len, ar_config.hidden_size),
            device=device,
            dtype=self.ar.prompt_fc.weight.dtype,
        )

        predicting_semantic = True
        semantic_length = None
        for step in range(max_new_tokens):
            outputs = self._ar_step(model_kwargs, "base", text_embeds)
            text_embeds = outputs.text_embeds
            logits = outputs.logits

            if predicting_semantic:
                if guidance_scale_semantic_on_text != 1.0:
                    free = self._ar_step(model_kwargs, "semantic_on_text", free_text_embeds).logits
                    logits = free + (logits - free) * guidance_scale_semantic_on_text
                logits = logits.clone()
                logits[..., ar_config.bos_token_id] = -1e5
                # The released band boundary sits at `semantic_vocab_size + 1`, two below the true end of the
                # semantic ids, so the last two of them are unreachable here.
                logits[..., ar_config.semantic_vocab_size + 1 : semantic_eos] = -1e5
                logits[..., acoustic_eos] = -1e5
                filtered = top_k_top_p_filtering(
                    logits[0, -1], top_k=semantic_top_k, top_p=semantic_top_p, temperature=temperature
                )
            else:
                if guidance_scale_acoustic_on_text != 1.0:
                    free = self._ar_step(model_kwargs, "acoustic_on_text", free_text_embeds).logits
                    logits = free + (logits - free) * guidance_scale_acoustic_on_text
                if guidance_scale_acoustic_on_semantic != 1.0:
                    free = self._ar_step(
                        model_kwargs, "acoustic_on_semantic", text_embeds, mask_semantic=True
                    ).logits
                    logits = free + (logits - free) * guidance_scale_acoustic_on_semantic
                logits = logits.clone()
                logits[..., ar_config.bos_token_id] = -1e5
                # Same boundary, which leaves the two semantic ids above it reachable on this branch.
                logits[..., : ar_config.semantic_vocab_size + 1] = -1e5
                logits[..., semantic_eos] = -1e5
                filtered = top_k_top_p_filtering(
                    logits[0, -1], top_k=acoustic_top_k, top_p=acoustic_top_p, temperature=temperature
                )

            sample = torch.multinomial(filtered.softmax(dim=-1), 1).unsqueeze(0)
            model_kwargs["input_ids"] = torch.cat([model_kwargs["input_ids"], sample], dim=1)
            segment = 1 if predicting_semantic else 2
            model_kwargs["segment_ids"] = torch.cat(
                [model_kwargs["segment_ids"], torch.full_like(sample, segment)], dim=1
            )

            if sample.item() == semantic_eos:
                predicting_semantic = False
                model_kwargs["input_ids"] = torch.cat(
                    [model_kwargs["input_ids"], acoustic_prompt_ids[:, 0]], dim=1
                )
                model_kwargs["segment_ids"] = torch.cat(
                    [model_kwargs["segment_ids"], torch.full((1, prompt_frames), 2, device=device, dtype=torch.long)],
                    dim=1,
                )
                semantic_length = int((model_kwargs["segment_ids"] == 1).sum())
                # The appended prompt has to run through the decoder, so the cache of the base branch is dropped.
                model_kwargs["base"] = None
            elif sample.item() == acoustic_eos:
                break
            elif step == max_new_tokens - 1:
                model_kwargs["input_ids"] = torch.cat(
                    [model_kwargs["input_ids"], torch.full_like(sample, acoustic_eos)], dim=1
                )
                model_kwargs["segment_ids"] = torch.cat(
                    [model_kwargs["segment_ids"], torch.full_like(sample, 2)], dim=1
                )

        if semantic_length is None:
            raise RuntimeError(
                "The autoregressive stage never closed the semantic span, so there is nothing to decode."
            )

        sequences = model_kwargs["input_ids"]
        segment_ids = model_kwargs["segment_ids"]
        codes = self._generate_residual_codebooks(
            sequences=sequences,
            segment_ids=segment_ids,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            acoustic_prompt_ids=acoustic_prompt_ids,
            semantic_length=semantic_length,
            num_iterations=num_nar_iterations,
        )

        known_length = semantic_length + prompt_frames
        # The trailing position holds the acoustic end of sequence token, which is not part of the speech.
        codes = codes[:, :, known_length:-1] - ar_config.acoustic_token_offset
        codes = codes.clamp(0, ar_config.acoustic_vocab_size - 1)

        audio_values = None
        if return_audio:
            audio_values = self.decode(codes, vocoder=vocoder)

        return VoxInstructGenerateOutput(
            audio_values=audio_values,
            audio_codes=codes,
            sequences=sequences,
            segment_ids=segment_ids,
        )

    def _generate_residual_codebooks(
        self,
        sequences,
        segment_ids,
        text_input_ids,
        text_attention_mask,
        acoustic_prompt_ids,
        semantic_length,
        num_iterations,
    ):
        """Fills codebooks `1 .. num_codebooks - 1` given the first one, by iterative confidence ordered decoding."""
        nar_config = self.config.nar_config
        device = sequences.device
        num_codebooks = self.config.num_codebooks
        length = sequences.shape[1]
        prompt_frames = acoustic_prompt_ids.shape[-1]
        known_length = semantic_length + prompt_frames

        codes = sequences.unsqueeze(1).repeat(1, num_codebooks, 1)
        codes[:, :, semantic_length:known_length] = acoustic_prompt_ids
        positions = torch.arange(length, device=device).unsqueeze(0)
        known = positions < known_length
        # Only the first codebook is known past the prompt, so every other one is blanked there.
        visible = torch.zeros(num_codebooks, dtype=torch.bool, device=device)
        visible[0] = True
        codes = torch.where(visible[None, :, None] | known[:, None, :], codes, torch.zeros_like(codes))

        text_embeds = None
        for codebook_index in range(1, num_codebooks):
            to_predict = positions >= known_length
            predicted_so_far = 0
            total = int(to_predict.sum())
            for iteration in range(num_iterations):
                target = math.ceil(
                    total * (1 - math.cos(math.pi / 2 * (iteration + 1) / num_iterations)) - predicted_so_far
                )
                outputs = self.nar(
                    input_ids=codes.transpose(1, 2),
                    segment_ids=segment_ids,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    text_embeds=text_embeds,
                    codebook_index=codebook_index,
                )
                text_embeds = outputs.text_embeds
                logits = outputs.logits

                confidence = F.softmax(logits, dim=-1).max(dim=-1).values * to_predict.float()
                order = torch.argsort(confidence, dim=1, descending=True)
                selected = order[:, :target]
                predictions = logits.argmax(dim=-1) + nar_config.acoustic_token_offset

                codes[:, codebook_index] = codes[:, codebook_index].scatter(
                    1, selected, predictions.gather(1, order)[:, :target]
                )
                to_predict = to_predict.scatter(1, selected, torch.zeros_like(to_predict))
                predicted_so_far += target

        return codes


__all__ = ["VoxInstructGenerationMixin", "VoxInstructGenerateOutput", "top_k_top_p_filtering"]
