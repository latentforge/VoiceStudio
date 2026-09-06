# coding=utf-8
# Copyright 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du) and the HuggingFace Inc. team. All rights reserved.
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
"""Generation utilities for CosyVoice v1."""

from typing import Generator, Optional

import numpy as np
import torch
import torch.nn.functional as F


def nucleus_sampling(scores: torch.Tensor, top_p: float = 0.8, top_k: int = 25) -> int:
    """
    Samples one speech token from the smallest set of tokens whose cumulative probability stays below
    `top_p`, capped at `top_k` tokens.

    Args:
        scores (`torch.Tensor` of shape `(speech_vocab_size + 1,)`):
            Log probabilities of the next speech token.
        top_p (`float`, *optional*, defaults to 0.8):
            Cumulative probability the candidate set is grown to.
        top_k (`int`, *optional*, defaults to 25):
            Largest number of candidates.

    Returns:
        `int`: the sampled token id.
    """
    sorted_value, sorted_index = scores.softmax(dim=0).sort(descending=True, stable=True)
    cumulative = torch.cumsum(sorted_value, dim=0)
    keep = int(torch.searchsorted(cumulative, cumulative.new_tensor(top_p)).item()) + 1
    keep = max(1, min(keep, top_k, sorted_value.numel()))
    probabilities = sorted_value[:keep]
    return sorted_index[probabilities.multinomial(1, replacement=True)].item()


def random_sampling(scores: torch.Tensor) -> int:
    """
    Samples one speech token from the full distribution.

    Args:
        scores (`torch.Tensor` of shape `(speech_vocab_size + 1,)`):
            Log probabilities of the next speech token.

    Returns:
        `int`: the sampled token id.
    """
    return scores.softmax(dim=0).multinomial(1, replacement=True).item()


def repetition_aware_sampling(
    scores: torch.Tensor,
    decoded_tokens: list[int],
    top_p: float = 0.8,
    top_k: int = 25,
    window_size: int = 10,
    repetition_threshold: float = 0.1,
) -> int:
    """
    Nucleus sampling that falls back to sampling from the full distribution when the drawn token
    already occupies too much of the recent window, as introduced by VALL-E 2.

    Args:
        scores (`torch.Tensor` of shape `(speech_vocab_size + 1,)`):
            Log probabilities of the next speech token.
        decoded_tokens (`list[int]`):
            Speech tokens decoded so far.
        top_p (`float`, *optional*, defaults to 0.8):
            Cumulative probability the candidate set is grown to.
        top_k (`int`, *optional*, defaults to 25):
            Largest number of candidates.
        window_size (`int`, *optional*, defaults to 10):
            Number of recent tokens inspected for repetitions.
        repetition_threshold (`float`, *optional*, defaults to 0.1):
            Fraction of the window above which the drawn token is rejected.

    Returns:
        `int`: the sampled token id.
    """
    token_id = nucleus_sampling(scores, top_p=top_p, top_k=top_k)
    recent = decoded_tokens[-window_size:]
    repetitions = sum(1 for token in recent if token == token_id)
    if repetitions >= window_size * repetition_threshold:
        scores = scores.clone()
        scores[token_id] = -float("inf")
        token_id = random_sampling(scores)
    return token_id


def fade_in_out(fade_in: torch.Tensor, fade_out: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """
    Cross fades the head of a chunk with the tail of the previous one.

    Args:
        fade_in (`torch.Tensor`):
            Chunk whose head is faded in.
        fade_out (`torch.Tensor`):
            Tail of the previous chunk, faded out.
        window (`torch.Tensor` of shape `(2 * overlap,)`):
            Symmetric window, the first half fading in and the second half fading out.

    Returns:
        `torch.Tensor`: the cross faded chunk.
    """
    overlap = window.shape[0] // 2
    window = window.to(fade_in.device, fade_in.dtype)
    fade_in = fade_in.clone()
    fade_in[..., :overlap] = fade_in[..., :overlap] * window[:overlap] + fade_out[..., -overlap:] * window[overlap:]
    return fade_in


class CosyVoiceV1GenerationMixin:
    """
    Speech token decoding and token to waveform rendering for [`CosyVoiceV1ForConditionalGeneration`].

    The language model samples speech tokens one at a time with repetition aware sampling, the flow
    matching model turns them into a mel spectrogram and the vocoder renders the waveform. In streaming
    mode the mel spectrogram and the waveform of consecutive chunks are cross faded, and the flow
    matching cache keeps the prompt and the chunk overlap fixed across chunks.
    """

    token_overlap_len = 20
    mel_cache_len = 20
    stream_scale_factor = 1

    @torch.inference_mode()
    def generate_speech_tokens(
        self,
        input_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor] = None,
        prompt_speech_token_ids: Optional[torch.Tensor] = None,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
        top_p: float = 0.8,
        top_k: int = 25,
        window_size: int = 10,
        repetition_threshold: float = 0.1,
    ) -> Generator[int, None, None]:
        """
        Samples speech tokens for one utterance and yields them one at a time.

        Args:
            input_ids (`torch.Tensor` of shape `(1, text_length)`):
                Text token ids of the sentence to synthesize.
            speaker_embedding (`torch.Tensor` of shape `(1, speaker_embedding_dim)`):
                Utterance level speaker embedding, or an empty tensor to leave the speaker
                conditioning out.
            prompt_input_ids (`torch.Tensor` of shape `(1, prompt_text_length)`, *optional*):
                Text token ids of the prompt utterance.
            prompt_speech_token_ids (`torch.Tensor` of shape `(1, prompt_speech_length)`, *optional*):
                Speech tokens of the prompt utterance.
            min_token_text_ratio (`float`, *optional*, defaults to 2.0):
                Smallest number of speech tokens generated per text token.
            max_token_text_ratio (`float`, *optional*, defaults to 20.0):
                Largest number of speech tokens generated per text token.
            top_p (`float`, *optional*, defaults to 0.8):
                Cumulative probability the candidate set is grown to.
            top_k (`int`, *optional*, defaults to 25):
                Largest number of candidates.
            window_size (`int`, *optional*, defaults to 10):
                Number of recent tokens inspected for repetitions.
            repetition_threshold (`float`, *optional*, defaults to 0.1):
                Fraction of the window above which the drawn token is rejected.

        Yields:
            `int`: the next speech token id.
        """
        device = input_ids.device
        text_length = input_ids.shape[1]
        if prompt_input_ids is None:
            prompt_input_ids = input_ids.new_zeros(1, 0)
        input_ids = torch.concat([prompt_input_ids, input_ids], dim=1)
        input_lengths = torch.tensor([input_ids.shape[1]], dtype=torch.int32, device=device)

        text_hidden_states = self.llm.encode_text(input_ids, input_lengths)
        if speaker_embedding.shape[0] != 0:
            speaker_hidden_states = self.llm.encode_speaker(speaker_embedding)
        else:
            speaker_hidden_states = text_hidden_states.new_zeros(1, 0, self.config.lm_hidden_size)

        sos_embed = self.llm.llm_embedding.weight[self.llm.sos_index].reshape(1, 1, -1)
        task_id_embed = self.llm.llm_embedding.weight[self.llm.task_id_index].reshape(1, 1, -1)
        if prompt_speech_token_ids is not None and prompt_speech_token_ids.shape[1] != 0:
            prompt_hidden_states = self.llm.speech_embedding(prompt_speech_token_ids)
        else:
            prompt_hidden_states = text_hidden_states.new_zeros(1, 0, self.config.lm_hidden_size)
        hidden_states = torch.concat(
            [sos_embed, speaker_hidden_states, text_hidden_states, task_id_embed, prompt_hidden_states], dim=1
        )

        min_length = int(text_length * min_token_text_ratio)
        max_length = int(text_length * max_token_text_ratio)

        decoded_tokens: list[int] = []
        past_key_values = None
        for step in range(max_length):
            padding_mask = hidden_states.new_ones(
                1, hidden_states.shape[1] + (0 if past_key_values is None else past_key_values[0][0].shape[2]),
                dtype=torch.bool,
            )
            outputs, past_key_values = self.llm.llm(hidden_states, padding_mask, past_key_values)
            log_probs = self.llm.llm_decoder(outputs[:, -1]).log_softmax(dim=-1).squeeze(dim=0)
            if step < min_length:
                log_probs[self.llm.eos_token_id] = -float("inf")
            token_id = repetition_aware_sampling(
                log_probs, decoded_tokens, top_p, top_k, window_size, repetition_threshold
            )
            if token_id == self.llm.eos_token_id:
                break
            yield token_id
            decoded_tokens.append(token_id)
            hidden_states = self.llm.speech_embedding.weight[token_id].reshape(1, 1, -1)

    @torch.inference_mode()
    def token2wav(
        self,
        speech_token_ids: torch.Tensor,
        prompt_speech_token_ids: torch.Tensor,
        prompt_speech_feat: torch.Tensor,
        speaker_embedding: torch.Tensor,
        state: dict,
        finalize: bool = True,
        speed: float = 1.0,
    ) -> torch.Tensor:
        """
        Renders one chunk of speech tokens into a waveform.

        Args:
            speech_token_ids (`torch.Tensor` of shape `(1, speech_length)`):
                Speech tokens of the chunk.
            prompt_speech_token_ids (`torch.Tensor` of shape `(1, prompt_speech_length)`):
                Speech tokens of the prompt utterance.
            prompt_speech_feat (`torch.Tensor` of shape `(1, prompt_mel_length, flow_output_size)`):
                Mel spectrogram of the prompt utterance.
            speaker_embedding (`torch.Tensor` of shape `(1, speaker_embedding_dim)`):
                Utterance level speaker embedding.
            state (`dict`):
                Mutable streaming state holding the flow matching cache, the mel overlap and the
                vocoder cache.
            finalize (`bool`, *optional*, defaults to `True`):
                Whether this is the last chunk of the utterance.
            speed (`float`, *optional*, defaults to 1.0):
                Playback speed, only available on the last chunk of a non streaming call.

        Returns:
            `torch.Tensor` of shape `(1, num_samples)`: the waveform of the chunk.

        Raises:
            ValueError: If `speed` is changed while streaming.
        """
        device = speech_token_ids.device
        mel, state["flow_cache"] = self.flow.inference(
            speech_token_ids=speech_token_ids.to(device=device, dtype=torch.int32),
            speech_token_lengths=torch.tensor([speech_token_ids.shape[1]], dtype=torch.int32, device=device),
            prompt_token_ids=prompt_speech_token_ids.to(device=device, dtype=torch.int32),
            prompt_token_lengths=torch.tensor(
                [prompt_speech_token_ids.shape[1]], dtype=torch.int32, device=device
            ),
            prompt_feat=prompt_speech_feat.to(device),
            speaker_embedding=speaker_embedding.to(device),
            num_steps=self.config.num_flow_inference_steps,
            cache=state["flow_cache"],
        )

        if state["mel_overlap"].shape[2] != 0:
            mel = fade_in_out(mel, state["mel_overlap"], state["mel_window"])

        if state["hift_cache"] is not None:
            mel = torch.concat([state["hift_cache"]["mel"], mel], dim=2)
            cache_source = state["hift_cache"]["source"]
        else:
            cache_source = mel.new_zeros(1, 1, 0)

        if not finalize:
            state["mel_overlap"] = mel[:, :, -state["mel_overlap_len"] :]
            mel = mel[:, :, : -state["mel_overlap_len"]]
            waveform, source = self.hift.inference(mel, cache_source)
            if state["hift_cache"] is not None:
                waveform = fade_in_out(waveform, state["hift_cache"]["speech"], state["speech_window"])
            state["hift_cache"] = {
                "mel": mel[:, :, -self.mel_cache_len :],
                "source": source[:, :, -state["source_cache_len"] :],
                "speech": waveform[:, -state["source_cache_len"] :],
            }
            return waveform[:, : -state["source_cache_len"]]

        if speed != 1.0:
            if state["hift_cache"] is not None:
                raise ValueError("speed can only be changed in non streaming mode")
            mel = F.interpolate(mel, size=int(mel.shape[2] / speed), mode="linear")
        waveform, _ = self.hift.inference(mel, cache_source)
        if state["hift_cache"] is not None:
            waveform = fade_in_out(waveform, state["hift_cache"]["speech"], state["speech_window"])
        return waveform

    def init_streaming_state(self) -> dict:
        """
        Returns:
            `dict`: a fresh streaming state for [`~CosyVoiceV1GenerationMixin.token2wav`].
        """
        mel_overlap_len = int(self.token_overlap_len / self.config.flow_input_frame_rate * 22050 / 256)
        source_cache_len = int(self.mel_cache_len * 256)
        return {
            "flow_cache": None,
            "mel_overlap": torch.zeros(1, self.config.flow_output_size, 0),
            "mel_overlap_len": mel_overlap_len,
            "mel_window": torch.from_numpy(np.hamming(2 * mel_overlap_len).astype(np.float32)),
            "hift_cache": None,
            "source_cache_len": source_cache_len,
            "speech_window": torch.from_numpy(np.hamming(2 * source_cache_len).astype(np.float32)),
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor] = None,
        prompt_speech_token_ids: Optional[torch.Tensor] = None,
        flow_prompt_speech_token_ids: Optional[torch.Tensor] = None,
        prompt_speech_feat: Optional[torch.Tensor] = None,
        source_speech_token_ids: Optional[torch.Tensor] = None,
        stream: bool = False,
        speed: float = 1.0,
        **sampling_kwargs,
    ) -> "torch.Tensor | Generator[torch.Tensor, None, None]":
        """
        Synthesizes one utterance.

        Args:
            input_ids (`torch.Tensor` of shape `(1, text_length)`):
                Text token ids of the sentence to synthesize.
            speaker_embedding (`torch.Tensor` of shape `(1, speaker_embedding_dim)`):
                Utterance level speaker embedding.
            prompt_input_ids (`torch.Tensor` of shape `(1, prompt_text_length)`, *optional*):
                Text token ids of the prompt utterance, prepended to `input_ids` before the text
                encoder runs.
            prompt_speech_token_ids (`torch.Tensor` of shape `(1, prompt_speech_length)`, *optional*):
                Speech tokens of the prompt utterance the language model continues from. Leaving it
                out while `flow_prompt_speech_token_ids` is given is what makes the generation
                cross lingual: the language model then only sees the speaker embedding.
            flow_prompt_speech_token_ids (`torch.Tensor` of shape `(1, prompt_speech_length)`, *optional*):
                Speech tokens the flow matching model is conditioned on, alongside
                `prompt_speech_feat`. Defaults to `prompt_speech_token_ids`.
            prompt_speech_feat (`torch.Tensor` of shape `(1, prompt_mel_length, flow_output_size)`, *optional*):
                Mel spectrogram of the prompt utterance.
            source_speech_token_ids (`torch.Tensor` of shape `(1, speech_length)`, *optional*):
                Speech tokens to render directly, which bypasses the language model and performs voice
                conversion instead of text to speech.
            stream (`bool`, *optional*, defaults to `False`):
                Whether to return a generator yielding the waveform chunk by chunk instead of the
                whole waveform.
            speed (`float`, *optional*, defaults to 1.0):
                Playback speed, only available when `stream` is `False`.
            sampling_kwargs:
                Forwarded to [`~CosyVoiceV1GenerationMixin.generate_speech_tokens`].

        Returns:
            `torch.Tensor` of shape `(1, num_samples)`, or a generator of such tensors when `stream`
            is `True`: the generated waveform.
        """
        if flow_prompt_speech_token_ids is None:
            flow_prompt_speech_token_ids = prompt_speech_token_ids
        if stream:
            return self._generate_stream(
                input_ids,
                speaker_embedding,
                prompt_input_ids,
                prompt_speech_token_ids,
                flow_prompt_speech_token_ids,
                prompt_speech_feat,
                source_speech_token_ids,
                **sampling_kwargs,
            )
        return self._generate_single(
            input_ids,
            speaker_embedding,
            prompt_input_ids,
            prompt_speech_token_ids,
            flow_prompt_speech_token_ids,
            prompt_speech_feat,
            source_speech_token_ids,
            speed,
            **sampling_kwargs,
        )

    def _speech_token_stream(
        self,
        input_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor],
        prompt_speech_token_ids: Optional[torch.Tensor],
        source_speech_token_ids: Optional[torch.Tensor],
        **sampling_kwargs,
    ):
        if source_speech_token_ids is not None:
            return iter(source_speech_token_ids.flatten().tolist())
        return self.generate_speech_tokens(
            input_ids,
            speaker_embedding,
            prompt_input_ids=prompt_input_ids,
            prompt_speech_token_ids=prompt_speech_token_ids,
            **sampling_kwargs,
        )

    @torch.inference_mode()
    def _generate_single(
        self,
        input_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor],
        prompt_speech_token_ids: Optional[torch.Tensor],
        flow_prompt_speech_token_ids: Optional[torch.Tensor],
        prompt_speech_feat: Optional[torch.Tensor],
        source_speech_token_ids: Optional[torch.Tensor],
        speed: float,
        **sampling_kwargs,
    ) -> torch.Tensor:
        device = input_ids.device
        if flow_prompt_speech_token_ids is None:
            flow_prompt_speech_token_ids = input_ids.new_zeros(1, 0)
        if prompt_speech_feat is None:
            prompt_speech_feat = torch.zeros(1, 0, self.config.flow_output_size, device=device)

        state = self.init_streaming_state()
        token_stream = self._speech_token_stream(
            input_ids, speaker_embedding, prompt_input_ids, prompt_speech_token_ids,
            source_speech_token_ids, **sampling_kwargs,
        )
        tokens = list(token_stream)
        speech_token_ids = torch.tensor([tokens], dtype=torch.int32, device=device)
        return self.token2wav(
            speech_token_ids, flow_prompt_speech_token_ids, prompt_speech_feat, speaker_embedding, state,
            finalize=True, speed=speed,
        )

    @torch.inference_mode()
    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor],
        prompt_speech_token_ids: Optional[torch.Tensor],
        flow_prompt_speech_token_ids: Optional[torch.Tensor],
        prompt_speech_feat: Optional[torch.Tensor],
        source_speech_token_ids: Optional[torch.Tensor],
        **sampling_kwargs,
    ) -> Generator[torch.Tensor, None, None]:
        device = input_ids.device
        if flow_prompt_speech_token_ids is None:
            flow_prompt_speech_token_ids = input_ids.new_zeros(1, 0)
        if prompt_speech_feat is None:
            prompt_speech_feat = torch.zeros(1, 0, self.config.flow_output_size, device=device)

        state = self.init_streaming_state()
        token_stream = self._speech_token_stream(
            input_ids, speaker_embedding, prompt_input_ids, prompt_speech_token_ids,
            source_speech_token_ids, **sampling_kwargs,
        )

        token_hop_len = 2 * self.config.flow_input_frame_rate
        token_max_hop_len = 4 * self.config.flow_input_frame_rate
        tokens: list[int] = []
        for token_id in token_stream:
            tokens.append(token_id)
            while len(tokens) >= token_hop_len + self.token_overlap_len:
                chunk = torch.tensor([tokens[: token_hop_len + self.token_overlap_len]], dtype=torch.int32, device=device)
                yield self.token2wav(
                    chunk, flow_prompt_speech_token_ids, prompt_speech_feat, speaker_embedding, state,
                    finalize=False,
                )
                tokens = tokens[token_hop_len:]
                token_hop_len = min(token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
        chunk = torch.tensor([tokens], dtype=torch.int32, device=device)
        yield self.token2wav(
            chunk, flow_prompt_speech_token_ids, prompt_speech_feat, speaker_embedding, state, finalize=True
        )


__all__ = [
    "CosyVoiceV1GenerationMixin",
    "fade_in_out",
    "nucleus_sampling",
    "random_sampling",
    "repetition_aware_sampling",
]
