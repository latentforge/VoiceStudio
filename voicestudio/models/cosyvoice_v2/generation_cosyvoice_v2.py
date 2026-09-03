"""Generation utilities for CosyVoice v2."""

from typing import Generator, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..cosyvoice_v1.generation_cosyvoice_v1 import fade_in_out, repetition_aware_sampling


class CosyVoiceV2GenerationMixin:
    """
    Speech token decoding and token to waveform rendering for [`CosyVoiceV2ForConditionalGeneration`].

    The Qwen2 language model samples speech tokens one at a time with repetition aware sampling and a
    key value cache, the flow matching model turns the whole sequence decoded so far into a mel
    spectrogram, and the vocoder renders the waveform. In streaming mode each chunk re-encodes the
    sequence from its start and `token_offset` selects the mel frames that were not rendered yet, so
    only the vocoder carries a cache across chunks.
    """

    token_hop_len = 25
    stream_scale_factor = 2
    mel_cache_len = 8

    @torch.inference_mode()
    def generate_speech_tokens(
        self,
        input_ids: torch.Tensor,
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
            prompt_input_ids (`torch.Tensor` of shape `(1, prompt_text_length)`, *optional*):
                Text token ids of the prompt utterance, prepended to `input_ids`.
            prompt_speech_token_ids (`torch.Tensor` of shape `(1, prompt_speech_length)`, *optional*):
                Speech tokens of the prompt utterance the language model continues from.
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
        text_length = input_ids.shape[1]
        if prompt_input_ids is None:
            prompt_input_ids = input_ids.new_zeros(1, 0)
        input_ids = torch.concat([prompt_input_ids, input_ids], dim=1)
        text_embeds = self.llm.embed_text(input_ids)

        sos_embed = self.llm.llm_embedding.weight[self.llm.sos_index].reshape(1, 1, -1)
        task_id_embed = self.llm.llm_embedding.weight[self.llm.task_id_index].reshape(1, 1, -1)
        if prompt_speech_token_ids is not None and prompt_speech_token_ids.shape[1] != 0:
            prompt_embeds = self.llm.speech_embedding(prompt_speech_token_ids)
        else:
            prompt_embeds = text_embeds.new_zeros(1, 0, self.config.lm_hidden_size)
        hidden_states = torch.concat([sos_embed, text_embeds, task_id_embed, prompt_embeds], dim=1)

        min_length = int(text_length * min_token_text_ratio)
        max_length = int(text_length * max_token_text_ratio)

        decoded_tokens: list[int] = []
        past_key_values = None
        for step in range(max_length):
            logits, past_key_values = self.llm(hidden_states, past_key_values=past_key_values, use_cache=True)
            log_probs = logits[:, -1].log_softmax(dim=-1).squeeze(dim=0)
            if step < min_length:
                log_probs[self.llm.eos_token_id] = -float("inf")
            token_id = repetition_aware_sampling(
                log_probs, decoded_tokens, top_p, top_k, window_size, repetition_threshold
            )
            if token_id in self.llm.stop_token_ids:
                break
            yield token_id
            decoded_tokens.append(token_id)
            hidden_states = self.llm.speech_embedding.weight[token_id].reshape(1, 1, -1)

    def init_streaming_state(self) -> dict:
        """
        Returns:
            `dict`: a fresh streaming state for [`~CosyVoiceV2GenerationMixin.token2wav`].
        """
        source_cache_len = int(self.mel_cache_len * np.prod(self.config.vocoder_upsample_rates)
                               * self.config.vocoder_istft_hop_length)
        return {
            "hift_cache": None,
            "source_cache_len": source_cache_len,
            "speech_window": torch.from_numpy(np.hamming(2 * source_cache_len).astype(np.float32)),
        }

    @torch.inference_mode()
    def token2wav(
        self,
        speech_token_ids: torch.Tensor,
        prompt_speech_token_ids: torch.Tensor,
        prompt_speech_feat: torch.Tensor,
        speaker_embedding: torch.Tensor,
        state: dict,
        token_offset: int = 0,
        stream: bool = False,
        finalize: bool = True,
        speed: float = 1.0,
    ) -> torch.Tensor:
        """
        Renders the speech tokens decoded so far into the waveform of one chunk.

        Args:
            speech_token_ids (`torch.Tensor` of shape `(1, speech_length)`):
                Every speech token decoded so far, prompt excluded.
            prompt_speech_token_ids (`torch.Tensor` of shape `(1, prompt_speech_length)`):
                Speech tokens of the prompt utterance.
            prompt_speech_feat (`torch.Tensor` of shape `(1, prompt_mel_length, flow_output_size)`):
                Mel spectrogram of the prompt utterance.
            speaker_embedding (`torch.Tensor` of shape `(1, speaker_embedding_dim)`):
                Utterance level speaker embedding.
            state (`dict`):
                Mutable streaming state holding the vocoder cache.
            token_offset (`int`, *optional*, defaults to 0):
                Number of speech tokens whose mel frames were rendered by a previous call.
            stream (`bool`, *optional*, defaults to `False`):
                Whether the flow matching model attends within chunks only.
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
        mel = self.flow.inference(
            speech_token_ids=speech_token_ids.to(device=device, dtype=torch.int32),
            speech_token_lengths=torch.tensor([speech_token_ids.shape[1]], dtype=torch.int32, device=device),
            prompt_token_ids=prompt_speech_token_ids.to(device=device, dtype=torch.int32),
            prompt_token_lengths=torch.tensor(
                [prompt_speech_token_ids.shape[1]], dtype=torch.int32, device=device
            ),
            prompt_feat=prompt_speech_feat.to(device),
            speaker_embedding=speaker_embedding.to(device),
            num_steps=self.config.num_flow_inference_steps,
            streaming=stream,
            finalize=finalize,
        )
        mel = mel[:, :, token_offset * self.config.token_mel_ratio :]

        if state["hift_cache"] is not None:
            mel = torch.concat([state["hift_cache"]["mel"], mel], dim=2)
            cache_source = state["hift_cache"]["source"]
        else:
            cache_source = mel.new_zeros(1, 1, 0)

        if not finalize:
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
                Utterance level speaker embedding. It conditions the flow matching model only, since
                the v2 language model does not read it.
            prompt_input_ids (`torch.Tensor` of shape `(1, prompt_text_length)`, *optional*):
                Text token ids of the prompt utterance.
            prompt_speech_token_ids (`torch.Tensor` of shape `(1, prompt_speech_length)`, *optional*):
                Speech tokens of the prompt utterance the language model continues from.
            flow_prompt_speech_token_ids (`torch.Tensor` of shape `(1, prompt_speech_length)`, *optional*):
                Speech tokens the flow matching model is conditioned on, alongside
                `prompt_speech_feat`. Defaults to `prompt_speech_token_ids`.
            prompt_speech_feat (`torch.Tensor` of shape `(1, prompt_mel_length, flow_output_size)`, *optional*):
                Mel spectrogram of the prompt utterance.
            source_speech_token_ids (`torch.Tensor` of shape `(1, speech_length)`, *optional*):
                Speech tokens to render directly, which bypasses the language model and performs voice
                conversion instead of text to speech.
            stream (`bool`, *optional*, defaults to `False`):
                Whether to return a generator yielding the waveform chunk by chunk.
            speed (`float`, *optional*, defaults to 1.0):
                Playback speed, only available when `stream` is `False`.
            sampling_kwargs:
                Forwarded to [`~CosyVoiceV2GenerationMixin.generate_speech_tokens`].

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
        prompt_input_ids: Optional[torch.Tensor],
        prompt_speech_token_ids: Optional[torch.Tensor],
        source_speech_token_ids: Optional[torch.Tensor],
        **sampling_kwargs,
    ):
        if source_speech_token_ids is not None:
            return iter(source_speech_token_ids.flatten().tolist())
        return self.generate_speech_tokens(
            input_ids,
            prompt_input_ids=prompt_input_ids,
            prompt_speech_token_ids=prompt_speech_token_ids,
            **sampling_kwargs,
        )

    def _prompt_defaults(
        self,
        input_ids: torch.Tensor,
        flow_prompt_speech_token_ids: Optional[torch.Tensor],
        prompt_speech_feat: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flow_prompt_speech_token_ids is None:
            flow_prompt_speech_token_ids = input_ids.new_zeros(1, 0)
        if prompt_speech_feat is None:
            prompt_speech_feat = torch.zeros(1, 0, self.config.flow_output_size, device=input_ids.device)
        return flow_prompt_speech_token_ids, prompt_speech_feat

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
        flow_prompt_speech_token_ids, prompt_speech_feat = self._prompt_defaults(
            input_ids, flow_prompt_speech_token_ids, prompt_speech_feat
        )
        state = self.init_streaming_state()
        tokens = list(
            self._speech_token_stream(
                input_ids, prompt_input_ids, prompt_speech_token_ids, source_speech_token_ids,
                **sampling_kwargs,
            )
        )
        speech_token_ids = torch.tensor([tokens], dtype=torch.int32, device=input_ids.device)
        return self.token2wav(
            speech_token_ids, flow_prompt_speech_token_ids, prompt_speech_feat, speaker_embedding, state,
            token_offset=0, stream=False, finalize=True, speed=speed,
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
        flow_prompt_speech_token_ids, prompt_speech_feat = self._prompt_defaults(
            input_ids, flow_prompt_speech_token_ids, prompt_speech_feat
        )
        state = self.init_streaming_state()
        token_stream = self._speech_token_stream(
            input_ids, prompt_input_ids, prompt_speech_token_ids, source_speech_token_ids, **sampling_kwargs
        )

        lookahead = self.config.pre_lookahead_len
        token_hop_len = self.token_hop_len
        token_max_hop_len = 4 * self.token_hop_len
        prompt_length = flow_prompt_speech_token_ids.shape[1]
        prompt_pad = int(np.ceil(prompt_length / self.token_hop_len) * self.token_hop_len - prompt_length)

        tokens: list[int] = []
        token_offset = 0
        for token_id in token_stream:
            tokens.append(token_id)
            while True:
                hop = token_hop_len + prompt_pad if token_offset == 0 else token_hop_len
                if len(tokens) - token_offset < hop + lookahead:
                    break
                chunk = torch.tensor(
                    [tokens[: token_offset + hop + lookahead]], dtype=torch.int32, device=input_ids.device
                )
                yield self.token2wav(
                    chunk, flow_prompt_speech_token_ids, prompt_speech_feat, speaker_embedding, state,
                    token_offset=token_offset, stream=True, finalize=False,
                )
                token_offset += hop
                token_hop_len = min(token_max_hop_len, token_hop_len * self.stream_scale_factor)

        chunk = torch.tensor([tokens], dtype=torch.int32, device=input_ids.device)
        yield self.token2wav(
            chunk, flow_prompt_speech_token_ids, prompt_speech_feat, speaker_embedding, state,
            token_offset=token_offset, stream=False, finalize=True,
        )


__all__ = ["CosyVoiceV2GenerationMixin"]
