"""Generation utilities for CosyVoice v3."""

from types import GeneratorType
from typing import Generator, Iterator, Optional, Union

import torch
import torch.nn.functional as F

from ..cosyvoice_v2.generation_cosyvoice_v2 import CosyVoiceV2GenerationMixin


class CosyVoiceV3GenerationMixin(CosyVoiceV2GenerationMixin):
    """
    Speech token decoding and token to waveform rendering for [`CosyVoiceV3ForConditionalGeneration`].

    The language model loop is v2's unchanged, because v3 moves its control vectors into the speech
    token table without changing how a sequence is packed. Two things around it are new. Runs of the
    speech tokens upstream treats as silence or breath are thinned to `max_silent_run`, and the
    vocoder carries no cache: each chunk re-renders the whole mel spectrogram accumulated so far and
    returns only the samples a previous chunk did not already emit.
    """

    def _speech_token_stream(
        self,
        input_ids: Union[torch.Tensor, Iterator[torch.Tensor]],
        prompt_input_ids: Optional[torch.Tensor],
        prompt_speech_token_ids: Optional[torch.Tensor],
        source_speech_token_ids: Optional[torch.Tensor],
        **sampling_kwargs,
    ):
        stream = super()._speech_token_stream(
            input_ids, prompt_input_ids, prompt_speech_token_ids, source_speech_token_ids,
            **sampling_kwargs,
        )
        if source_speech_token_ids is not None:
            return stream
        return self._thin_silence(stream)

    def _thin_silence(self, stream) -> Generator[int, None, None]:
        """
        Drops the tail of any run of silence or breath tokens longer than `max_silent_run`.

        Args:
            stream (`Iterable[int]`):
                Speech tokens as the language model produces them.

        Yields:
            `int`: the next kept speech token id.
        """
        silent = set(self.config.silent_token_ids)
        run = 0
        for token_id in stream:
            if token_id in silent:
                run += 1
                if run > self.config.max_silent_run:
                    continue
            else:
                run = 0
            yield token_id

    def split_prompt_text(self, prompt_input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the prompt text of an interleaved decode at the end of prompt token, which v3 places
        at the head of the sequence rather than interleaving it with the speech tokens.

        Args:
            prompt_input_ids (`torch.Tensor` of shape `(1, prompt_text_length)`):
                Text token ids of the prompt utterance.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the instruction, up to and including the end of
            prompt token, and the transcript that follows it.

        Raises:
            ValueError: If the end of prompt token does not appear in `prompt_input_ids`.
        """
        positions = (prompt_input_ids.flatten() == self.config.end_of_prompt_token_id).nonzero()
        if positions.numel() == 0:
            raise ValueError(
                f"CosyVoice v3 expects the end of prompt token {self.config.end_of_prompt_token_id} "
                "in the prompt text of an interleaved decode, and it does not contain it."
            )
        index = int(positions[0]) + 1
        return prompt_input_ids[:, :index], prompt_input_ids[:, index:]

    def init_streaming_state(self) -> dict:
        """
        Returns:
            `dict`: a fresh streaming state for [`~CosyVoiceV3GenerationMixin.token2wav`].
        """
        return {"mel": None, "speech_offset": 0}

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
                Mutable streaming state holding the mel spectrogram accumulated so far and the
                number of samples already emitted.
            token_offset (`int`, *optional*, defaults to 0):
                Number of speech tokens whose mel frames were produced by a previous call.
            stream (`bool`, *optional*, defaults to `False`):
                Whether the flow matching model attends within chunks only.
            finalize (`bool`, *optional*, defaults to `True`):
                Whether this is the last chunk of the utterance.
            speed (`float`, *optional*, defaults to 1.0):
                Playback speed, only available on a single chunk non streaming call.

        Returns:
            `torch.Tensor` of shape `(1, num_samples)`: the waveform of the chunk.

        Raises:
            ValueError: If `speed` is changed on anything but a single final chunk.
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

        if state["mel"] is not None:
            mel = torch.concat([state["mel"], mel], dim=2)
        state["mel"] = mel

        if speed != 1.0:
            if token_offset != 0 or not finalize:
                raise ValueError("speed can only be changed on a single final chunk")
            mel = F.interpolate(mel, size=int(mel.shape[2] / speed), mode="linear")

        waveform, _ = self.hift.inference(mel, finalize=finalize)
        waveform = waveform[:, state["speech_offset"] :]
        state["speech_offset"] += waveform.shape[1]
        return waveform

    def generate(
        self,
        input_ids: Union[torch.Tensor, Iterator[torch.Tensor]],
        speaker_embedding: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "torch.Tensor | Generator[torch.Tensor, None, None]":
        """
        Synthesizes one utterance.

        Args:
            input_ids (`torch.Tensor` of shape `(1, text_length)`, or a generator of such tensors):
                Text token ids of the sentence to synthesize. Upstream requires the end of prompt
                token to be present in the text or the prompt text, and in the prompt text alone
                when the text arrives as a generator.
            speaker_embedding (`torch.Tensor` of shape `(1, speaker_embedding_dim)`):
                Utterance level speaker embedding, which conditions the flow matching model only.
            prompt_input_ids (`torch.Tensor` of shape `(1, prompt_text_length)`, *optional*):
                Text token ids of the prompt utterance.
            kwargs:
                Forwarded to [`~CosyVoiceV2GenerationMixin.generate`].

        Returns:
            `torch.Tensor` of shape `(1, num_samples)`, or a generator of such tensors when `stream`
            is `True`: the generated waveform.

        Raises:
            ValueError: If the end of prompt token appears in neither `input_ids` nor
                `prompt_input_ids`.
        """
        end_of_prompt = self.config.end_of_prompt_token_id
        present = not isinstance(input_ids, GeneratorType) and bool((input_ids == end_of_prompt).any())
        if prompt_input_ids is not None:
            present = present or bool((prompt_input_ids == end_of_prompt).any())
        if not present:
            raise ValueError(
                f"CosyVoice v3 expects the end of prompt token {end_of_prompt} in the text or the "
                "prompt text, and neither contains it."
            )
        return super().generate(
            input_ids, speaker_embedding, prompt_input_ids=prompt_input_ids, **kwargs
        )


__all__ = ["CosyVoiceV3GenerationMixin"]
