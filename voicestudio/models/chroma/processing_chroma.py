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
"""Processor class for Chroma."""

from typing import Optional, Union, Unpack

import numpy as np
import torch

from transformers.audio_utils import load_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.processing_utils import AudioKwargs, ProcessingKwargs


def _load_conversation_audio(source: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(source, np.ndarray):
        if source.ndim > 1:
            raise ValueError("Support only mono audio")
        return source
    if source.startswith("file://"):
        source = source[len("file://") :]
    return load_audio(source, sampling_rate=16000)


def process_audio_info(
    conversations: Union[list[dict], list[list[dict]]], use_audio_in_video: bool
) -> Optional[list[np.ndarray]]:
    """
    Collects every audio waveform referenced by a conversation.

    Args:
        conversations (`list[dict]` or `list[list[dict]]`):
            One conversation or a batch of them, in the chat template message format.
        use_audio_in_video (`bool`):
            Whether the audio track of `video` content is collected as well.

    Returns:
        `list[np.ndarray]` or `None`: The waveforms at the reasoner sample rate, or `None` when the conversations
        reference no audio.

    Raises:
        ValueError: If an `audio` or `video` content block carries no source.
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" not in ele:
                        raise ValueError(f"Unknown audio {ele}")
                    audios.append(_load_conversation_audio(ele["audio"]))
                if use_audio_in_video and ele["type"] == "video":
                    if "video" not in ele:
                        raise ValueError(f"Unknown video {ele}")
                    audios.append(_load_conversation_audio(ele["video"]))
    if len(audios) == 0:
        audios = None
    return audios


class ChromaAudioKwargs(AudioKwargs, total=False):
    target_sample_rate: Optional[int]


class ChromaProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: ChromaAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "target_sample_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class ChromaProcessor(Qwen2_5OmniProcessor):
    r"""
    Constructs a Chroma processor, which combines the reasoner inputs of a [`Qwen2_5OmniProcessor`] with the
    reference transcript and reference waveform the backbone clones the voice from. Reasoner keys are returned
    under a `thinker_` prefix, the reference transcript under the plain tokenizer keys, and the reference waveform
    as `input_values` and `input_values_cutoffs`.

    [`ChromaProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`], [`WhisperFeatureExtractor`]
    and [`Qwen2TokenizerFast`]. See [`~ChromaProcessor.__call__`] and [`~ChromaProcessor.decode`] for more
    information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor.
        video_processor ([`Qwen2VLVideoProcessor`], *optional*):
            The video processor.
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`str`, *optional*):
            The Jinja template used to format the conversation. If not provided, the default chat template is used.
    """

    valid_processor_kwargs = ChromaProcessorKwargs

    def __call__(
        self,
        conversations: list[list[dict]],
        prompt_audio: list[str],
        prompt_text: list[str],
        **kwargs: Unpack[ChromaProcessorKwargs],
    ) -> BatchFeature:
        """
        Args:
            conversations (`list[list[dict]]`):
                One conversation per batch entry, in the chat template message format.
            prompt_audio (`list[str]`):
                One reference waveform per batch entry, as an `http(s)://` URL, a local file path or a
                base64-encoded string.
            prompt_text (`list[str]`):
                One reference transcript per batch entry, matching `prompt_audio`.
            kwargs (`dict[str, Any]`, *optional*):
                Forwarded to the tokenizer, feature extractor and chat template. `target_sample_rate` selects the
                rate the reference waveforms are resampled to.

        Returns:
            [`~feature_extraction_utils.BatchFeature`]: The reasoner inputs under a `thinker_` prefix, the
            reference transcript ids as `input_ids` and `attention_mask`, the padded reference waveforms as
            `input_values` of shape `(batch_size, 1, audio_length)` and their unpadded lengths as
            `input_values_cutoffs`.

        Raises:
            ValueError: If `prompt_audio` or `prompt_text` is missing, or does not have one entry per conversation.
        """
        if prompt_audio is None:
            raise ValueError("prompt_audio can not be empty")
        if prompt_text is None:
            raise ValueError("prompt_text can not be empty")

        batch_size = len(conversations)
        if len(prompt_audio) != batch_size:
            raise ValueError(f"prompt_audio has {len(prompt_audio)} entries, expected {batch_size}")
        if len(prompt_text) != batch_size:
            raise ValueError(f"prompt_text has {len(prompt_text)} entries, expected {batch_size}")

        text, audios = self.apply_chat_template(conversations, **kwargs)
        thinker_inputs = super().__call__(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        thinker_inputs = {f"thinker_{k}": v for k, v in thinker_inputs.items()}

        inputs = super().__call__(text=prompt_text, return_tensors="pt", padding=True)
        prompt_audio_wavs = [self.load_audio(audio, kwargs.get("target_sample_rate", 24000)) for audio in prompt_audio]
        prompt_audio_cutoffs = torch.tensor([len(audio) for audio in prompt_audio_wavs], dtype=torch.long)
        prompt_audio_tensor = torch.nn.utils.rnn.pad_sequence(prompt_audio_wavs, batch_first=True).unsqueeze(1)

        return BatchFeature(
            data={
                **thinker_inputs,
                **inputs,
                "input_values": prompt_audio_tensor,
                "input_values_cutoffs": prompt_audio_cutoffs,
            },
            tensor_type=kwargs.get("return_tensors"),
        )

    def load_audio(self, audio_path: str, target_sample_rate: int = 24000) -> torch.Tensor:
        """
        Loads a mono audio waveform and resamples it to the target sample rate.

        Args:
            audio_path (`str`):
                An `http(s)://` URL, a local file path, or a base64-encoded string.
            target_sample_rate (`int`, *optional*, defaults to 24000):
                Sample rate the waveform is resampled to, matching the codec input rate.

        Returns:
            `torch.Tensor`: The mono waveform of shape `(num_samples,)`.
        """
        return torch.from_numpy(load_audio(audio_path, sampling_rate=target_sample_rate, backend="torchaudio"))

    def apply_chat_template(
        self,
        conversations: Union[list[dict], list[list[dict]]],
        chat_template: Optional[str] = None,
        **kwargs,
    ) -> tuple[Union[str, list[str]], Optional[list[np.ndarray]]]:
        """
        Formats conversations with the chat template and collects the audio they reference.

        Args:
            conversations (`list[dict]` or `list[list[dict]]`):
                One conversation or a batch of them, in the chat template message format.
            chat_template (`str`, *optional*):
                Jinja template overriding the tokenizer's own.
            kwargs (`dict[str, Any]`, *optional*):
                Forwarded to [`~PreTrainedTokenizerBase.apply_chat_template`].

        Returns:
            `tuple`: The formatted prompt, and the referenced waveforms or `None` when there are none.
        """
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        audios = process_audio_info(conversations, use_audio_in_video=False)
        return self.tokenizer.apply_chat_template(conversations, chat_template, **kwargs), audios


__all__ = ["ChromaProcessor"]
