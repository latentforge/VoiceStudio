# Copyright 2026 The Qwen team, Alibaba Group and the LatentForge team. All rights reserved.
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
"""Processor class for Qwen3-TTS."""

import json
from typing import Literal, Union

from transformers import AutoConfig
from transformers.conversion_mapping import (
    WeightRenaming,
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_tts.processing_qwen3_tts import Qwen3TTSProcessor as _Qwen3TTSProcessor
from transformers.models.qwen3_tts_tokenizer_multi_codebook.configuration_qwen3_tts_tokenizer_multi_codebook import (
    Qwen3TTSTokenizerMultiCodebookConfig,
)
from transformers.models.qwen3_tts_tokenizer_multi_codebook.modeling_qwen3_tts_tokenizer_multi_codebook import (
    Qwen3TTSTokenizerMultiCodebookModel,
)
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file


# `model_type` a checkpoint carries while it is still in the layout Qwen3-TTS-Tokenizer-12Hz publishes,
# whose configuration is of a schema [`Qwen3TTSTokenizerMultiCodebookConfig`] does not read.
_ORIGINAL_MODEL_TYPE = "qwen3_tts_tokenizer_12hz"

# Keyword arguments of [`~transformers.utils.hub.cached_file`] that select which copy of a repository is read.
_DOWNLOAD_KWARGS = ("revision", "token", "cache_dir", "local_files_only")

# The original layout keeps the decoder's two residual quantizers under `rvq_first` and `rvq_rest`, each
# codebook behind a `vq` module, and the codebook's running sum under `embedding_sum`. Loading it unrenamed
# leaves the whole decoder quantizer at its initialization, which is correctly shaped and meaningless: the
# codes a reference waveform encodes to and the waveform codes decode to are both wrong.
if get_checkpoint_conversion_mapping(Qwen3TTSTokenizerMultiCodebookConfig.model_type) is None:
    register_checkpoint_conversion_mapping(
        Qwen3TTSTokenizerMultiCodebookConfig.model_type,
        [
            WeightRenaming(
                source_patterns=r"decoder\.quantizer\.rvq_first\.",
                target_patterns="decoder.quantizer.semantic_residual_vector_quantizer.",
            ),
            WeightRenaming(
                source_patterns=r"decoder\.quantizer\.rvq_rest\.",
                target_patterns="decoder.quantizer.acoustic_residual_vector_quantizer.",
            ),
            WeightRenaming(source_patterns=r"\.vq\.layers\.", target_patterns=".layers."),
            WeightRenaming(source_patterns=r"\._codebook\.", target_patterns=".codebook."),
            WeightRenaming(source_patterns=r"\.embedding_sum", target_patterns=".embed_sum"),
        ],
    )


# Maps the task implied by a call's arguments to the `tts_model_type` value a
# checkpoint must be configured with to serve that call. Voice cloning is served by
# "base" checkpoints; there is no dedicated `tts_model_type` for it.
_IMPLIED_TASK_TO_MODEL_TYPE = {
    "base": "base",
    "voice_clone": "base",
    "custom_voice": "custom_voice",
    "voice_design": "voice_design",
}

# `non_streaming_mode` for each task. True puts the whole text in the talker's prefill; False
# puts only its first token there and feeds one further text token per generated codec frame,
# so the model starts speaking before it has seen the rest of the sentence. Qwen3-TTS uses the
# streaming form for voice cloning only.
_IMPLIED_TASK_TO_NON_STREAMING_MODE = {
    "base": False,
    "voice_clone": False,
    "custom_voice": True,
    "voice_design": True,
}


def load_audio_tokenizer(
    pretrained_model_name_or_path, subfolder: str, **kwargs
) -> Qwen3TTSTokenizerMultiCodebookModel:
    r"""
    Loads a bundled Qwen3-TTS-Tokenizer-12Hz as a [`Qwen3TTSTokenizerMultiCodebookModel`].

    A checkpoint still in the layout Qwen3-TTS-Tokenizer-12Hz publishes describes itself with a configuration
    [`Qwen3TTSTokenizerMultiCodebookConfig`] does not read, so the model is built from the defaults that layout
    implies and its keys are renamed as it loads.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            Repository id or local directory holding the checkpoint.
        subfolder (`str`):
            Directory inside it the audio tokenizer sits in.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`~PreTrainedModel.from_pretrained`].

    Returns:
        [`Qwen3TTSTokenizerMultiCodebookModel`]: The audio tokenizer.
    """
    config_file = cached_file(
        pretrained_model_name_or_path,
        CONFIG_NAME,
        subfolder=subfolder,
        **{key: value for key, value in kwargs.items() if key in _DOWNLOAD_KWARGS},
    )
    with open(config_file) as file:
        model_type = json.load(file).get("model_type")

    config = Qwen3TTSTokenizerMultiCodebookConfig() if model_type == _ORIGINAL_MODEL_TYPE else None
    return Qwen3TTSTokenizerMultiCodebookModel.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, config=config, **kwargs
    )


class Qwen3TTSProcessor(_Qwen3TTSProcessor):
    r"""
    Constructs a Qwen3TTS processor with task-dispatch methods on top of
    [`~Qwen3TTSProcessor.apply_chat_template`].

    Each Qwen3-TTS checkpoint is trained for a single task, given by `config.tts_model_type`
    (`"base"`, `"custom_voice"`, or `"voice_design"`). [`~Qwen3TTSProcessor.encode`] accepts arguments
    for any task and infers which task they imply, raising `RuntimeError` if the implied task does not
    match the loaded checkpoint's task. [`~Qwen3TTSProcessor.encode_voice_design`] and
    [`~Qwen3TTSProcessor.encode_custom_voice`] skip that inference and only accept the arguments valid
    for their task.

    The returned [`BatchFeature`] also carries the task's `non_streaming_mode`, so that splatting it
    into [`Qwen3TTSForConditionalGeneration.generate`] runs the text through the talker the way
    Qwen3-TTS runs it for that task.

    Args:
        task (`str`, *optional*):
            The task this checkpoint was configured for. Read from `config.tts_model_type` by
            [`~Qwen3TTSProcessor.from_pretrained`] when not given explicitly.
    """

    task: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        task: str | None = None,
        audio_tokenizer_subfolder: str = "speech_tokenizer",
        **kwargs,
    ):
        processor = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if getattr(processor, "audio_tokenizer", None) is None:
            processor.audio_tokenizer = load_audio_tokenizer(
                pretrained_model_name_or_path, audio_tokenizer_subfolder, **kwargs
            )
        if task is None:
            try:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
                task = getattr(config, "tts_model_type", None)
            except OSError:
                task = None
        processor.task = task
        return processor

    def _check_task(self, implied_task: str):
        required_model_type = _IMPLIED_TASK_TO_MODEL_TYPE[implied_task]
        if self.task is not None and self.task != required_model_type:
            raise RuntimeError(
                f"This processor is configured for the {self.task!r} task (tts_model_type), but the given "
                f"arguments imply the {implied_task!r} task, which requires a {required_model_type!r} checkpoint."
            )

    def encode(
        self,
        text: Union[str, list[str]],
        speaker: Union[str, list[str]] | None = None,
        instruct: Union[str, list[str]] | None = None,
        language: Union[str, list[str]] | None = None,
        return_tensors: Literal["pt", "np"] = "pt",
    ) -> BatchFeature:
        """
        Prepare inputs for the configured task, inferring the task from which of `speaker`/`instruct`
        are given.

        Args:
            text (`str` or `List[str]`):
                The text to synthesize.
            speaker (`str` or `List[str]`, *optional*):
                Speaker preset name(s). Implies the `"custom_voice"` task.
            instruct (`str` or `List[str]`, *optional*):
                Natural-language style instruction(s). Implies the `"voice_design"` task.
            language (`str` or `List[str]`, *optional*):
                Spoken language for each sample.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Tensor type to return.

        Returns:
            [`BatchFeature`]: Ready to be passed to [`Qwen3TTSForConditionalGeneration.generate`].

        Raises:
            `RuntimeError`: If the implied task does not match `self.task`.
        """
        if speaker is not None:
            implied_task = "custom_voice"
        elif instruct is not None:
            implied_task = "voice_design"
        else:
            implied_task = "base"
        self._check_task(implied_task)

        conversation = self._build_conversation(text=text, speaker=speaker, instruct=instruct, language=language)
        return self._encode_conversation(conversation, implied_task, return_tensors)

    def encode_voice_design(
        self,
        text: Union[str, list[str]],
        instruct: Union[str, list[str]],
        language: Union[str, list[str]] | None = None,
        return_tensors: Literal["pt", "np"] = "pt",
    ) -> BatchFeature:
        """
        Prepare inputs for the voice design task.

        Args:
            text (`str` or `List[str]`):
                The text to synthesize.
            instruct (`str` or `List[str]`):
                Natural-language voice/style description.
            language (`str` or `List[str]`, *optional*):
                Spoken language for each sample.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Tensor type to return.

        Returns:
            [`BatchFeature`]: Ready to be passed to [`Qwen3TTSForConditionalGeneration.generate`].
        """
        self._check_task("voice_design")
        conversation = self._build_conversation(text=text, instruct=instruct, language=language)
        return self._encode_conversation(conversation, "voice_design", return_tensors)

    def encode_custom_voice(
        self,
        text: Union[str, list[str]],
        speaker: Union[str, list[str]],
        instruct: Union[str, list[str]] | None = None,
        language: Union[str, list[str]] | None = None,
        return_tensors: Literal["pt", "np"] = "pt",
    ) -> BatchFeature:
        """
        Prepare inputs for the custom voice task.

        Args:
            text (`str` or `List[str]`):
                The text to synthesize.
            speaker (`str` or `List[str]`):
                Speaker preset name(s).
            instruct (`str` or `List[str]`, *optional*):
                Natural-language style instruction(s).
            language (`str` or `List[str]`, *optional*):
                Spoken language for each sample.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Tensor type to return.

        Returns:
            [`BatchFeature`]: Ready to be passed to [`Qwen3TTSForConditionalGeneration.generate`].
        """
        self._check_task("custom_voice")
        conversation = self._build_conversation(text=text, speaker=speaker, instruct=instruct, language=language)
        return self._encode_conversation(conversation, "custom_voice", return_tensors)

    def encode_voice_clone(self, *args, **kwargs):
        """
        Prepare inputs for the voice cloning task from a reference audio prompt.

        Raises:
            `NotImplementedError`: Always. Voice cloning from reference audio is not supported by
                [`Qwen3TTSProcessor.apply_chat_template`].
        """
        raise NotImplementedError(
            "Voice cloning from reference audio is not supported by the transformers-tts Qwen3TTSProcessor."
        )

    def _encode_conversation(self, conversation, implied_task, return_tensors):
        inputs = self.apply_chat_template(conversation, return_tensors=return_tensors)
        inputs["non_streaming_mode"] = _IMPLIED_TASK_TO_NON_STREAMING_MODE[implied_task]
        return inputs

    def _build_conversation(self, text, speaker=None, instruct=None, language=None):
        texts = text if isinstance(text, list) else [text]
        speakers = speaker if isinstance(speaker, list) else [speaker] * len(texts)
        instructs = instruct if isinstance(instruct, list) else [instruct] * len(texts)
        languages = language if isinstance(language, list) else [language] * len(texts)

        conversations = []
        for one_text, one_speaker, one_instruct, one_language in zip(texts, speakers, instructs, languages):
            user_message = {"role": "user", "content": one_text}
            if one_language is not None:
                user_message["language"] = one_language
            if one_speaker is not None:
                user_message["speaker"] = one_speaker

            messages = [user_message]
            if one_instruct is not None:
                messages.insert(0, {"role": "system", "content": one_instruct})
            conversations.append(messages)

        return conversations if len(conversations) > 1 else conversations[0]


__all__ = ["Qwen3TTSProcessor"]
