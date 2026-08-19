"""Processor class for Qwen3-TTS."""

from typing import Literal, Union

from transformers import AutoConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_tts.processing_qwen3_tts import Qwen3TTSProcessor as _Qwen3TTSProcessor
from transformers.models.qwen3_tts_tokenizer_multi_codebook.modeling_qwen3_tts_tokenizer_multi_codebook import (
    Qwen3TTSTokenizerMultiCodebookModel,
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
            processor.audio_tokenizer = Qwen3TTSTokenizerMultiCodebookModel.from_pretrained(
                pretrained_model_name_or_path, subfolder=audio_tokenizer_subfolder, **kwargs
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
        return self.apply_chat_template(conversation, return_tensors=return_tensors)

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
        return self.apply_chat_template(conversation, return_tensors=return_tensors)

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
        return self.apply_chat_template(conversation, return_tensors=return_tensors)

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
