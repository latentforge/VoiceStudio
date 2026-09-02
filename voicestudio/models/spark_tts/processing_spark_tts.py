"""Processor class for Spark-TTS."""

import re

import torch

from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.utils.import_utils import requires


logger = logging.get_logger(__name__)

TASK_TOKENS = {
    "vc": "<|task_vc|>",
    "tts": "<|task_tts|>",
    "asr": "<|task_asr|>",
    "s2s": "<|task_s2s|>",
    "t2s": "<|task_t2s|>",
    "understand": "<|task_understand|>",
    "caption": "<|task_cap|>",
    "controllable_tts": "<|task_controllable_tts|>",
    "prompt_tts": "<|task_prompt_tts|>",
    "speech_edit": "<|task_edit|>",
}

LEVELS = {"very_low": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4}

GENDERS = {"female": 0, "male": 1}

SEMANTIC_TOKEN_PATTERN = re.compile(r"^<\|bicodec_semantic_(\d+)\|>$")
GLOBAL_TOKEN_PATTERN = re.compile(r"^<\|bicodec_global_(\d+)\|>$")


class SparkTTSProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
        },
    }


@requires(backends=("torch",))
class SparkTTSProcessor(ProcessorMixin):
    r"""
    Constructs a Spark-TTS processor which wraps a [`SparkTTSFeatureExtractor`], an [`AutoTokenizer`] and a
    [`SparkTTSBiCodecModel`] into a single processor. See [`~SparkTTSProcessor.__call__`] and
    [`~SparkTTSProcessor.decode`] for more information.

    Spark-TTS drives two prompt layouts over one vocabulary. Passing `reference_audio` clones the voice of that clip
    by prefixing its BiCodec global tokens; passing `gender`, `pitch` and `speed` instead builds a voice from
    attribute labels, in which case the model emits its own global tokens.

    Args:
        feature_extractor ([`SparkTTSFeatureExtractor`]):
            The feature extractor, a required input.
        tokenizer ([`AutoTokenizer`]):
            The tokenizer, a required input.
        audio_tokenizer ([`SparkTTSBiCodecModel`]):
            The BiCodec audio tokenizer, a required input.
        chat_template (`str`, *optional*):
            Template string used to format chat-style inputs.
    """

    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "SparkTTSBiCodecModel"

    def __init__(self, feature_extractor=None, tokenizer=None, audio_tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        # `ProcessorMixin.__init__` type-checks `audio_tokenizer` against a hardcoded list of the audio tokenizer
        # classes that ship in `transformers`, so a codec defined outside it has to be attached afterwards. Every
        # other code path only needs the class to be registered in `AutoModelForAudioTokenization`.
        self.audio_tokenizer = audio_tokenizer

    def _build_voice_cloning_prompt(self, text: str, global_codes: torch.Tensor, prompt_text: str | None) -> str:
        global_tokens = "".join(f"<|bicodec_global_{code}|>" for code in global_codes.reshape(-1).tolist())
        prompt = [
            TASK_TOKENS["tts"],
            "<|start_content|>",
            text if prompt_text is None else prompt_text + text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]
        return "".join(prompt)

    def _build_voice_creation_prompt(self, text: str, gender: str, pitch: str, speed: str) -> str:
        if gender not in GENDERS:
            raise ValueError(f"`gender` must be one of {sorted(GENDERS)}, got {gender!r}.")
        if pitch not in LEVELS:
            raise ValueError(f"`pitch` must be one of {sorted(LEVELS)}, got {pitch!r}.")
        if speed not in LEVELS:
            raise ValueError(f"`speed` must be one of {sorted(LEVELS)}, got {speed!r}.")

        prompt = [
            TASK_TOKENS["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            f"<|gender_{GENDERS[gender]}|>",
            f"<|pitch_label_{LEVELS[pitch]}|>",
            f"<|speed_label_{LEVELS[speed]}|>",
            "<|end_style_label|>",
        ]
        return "".join(prompt)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        reference_audio: AudioInput | None = None,
        prompt_text: str | None = None,
        gender: str | None = None,
        pitch: str | None = None,
        speed: str | None = None,
        output_labels: bool = False,
        **kwargs: Unpack[SparkTTSProcessorKwargs],
    ) -> BatchFeature:
        """
        Build the prompt for [`SparkTTSForConditionalGeneration`], either by cloning `reference_audio` or by naming
        the speaker attributes to synthesize.

        Args:
            text (`str` or `list[str]`):
                The text to synthesize.
            reference_audio (`AudioInput`, *optional*):
                Clip whose voice is cloned. Mutually exclusive with `gender`/`pitch`/`speed`.
            prompt_text (`str`, *optional*):
                Transcript of `reference_audio`. When given, the reference clip's own semantic tokens are appended to
                the prompt so that generation continues it rather than starting from silence.
            gender (`str`, *optional*):
                One of `"female"` or `"male"`.
            pitch (`str`, *optional*):
                One of `"very_low"`, `"low"`, `"moderate"`, `"high"` or `"very_high"`.
            speed (`str`, *optional*):
                One of `"very_low"`, `"low"`, `"moderate"`, `"high"` or `"very_high"`.
            output_labels (`bool`, *optional*, defaults to `False`):
                Whether to additionally return `labels` for cross-entropy training.

        Returns:
            [`BatchFeature`]: With `input_ids` and `attention_mask`, ready to be passed to
            [`SparkTTSForConditionalGeneration`].

        Raises:
            ValueError: If neither or both prompt layouts are requested, if more than one prompt is passed alongside
                `reference_audio`, or if an attribute label is not one of the accepted values.
        """
        output_kwargs = self._merge_kwargs(
            SparkTTSProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(item, str) for item in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        attributes_given = gender is not None or pitch is not None or speed is not None
        if (reference_audio is None) == (not attributes_given):
            raise ValueError(
                "Provide either `reference_audio`, to clone a voice, or all of `gender`/`pitch`/`speed`, to build "
                "one from attribute labels, but not both."
            )

        if reference_audio is not None:
            reference_audio = make_list_of_audio(reference_audio)
            if len(reference_audio) != len(text):
                raise ValueError(
                    f"Got {len(reference_audio)} reference clips for {len(text)} prompts; pass one clip per prompt."
                )
            audio_inputs = self.feature_extractor(reference_audio, **audio_kwargs)
            audio_inputs = audio_inputs.to(self.audio_tokenizer.device)
            with torch.no_grad():
                encoded = self.audio_tokenizer.encode(**audio_inputs)

            prompts = []
            for index, prompt in enumerate(text):
                prompt = self._build_voice_cloning_prompt(prompt, encoded.global_codes[index], prompt_text)
                if prompt_text is not None:
                    semantic_tokens = "".join(
                        f"<|bicodec_semantic_{code}|>" for code in encoded.audio_codes[index].tolist()
                    )
                    prompt = prompt + "<|start_semantic_token|>" + semantic_tokens
                prompts.append(prompt)
        else:
            prompts = [self._build_voice_creation_prompt(prompt, gender, pitch, speed) for prompt in text]

        data = dict(self.tokenizer(prompts, return_tensors="pt", **text_kwargs))

        if output_labels:
            labels = data["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type="pt")

    def decode(self, sequences: torch.LongTensor, input_length: int = 0) -> torch.Tensor:
        """
        Turn a generated token sequence back into a waveform.

        Global tokens are read from the whole sequence, since a voice-cloning prompt carries them and a voice-creation
        prompt makes the model emit them. Semantic tokens are read from the continuation only, so that the reference
        clip's own semantic tokens do not leak into the output.

        Args:
            sequences (`torch.LongTensor` of shape `(1, sequence_length)`):
                Prompt and continuation as returned by [`~SparkTTSForConditionalGeneration.generate`].
            input_length (`int`, *optional*, defaults to 0):
                Length of the prompt, i.e. `inputs["input_ids"].shape[-1]`.

        Returns:
            `torch.Tensor` of shape `(sequence_length,)`: The synthesized waveform.

        Raises:
            ValueError: If more than one sequence is passed, or if the sequence carries no global or semantic tokens.
        """
        if sequences.shape[0] != 1:
            raise ValueError(
                f"Expecting a single sequence to be decoded but received {sequences.shape[0]} sequences instead."
            )

        global_codes = self._parse_codes(sequences[0], GLOBAL_TOKEN_PATTERN)
        audio_codes = self._parse_codes(sequences[0, input_length:], SEMANTIC_TOKEN_PATTERN)
        if not global_codes:
            raise ValueError("The sequence carries no `<|bicodec_global_*|>` tokens, so no voice can be rebuilt.")
        if not audio_codes:
            raise ValueError("The sequence carries no `<|bicodec_semantic_*|>` tokens, so there is nothing to decode.")

        device = self.audio_tokenizer.device
        global_codes = torch.tensor(global_codes, dtype=torch.long, device=device).reshape(1, 1, -1)
        audio_codes = torch.tensor(audio_codes, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            audio_values = self.audio_tokenizer.decode(audio_codes, global_codes).audio_values
        return audio_values.reshape(-1).cpu()

    def _parse_codes(self, sequence: torch.LongTensor, pattern: re.Pattern) -> list[int]:
        tokens = self.tokenizer.convert_ids_to_tokens(sequence.tolist())
        return [int(match.group(1)) for token in tokens if (match := pattern.match(token))]

    @property
    def model_input_names(self):
        return self.tokenizer.model_input_names


__all__ = ["SparkTTSProcessor"]
