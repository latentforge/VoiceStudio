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
            "padding": True,
            "padding_side": "left",
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

    attributes = ["feature_extractor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "SparkTTSBiCodecModel"

    def __init__(self, feature_extractor=None, tokenizer=None, audio_tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        # `ProcessorMixin.__init__` type-checks `audio_tokenizer` against a hardcoded list of the audio tokenizer
        # classes that ship in `transformers`, so a codec defined outside it has to be attached afterwards. Every
        # other code path only needs the class to be registered in `AutoModelForAudioTokenization`.
        self.audio_tokenizer = audio_tokenizer

    @staticmethod
    def _global_tokens(codes: torch.Tensor) -> str:
        return "".join(f"<|bicodec_global_{code}|>" for code in codes.reshape(-1).tolist())

    @staticmethod
    def _semantic_tokens(codes: torch.Tensor) -> str:
        return "".join(f"<|bicodec_semantic_{code}|>" for code in codes.reshape(-1).tolist())

    def _build_voice_cloning_prompt(self, text: str, global_codes: torch.Tensor, prompt_text: str | None) -> str:
        prompt = [
            TASK_TOKENS["tts"],
            "<|start_content|>",
            text if prompt_text is None else prompt_text + text,
            "<|end_content|>",
            "<|start_global_token|>",
            self._global_tokens(global_codes),
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

    def _build_voice_cloning_target(self, semantic_codes: torch.Tensor) -> str:
        return "".join(
            [
                "<|start_semantic_token|>",
                self._semantic_tokens(semantic_codes),
                self.tokenizer.eos_token,
            ]
        )

    def _build_voice_creation_target(
        self,
        global_codes: torch.Tensor,
        semantic_codes: torch.Tensor,
        pitch_value: int,
        speed_value: int,
    ) -> str:
        return "".join(
            [
                "<|start_acoustic_token|>",
                f"<|pitch_value_{min(max(int(pitch_value), 0), 1000)}|>",
                f"<|speed_value_{min(max(int(speed_value), 0), 10)}|>",
                "<|end_acoustic_token|>",
                "<|start_global_token|>",
                self._global_tokens(global_codes),
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                self._semantic_tokens(semantic_codes),
                self.tokenizer.eos_token,
            ]
        )

    def _pad(self, prompt_ids: list[list[int]], target_ids: list[list[int]]) -> dict[str, torch.Tensor]:
        """
        Concatenate each prompt with its supervised continuation and pad the batch, on the side the tokenizer is
        configured for, into `input_ids`, `attention_mask` and `labels`. `labels` is `-100` over the prompt and over
        the padding, so the cross entropy only sees the continuation.

        Args:
            prompt_ids (`list[list[int]]`):
                Token ids of the prompt of each example.
            target_ids (`list[list[int]]`):
                Token ids of the supervised continuation of each example.

        Returns:
            `dict[str, torch.Tensor]`: The padded batch.
        """
        rows = [prompt + target for prompt, target in zip(prompt_ids, target_ids)]
        width = max(len(row) for row in rows)
        pad_id = self.tokenizer.pad_token_id
        left = self.tokenizer.padding_side == "left"

        input_ids, attention_mask, labels = [], [], []
        for index, row in enumerate(rows):
            padding = width - len(row)
            supervised = [-100] * len(prompt_ids[index]) + target_ids[index]
            if left:
                input_ids.append([pad_id] * padding + row)
                attention_mask.append([0] * padding + [1] * len(row))
                labels.append([-100] * padding + supervised)
            else:
                input_ids.append(row + [pad_id] * padding)
                attention_mask.append([1] * len(row) + [0] * padding)
                labels.append(supervised + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        reference_audio: AudioInput | None = None,
        prompt_text: str | None = None,
        gender: str | None = None,
        pitch: str | None = None,
        speed: str | None = None,
        output_labels: bool = False,
        pitch_value: int | None = None,
        speed_value: int | None = None,
        **kwargs: Unpack[SparkTTSProcessorKwargs],
    ) -> BatchFeature:
        """
        Build the prompt for [`SparkTTSForConditionalGeneration`], either by cloning `reference_audio` or by naming
        the speaker attributes to synthesize.

        Args:
            text (`str` or `list[str]`):
                The text to synthesize.
            reference_audio (`AudioInput`, *optional*):
                Clip whose voice is cloned. Mutually exclusive with `gender`/`pitch`/`speed`, unless `output_labels`
                is set, in which case the attribute layout needs it as the clip the target codes come from.
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
                Whether to also encode `reference_audio` as the supervised continuation and return `labels` for
                cross-entropy training, masked with `-100` over the prompt and the padding.
            pitch_value (`int`, *optional*):
                Fine-grained pitch of `reference_audio`, in `[0, 1000]`. Part of the attribute layout's supervised
                continuation, so it is required by `output_labels` together with `gender`/`pitch`/`speed`.
            speed_value (`int`, *optional*):
                Fine-grained speaking rate of `reference_audio`, in `[0, 10]`, required alongside `pitch_value`.

        Returns:
            [`BatchFeature`]: With `input_ids` and `attention_mask`, ready to be passed to
            [`SparkTTSForConditionalGeneration`], plus `labels` when `output_labels` is set.

        Raises:
            ValueError: If neither or both prompt layouts are requested, if the number of reference clips does not
                match the number of prompts, if an attribute label is not one of the accepted values, or if
                `output_labels` is set without the inputs the supervised continuation needs.
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
        if not attributes_given and reference_audio is None:
            raise ValueError(
                "Provide either `reference_audio`, to clone a voice, or all of `gender`/`pitch`/`speed`, to build "
                "one from attribute labels."
            )
        if attributes_given and reference_audio is not None and not output_labels:
            raise ValueError(
                "`reference_audio` and `gender`/`pitch`/`speed` select different prompt layouts, so they can only "
                "be combined when `output_labels` is set, where the attribute layout supervises the clip's codes."
            )
        if output_labels and reference_audio is None:
            raise ValueError("`output_labels` needs `reference_audio`, the clip whose codes are the target.")
        if output_labels and attributes_given and (pitch_value is None or speed_value is None):
            raise ValueError(
                "The attribute layout supervises fine-grained pitch and speed, so `output_labels` needs "
                "`pitch_value` and `speed_value` alongside `gender`/`pitch`/`speed`."
            )

        encoded = None
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

        prompts, targets = [], []
        for index, prompt in enumerate(text):
            if attributes_given:
                prompts.append(self._build_voice_creation_prompt(prompt, gender, pitch, speed))
                if output_labels:
                    targets.append(
                        self._build_voice_creation_target(
                            encoded.global_codes[index], encoded.audio_codes[index], pitch_value, speed_value
                        )
                    )
            else:
                prompt = self._build_voice_cloning_prompt(prompt, encoded.global_codes[index], prompt_text)
                if prompt_text is not None:
                    prompt = prompt + "<|start_semantic_token|>" + self._semantic_tokens(encoded.audio_codes[index])
                prompts.append(prompt)
                if output_labels:
                    targets.append(self._build_voice_cloning_target(encoded.audio_codes[index]))

        if not output_labels:
            data = dict(self.tokenizer(prompts, return_tensors="pt", **text_kwargs))
            return BatchFeature(data=data, tensor_type="pt")

        prompt_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(targets, add_special_tokens=False)["input_ids"]
        return BatchFeature(data=self._pad(prompt_ids, target_ids), tensor_type="pt")

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
        """
        Read the codes carried by every token of a sequence whose name matches `pattern`.

        Args:
            sequence (`torch.LongTensor` of shape `(sequence_length,)`):
                Token ids to read.
            pattern (`re.Pattern`):
                Pattern whose first group is the code, matched against each token's string form.

        Returns:
            `list[int]`: The codes, in the order they appear.
        """
        tokens = self.tokenizer.convert_ids_to_tokens(sequence.tolist())
        return [int(match.group(1)) for token in tokens if (match := pattern.match(token))]

    @property
    def model_input_names(self):
        return self.tokenizer.model_input_names


__all__ = ["SparkTTSProcessor"]
