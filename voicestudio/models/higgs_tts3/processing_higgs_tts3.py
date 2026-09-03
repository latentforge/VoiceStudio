# Copyright 2026 Boson AI and the LatentForge team. All rights reserved.
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
"""Processor class for Higgs TTS 3."""

import torch

from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.utils.import_utils import requires


logger = logging.get_logger(__name__)


class HiggsTTS3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "padding": False,
            "sampling_rate": 24000,
            "return_tensors": "pt",
        },
    }


# Higgs TTS 3 prompts are framed by dedicated specials rather than by the text tokenizer's chat
# template: `<|tts|>` opens the prompt, an optional `<|ref_text|>`/`<|ref_audio|>` pair carries the
# voice-cloning reference, `<|text|>` introduces the target text, and `<|audio|>` opens the audio
# stream the model continues with codebook frames.
_TTS_TOKEN = "<|tts|>"
_REF_TEXT_TOKEN = "<|ref_text|>"
_REF_AUDIO_TOKEN = "<|ref_audio|>"
_TEXT_TOKEN = "<|text|>"
_AUDIO_TOKEN = "<|audio|>"


@requires(backends=("torch",))
class HiggsTTS3Processor(ProcessorMixin):
    r"""
    Constructs a Higgs TTS 3 processor which wraps a [`DacFeatureExtractor`], an [`AutoTokenizer`], and a
    [`HiggsAudioV2TokenizerModel`] into a single processor. See [`~HiggsTTS3Processor.__call__`] and
    [`~HiggsTTS3Processor.decode`] for more information.

    Args:
        feature_extractor (`DacFeatureExtractor`):
            An instance of [`DacFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
        audio_tokenizer (`HiggsAudioV2TokenizerModel`):
            An instance of [`HiggsAudioV2TokenizerModel`]. The audio tokenizer is a required input.
        chat_template (`str`, *optional*):
            A template string for chat formatting when combining text and audio interactions.
        audio_token_id (`int`, *optional*, defaults to -100):
            Placeholder id inserted into `input_ids` at audio-frame positions.
        audio_stream_bos_id (`int`, *optional*, defaults to 1024):
            Id, within a codebook's vocabulary, of the beginning-of-codebook special used by the delay pattern.
        audio_stream_eos_id (`int`, *optional*, defaults to 1025):
            Id, within a codebook's vocabulary, of the end-of-codebook special used by the delay pattern.
    """

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "HiggsAudioV2TokenizerModel"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        audio_tokenizer=None,
        chat_template=None,
        audio_token_id=-100,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
    ):
        self.audio_token_id = audio_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id

        if feature_extractor is not None and audio_tokenizer is not None:
            super().__init__(
                feature_extractor,
                tokenizer,
                audio_tokenizer=audio_tokenizer,
                chat_template=chat_template,
            )
        else:
            # `ProcessorMixin.__init__` requires every declared attribute (`feature_extractor`,
            # `audio_tokenizer`) to be a real instance of the matching class, so it cannot be used
            # to build a tokenizer-only processor for checkpoints that ship neither. Wire up the
            # tokenizer-only case by hand instead.
            self.feature_extractor = feature_extractor
            self.audio_tokenizer = audio_tokenizer
            self.tokenizer = tokenizer
            self.chat_template = chat_template

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Some `bosonai/higgs-tts-3-*` checkpoints ship only the text tokenizer (no
        `preprocessor_config.json` for the `DacFeatureExtractor`, and no bundled
        `HiggsAudioV2TokenizerModel`). For those, the feature extractor and audio tokenizer are
        loaded from the codec repository named by the model config's `audio_tokenizer_id`, and, if
        that repository is unreachable too, a tokenizer-only processor is returned rather than
        raising, since `feature_extractor`/`audio_tokenizer` are optional.
        """
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            from transformers import AutoConfig, AutoTokenizer, DacFeatureExtractor
            from transformers.models.higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerModel

            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            audio_tokenizer_id = None
            try:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
                audio_tokenizer_id = getattr(config, "audio_tokenizer_id", None)
            except OSError:
                pass

            if audio_tokenizer_id is not None:
                try:
                    return cls(
                        feature_extractor=DacFeatureExtractor.from_pretrained(audio_tokenizer_id),
                        tokenizer=tokenizer,
                        audio_tokenizer=HiggsAudioV2TokenizerModel.from_pretrained(audio_tokenizer_id),
                    )
                except OSError:
                    pass

            logger.warning_once(
                f"'{pretrained_model_name_or_path}' does not ship a feature extractor or audio tokenizer config, "
                "and none could be loaded from its config's `audio_tokenizer_id`. Loading a tokenizer-only "
                "`HiggsTTS3Processor`; reference-audio conditioning and `decode` will be unavailable until "
                "`feature_extractor`/`audio_tokenizer` are set."
            )
            return cls(tokenizer=tokenizer)

    def _prompt_token_id(self, token: str) -> int:
        """
        Resolve one of Higgs TTS 3's prompt-framing specials to its id.

        Args:
            token (`str`):
                The special token to look up.

        Returns:
            `int`: The token's id in this processor's tokenizer.

        Raises:
            `ValueError`: If the tokenizer does not define `token`.
        """
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"This processor's tokenizer does not define the Higgs TTS 3 special {token!r}, so a "
                "prompt cannot be built for it."
            )
        return token_id

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        reference_audio: AudioInput | None = None,
        reference_text: str | None = None,
        output_labels: bool = False,
        **kwargs: Unpack[HiggsTTS3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare one or more text prompts, optionally paired with a reference audio clip for zero-shot voice
        cloning, for [`HiggsTTS3ForConditionalGeneration`].

        Args:
            text (`str`, `List[str]`, `List[int]`, or `List[List[int]]`):
                The text to synthesize.
            reference_audio (`AudioInput`, *optional*):
                Reference audio clip(s) to condition generation on, one per prompt in `text`.
            reference_text (`str`, *optional*):
                Transcript of `reference_audio`, prepended to it under `<|ref_text|>`.
            output_labels (`bool`, *optional*, defaults to `False`):
                Whether to additionally return `labels` and `audio_labels` for cross-entropy training.

        Returns:
            [`BatchFeature`]: Ready to be passed to [`HiggsTTS3ForConditionalGeneration`], with `input_ids`,
            `attention_mask`, and, when `reference_audio` is provided, `audio_input_ids`/`audio_input_ids_mask`.
        """
        output_kwargs = self._merge_kwargs(HiggsTTS3ProcessorKwargs, **kwargs)
        audio_kwargs = output_kwargs["audio_kwargs"]

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        audio_input_ids = None
        audio_input_ids_mask = None
        if reference_audio is not None:
            if self.feature_extractor is None or self.audio_tokenizer is None:
                raise ValueError(
                    "`reference_audio` was provided but this `HiggsTTS3Processor` has no `feature_extractor`/"
                    "`audio_tokenizer` (the checkpoint it was loaded from does not ship one). Text-only synthesis "
                    "is still supported."
                )
            if len(text) != 1:
                raise ValueError(
                    "HiggsTTS3Processor only supports a single prompt at a time when `reference_audio` is given."
                )
            reference_audio = make_list_of_audio(reference_audio)
            if len(reference_audio) != 1:
                raise ValueError("Provide exactly one reference audio clip.")

            audio_inputs = self.feature_extractor(reference_audio[0], **audio_kwargs)
            audio_inputs.pop("padding_mask", None)
            audio_inputs.to(self.audio_tokenizer.device)
            codes = self.audio_tokenizer.encode(**audio_inputs).audio_codes
            codes = self.build_delay_pattern(codes)[0].transpose(0, 1)

            audio_input_ids = codes.unsqueeze(0)
            audio_input_ids_mask = torch.ones((1, codes.shape[0]), dtype=torch.bool)

        sequences = []
        for one_text in text:
            prompt_ids = [self._prompt_token_id(_TTS_TOKEN)]
            if audio_input_ids is not None:
                if reference_text is not None:
                    prompt_ids.append(self._prompt_token_id(_REF_TEXT_TOKEN))
                    prompt_ids.extend(self.tokenizer.encode(reference_text, add_special_tokens=False))
                prompt_ids.append(self._prompt_token_id(_REF_AUDIO_TOKEN))
                prompt_ids.extend([self.audio_token_id] * audio_input_ids.shape[1])
            prompt_ids.append(self._prompt_token_id(_TEXT_TOKEN))
            prompt_ids.extend(self.tokenizer.encode(one_text, add_special_tokens=False))
            prompt_ids.append(self._prompt_token_id(_AUDIO_TOKEN))
            sequences.append(prompt_ids)

        max_length = max(len(prompt_ids) for prompt_ids in sequences)
        input_ids = torch.full((len(sequences), max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
        for row, prompt_ids in enumerate(sequences):
            input_ids[row, : len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long)
            attention_mask[row, : len(prompt_ids)] = 1

        data = {"input_ids": input_ids, "attention_mask": attention_mask}
        if audio_input_ids is not None:
            data["audio_input_ids"] = audio_input_ids
            data["audio_input_ids_mask"] = audio_input_ids_mask

        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

            if audio_input_ids is not None:
                audio_labels = audio_input_ids.clone()
                audio_labels[audio_labels == self.audio_stream_bos_id] = -100
                audio_labels[audio_labels == self.audio_stream_eos_id] = -100
                data["audio_labels"] = audio_labels

        return BatchFeature(data=data, tensor_type="pt")

    def decode(self, audio_input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Decode a single generated audio codebook sequence into a waveform.

        Args:
            audio_input_ids (`torch.LongTensor` of shape `(1, sequence_length, num_codebooks)`):
                Delayed audio codes as produced by [`HiggsTTS3ForConditionalGeneration.generate`].

        Returns:
            `torch.Tensor`: A 1D waveform tensor.
        """
        if self.audio_tokenizer is None:
            raise ValueError(
                "This `HiggsTTS3Processor` has no `audio_tokenizer` (the checkpoint it was loaded from does not "
                "ship one), so generated audio codes cannot be decoded to a waveform."
            )
        if audio_input_ids.shape[0] != 1:
            raise ValueError(
                f"Expecting a single output to be decoded but received {audio_input_ids.shape[0]} samples instead."
            )
        codes = self.revert_delay_pattern(audio_input_ids[0]).clip(0, self.audio_stream_bos_id - 1)
        codes = codes.to(self.audio_tokenizer.device)
        with torch.no_grad():
            return self.audio_tokenizer.decode(codes.transpose(0, 1).unsqueeze(0)).audio_values.cpu().squeeze()

    def build_delay_pattern(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        bsz, num_codebooks, seq_len = input_ids.shape
        new_seq_len = seq_len + num_codebooks - 1

        output = torch.ones((bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device)
        bos_mask = torch.tril(output, -1) > 0
        eos_mask = torch.triu(output, seq_len) > 0
        data_mask = ~(bos_mask | eos_mask)

        output[bos_mask] = self.audio_stream_bos_id
        output[data_mask] = input_ids.reshape(-1)
        output[eos_mask] = self.audio_stream_eos_id
        return output

    def revert_delay_pattern(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        seq_len, num_codebooks = input_ids.shape
        slices = []
        for i in range(num_codebooks):
            end_idx = seq_len - num_codebooks + 1 + i
            slices.append(input_ids[i:end_idx, i : i + 1])
        return torch.cat(slices, dim=1)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return tokenizer_input_names + ["audio_input_ids", "audio_input_ids_mask"]


__all__ = ["HiggsTTS3Processor"]
