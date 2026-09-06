# Copyright 2026 RESONIA, INC., Sesame, The HuggingFace Inc. team and the LatentForge team.
# All rights reserved.
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
"""Processor class for Breeze TTS 2."""

import torch
import torch.nn.functional as F

from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import TextInput
from transformers.utils import logging
from transformers.utils.import_utils import requires


logger = logging.get_logger(__name__)


# A Breeze TTS 2 prompt is a flat token sequence built from text and audio segments: `<|AUDIO|>` reserves one
# position per audio frame, `<|audio_eos|>` closes an audio span, an instruction is wrapped in
# `<ins_bos>`/`<ins_eos>`, and every text span opens with a `[S<i>]` speaker token.
AUDIO_TOKEN = "<|AUDIO|>"
AUDIO_EOS_TOKEN = "<|audio_eos|>"
INSTRUCTION_BOS_TOKEN = "<ins_bos>"
INSTRUCTION_EOS_TOKEN = "<ins_eos>"

# `BreezeBlue/breeze-tts-2` keeps its audio tokenizer in a subfolder rather than in a repository of its own.
AUDIO_TOKENIZER_SUBFOLDER = "audio_tokenizer"


class BreezeTTSProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"return_tensors": "pt"},
        "audio_kwargs": {"sampling_rate": 24000},
    }


@requires(backends=("torch",))
class BreezeTTSProcessor(ProcessorMixin):
    r"""
    Constructs a Breeze TTS 2 processor which wraps an [`AutoTokenizer`] and a
    [`Qwen3TTSTokenizerMultiCodebookModel`] into a single processor. See [`~BreezeTTSProcessor.__call__`] and
    [`~BreezeTTSProcessor.decode`] for more information.

    Args:
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
        audio_tokenizer (`Qwen3TTSTokenizerMultiCodebookModel`):
            An instance of [`Qwen3TTSTokenizerMultiCodebookModel`], which turns reference waveforms into codebook
            frames and generated codebook frames back into waveforms.
        chat_template (`str`, *optional*):
            A template string for chat formatting when combining text and audio interactions.
        audio_token_id (`int`, *optional*, defaults to 262144):
            Id of `<|AUDIO|>`, the token reserving one position per audio frame.
        audio_eos_token_id (`int`, *optional*, defaults to 262145):
            Id of `<|audio_eos|>`, the token closing an audio span.
        num_codebooks (`int`, *optional*, defaults to 16):
            Number of codebooks the audio tokenizer produces per frame.
        default_speaker (`str`, *optional*, defaults to `"S0"`):
            Speaker token every text span opens with when none is given.
    """

    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "Qwen3TTSTokenizerMultiCodebookModel"

    def __init__(
        self,
        tokenizer=None,
        audio_tokenizer=None,
        chat_template=None,
        audio_token_id=262144,
        audio_eos_token_id=262145,
        num_codebooks=16,
        default_speaker="S0",
    ):
        self.audio_token_id = audio_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.num_codebooks = num_codebooks
        self.default_speaker = default_speaker
        super().__init__(tokenizer, audio_tokenizer=audio_tokenizer, chat_template=chat_template)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Loads the processor of a Breeze TTS 2 checkpoint, together with the audio tokenizer bundled in its
        `audio_tokenizer` subfolder, which is a Qwen3-TTS-Tokenizer-12Hz in the layout that model publishes.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Repository id or local path of the checkpoint.
            kwargs (`dict[str, Any]`, *optional*):
                Forwarded to the tokenizer and audio tokenizer loaders.

        Returns:
            [`BreezeTTSProcessor`]: the loaded processor. Its `audio_tokenizer` is `None` when the checkpoint
            bundles none, in which case reference audio and [`~BreezeTTSProcessor.decode`] are unavailable.
        """
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if getattr(processor, "audio_tokenizer", None) is None:
            from ..qwen3_tts.processing_qwen3_tts import load_audio_tokenizer

            try:
                processor.audio_tokenizer = load_audio_tokenizer(
                    pretrained_model_name_or_path, AUDIO_TOKENIZER_SUBFOLDER
                )
            except OSError:
                logger.warning_once(
                    f"'{pretrained_model_name_or_path}' bundles no '{AUDIO_TOKENIZER_SUBFOLDER}' audio "
                    "tokenizer. Reference-audio conditioning and `decode` will be unavailable until "
                    "`audio_tokenizer` is set."
                )
        return processor

    def _speaker_prefix(self, speaker: str | None) -> str:
        """
        Renders a speaker id as the token every text span opens with.

        Args:
            speaker (`str`, *optional*):
                Speaker id, with or without its brackets. An empty string opens the span with no speaker token.

        Returns:
            `str`: the speaker token, or the empty string.
        """
        if speaker is None:
            speaker = self.default_speaker
        if speaker == "":
            return ""
        return speaker if speaker.startswith("[") and speaker.endswith("]") else f"[{speaker}]"

    def _build_segments(
        self,
        text: str,
        instruction: str | None,
        reference_text: str | None,
        has_reference_audio: bool,
        speaker: str | None,
        with_instruction: bool = True,
    ) -> list[dict]:
        """
        Builds the segment list of one prompt.

        A prompt without reference audio is a single text segment. A prompt with reference audio is the reference
        transcript, the reference audio span, then the target text. The instruction, when present, is wrapped in
        `<ins_bos>`/`<ins_eos>` right before the target text.

        Args:
            text (`str`):
                Text to synthesize.
            instruction (`str`, *optional*):
                Natural-language voice description or direction.
            reference_text (`str`, *optional*):
                Exact transcript of the reference audio.
            has_reference_audio (`bool`):
                Whether the prompt carries a reference audio span.
            speaker (`str`, *optional*):
                Speaker id every text span opens with.
            with_instruction (`bool`, *optional*, defaults to `True`):
                Whether `instruction` is part of the prompt. The guidance branches reuse this builder with the
                instruction, the reference audio, or both dropped.

        Returns:
            `list[dict]`: the segments, each either `{"type": "text", "text": ...}` or `{"type": "audio"}`.
        """
        prefix = self._speaker_prefix(speaker)
        target = f"{prefix}{text}"
        if with_instruction and instruction:
            target = f"{prefix}{INSTRUCTION_BOS_TOKEN}{instruction}{INSTRUCTION_EOS_TOKEN}{text}"

        if not has_reference_audio:
            return [{"type": "text", "text": target}]
        return [
            {"type": "text", "text": f"{prefix}{reference_text}"},
            {"type": "audio"},
            {"type": "text", "text": target},
        ]

    def _prepare_one(self, segments: list[dict], audio_codes: list[torch.Tensor]) -> dict:
        """
        Renders one prompt's segments to token ids and the masks the text encoder needs.

        Args:
            segments (`list[dict]`):
                Segments of the prompt, as built by [`~BreezeTTSProcessor._build_segments`].
            audio_codes (`list[torch.Tensor]`):
                Codebook frames of each audio segment, of shape `(num_frames, num_codebooks)`.

        Returns:
            `dict`: the prompt's `input_ids`, `attention_mask`, `text_ids_mask`, `text_ids_len` and
            `audio_tokens`.
        """
        rendered = []
        audio_iter = iter(audio_codes)
        used_audio_codes = []
        for segment in segments:
            if segment["type"] == "text":
                encoded = self.tokenizer(segment["text"], add_special_tokens=True)
                rendered.append(
                    {
                        "type": "text",
                        "value": self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=False),
                    }
                )
                continue
            if segment["type"] != "audio":
                raise ValueError(f"Unknown segment type: {segment['type']}")

            codes = next(audio_iter)
            used_audio_codes.append(codes)
            placeholders = AUDIO_TOKEN * codes.shape[0]
            if segment.get("append_eos", True):
                placeholders += AUDIO_EOS_TOKEN
            rendered.append({"type": "audio", "value": placeholders})

        encoded = self.tokenizer(
            "".join(segment["value"] for segment in rendered), add_special_tokens=False, return_tensors="pt"
        )

        text_ids_mask = []
        text_ids_len = []
        for segment in rendered:
            segment_len = len(self.tokenizer(segment["value"], add_special_tokens=False)["input_ids"])
            text_ids_mask.extend([segment["type"] == "text"] * segment_len)
            if segment["type"] == "text":
                text_ids_len.append(segment_len)

        if used_audio_codes:
            audio_tokens = torch.cat(used_audio_codes, dim=0).unsqueeze(0)
        else:
            audio_tokens = torch.zeros((1, 0, self.num_codebooks), dtype=torch.long)

        encoded["text_ids_mask"] = torch.tensor([text_ids_mask], dtype=torch.bool)
        encoded["text_ids_len"] = torch.tensor(text_ids_len, dtype=torch.long)
        encoded["audio_tokens"] = audio_tokens
        return encoded

    def _collate(self, prompts: list[dict]) -> dict:
        """
        Left-pads a batch of rendered prompts to a common length.

        Args:
            prompts (`list[dict]`):
                Rendered prompts, as returned by [`~BreezeTTSProcessor._prepare_one`].

        Returns:
            `dict`: the batched `input_ids`, `attention_mask`, `text_ids_mask`, `text_ids_len` and
            `input_values`. `text_ids_len` and `input_values` are concatenated over the batch rather than stacked,
            since the model consumes them in the row-major order of `text_ids_mask` and of the `<|AUDIO|>`
            positions.
        """
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("This processor's tokenizer defines neither `pad_token_id` nor `eos_token_id`.")

        max_len = max(prompt["input_ids"].shape[1] for prompt in prompts)
        input_ids, attention_mask, text_ids_mask = [], [], []
        for prompt in prompts:
            pad_len = max_len - prompt["input_ids"].shape[1]
            input_ids.append(F.pad(prompt["input_ids"], (pad_len, 0), value=pad_token_id))
            attention_mask.append(F.pad(prompt["attention_mask"], (pad_len, 0), value=0))
            text_ids_mask.append(F.pad(prompt["text_ids_mask"], (pad_len, 0), value=False))

        audio_tokens = torch.cat([prompt["audio_tokens"] for prompt in prompts], dim=1)
        return {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
            "text_ids_mask": torch.cat(text_ids_mask, dim=0),
            "text_ids_len": torch.cat([prompt["text_ids_len"] for prompt in prompts], dim=0),
            "input_values": audio_tokens if audio_tokens.shape[1] > 0 else None,
        }

    def _encode_reference_audio(self, reference_audio: AudioInput, sampling_rate: int) -> list[torch.Tensor]:
        """
        Encodes reference waveforms into codebook frames.

        Args:
            reference_audio (`AudioInput`):
                One reference waveform per prompt, mono and sampled at the audio tokenizer's input rate.
            sampling_rate (`int`):
                Sample rate of `reference_audio`.

        Returns:
            `list[torch.Tensor]`: codebook frames of each clip, of shape `(num_frames, num_codebooks)`.

        Raises:
            ValueError: if this processor has no audio tokenizer, or `sampling_rate` is not the one it expects.
        """
        if getattr(self, "audio_tokenizer", None) is None:
            raise ValueError(
                "`reference_audio` was provided but this `BreezeTTSProcessor` has no `audio_tokenizer`."
            )
        expected_sampling_rate = self.audio_tokenizer.get_input_sample_rate()
        if sampling_rate != expected_sampling_rate:
            raise ValueError(
                f"`reference_audio` must be sampled at {expected_sampling_rate} Hz, got {sampling_rate} Hz."
            )

        codes = []
        for waveform in make_list_of_audio(reference_audio):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.to(self.audio_tokenizer.device)
            with torch.no_grad():
                encoded = self.audio_tokenizer.encode(waveform.unsqueeze(0), return_dict=True)
            codes.append(encoded.audio_codes[0].to(torch.long).cpu())
        return codes

    def __call__(
        self,
        text: TextInput | list[TextInput],
        instruction: TextInput | list[TextInput] | None = None,
        reference_audio: AudioInput | None = None,
        reference_text: TextInput | list[TextInput] | None = None,
        speaker: str | list[str] | None = None,
        guidance_scale: float | None = None,
        guidance_scale_ref: float | None = None,
        guidance_scale_ins: float | None = None,
        output_labels: bool = False,
        depth_decoder_labels_ratio: float = 1.0,
        **kwargs: Unpack[BreezeTTSProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Builds Breeze TTS 2 prompts, and the guidance branches those prompts are decoded against.

        Three prompt shapes are supported, matching the three ways the model is used: voice clone from
        `reference_audio` and its `reference_text`, voice design from an `instruction` alone, and voice direction
        from both.

        Args:
            text (`str` or `list[str]`):
                Text to synthesize.
            instruction (`str` or `list[str]`, *optional*):
                Natural-language description of the voice to design, or direction to apply to the reference voice.
            reference_audio (`AudioInput`, *optional*):
                Reference waveform per prompt, mono at the audio tokenizer's input sample rate.
            reference_text (`str` or `list[str]`, *optional*):
                Exact transcript of `reference_audio`. Required whenever `reference_audio` is given.
            speaker (`str` or `list[str]`, *optional*):
                Speaker token every text span opens with. Defaults to `default_speaker`.
            guidance_scale (`float`, *optional*):
                Classifier-free guidance scale of the instruction. Any value other than 1 also builds the negative
                prompt it is guided against, which is the same prompt with the instruction dropped.
            guidance_scale_ref (`float`, *optional*):
                Guidance scale of the reference audio, in the dual-guidance regime. Requires
                `guidance_scale_ins`, `reference_audio` and `instruction`.
            guidance_scale_ins (`float`, *optional*):
                Guidance scale of the instruction, in the dual-guidance regime.
            output_labels (`bool`, *optional*, defaults to `False`):
                Whether `labels` are returned for training. Indices are in
                `[config.audio_token_id, config.audio_eos_token_id, -100, -101]`, as documented on
                [`~BreezeTTSForConditionalGeneration.forward`].
            depth_decoder_labels_ratio (`float`, *optional*, defaults to 1.0):
                Fraction of the audio frames the depth decoder is scored on. The remaining frames are marked
                `-101` and score the backbone only.

        Returns:
            [`BatchFeature`]: the `input_ids`, `attention_mask`, `text_ids_mask`, `text_ids_len` and
            `input_values` of the prompts, the `cfg_`-prefixed inputs of the guidance branches, and `labels` when
            `output_labels=True`.
        """
        output_kwargs = self._merge_kwargs(BreezeTTSProcessorKwargs, **kwargs)
        return_tensors = output_kwargs["text_kwargs"].get("return_tensors", "pt")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")
        sampling_rate = output_kwargs["audio_kwargs"]["sampling_rate"]

        if isinstance(text, str):
            text = [text]
        batch_size = len(text)

        def _broadcast(value, name):
            if value is None or isinstance(value, str):
                return [value] * batch_size
            if len(value) != batch_size:
                raise ValueError(f"`{name}` has {len(value)} entries but `text` has {batch_size}.")
            return list(value)

        instruction = _broadcast(instruction, "instruction")
        reference_text = _broadcast(reference_text, "reference_text")
        speaker = _broadcast(speaker, "speaker")

        audio_codes = [None] * batch_size
        if reference_audio is not None:
            encoded_codes = self._encode_reference_audio(reference_audio, sampling_rate)
            if len(encoded_codes) != batch_size:
                raise ValueError(
                    f"`reference_audio` holds {len(encoded_codes)} clips but `text` has {batch_size} prompts."
                )
            if any(transcript is None for transcript in reference_text):
                raise ValueError("`reference_text` is required whenever `reference_audio` is given.")
            audio_codes = encoded_codes

        def _render(with_instruction: bool, with_reference_audio: bool) -> dict:
            prompts = [
                self._prepare_one(
                    self._build_segments(
                        text[i],
                        instruction[i],
                        reference_text[i],
                        has_reference_audio=with_reference_audio and audio_codes[i] is not None,
                        speaker=speaker[i],
                        with_instruction=with_instruction,
                    ),
                    [audio_codes[i]] if with_reference_audio and audio_codes[i] is not None else [],
                )
                for i in range(batch_size)
            ]
            return self._collate(prompts)

        data = _render(with_instruction=True, with_reference_audio=True)

        use_dual_guidance = guidance_scale_ref is not None and guidance_scale_ins is not None
        if use_dual_guidance:
            if reference_audio is None or not any(instruction):
                raise ValueError(
                    "Dual guidance needs both a reference audio branch and an instruction branch, so it requires "
                    "`reference_audio` and `instruction`."
                )
            branches = {
                "uncond": _render(with_instruction=False, with_reference_audio=False),
                "ref": _render(with_instruction=False, with_reference_audio=True),
                "ins": _render(with_instruction=True, with_reference_audio=False),
            }
            for branch, branch_data in branches.items():
                data[f"cfg_{branch}_prompt_ids"] = branch_data["input_ids"]
                data[f"cfg_{branch}_prompt_attention_mask"] = branch_data["attention_mask"]
                data[f"cfg_{branch}_text_ids_mask"] = branch_data["text_ids_mask"]
                data[f"cfg_{branch}_text_ids_len"] = branch_data["text_ids_len"]
            data["cfg_scale_ref"] = guidance_scale_ref
            data["cfg_scale_ins"] = guidance_scale_ins
        elif guidance_scale is not None and guidance_scale != 1.0:
            negative = _render(with_instruction=False, with_reference_audio=True)
            data["cfg_negative_prompt_ids"] = negative["input_ids"]
            data["cfg_negative_prompt_attention_mask"] = negative["attention_mask"]
            data["cfg_negative_text_ids_mask"] = negative["text_ids_mask"]
            data["cfg_negative_text_ids_len"] = negative["text_ids_len"]
            if negative["input_values"] is not None:
                data["cfg_negative_input_values"] = negative["input_values"]
            data["cfg_scale"] = guidance_scale

        if output_labels:
            if data["input_values"] is None:
                raise ValueError(
                    "`output_labels=True` scores the model on the audio frames of the prompt, but this prompt "
                    "carries none. Pass `reference_audio` and its `reference_text`."
                )
            labels = torch.where(
                (data["input_ids"] == self.audio_token_id) | (data["input_ids"] == self.audio_eos_token_id),
                data["input_ids"],
                -100,
            )
            audio_frame_idxs = (data["input_ids"] == self.audio_token_id).nonzero()
            n_audio_frames = audio_frame_idxs.shape[0]
            n_backbone_only = int(n_audio_frames * (1 - depth_decoder_labels_ratio))
            if n_backbone_only > 0:
                skip_idxs = audio_frame_idxs[torch.randperm(n_audio_frames)[:n_backbone_only]]
                labels[skip_idxs[:, 0], skip_idxs[:, 1]] = -101
            data["labels"] = labels

        data = {key: value for key, value in data.items() if value is not None}
        return BatchFeature(data=data, tensor_type=None)

    def decode(self, audio_codes: torch.LongTensor) -> list[torch.Tensor]:
        """
        Decodes generated codebook frames into waveforms.

        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames, num_codebooks)`):
                Frames produced by [`~BreezeTTSForConditionalGeneration.generate`], padded with
                `config.codebook_pad_token_id` past the end of each sample.

        Returns:
            `list[torch.Tensor]`: one mono waveform per sample.

        Raises:
            ValueError: if this processor has no audio tokenizer.
        """
        if getattr(self, "audio_tokenizer", None) is None:
            raise ValueError(
                "This `BreezeTTSProcessor` has no `audio_tokenizer`, so generated codes cannot be decoded."
            )
        codebook_size = self.audio_tokenizer.config.encoder_config.codebook_size

        waveforms = []
        for sample_codes in audio_codes:
            valid_frames = ((sample_codes >= 0) & (sample_codes < codebook_size)).all(dim=-1)
            num_frames = int(valid_frames.to(torch.int32).cumprod(dim=0).sum())
            sample_codes = sample_codes[:num_frames].to(self.audio_tokenizer.device)
            with torch.no_grad():
                decoded = self.audio_tokenizer.decode(sample_codes.unsqueeze(0), return_dict=True)
            waveforms.append(decoded.audio_values[0].cpu())
        return waveforms

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "input_values", "text_ids_mask", "text_ids_len"]


__all__ = ["BreezeTTSProcessor"]
