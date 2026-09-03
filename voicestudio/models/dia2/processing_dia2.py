# Copyright 2026 Nari Labs and the LatentForge team. All rights reserved.
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
"""Processor class for Dia2."""

import re

import numpy as np
import torch
import torchaudio

from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.mimi.modeling_mimi import MimiModel
from transformers.models.whisper.generation_whisper import _dynamic_time_warping, _median_filter
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from transformers.models.whisper.processing_whisper import WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES as WHISPER_LANGUAGES
from transformers.models.whisper.tokenization_whisper import _combine_tokens_into_words
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.utils.import_utils import requires


logger = logging.get_logger(__name__)

SPEAKER_TOKENS = ("[S1]", "[S2]")

# A word boundary is either a `<break time="1.5s"/>` silence directive or plain whitespace.
WORD_SEPARATOR = re.compile(r"(?:<break\s+time=\"([0-9]+(?:\.[0-9]*)?)s\"\s*/?>)|(?:\s+)")

DEFAULT_WHISPER_CHECKPOINT = "openai/whisper-large-v3"
DEFAULT_REFINE_PRECISION = 0.5

# One Whisper cross-attention frame covers 20 ms, and the 30 s window the encoder reads holds 1500 of them.
WHISPER_FRAME_DURATION = 0.02
WHISPER_WINDOW_FRAMES = 1500
WHISPER_MEDIAN_FILTER_WIDTH = 9
WHISPER_MIN_WORD_DURATION = 0.02


def _ensure_increasing_positions(alignment: list[dict], min_duration: float) -> list[dict]:
    """
    Pushes word boundaries apart in place until `start` and `end` come in increasing order.

    Args:
        alignment (`list[dict]`):
            Word-level alignment, as `{"text": str, "start": float, "end": float}` dicts.
        min_duration (`float`):
            Shortest duration a word may keep, in seconds.

    Returns:
        `list[dict]`: The same list, with every boundary rounded to the frame grid.
    """
    modified_backward = True
    while modified_backward:
        modified_backward = False
        previous_end = 0.0
        for index, word in enumerate(alignment):
            if word["start"] < previous_end:
                start = round((previous_end + word["start"]) / 2, 2)
                if start < alignment[index - 1]["start"] + min_duration:
                    start = previous_end
                else:
                    alignment[index - 1]["end"] = start
                    modified_backward = True
                word["start"] = start
            if word["end"] <= word["start"] + min_duration:
                word["end"] = word["start"] + min_duration
            previous_end = word["end"]

    for word in alignment:
        word["start"] = round(word["start"], 2)
        word["end"] = round(word["end"], 2)
    return alignment


class Dia2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@requires(backends=("torch",))
class Dia2Processor(ProcessorMixin):
    r"""
    Constructs a Dia2 processor which wraps an [`EncodecFeatureExtractor`], a tokenizer and a [`MimiModel`] into
    a single processor.

    Dia2 speaks one word per state machine step rather than one token per step, so the processor does not return
    a plain token sequence: it splits the script into words, tokenizes each of them, and reports how many tokens
    and how many frames of hold time each word owns.

    Conditioning audio needs a word-level alignment to place its words on the frame grid. A caller can supply
    one directly through `transcript`, or leave it out and let the processor derive it from `audio` itself with
    a Whisper model, loaded on first use from `whisper_checkpoint`.

    Args:
        feature_extractor ([`EncodecFeatureExtractor`], *optional*):
            Feature extractor of the Mimi codec, used to prepare conditioning audio.
        tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            Text tokenizer of the model.
        audio_tokenizer ([`MimiModel`], *optional*):
            Codec that turns conditioning audio into codes and generated codes back into a waveform.
        chat_template (`str`, *optional*):
            Template string used by [`~ProcessorMixin.apply_chat_template`].
        whisper_checkpoint (`str`, *optional*, defaults to `"openai/whisper-large-v3"`):
            Hub id or local path of the Whisper checkpoint used to align `audio` when `transcript` is not
            given. Only loaded the first time that path is taken.
        refine_whisper_precision (`float`, *optional*, defaults to 0.5):
            Margin in seconds, a multiple of 0.02, added on both sides of the segment boundaries Whisper
            predicts for itself. Each segment is aligned again inside that window. `None` keeps the single
            whole-window alignment.
    """

    # `ProcessorMixin` only accepts an `audio_tokenizer` argument whose class is registered for audio
    # tokenization, which `MimiModel` is not, so the codec is held outside the sub-processor machinery. Its
    # class-level default keeps it out of `__dict__`, and therefore out of `to_dict`, while it is unset.
    audio_tokenizer = None

    # Cache for the Whisper model and processor backing automatic alignment, built on first use by
    # `_require_whisper`. Kept off `__init__`'s signature so it is never part of `to_dict`.
    _whisper_model = None
    _whisper_processor = None

    feature_extractor_class = "EncodecFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        audio_tokenizer=None,
        chat_template=None,
        whisper_checkpoint: str = DEFAULT_WHISPER_CHECKPOINT,
        refine_whisper_precision: float | None = DEFAULT_REFINE_PRECISION,
    ):
        if audio_tokenizer is not None:
            self.audio_tokenizer = audio_tokenizer
        self.whisper_checkpoint = whisper_checkpoint
        self.refine_whisper_precision = refine_whisper_precision
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @property
    def frame_rate(self) -> float:
        return self._require_audio_tokenizer().config.frame_rate

    @property
    def sampling_rate(self) -> int:
        return self._require_audio_tokenizer().config.sampling_rate

    def _require_audio_tokenizer(self) -> MimiModel:
        if self.audio_tokenizer is None:
            raise ValueError(
                "This Dia2Processor has no `audio_tokenizer`. Set one to a `MimiModel` before splitting a "
                "script, encoding conditioning audio or decoding generated codes."
            )
        return self.audio_tokenizer

    def _require_whisper(self) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
        if self._whisper_model is None:
            self._whisper_processor = WhisperProcessor.from_pretrained(self.whisper_checkpoint)
            self._whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_checkpoint)
            self._whisper_model.to(self._require_audio_tokenizer().device)
            self._whisper_model.eval()
        return self._whisper_model, self._whisper_processor

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | None = None,
        audio: AudioInput | None = None,
        transcript: list[list[dict]] | None = None,
        **kwargs: Unpack[Dia2ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Args:
            text (`str` or `list[str]`):
                Script to speak, using `[S1]` / `[S2]` speaker tags and optional `<break time="1.5s"/>`
                directives. A list is joined into a single script, one line per element.
            audio (`AudioInput`, *optional*):
                One conditioning waveform per speaker, at `sampling_rate`.
            transcript (`list[list[dict]]`, *optional*):
                Word-level alignment of each `audio` entry, as a list of `{"text": str, "start": float,
                "end": float}` dicts with times in seconds. When `audio` is given and this is not, it is
                derived automatically with the Whisper model from `whisper_checkpoint`.

        Returns:
            [`~feature_extraction_utils.BatchFeature`]: A dictionary holding

            - **input_ids** -- Text tokens of every word, concatenated.
            - **word_lengths** -- Number of `input_ids` tokens belonging to each word.
            - **word_paddings** -- Number of frames each word must be held before the next one may start.
            - **prefix_audio_codes** -- Time-aligned codes of the conditioning audio, when `audio` is given.
            - **prefix_word_start_frames** -- Frame at which each conditioning word starts, when `audio` is given.

        Raises:
            ValueError: If `text` is missing, or if `audio` and `transcript` do not describe the same speakers.
        """
        if text is None:
            raise ValueError("Dia2Processor requires a `text` script.")
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(line).strip() for line in text)
        text = str(text).strip()

        output_kwargs = self._merge_kwargs(
            Dia2ProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        add_special_tokens = output_kwargs["text_kwargs"].get("add_special_tokens", False)
        return_tensors = output_kwargs["text_kwargs"].get("return_tensors", "pt")

        data = {}
        prefix_words = []
        if audio is not None:
            audio = make_list_of_audio(audio)
            if transcript is None:
                transcript = [self._transcribe(waveform) for waveform in audio]
            elif len(audio) != len(transcript):
                raise ValueError(
                    f"Got {len(audio)} conditioning waveforms but {len(transcript)} transcripts; they must match."
                )
            prefix_words, prefix_codes, prefix_start_frames = self._encode_prefix(
                audio, transcript, add_special_tokens
            )
            data["prefix_audio_codes"] = prefix_codes
            data["prefix_word_start_frames"] = prefix_start_frames

        words = prefix_words + self._split_script(text, add_special_tokens)
        data["input_ids"] = [[token_id for token_ids, _ in words for token_id in token_ids]]
        data["word_lengths"] = [[len(token_ids) for token_ids, _ in words]]
        data["word_paddings"] = [[padding for _, padding in words]]

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _split_script(self, text: str, add_special_tokens: bool) -> list[tuple[list[int], int]]:
        """
        Splits a script into `(token_ids, padding)` words, resolving speaker tags and break directives.
        """
        words: list[tuple[list[int], int]] = []
        speaker_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in SPEAKER_TOKENS]
        remaining = text.replace("’", "'").replace(":", " ")
        pending_speaker = None
        is_first_word = True

        while remaining:
            match = WORD_SEPARATOR.search(remaining)
            if match is None:
                segment, remaining = remaining, ""
            else:
                segment, remaining = remaining[: match.start()], remaining[match.end() :]

            for raw_word in segment.split():
                if raw_word in SPEAKER_TOKENS:
                    pending_speaker = raw_word
                    continue
                token_ids = self._encode_word(raw_word, pending_speaker, add_special_tokens)
                pending_speaker = None
                if is_first_word:
                    if speaker_token_ids[0] is not None and (not token_ids or token_ids[0] != speaker_token_ids[0]):
                        token_ids.insert(0, speaker_token_ids[0])
                    is_first_word = False
                words.append((token_ids, len(token_ids)))

            if match is not None and match.group(1):
                padding = int(round(float(match.group(1)) * self.frame_rate))
                if padding > 0:
                    words.append(([], padding))

        return words

    def _encode_word(self, word: str, speaker: str | None, add_special_tokens: bool) -> list[int]:
        text = f"{speaker} {word}" if speaker else word
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def _transcribe(self, waveform) -> list[dict]:
        """
        Runs `whisper_checkpoint` over one conditioning waveform and returns its word-level alignment, in the
        same `{"text", "start", "end"}` schema `_align_words` consumes.
        """
        model, whisper_processor = self._require_whisper()
        feature_extractor = whisper_processor.feature_extractor
        tokenizer = whisper_processor.tokenizer

        audio = torch.as_tensor(waveform, dtype=torch.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1) if audio.shape[0] == 1 else audio.mean(dim=0)
        if self.sampling_rate != feature_extractor.sampling_rate:
            audio = torchaudio.functional.resample(audio, self.sampling_rate, feature_extractor.sampling_rate)

        inputs = feature_extractor(
            audio.numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs["input_features"].to(device=model.device, dtype=model.dtype)
        attention_mask = inputs["attention_mask"].to(model.device)

        refine = self.refine_whisper_precision is not None
        with torch.no_grad():
            outputs = model.generate(
                input_features,
                attention_mask=attention_mask,
                return_timestamps=refine,
                return_token_timestamps=True,
                return_dict_in_generate=True,
            )
        sequence = (outputs["sequences"] if isinstance(outputs, dict) else outputs.sequences)[0].tolist()
        token_timestamps = (
            outputs["token_timestamps"] if isinstance(outputs, dict) else outputs.token_timestamps
        )[0].tolist()
        segments = outputs["segments"][0] if isinstance(outputs, dict) and "segments" in outputs else []

        # Asking for timestamps drops the decoder prompt from `sequences`, so the language token has to be
        # read off the sequence of the generation call each segment came out of. Grouping tokens into words
        # needs the language, since it decides whether words are split on whitespace or Unicode boundaries.
        prompted_sequence = segments[0]["result"]["sequences"].tolist() if segments else sequence
        language = None
        lang_to_id = getattr(model.generation_config, "lang_to_id", None)
        if lang_to_id:
            id_to_code = {token_id: token.strip("<|>") for token, token_id in lang_to_id.items()}
            language = next(
                (
                    WHISPER_LANGUAGES.get(id_to_code[token_id])
                    for token_id in prompted_sequence
                    if token_id in id_to_code
                ),
                None,
            )

        if segments:
            token_timestamps = self._refine_token_timestamps(segments, token_timestamps)

        timestamp_begin = model.generation_config.no_timestamps_token_id + 1
        special_ids = set(tokenizer.all_special_ids)
        content_positions = [
            position
            for position, token_id in enumerate(sequence)
            if token_id not in special_ids and token_id < timestamp_begin
        ]
        content_tokens = [sequence[position] for position in content_positions]
        words, _, token_groups = _combine_tokens_into_words(tokenizer, content_tokens, language=language)

        alignment = []
        for word, group in zip(words, token_groups):
            start_position = content_positions[group[0]]
            end_position = content_positions[group[-1]]
            start = token_timestamps[start_position - 1] if start_position > 0 else 0.0
            alignment.append({"text": word.strip(), "start": start, "end": token_timestamps[end_position]})
        if segments:
            _ensure_increasing_positions(alignment, WHISPER_MIN_WORD_DURATION)
        return alignment

    def _refine_token_timestamps(self, segments: list[dict], token_timestamps: list[float]) -> list[float]:
        r"""
        Aligns every Whisper segment again inside a `refine_whisper_precision` window around the segment
        boundaries the decoder predicted for itself, and replaces the times of the tokens it covers.

        Args:
            segments (`list[dict]`):
                Segments returned by [`~WhisperForConditionalGeneration.generate`], each holding its tokens,
                their span in the underlying generation call and that call's outputs.
            token_timestamps (`list[float]`):
                Whole-window alignment times, one per token of the returned sequence.

        Returns:
            `list[float]`: The times, with every position of a realigned segment replaced.

        Raises:
            ValueError: If `refine_whisper_precision` is negative or is not a multiple of 0.02.
        """
        margin = self.refine_whisper_precision / WHISPER_FRAME_DURATION
        if self.refine_whisper_precision < 0 or margin != round(margin):
            raise ValueError(
                f"`refine_whisper_precision` must be a non-negative multiple of {WHISPER_FRAME_DURATION}, got "
                f"{self.refine_whisper_precision}."
            )
        margin = round(margin)

        model, _ = self._require_whisper()
        alignment_heads = getattr(model.generation_config, "alignment_heads", None)
        timestamp_begin = model.generation_config.no_timestamps_token_id + 1
        if not alignment_heads or sum(len(segment["tokens"]) for segment in segments) != len(token_timestamps):
            return token_timestamps

        refined = list(token_timestamps)
        weights = None
        aligned_call = None
        position = 0
        for segment in segments:
            tokens = segment["tokens"].tolist()
            if segment["result"] is not aligned_call:
                aligned_call = segment["result"]
                weights = torch.stack(
                    [
                        torch.cat([step[layer][0, head] for step in aligned_call["cross_attentions"]], dim=0)
                        for layer, head in alignment_heads
                    ]
                ).float()
            times = self._align_segment(tokens, weights, segment["idxs"], margin, timestamp_begin)
            if times is None:
                position += len(tokens)
                continue
            # The cross-attention of a token is read while the next one is predicted, so the time of the
            # token at `index` lands on the entry of the token before it.
            for index, time in enumerate(times, start=position - 1):
                if 0 <= index < len(refined):
                    refined[index] = time
            position += len(tokens)
        return refined

    @staticmethod
    def _align_segment(
        tokens: list[int], weights: torch.Tensor, span: tuple[int, int], margin: int, timestamp_begin: int
    ) -> list[float] | None:
        r"""
        Warps one segment's cross-attention against the frames its own timestamps span, widened by `margin`.

        Args:
            tokens (`list[int]`):
                Segment tokens, opened and closed by the timestamp tokens Whisper predicted.
            weights (`torch.FloatTensor` of shape `(num_alignment_heads, num_rows, 1500)`):
                Cross-attention of the whole generation call, over the heads calibrated for alignment.
            span (`tuple[int, int]`):
                Start and end index of `tokens` in the sequence of that generation call.
            margin (`int`):
                Number of frames added on both sides of the segment.
            timestamp_begin (`int`):
                Token id of `<|0.00|>`.

        Returns:
            `list[float]` or `None`: Time of every token, followed by the time the segment ends at, or `None`
            when the segment carries no usable timestamps.
        """
        start_frame = tokens[0] - timestamp_begin
        end_frame = tokens[-1] - timestamp_begin
        if start_frame < 0:
            return None
        if end_frame < 0:
            end_frame = WHISPER_WINDOW_FRAMES

        end_frame = min(WHISPER_WINDOW_FRAMES, max(end_frame, start_frame + len(tokens)))
        if margin > 0:
            start_frame = max(start_frame - margin, 0)
            end_frame = min(end_frame + margin, WHISPER_WINDOW_FRAMES)
        row_start, row_end = span[0] - 1, span[1] - 1
        num_frames = end_frame - start_frame
        if num_frames < max(len(tokens), WHISPER_MEDIAN_FILTER_WIDTH) or row_start < 0 or row_end > weights.shape[-2]:
            return None

        matrix = _median_filter(weights[:, row_start:row_end, start_frame:end_frame], WHISPER_MEDIAN_FILTER_WIDTH)
        matrix = matrix.softmax(dim=-1).mean(dim=0)
        matrix = -(matrix / matrix.norm(dim=-2, keepdim=True)).double().numpy()
        matrix[0, 0] = matrix.min()

        text_indices, time_indices = _dynamic_time_warping(matrix)
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jumps = np.pad(time_indices[jumps], (0, 1), constant_values=time_indices[-1])
        times = [round((start_frame + jump) * WHISPER_FRAME_DURATION, 2) for jump in jumps]
        if margin == 0:
            times[1] = times[0]
            times[-2] = times[-1]
        return times

    def _encode_prefix(
        self, audio: list, transcript: list[list[dict]], add_special_tokens: bool
    ) -> tuple[list[tuple[list[int], int]], torch.Tensor, list[int]]:
        """
        Encodes the conditioning waveforms into codes and turns their alignments into script words.
        """
        words: list[tuple[list[int], int]] = []
        start_frames: list[int] = []
        codes = []
        # The state machine sees three leading frames of beginning-of-stream padding before the first
        # conditioning frame, which the first speaker's word alignment has to be shifted past.
        offset = 3

        for speaker_index, (waveform, alignment) in enumerate(zip(audio, transcript)):
            speaker_words, speaker_start_frames = self._align_words(
                alignment, SPEAKER_TOKENS[min(speaker_index, len(SPEAKER_TOKENS) - 1)], add_special_tokens
            )
            words.extend(speaker_words)
            start_frames.extend(frame + offset for frame in speaker_start_frames)

            inputs = self.feature_extractor(
                raw_audio=waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
            )
            input_values = inputs["input_values"].to(self._require_audio_tokenizer().device)
            if input_values.ndim == 2:
                input_values = input_values.unsqueeze(1)
            with torch.no_grad():
                speaker_codes = self._require_audio_tokenizer().encode(input_values).audio_codes[0].to(torch.long)
            codes.append(speaker_codes)
            offset = sum(chunk.shape[-1] for chunk in codes)

        return words, torch.cat(codes, dim=-1).unsqueeze(0), start_frames

    def _align_words(
        self, alignment: list[dict], speaker: str, add_special_tokens: bool
    ) -> tuple[list[tuple[list[int], int]], list[int]]:
        """
        Turns a word-level alignment into `(token_ids, padding)` words and the frames they start at.
        """
        words: list[tuple[list[int], int]] = []
        start_frames: list[int] = []
        pending_speaker = speaker
        cursor = 0

        for index, word in enumerate(alignment):
            token_ids = self._encode_word(str(word["text"]).strip(), pending_speaker, add_special_tokens)
            pending_speaker = None
            start_frame = max(cursor + 1, int(round(float(word["start"]) * self.frame_rate)))
            end_frame = start_frame + len(token_ids)
            start_frames.append(start_frame - 1)

            if index < len(alignment) - 1:
                following = int(round(float(alignment[index + 1]["start"]) * self.frame_rate))
            else:
                following = int(round(float(alignment[-1]["end"]) * self.frame_rate))
            next_start = max(end_frame + 1, following)

            words.append((token_ids, max(0, next_start - start_frame - 1)))
            cursor = end_frame

        return words, start_frames

    def decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
                Time-aligned codes returned by [`~Dia2ForConditionalGeneration.generate`].

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 1, num_samples)`: The decoded waveform, clamped to
            `[-1, 1]`.
        """
        audio_codes = audio_codes.to(self._require_audio_tokenizer().device)
        with torch.no_grad():
            audio_values = self._require_audio_tokenizer().decode(audio_codes).audio_values
        return audio_values.clamp(-1.0, 1.0)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


__all__ = ["Dia2Processor"]
