"""Processor class for Dia2."""

import re

import torch

from transformers.audio_utils import AudioInput, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.mimi.modeling_mimi import MimiModel
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.utils.import_utils import requires


logger = logging.get_logger(__name__)

SPEAKER_TOKENS = ("[S1]", "[S2]")

# A word boundary is either a `<break time="1.5s"/>` silence directive or plain whitespace.
WORD_SEPARATOR = re.compile(r"(?:<break\s+time=\"([0-9]+(?:\.[0-9]*)?)s\"\s*/?>)|(?:\s+)")


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

    Args:
        feature_extractor ([`EncodecFeatureExtractor`], *optional*):
            Feature extractor of the Mimi codec, used to prepare conditioning audio.
        tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            Text tokenizer of the model.
        audio_tokenizer ([`MimiModel`], *optional*):
            Codec that turns conditioning audio into codes and generated codes back into a waveform.
        chat_template (`str`, *optional*):
            Template string used by [`~ProcessorMixin.apply_chat_template`].
    """

    # `ProcessorMixin` only accepts an `audio_tokenizer` argument whose class is registered for audio
    # tokenization, which `MimiModel` is not, so the codec is held outside the sub-processor machinery. Its
    # class-level default keeps it out of `__dict__`, and therefore out of `to_dict`, while it is unset.
    audio_tokenizer = None

    feature_extractor_class = "EncodecFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, audio_tokenizer=None, chat_template=None):
        if audio_tokenizer is not None:
            self.audio_tokenizer = audio_tokenizer
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
                One conditioning waveform per speaker, at `sampling_rate`. Requires `transcript`.
            transcript (`list[list[dict]]`, *optional*):
                Word-level alignment of each `audio` entry, as a list of `{"text": str, "start": float,
                "end": float}` dicts with times in seconds.

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
            if transcript is None:
                raise ValueError("`audio` requires the matching word-level `transcript`.")
            audio = make_list_of_audio(audio)
            if len(audio) != len(transcript):
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
