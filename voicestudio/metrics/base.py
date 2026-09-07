"""Shared normalization, alignment readout and transcription for the error rate metrics."""

import soundfile
import torch
import torchaudio.functional as audio_functional
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_audio(path: str, sampling_rate: int) -> torch.Tensor:
    """Reads one clip as a mono waveform at the requested sampling rate.

    Args:
        path (`str`):
            Path to the audio file.
        sampling_rate (`int`):
            Rate the returned waveform is resampled to.

    Returns:
        `torch.Tensor`: The mono waveform, shaped `(samples,)`.
    """
    # `soundfile` decodes without the FFmpeg bindings `torchaudio.load` reaches for.
    samples, source_rate = soundfile.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).mean(1)
    if source_rate != sampling_rate:
        waveform = audio_functional.resample(waveform, source_rate, sampling_rate)
    return waveform


def load_batch(paths: list[str], sampling_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reads several clips into one padded batch.

    Args:
        paths (`list[str]`):
            Paths to the audio files.
        sampling_rate (`int`):
            Rate the returned waveforms are resampled to.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The waveforms right-padded with zeros into a
        `(batch, samples)` tensor, and the unpadded length of each.
    """
    waveforms = [load_audio(path, sampling_rate) for path in paths]
    lengths = torch.tensor([waveform.shape[0] for waveform in waveforms])
    padded = torch.zeros(len(waveforms), int(lengths.max()), dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        padded[index, : waveform.shape[0]] = waveform
    return padded, lengths


def normalizing_transform(reduction):
    """Builds a jiwer transform that normalizes before reducing to tokens.

    Args:
        reduction (`jiwer.AbstractTransform`):
            The reduction the caller's metric aligns on, such as
            [`jiwer.ReduceToListOfListOfWords`] or [`jiwer.ReduceToListOfListOfChars`].

    Returns:
        `jiwer.Compose`: The transform to pass as both `reference_transform` and `hypothesis_transform`.
    """
    import jiwer

    # jiwer's own defaults only collapse whitespace. Transcribers disagree on casing and punctuation far
    # more often than they disagree on words, so comparing without stripping those reports differences
    # that are not errors.
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            reduction,
        ]
    )


def per_utterance_rates(output):
    """Recovers a per-utterance error rate from an already computed alignment.

    Args:
        output (`jiwer.WordOutput` or `jiwer.CharacterOutput`):
            The result of `jiwer.process_words` or `jiwer.process_characters`.

    Returns:
        `list[float]`: One rate per utterance, in input order, `inf` where the reference is empty.
    """
    rates = []
    for reference, alignment in zip(output.references, output.alignments):
        # An insertion spans hypothesis tokens and has no width on the reference side.
        errors = sum(
            chunk.hyp_end_idx - chunk.hyp_start_idx
            if chunk.type == "insert"
            else chunk.ref_end_idx - chunk.ref_start_idx
            for chunk in alignment
            if chunk.type != "equal"
        )
        rates.append(errors / len(reference) if reference else float("inf"))
    return rates


class TranscriberMixin:
    """Turns paths to generated audio into text with Whisper, for a metric that scores transcripts.

    Args:
        transcriber (`str`, *optional*):
            Repository id of a Whisper checkpoint, or a bare size such as `"large-v3"`. Given one,
            `predictions` are read as paths to generated audio and transcribed before scoring rather
            than taken as text.
        language (`str`, *optional*, defaults to `"en"`):
            Language passed to `generate`. Letting Whisper detect it instead makes the transcript
            depend on the audio being scored, which is the thing under test.
        device (`str`, *optional*):
            Device the transcriber runs on. Defaults to CUDA where it is available.
        dtype (`str` or `torch.dtype`, *optional*, defaults to `torch.float16`):
            Precision the transcriber is loaded in.
        batch_size (`int`, *optional*, defaults to 8):
            Clips per `generate` call.
    """

    def __init__(
        self,
        transcriber: str | None = None,
        language: str = "en",
        device: str | None = None,
        dtype: str | torch.dtype = torch.float16,
        batch_size: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transcriber = transcriber
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def load_transcriber(self) -> None:
        """Loads the Whisper checkpoint named by `transcriber`, once.

        Raises:
            ValueError: If no `transcriber` was configured.
        """
        if self._model is not None:
            return
        if self.transcriber is None:
            raise ValueError("No transcriber was configured, so predictions must already be text.")
        # A bare size such as "large-v3" names an OpenAI release; anything with an owner is taken as is.
        model_id = self.transcriber if "/" in self.transcriber else f"openai/whisper-{self.transcriber}"
        self._processor = WhisperProcessor.from_pretrained(model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype)
        self._model.to(self.device).eval()

    def release(self) -> None:
        """Drops the loaded transcriber and frees the memory it held."""
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, audio_paths: list[str]) -> list[str]:
        """Transcribes generated audio in batches.

        Args:
            audio_paths (`list[str]`):
                Paths to the generated audio, one per utterance.

        Returns:
            `list[str]`: One transcript per input path, in input order.
        """
        self.load_transcriber()
        sampling_rate = self._processor.feature_extractor.sampling_rate

        # A path repeated across utterances is transcribed once. Whisper dominates the runtime here.
        unique_paths = list(dict.fromkeys(audio_paths))
        transcripts = {}
        for start in range(0, len(unique_paths), self.batch_size):
            batch = unique_paths[start : start + self.batch_size]
            waveforms = [load_audio(path, sampling_rate).numpy() for path in batch]
            # Whisper's pad token is its eos token, so a batch without an explicit mask decodes
            # against padding it cannot identify.
            features = self._processor(
                waveforms, sampling_rate=sampling_rate, return_tensors="pt", return_attention_mask=True
            )
            with torch.no_grad():
                predicted_ids = self._model.generate(
                    features.input_features.to(device=self.device, dtype=self._model.dtype),
                    attention_mask=features.attention_mask.to(self.device),
                    language=self.language,
                    task="transcribe",
                )
            decoded = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcripts.update(zip(batch, (text.strip() for text in decoded)))

        return [transcripts[path] for path in audio_paths]

__all__ = ["TranscriberMixin", "load_audio", "load_batch", "normalizing_transform", "per_utterance_rates"]
