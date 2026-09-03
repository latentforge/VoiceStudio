"""Processor class for F5-TTS."""

import re

import numpy as np
import torch
import torchaudio

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging


logger = logging.get_logger(__name__)


class F5TTSProcessor(ProcessorMixin):
    r"""
    Constructs an F5-TTS processor which wraps an [`F5TTSFeatureExtractor`] and an [`F5TTSTokenizer`] into a single
    processor.

    It turns a reference clip and its transcription, together with the text to speak, into the reference log mel
    spectrogram, the character ids of the reference transcription followed by the text, and the number of mel frames
    to generate. It also owns the inverse steps, vocoding the generated spectrogram through the model's vocoder,
    undoing the loudness normalization of the reference clip and cross fading the waveforms of consecutive text
    chunks back together.

    Args:
        feature_extractor ([`F5TTSFeatureExtractor`]):
            Feature extractor computing the log mel spectrogram.
        tokenizer ([`F5TTSTokenizer`]):
            Tokenizer mapping characters onto vocabulary ids.
        target_rms (`float`, *optional*, defaults to 0.1):
            Loudness the reference clip is scaled to when it is quieter than this.
        speed (`float`, *optional*, defaults to 1.0):
            Divisor of the estimated number of frames to generate. Values above 1 speak faster.
        cross_fade_duration (`float`, *optional*, defaults to 0.15):
            Length in seconds of the overlap two consecutive chunk waveforms are cross faded over.
    """

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        target_rms: float = 0.1,
        speed: float = 1.0,
        cross_fade_duration: float = 0.15,
        **kwargs,
    ):
        self.target_rms = target_rms
        self.speed = speed
        self.cross_fade_duration = cross_fade_duration
        super().__init__(feature_extractor, tokenizer, **kwargs)

    @staticmethod
    def chunk_text(text: str, max_chars: int) -> list[str]:
        r"""
        Splits text on sentence punctuation into chunks that each stay under a byte budget.

        Args:
            text (`str`):
                Text to split.
            max_chars (`int`):
                Largest number of UTF-8 bytes a chunk may hold.

        Returns:
            `list[str]`: The chunks, in order.
        """
        chunks = []
        current_chunk = ""
        sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

        for sentence in sentences:
            if not sentence:
                continue
            if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
                current_chunk += (
                    sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
                )
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = (
                    sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
                )

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def max_chunk_chars(self, ref_text: str, reference_seconds: float, speed: float | None = None) -> int:
        r"""
        Estimates the byte budget one chunk may hold so that the reference clip plus the chunk stay under the
        22 seconds the model was trained on.

        Args:
            ref_text (`str`):
                Transcription of the reference clip.
            reference_seconds (`float`):
                Length of the reference clip, in seconds.
            speed (`float`, *optional*):
                Divisor of the estimated number of frames to generate. Defaults to the value set on the processor.

        Returns:
            `int`: Largest number of UTF-8 bytes a chunk may hold.
        """
        speed = self.speed if speed is None else speed
        return int(len(ref_text.encode("utf-8")) / reference_seconds * (22 - reference_seconds) * speed)

    @staticmethod
    def prepare_reference_text(ref_text: str) -> str:
        r"""
        Appends the sentence final punctuation and trailing space the reference transcription is expected to end
        with.

        Args:
            ref_text (`str`):
                Transcription of the reference clip.

        Returns:
            `str`: The transcription, terminated.
        """
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            ref_text = ref_text + " " if ref_text.endswith(".") else ref_text + ". "
        return ref_text

    def _prepare_reference_audio(self, audio, sampling_rate: int | None) -> tuple[torch.Tensor, float]:
        waveform = torch.as_tensor(np.asarray(audio) if not torch.is_tensor(audio) else audio, dtype=torch.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        waveform = waveform.reshape(-1)

        rms = float(torch.sqrt(torch.mean(torch.square(waveform))))
        if rms < self.target_rms and rms > 0:
            waveform = waveform * self.target_rms / rms

        if sampling_rate is not None and sampling_rate != self.feature_extractor.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform, sampling_rate, self.feature_extractor.sampling_rate
            )

        return waveform, rms

    def __call__(
        self,
        text: str | list[str],
        audio=None,
        ref_text: str = "",
        sampling_rate: int | None = None,
        speed: float | None = None,
        fix_duration: float | list[float] | None = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        r"""
        Args:
            text (`str` or `list[str]`):
                Text to speak. A list gives one batch entry per element, all sharing the reference clip, which is
                how a long text split with [`~F5TTSProcessor.chunk_text`] is passed in.
            audio (`np.ndarray` or `torch.Tensor`):
                Waveform of the reference clip, mono or multichannel.
            ref_text (`str`, *optional*, defaults to `""`):
                Transcription of the reference clip.
            sampling_rate (`int`, *optional*):
                Sampling rate of `audio`. The clip is resampled when it differs from the feature extractor's.
            speed (`float`, *optional*):
                Divisor of the estimated number of frames to generate. Defaults to the value set on the processor.
            fix_duration (`float` or `list[float]`, *optional*):
                Total length in seconds, reference clip included, to generate instead of estimating it.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Framework of the returned tensors. Only `"pt"` is supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with `input_ids`, `input_features`, `attention_mask` and
            `duration` entries, plus the `reference_rms` and `reference_length` needed to undo the loudness
            normalization and to cut the reference frames off the generated spectrogram.

        Raises:
            ValueError: If `audio` is not given, if `ref_text` is empty, or if `return_tensors` is not `"pt"`.
        """
        if audio is None:
            raise ValueError("F5-TTS conditions on a reference clip, so `audio` is required.")
        if not ref_text.strip():
            raise ValueError(
                "F5-TTS conditions on the transcription of the reference clip, so `ref_text` is required."
            )
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only returns PyTorch tensors, got {return_tensors}.")

        speed = self.speed if speed is None else speed
        texts = [text] if isinstance(text, str) else list(text)

        waveform, rms = self._prepare_reference_audio(audio, sampling_rate)
        ref_text = self.prepare_reference_text(ref_text)
        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "

        features = self.feature_extractor(waveform, sampling_rate=self.feature_extractor.sampling_rate)
        reference_length = int(features["input_features"].shape[1])

        hop_length = self.feature_extractor.hop_length
        target_sampling_rate = self.feature_extractor.sampling_rate
        ref_text_len = len(ref_text.encode("utf-8"))

        if fix_duration is None:
            fix_durations = [None] * len(texts)
        elif isinstance(fix_duration, (int, float)):
            fix_durations = [float(fix_duration)] * len(texts)
        else:
            fix_durations = list(fix_duration)

        durations = []
        for gen_text, fixed in zip(texts, fix_durations):
            if fixed is not None:
                durations.append(int(fixed * target_sampling_rate / hop_length))
                continue
            local_speed = 0.3 if len(gen_text.encode("utf-8")) < 10 else speed
            gen_text_len = len(gen_text.encode("utf-8"))
            durations.append(reference_length + int(reference_length / ref_text_len * gen_text_len / local_speed))

        encoded = self.tokenizer([ref_text + gen_text for gen_text in texts], padding=True, return_tensors="pt")

        data = {
            "input_ids": encoded["input_ids"],
            "input_features": features["input_features"].expand(len(texts), -1, -1),
            "attention_mask": features["attention_mask"].expand(len(texts), -1),
            "duration": torch.tensor(durations, dtype=torch.long),
            "reference_length": reference_length,
            "reference_rms": rms,
        }
        return BatchFeature(data=data)

    def batch_decode(
        self,
        mel_spectrogram: torch.Tensor,
        vocoder,
        duration: torch.Tensor | None = None,
        reference_length: int = 0,
        reference_rms: float | None = None,
        cross_fade_duration: float | None = None,
    ) -> np.ndarray:
        r"""
        Vocodes the generated spectrograms of consecutive text chunks, joins them into one signal and undoes the
        loudness normalization applied to the reference clip.

        Args:
            mel_spectrogram (`torch.Tensor`):
                Generated log mel spectrogram of shape `(batch_size, sequence_length, mel_dim)`, one entry per
                chunk, in order.
            vocoder ([`VocosModel`] or [`BigVGANModel`]):
                Vocoder turning a generated log mel spectrogram back into a waveform, which is
                `F5TTSForConditionalGeneration.vocoder`. It has to match the mel front end the feature extractor
                is configured with.
            duration (`torch.Tensor`, *optional*):
                Number of frames of each entry as returned by [`~F5TTSProcessor.__call__`]. Given, the batch
                padding past each entry's own duration is cut before vocoding.
            reference_length (`int`, *optional*, defaults to 0):
                Number of leading frames carrying the reference speech, as returned by
                [`~F5TTSProcessor.__call__`]. They are cut off before vocoding.
            reference_rms (`float`, *optional*):
                Loudness of the reference clip as returned by [`~F5TTSProcessor.__call__`]. Given, the output is
                scaled back by the factor the reference clip was scaled up by.
            cross_fade_duration (`float`, *optional*):
                Length in seconds of the cross faded overlap. Defaults to the value set on the processor. `0`
                concatenates without a fade.

        Returns:
            `np.ndarray`: The joined waveform.
        """
        cross_fade_duration = self.cross_fade_duration if cross_fade_duration is None else cross_fade_duration
        sampling_rate = self.feature_extractor.sampling_rate
        dtype = next(vocoder.parameters()).dtype
        device = next(vocoder.parameters()).device

        waveforms = []
        for index, spectrogram in enumerate(mel_spectrogram):
            if duration is not None:
                spectrogram = spectrogram[: int(duration[index])]
            spectrogram = spectrogram[reference_length:].permute(1, 0).unsqueeze(0)
            with torch.no_grad():
                waveform = vocoder(input_features=spectrogram.to(device=device, dtype=dtype)).audio_values
            waveforms.append(waveform.squeeze(0).to(torch.float32).cpu().numpy())

        if reference_rms is not None and 0 < reference_rms < self.target_rms:
            waveforms = [waveform * reference_rms / self.target_rms for waveform in waveforms]

        if not waveforms:
            return np.zeros(0, dtype=np.float32)

        final_wave = waveforms[0]
        for next_wave in waveforms[1:]:
            cross_fade_samples = int(cross_fade_duration * sampling_rate)
            cross_fade_samples = min(cross_fade_samples, len(final_wave), len(next_wave))
            if cross_fade_samples <= 0:
                final_wave = np.concatenate([final_wave, next_wave])
                continue

            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)
            overlap = final_wave[-cross_fade_samples:] * fade_out + next_wave[:cross_fade_samples] * fade_in
            final_wave = np.concatenate(
                [final_wave[:-cross_fade_samples], overlap, next_wave[cross_fade_samples:]]
            )

        return final_wave


__all__ = ["F5TTSProcessor"]
