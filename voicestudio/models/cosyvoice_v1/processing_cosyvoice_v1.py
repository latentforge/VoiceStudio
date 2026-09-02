"""Processor class for CosyVoice v1."""

import numpy as np
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import cached_file


def _whisper_log_mel_spectrogram(
    waveform: torch.Tensor, n_mels: int = 128, n_fft: int = 400, hop_length: int = 160
) -> torch.Tensor:
    r"""
    Log-mel spectrogram matching `openai-whisper`'s `audio.log_mel_spectrogram` (16 kHz sampling rate, a
    400-sample/25ms Hann-windowed STFT with 160-sample/10ms hop, log10-compressed and dynamic-range-clamped to
    the top 80dB, `(x + 4) / 4` normalized): the feature extraction the original CosyVoice repo's
    `speech_tokenizer_v1.onnx` was trained and exported with.
    """
    import librosa

    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels)).to(waveform.device, waveform.dtype)
    mel_spec = mel_filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


class CosyVoiceV1Processor(ProcessorMixin):
    r"""
    Constructs a CosyVoice v1 processor which wraps a text tokenizer.

    [`CosyVoiceV1Processor`] tokenizes text into the ids consumed by [`CosyVoiceV1LLM`] and renders a generated
    mel spectrogram into a waveform with [`CosyVoiceV1HiFTGenerator`]. Discrete speech token extraction
    (`extract_speech_token`) and speaker embedding extraction (`extract_speaker_embedding`) for voice cloning are
    done via the original repository's `speech_tokenizer_v1.onnx`/`campplus.onnx` ONNX models, which have no
    `transformers` equivalent and are run directly through `onnxruntime.InferenceSession` (loaded lazily by
    [`~CosyVoiceV1Processor.from_pretrained`] alongside the tokenizer) rather than reimplemented as torch modules.

    Args:
        tokenizer ([`PreTrainedTokenizerBase`]):
            The text tokenizer.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.speech_tokenizer_session = None
        self.campplus_session = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        onnx_providers: list[str] | None = None,
        speech_tokenizer_filename: str = "speech_tokenizer_v1.onnx",
        campplus_filename: str = "campplus.onnx",
        **kwargs,
    ) -> "CosyVoiceV1Processor":
        r"""
        Loads the text tokenizer as usual, then additionally loads `speech_tokenizer_v1.onnx` and
        `campplus.onnx` from the same `pretrained_model_name_or_path` as `onnxruntime.InferenceSession`s so
        `extract_speech_token`/`extract_speaker_embedding` work out of the box.

        Args:
            onnx_providers (`list[str]`, *optional*):
                `onnxruntime` execution providers to request, in priority order. Defaults to
                `["CUDAExecutionProvider", "CPUExecutionProvider"]`, falling back automatically to whatever
                `onnxruntime` actually has available.
        """
        import onnxruntime

        processor = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        cache_kwargs = {k: v for k, v in kwargs.items() if k in ("cache_dir", "force_download", "proxies", "token", "revision", "local_files_only", "subfolder")}
        speech_tokenizer_path = cached_file(pretrained_model_name_or_path, speech_tokenizer_filename, **cache_kwargs)
        campplus_path = cached_file(pretrained_model_name_or_path, campplus_filename, **cache_kwargs)

        available = onnxruntime.get_available_providers()
        providers = [p for p in (onnx_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]) if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        processor.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_path, providers=providers)
        processor.campplus_session = onnxruntime.InferenceSession(campplus_path, providers=providers)
        return processor

    def __call__(self, text: str | list[str], **kwargs) -> BatchFeature:
        """
        Args:
            text (`str` or `list[str]`):
                Input text to tokenize.

        Returns:
            [`BatchFeature`] with `text_token` and `text_token_len`.
        """
        kwargs.setdefault("return_tensors", "pt")
        encoded = self.tokenizer(text, **kwargs)
        lengths = encoded["attention_mask"].sum(dim=-1) if "attention_mask" in encoded else torch.tensor([encoded["input_ids"].shape[-1]])
        return BatchFeature(data={"text_token": encoded["input_ids"], "text_token_len": lengths})

    def _resample_to_16k(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == 16000:
            return waveform
        import torchaudio

        return torchaudio.functional.resample(waveform, sample_rate, 16000)

    def extract_speech_token(self, waveform: torch.Tensor, sample_rate: int = 16000) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Extracts discrete speech tokens from a reference waveform via the real `speech_tokenizer_v1.onnx` model,
        for use as `prompt_speech_token` in [`~CosyVoiceV1ForConditionalGeneration.generate`].

        Args:
            waveform (`torch.FloatTensor` of shape `(num_samples,)` or `(1, num_samples)`):
                Mono reference waveform.
            sample_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of `waveform`; resampled to 16kHz if different.

        Returns:
            `tuple(torch.LongTensor, torch.LongTensor)`: `speech_token` of shape `(1, token_len)` and its length.
        """
        if self.speech_tokenizer_session is None:
            raise ValueError("No `speech_tokenizer_v1.onnx` session loaded; load this processor via `from_pretrained`.")
        waveform = waveform.reshape(-1)
        waveform = self._resample_to_16k(waveform, sample_rate)
        feats = _whisper_log_mel_spectrogram(waveform).unsqueeze(0).numpy().astype(np.float32)
        feats_length = np.array([feats.shape[-1]], dtype=np.int32)
        input_names = [inp.name for inp in self.speech_tokenizer_session.get_inputs()]
        indices = self.speech_tokenizer_session.run(None, {input_names[0]: feats, input_names[1]: feats_length})[0]
        speech_token = torch.from_numpy(indices).long().reshape(1, -1)
        speech_token_len = torch.tensor([speech_token.shape[-1]], dtype=torch.long)
        return speech_token, speech_token_len

    def extract_speaker_embedding(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        r"""
        Extracts a speaker embedding from a reference waveform via the real `campplus.onnx` model, for use as
        `prompt_spk_embedding` in [`~CosyVoiceV1ForConditionalGeneration.generate`].

        Args:
            waveform (`torch.FloatTensor` of shape `(num_samples,)` or `(1, num_samples)`):
                Mono reference waveform.
            sample_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of `waveform`; resampled to 16kHz if different.

        Returns:
            `torch.FloatTensor` of shape `(1, 192)`: The speaker embedding.
        """
        if self.campplus_session is None:
            raise ValueError("No `campplus.onnx` session loaded; load this processor via `from_pretrained`.")
        import torchaudio.compliance.kaldi as kaldi

        waveform = waveform.reshape(1, -1)
        waveform = self._resample_to_16k(waveform, sample_rate)
        feat = kaldi.fbank(waveform, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        feat = feat.unsqueeze(0).numpy().astype(np.float32)
        input_name = self.campplus_session.get_inputs()[0].name
        embedding = self.campplus_session.run(None, {input_name: feat})[0]
        return torch.from_numpy(embedding)

    def decode(self, waveform: torch.Tensor, sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
        """
        Args:
            waveform (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
                Waveform produced by [`~CosyVoiceV1ForConditionalGeneration.generate`].
            sample_rate (`int`, *optional*):
                Overrides the model's configured output sample rate.

        Returns:
            `tuple(torch.FloatTensor, int)`: The waveform and its sample rate.
        """
        return waveform, sample_rate or 22050


__all__ = ["CosyVoiceV1Processor"]
