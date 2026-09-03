"""Processor class for CosyVoice v1."""

from typing import Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from librosa.filters import mel as librosa_mel

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin


class CosyVoiceV1FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CosyVoice v1 feature extractor, which turns a waveform into the log mel spectrogram
    the flow matching model is conditioned on and trained against.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            Number of mel bins.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Rate the incoming waveform is resampled to before the mel spectrogram is taken.
        mel_sampling_rate (`int`, *optional*, defaults to 22050):
            Rate the mel filter bank is built for.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 256):
            Hop between two consecutive frames.
        win_length (`int`, *optional*, defaults to 1024):
            Size of the analysis window.
        fmin (`float`, *optional*, defaults to 0.0):
            Lowest frequency of the mel filter bank.
        fmax (`float`, *optional*, defaults to 8000.0):
            Highest frequency of the mel filter bank.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to pad batches of spectrograms.
    """

    model_input_names = ["speech_feat"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 24000,
        mel_sampling_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs
        )
        self.mel_sampling_rate = mel_sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.mel_filters = librosa_mel(
            sr=mel_sampling_rate, n_fft=n_fft, n_mels=feature_size, fmin=fmin, fmax=fmax
        )

    def _mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (`torch.Tensor` of shape `(batch_size, num_samples)`):
                Waveform sampled at `sampling_rate`.

        Returns:
            `torch.Tensor` of shape `(batch_size, feature_size, num_frames)`: the log mel spectrogram.
        """
        padding = int((self.n_fft - self.hop_length) / 2)
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=waveform.device),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spectrogram = torch.view_as_real(spectrogram)
        spectrogram = torch.sqrt(spectrogram.pow(2).sum(-1) + 1e-9)
        mel_filters = torch.from_numpy(self.mel_filters).to(spectrogram)
        return torch.log(torch.clamp(torch.matmul(mel_filters, spectrogram), min=1e-5))

    def __call__(
        self,
        raw_speech: Union[np.ndarray, torch.Tensor, list],
        sampling_rate: Optional[int] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            raw_speech (`np.ndarray`, `torch.Tensor` or `list`):
                Mono waveform of shape `(num_samples,)` or `(1, num_samples)`.
            sampling_rate (`int`, *optional*):
                Rate of `raw_speech`. It is resampled to `self.sampling_rate` when the two differ.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Only `"pt"` is supported.

        Returns:
            [`BatchFeature`]: `speech_feat` of shape `(1, num_frames, feature_size)` and
            `speech_feat_lengths` of shape `(1,)`.

        Raises:
            ValueError: If `return_tensors` is not `"pt"`.
        """
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports return_tensors='pt'")
        waveform = raw_speech if isinstance(raw_speech, torch.Tensor) else torch.as_tensor(np.asarray(raw_speech))
        waveform = waveform.reshape(1, -1).float()
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sampling_rate, self.sampling_rate)
        speech_feat = self._mel_spectrogram(waveform).transpose(1, 2)
        lengths = torch.tensor([speech_feat.shape[1]], dtype=torch.int32)
        return BatchFeature({"speech_feat": speech_feat, "speech_feat_lengths": lengths})


class CosyVoiceV1Processor(ProcessorMixin):
    r"""
    Constructs a CosyVoice v1 processor, which wraps a Whisper tokenizer, the mel spectrogram feature
    extractor of the flow matching model, the supervised semantic speech tokenizer and the speaker
    encoder into a single object.

    The speech tokenizer and the speaker encoder are the ONNX graphs shipped with the released model
    directory, `speech_tokenizer_v1.onnx` and `campplus.onnx`. Upstream publishes no PyTorch weights
    for either, so `onnxruntime` and both graph files are required to derive a prompt from a
    waveform. They are opened lazily, so text tokenization and the speaker table work without them.

    Args:
        feature_extractor ([`CosyVoiceV1FeatureExtractor`]):
            Mel spectrogram extractor of the flow matching model.
        tokenizer ([`WhisperTokenizer`]):
            Text tokenizer.
        speech_token_model_path (`str`, *optional*):
            Path of `speech_tokenizer_v1.onnx`.
        speaker_encoder_model_path (`str`, *optional*):
            Path of `campplus.onnx`.
        speaker_info_path (`str`, *optional*):
            Path of a `spk2info.pt`, the table of precomputed prompts the SFT and Instruct
            checkpoints ship. Read lazily by [`~CosyVoiceV1Processor.get_speaker`].
        speech_token_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel bins of the log mel spectrogram the speech tokenizer consumes.
    """

    speech_tokenizer_sampling_rate = 16000
    speaker_encoder_sampling_rate = 16000
    speaker_encoder_max_seconds = 10

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        speech_token_model_path: Optional[str] = None,
        speaker_encoder_model_path: Optional[str] = None,
        speaker_info_path: Optional[str] = None,
        speech_token_mel_bins: int = 128,
        **kwargs,
    ):
        super().__init__(feature_extractor, tokenizer, **kwargs)
        self.speech_token_model_path = speech_token_model_path
        self.speaker_encoder_model_path = speaker_encoder_model_path
        self.speaker_info_path = speaker_info_path
        self.speech_token_mel_bins = speech_token_mel_bins
        self._speech_token_features = None
        self._speech_tokenizer_session = None
        self._speaker_encoder_session = None
        self._speaker_info = None

    @property
    def speech_token_feature_extractor(self):
        """
        Returns:
            [`WhisperFeatureExtractor`]: The log mel extractor feeding the speech tokenizer.
        """
        if self._speech_token_features is None:
            from transformers import WhisperFeatureExtractor

            self._speech_token_features = WhisperFeatureExtractor(
                feature_size=self.speech_token_mel_bins, sampling_rate=self.speech_tokenizer_sampling_rate
            )
        return self._speech_token_features

    @property
    def speakers(self) -> list[str]:
        """
        Returns:
            `list[str]`: Names of the speakers `speaker_info_path` holds, empty when it is unset.
        """
        if self.speaker_info_path is None:
            return []
        if self._speaker_info is None:
            self._speaker_info = torch.load(self.speaker_info_path, map_location="cpu", weights_only=True)
        return list(self._speaker_info)

    def get_speaker(self, name: str) -> BatchFeature:
        """
        Reads one precomputed prompt out of `speaker_info_path`.

        Args:
            name (`str`):
                Name of the speaker, one of [`~CosyVoiceV1Processor.speakers`].

        Returns:
            [`BatchFeature`]: `speaker_embedding`, and `prompt_speech_token_ids` plus `speech_feat`
            when the table carries them.

        Raises:
            ValueError: If no speaker table is configured or `name` is not in it.
        """
        if not self.speakers:
            raise ValueError("this processor has no `speaker_info_path`, so it has no speaker table")
        if name not in self._speaker_info:
            raise ValueError(f"{name} is not one of {list(self._speaker_info)}")
        entry = self._speaker_info[name]
        data = {"speaker_embedding": entry["embedding"]}
        if "speech_token" in entry:
            data["prompt_speech_token_ids"] = entry["speech_token"].to(torch.int32)
            data["prompt_speech_token_lengths"] = torch.tensor(
                [entry["speech_token"].shape[1]], dtype=torch.int32
            )
        if "speech_feat" in entry:
            data["speech_feat"] = entry["speech_feat"]
            data["speech_feat_lengths"] = torch.tensor([entry["speech_feat"].shape[1]], dtype=torch.int32)
        return BatchFeature(data)

    @staticmethod
    def _open_onnx_session(path: str, providers: list[str]):
        """
        Args:
            path (`str`):
                Path of the ONNX graph.
            providers (`list[str]`):
                Execution providers, in order of preference.

        Returns:
            `onnxruntime.InferenceSession`: the opened session.

        Raises:
            ImportError: If `onnxruntime` is not installed.
        """
        try:
            import onnxruntime
        except ImportError as error:
            raise ImportError(
                "the CosyVoice v1 speech tokenizer and speaker encoder are published as ONNX graphs "
                "only, so deriving a prompt from a waveform needs `onnxruntime`, which this package "
                "does not depend on. Use `get_speaker` with a `spk2info.pt` instead, or install "
                "`onnxruntime` yourself."
            ) from error
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        return onnxruntime.InferenceSession(path, sess_options=options, providers=providers)

    def _resample(self, waveform: torch.Tensor, sampling_rate: Optional[int], target_rate: int) -> torch.Tensor:
        waveform = waveform.reshape(1, -1).float()
        if sampling_rate is not None and sampling_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, sampling_rate, target_rate)
        return waveform

    def encode_speech_tokens(
        self, audio: Union[np.ndarray, torch.Tensor], sampling_rate: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Turns a waveform into supervised semantic speech tokens.

        Args:
            audio (`np.ndarray` or `torch.Tensor`):
                Mono waveform.
            sampling_rate (`int`, *optional*):
                Rate of `audio`.

        Returns:
            `tuple(torch.Tensor)`: the speech token ids of shape `(1, speech_length)` and their length.

        Raises:
            ValueError: If the waveform is longer than thirty seconds.
        """
        if self._speech_tokenizer_session is None:
            providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            self._speech_tokenizer_session = self._open_onnx_session(self.speech_token_model_path, providers)
        waveform = self._resample(
            audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio)),
            sampling_rate,
            self.speech_tokenizer_sampling_rate,
        )
        if waveform.shape[1] / self.speech_tokenizer_sampling_rate > 30:
            raise ValueError("the CosyVoice v1 speech tokenizer does not support audio longer than 30 seconds")
        features = self.speech_token_feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.speech_tokenizer_sampling_rate,
            padding=False,
            return_tensors="np",
        ).input_features
        inputs = self._speech_tokenizer_session.get_inputs()
        speech_token = self._speech_tokenizer_session.run(
            None,
            {
                inputs[0].name: features.astype(np.float32),
                inputs[1].name: np.array([features.shape[2]], dtype=np.int32),
            },
        )[0].flatten()
        speech_token_ids = torch.tensor([speech_token], dtype=torch.int32)
        return speech_token_ids, torch.tensor([speech_token_ids.shape[1]], dtype=torch.int32)

    def encode_speaker(
        self, audio: Union[np.ndarray, torch.Tensor], sampling_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Turns a waveform into an utterance level speaker embedding.

        Args:
            audio (`np.ndarray` or `torch.Tensor`):
                Mono waveform.
            sampling_rate (`int`, *optional*):
                Rate of `audio`.

        Returns:
            `torch.Tensor` of shape `(1, speaker_embedding_dim)`: the speaker embedding.
        """
        if self._speaker_encoder_session is None:
            self._speaker_encoder_session = self._open_onnx_session(
                self.speaker_encoder_model_path, ["CPUExecutionProvider"]
            )
        waveform = self._resample(
            audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio)),
            sampling_rate,
            self.speaker_encoder_sampling_rate,
        )
        features = kaldi.fbank(
            waveform, num_mel_bins=80, dither=0, sample_frequency=self.speaker_encoder_sampling_rate
        )
        features = features - features.mean(dim=0, keepdim=True)
        embedding = self._speaker_encoder_session.run(
            None, {self._speaker_encoder_session.get_inputs()[0].name: features.unsqueeze(dim=0).numpy()}
        )[0].flatten()
        return torch.tensor([embedding])

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sampling_rate: Optional[int] = None,
        prompt_text: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            text (`str` or `list[str]`, *optional*):
                Text to synthesize.
            audio (`np.ndarray` or `torch.Tensor`, *optional*):
                Mono waveform of the prompt utterance, turned into speech tokens, a mel spectrogram and
                a speaker embedding.
            sampling_rate (`int`, *optional*):
                Rate of `audio`.
            prompt_text (`str` or `list[str]`, *optional*):
                Transcript of the prompt utterance.

        Returns:
            [`BatchFeature`]: `input_ids` and `input_lengths` for the text, plus
            `prompt_input_ids`, `prompt_speech_token_ids`, `prompt_speech_token_lengths`,
            `speech_feat`, `speech_feat_lengths` and `speaker_embedding` when a prompt is given.
        """
        data = {}
        if text is not None:
            encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt", **kwargs)
            data["input_ids"] = encoded["input_ids"].to(torch.int32)
            data["input_lengths"] = torch.tensor([data["input_ids"].shape[1]], dtype=torch.int32)
        if prompt_text is not None:
            encoded = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt", **kwargs)
            data["prompt_input_ids"] = encoded["input_ids"].to(torch.int32)
            data["prompt_input_lengths"] = torch.tensor([data["prompt_input_ids"].shape[1]], dtype=torch.int32)
        if audio is not None:
            speech_token_ids, speech_token_lengths = self.encode_speech_tokens(audio, sampling_rate)
            data["prompt_speech_token_ids"] = speech_token_ids
            data["prompt_speech_token_lengths"] = speech_token_lengths
            data.update(self.feature_extractor(audio, sampling_rate=sampling_rate))
            data["speaker_embedding"] = self.encode_speaker(audio, sampling_rate)
        return BatchFeature(data)


__all__ = ["CosyVoiceV1FeatureExtractor", "CosyVoiceV1Processor"]
