from typing import Optional, Union, Any, Literal, Unpack
from dataclasses import dataclass
import os

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.processing_utils import ImageInput, TextInput, VideoInput, AudioInput, PreTokenizedInput
from transformers import AutoConfig, AutoModel, AutoFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2 import Qwen2Tokenizer

import numpy as np
import librosa

try:
    from ..._qwen3_tts.core.models.processing_qwen3_tts import (
        Qwen3TTSProcessor as _Qwen3TTSProcessor, Qwen3TTSProcessorKwargs as _Qwen3TTSProcessorKwargs
    )
    from ..._qwen3_tts.core import Qwen3TTSTokenizerV1Model, Qwen3TTSTokenizerV2Model
    from ..._qwen3_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer as _Qwen3TTSTokenizer
    from ..._qwen3_tts.inference.qwen3_tts_model import Qwen3TTSModel as _Qwen3TTSModel
except ImportError:
    from voicestudio._qwen3_tts.core.models.processing_qwen3_tts import (
        Qwen3TTSProcessor as _Qwen3TTSProcessor, Qwen3TTSProcessorKwargs as _Qwen3TTSProcessorKwargs
    )
    from voicestudio._qwen3_tts.core import Qwen3TTSTokenizerV1Model, Qwen3TTSTokenizerV2Model
    from voicestudio._qwen3_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer as _Qwen3TTSTokenizer
    from voicestudio._qwen3_tts.inference.qwen3_tts_model import Qwen3TTSModel as _Qwen3TTSModel

from .configuration_qwen3_tts import Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV2Config


AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)

AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)


AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    tuple[np.ndarray, int],  # (waveform, sr)
]


@dataclass
class VoiceClonePrompt:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """
    prompt_code: Optional[torch.Tensor]                 # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    prompt_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    prompt_text: Optional[str] = None


class Qwen3TTSProcessorKwargs(_Qwen3TTSProcessorKwargs, total=False):
    """Extended kwargs for Qwen3TTS processor with audio support."""
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "audio_kwargs": {}
    }


class Qwen3TTSProcessor(_Qwen3TTSProcessor):
    """
    Qwen3TTS multi-modal processor with text_tokenizer and audio_tokenizer

    This processor combines:
    - tokenizer: Text tokenization (Qwen2 Tokenizer)
    - feature_extractor: Audio preprocessing
    - audio_tokenizer: Audio neural codec model (special handling with weights)

    The audio_tokenizer is treated as a special component with model weights,
    following ProcessorMixin's pattern for such components.

    Args:
        tokenizer ([`Qwen2Tokenizer`], *optional*):
            The text tokenizer.
        feature_extractor ([`FeatureExtractor`], *optional*):
            The audio feature extractor for preprocessing audio inputs.
        audio_tokenizer ([`PreTrainedModel`], *optional*):
            The audio tokenizer model with weights (Qwen3TTSTokenizerV1Model or V2Model).
        chat_template (`str`, *optional*):
            The Jinja template for formatting conversations.
    """
    attributes = ["tokenizer", "feature_extractor"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    feature_extractor_class = "AutoFeatureExtractor"
    valid_processor_kwargs = Qwen3TTSProcessorKwargs

    _auto_class = "AutoProcessor"

    def __init__(self, tokenizer=None, feature_extractor=None, audio_tokenizer=None, chat_template=None):
        """
        Initialize processor with text tokenizer, audio feature extractor, and audio tokenizer model.

        audio_tokenizer is handled specially by ProcessorMixin's __init__ (not part of attributes).
        """
        # ProcessorMixin.__init__ will handle audio_tokenizer specially
        self.tokenizer: Qwen2Tokenizer
        self.feature_extractor: AutoFeatureExtractor
        self.audio_tokenizer: Qwen3TTSTokenizerV1Model | Qwen3TTSTokenizerV2Model
        super(_Qwen3TTSProcessor, self).__init__(
            tokenizer,
            feature_extractor,
            audio_tokenizer=audio_tokenizer,
            chat_template=chat_template
        )

    @property
    def device(self) -> torch.device:
        """Get device from audio_tokenizer."""
        device = getattr(self.audio_tokenizer, "device", None)
        if device is None:
            try:
                device = next(self.audio_tokenizer.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        fix_mistral_regex: bool = True,
        **kwargs,
    ):
        """
        Load Qwen3TTS processor with text tokenizer, feature extractor, and audio tokenizer.

        Returns:
            Qwen3TTSProcessor: Initialized processor instance.
        """
        kwargs = dict(
            cache_dir=cache_dir, force_download=force_download,
            local_files_only=local_files_only, token=token, revision=revision,
            fix_mistral_regex=fix_mistral_regex
        )

        # If processor_config.json has audio_tokenizer info, it will be autoloaded
        processor = super(_Qwen3TTSProcessor, cls).from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # If audio_tokenizer wasn't loaded (first time from Qwen repo), load it manually
        if not hasattr(processor, "audio_tokenizer") or processor.audio_tokenizer is None:
            model_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ["device_map", "torch_dtype", "attn_implementation", "trust_remote_code"]
            }
            processor.audio_tokenizer = AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                **model_kwargs
            )

        # If feature_extractor wasn't loaded, load it manually
        if not hasattr(processor, "feature_extractor") or processor.feature_extractor is None:
            processor.feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path
            )

        return processor

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save processor components following transformers standards.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory to save processor components.
            **kwargs:
                Additional arguments passed to parent's save_pretrained.
        """
        # Parent saves tokenizer, feature_extractor config, processor_config.json
        output_files = super().save_pretrained(save_directory, **kwargs)

        # audio_tokenizer is not in attributes, so we must save it manually
        if hasattr(self, "audio_tokenizer") and self.audio_tokenizer is not None:
            self.audio_tokenizer.save_pretrained(save_directory)

        return output_files

    @property
    def model_input_names(self):
        """Combine input names from all sub-processors."""
        names = list(super().model_input_names)

        if hasattr(self, "feature_extractor"):
            names.extend(["input_values", "padding_mask"])

        if hasattr(self, "audio_tokenizer"):
            names.extend(["audio_codes", "xvectors", "ref_mels"])

        return list(dict.fromkeys(names))

    def get_model_type(self) -> str:
        """
        Get the underlying tokenizer model type.

        Returns:
            str: Model type string from `self.model.config.model_type`
                (e.g. "qwen3_tts_tokenizer_25hz" / "qwen3_tts_tokenizer_12hz").
        """
        return self.audio_tokenizer.get_model_type()

    def get_input_sample_rate(self) -> int:
        """
        Get the expected input sample rate for encoding.

        Returns:
            int: Input sample rate (Hz).
        """
        return int(self.audio_tokenizer.get_input_sample_rate())

    def get_output_sample_rate(self) -> int:
        """
        Get the output sample rate for decoded waveforms.

        Returns:
            int: Output sample rate (Hz).
        """
        return int(self.audio_tokenizer.get_output_sample_rate())

    def get_encode_downsample_rate(self) -> int:
        """
        Get the encoder downsample rate (waveform samples per code step).

        Returns:
            int: Encode downsample rate.
        """
        return int(self.audio_tokenizer.get_encode_downsample_rate())

    def get_decode_upsample_rate(self) -> int:
        """
        Get the decoder upsample rate (waveform samples per code step).

        Returns:
            int: Decode upsample rate.
        """
        return int(self.audio_tokenizer.get_decode_upsample_rate())

    def encode_voice_clone(
        self,
        text: Union[str, list[str]],
        instruct: Union[dict[str, Any], list[VoiceClonePrompt]] | None = None,
        language: Union[str, list[str]] = None,
        prompt_audio: Union[AudioLike, list[AudioLike]] | None = None,
        prompt_text: Union[str, list[Optional[str]]] | None = None,
        x_vector_only_mode: Union[bool, list[bool]] = False,
        sampling_rate: int | None = None,
        return_tensors: Literal["pt", "np"] = "pt",
    ):
        """Encoding parameter guide for voice cloning task"""
        if instruct is None and prompt_audio is None:
            raise ValueError("You need to specify either `instruct` or `prompt_audio` input.")
        elif instruct is None and prompt_audio is not None:
            return self(text=text, audio=prompt_audio, language=language, prompt_text=prompt_text, x_vector_only_mode=x_vector_only_mode, sampling_rate=sampling_rate, return_tensors=return_tensors)
        else:
            return self.encode(text=text, instruct=instruct, language=language, return_tensors=return_tensors)

    def encode_custom_voice(
        self,
        text: Union[str, list[str]],
        speaker: Union[str, list[str]],
        instruct: Union[str, list[str]],
        language: Union[str, list[str]] = None,
        return_tensors: Literal["pt", "np"] = "pt",
    ):
        """Encoding parameter guide for voice editing task"""
        return self.encode(text=text, speaker=speaker, language=language, instruct=instruct, return_tensors=return_tensors)

    def encode_voice_design(
        self,
        text: Union[str, list[str]],
        instruct: Union[str, list[str]],
        language: Union[str, list[str]] = None,
        return_tensors: Literal["pt", "np"] = "pt",
    ):
        """Encoding parameter guide for voice design task"""
        return self.encode(text=text, speaker=speaker, language=language, instruct=instruct, return_tensors=return_tensors)

    def encode(
        self,
        text: Union[str, list[str]],
        speaker: Union[str, list[str]] | None = None,
        instruct: Union[str, list[str], dict[str, Any], list[VoiceClonePrompt]] | None = None,
        language: Union[str, list[str]] = None,
        return_tensors: Literal["pt", "np"] = "pt",
    ):
        """Encoding parameter guide for text-to-speech task"""
        return self(text=text, speaker=speaker, language=language, instruct=instruct, return_tensors=return_tensors)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        return_tensors: Literal["pt", "np"] = "pt",
        **kwargs: Unpack[Qwen3TTSProcessorKwargs],
    ) -> BatchFeature:
        """
        Process text and/or audio inputs.

        Args:
            text (`str`, `List[str]`, *optional*):
                Text input(s) for tokenization.
            audio (`str`, `np.ndarray`, `List[str]`, `List[np.ndarray]`, *optional*):
                Audio input(s). Can be:
                - str: wav file path or base64 audio string
                - np.ndarray: raw waveform (requires `sr` parameter)
                - List of above types
            **kwargs:
                Additional arguments passed to tokenizer and feature_extractor.
                sampling_rate (`int`, *optional*):
                    Sampling rate for numpy array inputs. Required when audio is np.ndarray.

        Returns:
            [`BatchFeature`]: Processed inputs containing text and/or audio features.
        """
        speaker = kwargs.pop("speaker", None)
        language = kwargs.pop("language", None)
        instruct = kwargs.pop("instruct", None)

        # Validate unsupported modalities
        if images is not None or videos is not None:
            raise ValueError(f"{self.__class__.__name__} does not support image or video inputs.")

        if text is None and audio is None:
            raise ValueError("You need to specify either `text` or `audio` input.")

        output_kwargs = self._merge_kwargs(
            Qwen3TTSProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )

        # Input validation and normalization
        #
        ## Text is required for every task
        texts = self._ensure_list(text)
        #
        ## Language is optional, default to "Auto"
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")
        #
        ## Voice clone prompt is required for voice cloning task
        ## but can be constructed from prompt_audio(audio) and prompt_text
        voice_clone_prompt = instruct if isinstance(instruct, list) and isinstance(instruct[0], VoiceClonePrompt) else None
        style_prompt = instruct if voice_clone_prompt is None else None
        if voice_clone_prompt is None:
            prompt_audio = kwargs.pop("prompt_audio", audio)  # use audio as default prompt_audio
            prompt_text = kwargs.pop("prompt_text", None)
            x_vector_only_mode = kwargs.pop("x_vector_only_mode", False)
            sampling_rate = kwargs.pop("sampling_rate", None)

            if prompt_audio is None:  # need to check if it is voice cloning task
                if speaker is not None:
                    pass  # then this is voice editing task
                elif style_prompt is not None:
                    pass  # then this is voice design task
                else:
                    raise ValueError("You need to specify either `voice_clone_prompt` or `prompt_audio` input.")
            voice_clone_prompt = self.create_voice_clone_prompt(
                prompt_audio=prompt_audio, prompt_text=prompt_text,
                x_vector_only_mode=x_vector_only_mode, sampling_rate=sampling_rate,
                return_tensors=return_tensors, **kwargs
            )
        if voice_clone_prompt is not None:
            voice_clone_prompt = self._ensure_list(voice_clone_prompt)
            if len(voice_clone_prompt) == 1 and len(texts) > 1:
                voice_clone_prompt = voice_clone_prompt * len(texts)
            if len(voice_clone_prompt) != len(texts):
                raise ValueError(f"Batch size mismatch: voice_clone_prompt={len(voice_clone_prompt)}, text={len(texts)}")
        #
        ## Style prompt will be used for voice design task and voice editing task
        if style_prompt is not None:
            style_prompt = self._ensure_list(style_prompt) if isinstance(style_prompt, list) else ([style_prompt] * len(texts) if style_prompt is not None else [""] * len(texts))
            if len(style_prompt) == 1 and len(texts) > 1:
                style_prompt = style_prompt * len(texts)
            if len(style_prompt) != len(texts):
                raise ValueError(f"Batch size mismatch: style_prompt={len(style_prompt)}, text={len(texts)}")
        #
        ## Speaker will be used for voice editing task
        if speakers is not None:
            speakers = self._ensure_list(speaker)
            if len(speakers) == 1 and len(texts) > 1:
                speakers = speakers * len(texts)
            if len(speakers) != len(texts):
                raise ValueError(f"Batch size mismatch: speaker={len(speakers)}, text={len(texts)}")

        outputs = {}

        # Process text
        texts = [self._build_assistant_text(t) for t in texts]
        text_inputs = self.tokenizer(texts, return_tensors=return_tensors, **output_kwargs["text_kwargs"])[0]
        outputs['input_ids'] = text_inputs['input_ids']

        # Process cloning prompt
        if voice_clone_prompt:
            voice_clone_prompt_dict = dict(
                ref_code=[it.prompt_code for it in voice_clone_prompt],
                ref_spk_embedding=[it.prompt_spk_embedding for it in voice_clone_prompt],
                x_vector_only_mode=[it.x_vector_only_mode for it in voice_clone_prompt],
                icl_mode=[it.icl_mode for it in voice_clone_prompt],
            )
            prompt_texts = [it.prompt_text for it in voice_clone_prompt]
            ref_ids = []
            for i, rt in enumerate(prompt_texts):
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self.tokenizer([self._build_ref_text(rt)], return_tensors=return_tensors, **output_kwargs["text_kwargs"])[0]
                    ref_ids.append(ref_tok)
            outputs['voice_clone_prompt'] = voice_clone_prompt_dict
            outputs['ref_ids'] = ref_ids

        # Process style prompt
        if style_prompt:
            instruct_ids: List[Optional[torch.Tensor]] = []
            for ins in style_prompt:
                if ins is None or ins == "":
                    instruct_ids.append(None)
                else:
                    instruct_ids.append(self.tokenizer([self._build_instruct_text(ins)], return_tensors=return_tensors, **output_kwargs["text_kwargs"])[0])
            outputs['instruct_ids'] = instruct_ids

        # Additional args
        if languages:
            outputs['languages'] = languages
        if speakers:
            outputs['speakers'] = speakers

        return BatchFeature(
            **outputs,
            tensor_type=return_tensors,
        )

    def create_voice_clone_prompt(
        self,
        prompt_audio: Union[AudioLike, list[AudioLike]] | None = None,
        prompt_text: Union[str, list[Optional[str]]] | None = None,
        x_vector_only_mode: Union[bool, list[bool]] = False,
        sampling_rate: int | None = None,
        return_tensors: Literal["pt", "np"] = "pt",
        **kwargs: Unpack[Qwen3TTSProcessorKwargs]
    ) -> list[VoiceClonePrompt]:
        sampling_rate = self.audio_tokenizer.sampling_rate if sampling_rate is None else sampling_rate
        prompt_audio_list = self._ensure_list(prompt_audio)
        prompt_text_list = self._ensure_list(prompt_text) if isinstance(prompt_text, list) else ([prompt_text] * len(prompt_audio_list))
        x_vector_list = self._ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(prompt_audio_list))

        if len(prompt_text_list) != len(prompt_audio_list) or len(x_vector_list) != len(prompt_audio_list):
            raise ValueError(
                f"Batch size mismatch: prompt_audio={len(prompt_audio_list)}, prompt_text={len(prompt_text_list)}, x_vector_only_mode={len(x_vector_list)}"
            )

        # Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        normalized: List[Tuple[np.ndarray, int]] = []
        for a in prompt_audio_list:
            if isinstance(a, str):
                normalized.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                normalized.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        for i, a in enumerate(normalized):
            if a[0].ndim > 1:
                a[0] = np.mean(a[0], axis=-1).astype(np.float32)
                normalized[i] = (a[0], a[1])

        prompt_wavs_for_code: list[np.ndarray] = []
        prompt_sr_for_code: list[int] = []
        for wav, sr in normalized:
            prompt_wavs_for_code.append(wav)
            prompt_sr_for_code.append(sr)

        prompt_codes = []
        for wav, sr in ([(prompt_wavs_for_code, prompt_sr_for_code[0])] if len(set(prompt_sr_for_code)) == 1 else normalized):
            # Normalize audio inputs (handles paths, base64, numpy arrays)
            audio_list = self._normalize_audio_inputs(wav, sr)

            # Use feature_extractor to preprocess
            feature_inputs = self.feature_extractor(
                raw_audio=audio_list,
                sampling_rate=int(self.feature_extractor.sampling_rate),
                return_tensors=return_tensors,
                **kwargs.get("audio_kwargs", {})
            )

            # Move to model device and dtype
            feature_inputs = feature_inputs.to(self.device).to(self.audio_tokenizer.dtype)

            # Encode with audio_tokenizer model
            with torch.inference_mode():
                # audio_tokenizer.encode expects (B, T) and (B, T)
                audio_outputs = self.audio_tokenizer.encode(
                    feature_inputs["input_values"].squeeze(1),
                    feature_inputs["padding_mask"].squeeze(1),
                    return_dict=True,
                )
                audio_codes = audio_outputs.audio_codes
            prompt_codes.append(audio_codes if len(set(prompt_sr_for_code)) == 1 else audio_codes[0])

        items: list[VoiceClonePrompt] = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, prompt_codes, prompt_text_list, x_vector_list)):
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(f"prompt_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            wav_resample = wav
            if sr != self.audio_tokenizer.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(
                    y=wav_resample.astype(np.float32), 
                    orig_sr=int(sr), 
                    target_sr=self.audio_tokenizer.speaker_encoder_sample_rate
                )

            spk_emb = self.audio_tokenizer.extract_speaker_embedding(
                audio=wav_resample,
                sr=self.audio_tokenizer.speaker_encoder_sample_rate
            )

            items.append(VoiceClonePrompt(
                prompt_code=None if xvec_only else code,
                prompt_spk_embedding=spk_emb,
                x_vector_only_mode=bool(xvec_only),
                icl_mode=bool(not xvec_only),
                prompt_text=rtext,
            ))
        return items

    def batch_decode(self):
        pass

    def decode(self, encoded) -> tuple[list[np.ndarray], int]:
        """
        Decode audio codes back to waveforms.

        Args:
            encoded:
                Can be:
                - ModelOutput from encode() with audio_codes (and xvectors/ref_mels for 25Hz)
                - dict with same fields
                - list[dict] for batch

        Returns:
            Tuple[List[np.ndarray], int]:
                - List of decoded waveforms (float32 numpy arrays)
                - Output sampling rate
        """
        model_type = self.audio_tokenizer.get_model_type()
        device = self.device

        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x
            x = np.asarray(x)
            t = torch.from_numpy(x)
            if dtype is not None:
                t = t.to(dtype)
            return t

        # Extract fields
        if hasattr(encoded, "audio_codes"):
            audio_codes_list = encoded.audio_codes
            xvectors_list = getattr(encoded, "xvectors", None)
            ref_mels_list = getattr(encoded, "ref_mels", None)
        elif isinstance(encoded, dict):
            audio_codes_list = encoded["audio_codes"]
            xvectors_list = encoded.get("xvectors", None)
            ref_mels_list = encoded.get("ref_mels", None)
        elif isinstance(encoded, list):
            audio_codes_list = [e["audio_codes"] for e in encoded]
            xvectors_list = [e["xvectors"] for e in encoded] if ("xvectors" in encoded[0]) else None
            ref_mels_list = [e["ref_mels"] for e in encoded] if ("ref_mels" in encoded[0]) else None
        else:
            raise TypeError("`encoded` must be an encode output, a dict, or a list of dicts.")

        # Prepare audio_codes
        if isinstance(audio_codes_list, torch.Tensor):
            # Could be a single sample tensor or an already padded batch tensor.
            t = audio_codes_list
            if t.dim() == 1:
                # 25Hz single sample: (C,) -> (1, C)
                t = t.unsqueeze(0)
            elif t.dim() == 2:
                # 12Hz single sample: (C, Q) -> (1, C, Q)
                t = t.unsqueeze(0)
            audio_codes_padded = t.to(device)
        else:
            audio_codes_list = [_to_tensor(c, dtype=torch.long) for c in audio_codes_list]
            audio_codes_padded = pad_sequence(audio_codes_list, batch_first=True, padding_value=0).to(device)

        with torch.inference_mode():
            if model_type == "qwen3_tts_tokenizer_25hz":
                if xvectors_list is None or ref_mels_list is None:
                    raise ValueError("25Hz decode requires `xvectors` and `ref_mels`.")

                # Prepare xvectors
                if isinstance(xvectors_list, torch.Tensor):
                    xvectors_batch = xvectors_list
                    if xvectors_batch.dim() == 1:  # (D,) -> (1, D)
                        xvectors_batch = xvectors_batch.unsqueeze(0)
                    xvectors_batch = xvectors_batch.to(device).to(self.audio_tokenizer.dtype)
                else:
                    xvectors_list = [_to_tensor(x, dtype=torch.float32) for x in xvectors_list]
                    xvectors_batch = torch.stack(xvectors_list, dim=0).to(device).to(self.audio_tokenizer.dtype)

                # Prepare ref_mels
                if isinstance(ref_mels_list, torch.Tensor):
                    ref_mels_padded = ref_mels_list
                    if ref_mels_padded.dim() == 2:  # (T, M) -> (1, T, M)
                        ref_mels_padded = ref_mels_padded.unsqueeze(0)
                    ref_mels_padded = ref_mels_padded.to(device).to(self.audio_tokenizer.dtype)
                else:
                    ref_mels_list = [_to_tensor(m, dtype=torch.float32) for m in ref_mels_list]
                    ref_mels_padded = pad_sequence(ref_mels_list, batch_first=True, padding_value=0).to(device).to(self.audio_tokenizer.dtype)

                dec = self.audio_tokenizer.decode(audio_codes_padded, xvectors_batch, ref_mels_padded, return_dict=True)
                wav_tensors = dec.audio_values

            elif model_type == "qwen3_tts_tokenizer_12hz":
                dec = self.audio_tokenizer.decode(audio_codes_padded, return_dict=True)
                wav_tensors = dec.audio_values

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
        output_sr = int(self.audio_tokenizer.get_output_sample_rate())

        return wavs, output_sr

    def _ensure_list(self, x: list | Any) -> list[Any]:
        return _Qwen3TTSModel._ensure_list(self, x)

    def _build_assistant_text(self, text: str) -> str:
        return _Qwen3TTSModel._build_assistant_text(self, text)

    def _build_ref_text(self, text: str) -> str:
        return _Qwen3TTSModel._build_ref_text(self, text)

    def _build_instruct_text(self, instruct: str) -> str:
        return _Qwen3TTSModel._build_instruct_text(self, instruct)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        return _Qwen3TTSTokenizer._load_audio_to_np(self, x)

    def load_audio(
        self,
        x: str,
        target_sr: int,
    ) -> np.ndarray:
        return _Qwen3TTSTokenizer.load_audio(self, x, target_sr)

    def _is_probably_base64(self, s: str) -> bool:
        return _Qwen3TTSTokenizer._is_probably_base64(self, s)

    def _is_url(self, s: str) -> bool:
        return _Qwen3TTSTokenizer._is_url(self, s)

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        return _Qwen3TTSTokenizer._decode_base64_to_wav_bytes(self, b64)

    def _normalize_audio_inputs(
        self,
        audios: AudioInput,
        sr: Optional[int],
    ) -> list[np.ndarray]:
        return _Qwen3TTSTokenizer._normalize_audio_inputs(self, audios, sr)
