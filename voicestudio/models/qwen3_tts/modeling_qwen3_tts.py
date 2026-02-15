from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

try:
    from ..._qwen3_tts.inference.qwen3_tts_model import AudioLike, VoiceClonePromptItem, Qwen3TTSModel as _Qwen3TTSModel
    from ..._qwen3_tts.core.models import modeling_qwen3_tts
    from ..._qwen3_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSPreTrainedModel,
        Qwen3TTSTalkerTextPreTrainedModel,
        Qwen3TTSTalkerCodePredictorModel,
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        Qwen3TTSTalkerModel,
        Qwen3TTSTalkerForConditionalGeneration,
        Qwen3TTSForConditionalGeneration as _Qwen3TTSForConditionalGeneration,
    )
except ImportError:
    from voicestudio._qwen3_tts.inference.qwen3_tts_model import AudioLike, VoiceClonePromptItem, Qwen3TTSModel as _Qwen3TTSModel
    from voicestudio._qwen3_tts.core.models import modeling_qwen3_tts
    from voicestudio._qwen3_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSPreTrainedModel,
        Qwen3TTSTalkerTextPreTrainedModel,
        Qwen3TTSTalkerCodePredictorModel,
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        Qwen3TTSTalkerModel,
        Qwen3TTSTalkerForConditionalGeneration,
        Qwen3TTSForConditionalGeneration as _Qwen3TTSForConditionalGeneration,
    )

from transformers import logging
from .processing_qwen3_tts import Qwen3TTSProcessor


logger = logging.get_logger(__name__)


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        pass


modeling_qwen3_tts.Qwen3TTSTokenizer = DummyTokenizer


class Qwen3TTSForConditionalGeneration(_Qwen3TTSForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_speech_tokenizer(self, *args, **kwargs):
        pass

    def _supported_languages_set(self) -> Optional[set]:
        """Get set of supported languages from parent model."""
        v = self.get_supported_languages()
        if v is None:
            return None
        return set([str(x).lower() for x in v])

    def _supported_speakers_set(self) -> Optional[set]:
        """Get set of supported speakers from parent model."""
        v = self.get_supported_speakers()
        if v is None:
            return None
        return set([str(x).lower() for x in v])

    def _validate_languages(self, languages: list[str]) -> None:
        """
        Validate that requested languages are supported by the model.

        Args:
            languages: Language names for each sample.

        Raises:
            ValueError: If any language is not supported.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad = []
        for lang in languages:
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: list[Optional[str]]) -> None:
        """
        Validate that requested speakers are supported by the model.

        Args:
            speakers: Speaker names for each sample.

        Raises:
            ValueError: If any speaker is not supported.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge user-provided generation arguments with defaults from generate_config.

        Rule:
          - If the user explicitly passes a value (not None), use it.
          - Otherwise, use the value from generate_config if present.
          - Otherwise, fall back to hard defaults.

        Args:
            do_sample, top_k, top_p, temperature, repetition_penalty,
            subtalker_dosample, subtalker_top_k, subtalker_top_p, 
            subtalker_temperature, max_new_tokens: Common generation parameters.
            **kwargs: Other arguments forwarded to generate().

        Returns:
            Dict with final kwargs to pass to parent generate().
        """
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )

        generate_defaults = getattr(self, "generate_config", None) or {}

        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in generate_defaults:
                return generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    # voice clone model
    @torch.no_grad()
    def generate_voice_clone(
        self,
        input_ids: list[torch.Tensor],
        voice_clone_prompt: dict,
        ref_ids: Optional[list[torch.Tensor]] = None,
        languages: Optional[list[str]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Any]:
        """
        Voice Clone task-specific generate method.

        Args:
            input_ids: Text input IDs
            voice_clone_prompt: Voice clone prompt dict from processor
            ref_ids: Reference text IDs (for ICL mode)
            languages: Language for each sample
            non_streaming_mode: Whether to use non-streaming mode
            **kwargs: Additional generate parameters

        Returns:
            Tuple[List[Tensor], Any]: (talker_codes, generation_info)
        """
        return self.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **kwargs,
        )

    # custom voice model
    @torch.no_grad()
    def generate_custom_voice(
        self,
        input_ids: list[torch.Tensor],
        speakers: list[str],
        instruct_ids: Optional[list[torch.Tensor]] = None,
        languages: Optional[list[str]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Any]:
        """
        Custom Voice task-specific generate method.

        Args:
            input_ids: Text input IDs
            speakers: Speaker names
            instruct_ids: Instruction IDs for style control (optional)
            languages: Language for each sample
            non_streaming_mode: Whether to use non-streaming mode
            **kwargs: Additional generate parameters

        Returns:
            Tuple[List[Tensor], Any]: (talker_codes, generation_info)
        """
        return self.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            speakers=speakers,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **kwargs,
        )

    # voice design model
    @torch.no_grad()
    def generate_voice_design(
        self,
        input_ids: list[torch.Tensor],
        instruct_ids: list[torch.Tensor],
        languages: Optional[list[str]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Any]:
        """
        Voice Design task-specific generate method.

        Args:
            input_ids: Text input IDs
            instruct_ids: Instruction IDs for voice style control
            languages: Language for each sample
            non_streaming_mode: Whether to use non-streaming mode
            **kwargs: Additional generate parameters

        Returns:
            Tuple[List[Tensor], Any]: (talker_codes, generation_info)
        """
        return self.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: Optional[dict] = None,
        languages: Optional[list[str]] = None,
        speakers: Optional[list[str]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], Any]:
        """
        Unified generate method that handles all TTS tasks.

        Task detection based on provided parameters:
        - Voice Clone: voice_clone_prompt is provided
        - Voice Design: instruct_ids provided, speakers not provided
        - Custom Voice: speakers provided

        Args:
            input_ids: Text input IDs
            instruct_ids: Instruction IDs (voice_design/custom_voice)
            ref_ids: Reference text IDs (voice_clone)
            voice_clone_prompt: Voice clone prompt dict (voice_clone)
            languages: Language for each sample
            speakers: Speaker names (custom_voice)
            non_streaming_mode: Whether to use non-streaming mode
            **kwargs: Generation parameters including:
                max_new_tokens (int): Maximum tokens to generate
                do_sample (bool): Whether to use sampling
                top_k (int): Top-k sampling parameter
                top_p (float): Top-p sampling parameter
                temperature (float): Sampling temperature
                subtalker_dosample (bool): Sub-talker sampling (12Hz tokenizer)
                subtalker_top_k (int): Sub-talker top-k
                subtalker_top_p (float): Sub-talker top-p
                subtalker_temperature (float): Sub-talker temperature
                eos_token_id (int): EOS token ID
                repetition_penalty (float): Repetition penalty
                And any other parameters supported by parent generate()

        Returns:
            Tuple[List[Tensor], Any]: (talker_codes, generation_info)
        """
        # Validate model type
        if speakers is not None and self.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.tokenizer_type}\n"
                f"tts_model_size: {self.tts_model_size}\n"
                f"tts_model_type: {self.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )
        elif instruct_ids is not None and self.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.tokenizer_type}\n"
                f"tts_model_size: {self.tts_model_size}\n"
                f"tts_model_type: {self.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )
        elif voice_clone_prompt is not None and self.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.tokenizer_type}\n"
                f"tts_model_size: {self.tts_model_size}\n"
                f"tts_model_type: {self.tts_model_type}\n"
                "does not support generate_voice_clone, Please check Model Card or Readme for more details."
            )

        # Validate languages and speakers before generation
        if languages is not None:
            self._validate_languages(languages)

        if speakers is not None:
            self._validate_speakers(speakers)

            if self.tts_model_size == "0.6b":  # for 0.6b model, instruct is not supported
                logger.warning("instruct_ids is not supported for 0.6b model, please remove it")
                instruct_ids = None

        # Merge generation kwargs with defaults from generate_config
        merged_kwargs = self._merge_generate_kwargs(**kwargs)

        # Call parent generate with validated and merged parameters
        return super().generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **merged_kwargs,
        )
