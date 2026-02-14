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


from .processing_qwen3_tts import Qwen3TTSProcessor

modeling_qwen3_tts.Qwen3TTSTokenizer = Qwen3TTSProcessor


class ModelProxy:
    def __init__(self, model):
        self.model = model


class Qwen3TTSForConditionalGeneration(_Qwen3TTSForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proxy = ModelProxy(self)

    # voice clone model
    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        return _Qwen3TTSModel.generate_voice_clone(
            self.proxy, text, language, ref_audio, ref_text,
            x_vector_only_mode, voice_clone_prompt,
            non_streaming_mode, **kwargs
        )

    # voice design model
    @torch.no_grad()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        return _Qwen3TTSModel.generate_voice_design(
            self.proxy, text, instruct, language, non_streaming_mode, **kwargs
        )

    # custom voice model
    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        return _Qwen3TTSModel.generate_custom_voice(
            self.proxy, text, speaker, language, instruct, non_streaming_mode, **kwargs
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        pass
