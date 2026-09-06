from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor

from .configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from .modeling_qwen3_tts import (
    Qwen3TTSBasePreTrainedModel,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSPreTrainedModel,
    Qwen3TTSTalkerCodePredictorModel,
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    Qwen3TTSTalkerModel,
    Qwen3TTSTalkerTextPreTrainedModel,
)
from .processing_qwen3_tts import Qwen3TTSProcessor


AutoConfig.register(Qwen3TTSConfig.model_type, Qwen3TTSConfig, exist_ok=True)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, exist_ok=True)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor, exist_ok=True)


__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSBasePreTrainedModel",
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerCodePredictorModel",
    "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSTalkerTextPreTrainedModel",
    "Qwen3TTSProcessor",
]
