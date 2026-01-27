from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_tts import (
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
    Qwen3TTSConfig,
)
from .modeling_qwen3_tts import (
    Qwen3TTSPreTrainedModel,
    Qwen3TTSTalkerTextPreTrainedModel,
    Qwen3TTSTalkerCodePredictorModel,
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    Qwen3TTSTalkerModel,
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSForConditionalGeneration,
)
from .processing_qwen3_tts import (
    Qwen3TTSProcessor,
    Qwen3TTSProcessorKwargs,
)


AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoConfig.register("qwen3_tts_talker", Qwen3TTSTalkerConfig)
AutoConfig.register("qwen3_tts_talker_code_predictor", Qwen3TTSTalkerCodePredictorConfig)
AutoConfig.register("qwen3_tts_speaker_encoder", Qwen3TTSSpeakerEncoderConfig)

AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoModel.register(Qwen3TTSTalkerConfig, Qwen3TTSTalkerForConditionalGeneration)
AutoModel.register(Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerCodePredictorModelForConditionalGeneration)

AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
