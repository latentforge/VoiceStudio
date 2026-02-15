from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_tts import (
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerConfig,
    Qwen3TTSConfig,
    Qwen3TTSTokenizerV1Config,
    Qwen3TTSTokenizerV2Config,
)
from .modeling_qwen3_tts import (
    Qwen3TTSPreTrainedModel,
    Qwen3TTSTalkerModel,
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSForConditionalGeneration,
)
from .processing_qwen3_tts import Qwen3TTSProcessor


AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
