from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.models.auto.modeling_auto import AutoModel

from .configuration_f5_tts import F5TTSConfig
from .modeling_f5_tts import (
    F5TTSForConditionalGeneration,
    F5TTSModel,
    F5TTSOutput,
    F5TTSPreTrainedModel,
)
from .processing_f5_tts import F5TTSProcessor
from .tokenization_f5_tts import F5TTSTokenizer
from .weight_conversion import build_f5_tts_weight_conversion_mapping


AutoConfig.register("f5_tts", F5TTSConfig)
AutoTokenizer.register(F5TTSConfig, slow_tokenizer_class=F5TTSTokenizer)
AutoModel.register(F5TTSConfig, F5TTSForConditionalGeneration)
AutoProcessor.register(F5TTSConfig, F5TTSProcessor)
register_checkpoint_conversion_mapping(
    "f5_tts", build_f5_tts_weight_conversion_mapping(), overwrite=True
)


__all__ = [
    "F5TTSConfig",
    "F5TTSForConditionalGeneration",
    "F5TTSModel",
    "F5TTSOutput",
    "F5TTSPreTrainedModel",
    "F5TTSProcessor",
    "F5TTSTokenizer",
]
