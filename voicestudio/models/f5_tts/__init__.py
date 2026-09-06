from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForTextToWaveform,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source path, rewrites
# the underscores of a name missing from `CONFIG_MAPPING_NAMES` into hyphens, and looks the result up in
# `HARDCODED_CONFIG_FOR_MODELS` at decoration time, i.e. while `.modeling_f5_tts` below is being imported.
HARDCODED_CONFIG_FOR_MODELS["f5-tts"] = "F5TTSConfig"

from .configuration_f5_tts import F5TTSConfig
from .feature_extraction_f5_tts import F5TTSFeatureExtractor
from .generation_f5_tts import F5TTSFixedStepODESolver, F5TTSGenerationMixin, F5TTSGenerationOutput
from .modeling_f5_tts import (
    F5TTSForConditionalGeneration,
    F5TTSModel,
    F5TTSOutput,
    F5TTSPreTrainedModel,
    F5TTSUNetModel,
)
from .processing_f5_tts import F5TTSProcessor
from .tokenization_f5_tts import F5TTSTokenizer


AutoConfig.register(F5TTSConfig.model_type, F5TTSConfig, exist_ok=True)
AutoModel.register(F5TTSConfig, F5TTSForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(F5TTSConfig, F5TTSForConditionalGeneration, exist_ok=True)
AutoFeatureExtractor.register(F5TTSConfig, F5TTSFeatureExtractor, exist_ok=True)
AutoTokenizer.register(F5TTSConfig, F5TTSTokenizer, exist_ok=True)
AutoProcessor.register(F5TTSConfig, F5TTSProcessor, exist_ok=True)


__all__ = [
    "F5TTSConfig",
    "F5TTSFeatureExtractor",
    "F5TTSFixedStepODESolver",
    "F5TTSForConditionalGeneration",
    "F5TTSGenerationMixin",
    "F5TTSGenerationOutput",
    "F5TTSModel",
    "F5TTSOutput",
    "F5TTSPreTrainedModel",
    "F5TTSProcessor",
    "F5TTSTokenizer",
    "F5TTSUNetModel",
]
