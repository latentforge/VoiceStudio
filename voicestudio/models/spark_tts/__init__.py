import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioTokenization,
    AutoModelForCausalLM,
    AutoModelForTextToWaveform,
    AutoProcessor,
)
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source
# path and looks it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time
# (i.e. while `.modeling_spark_tts` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["spark-tts"] = "SparkTTSConfig"

from .configuration_spark_tts import SparkTTSBiCodecConfig, SparkTTSConfig
from .feature_extraction_spark_tts import SparkTTSFeatureExtractor
from .modeling_spark_tts import (
    SparkTTSBiCodecModel,
    SparkTTSBiCodecPreTrainedModel,
    SparkTTSForConditionalGeneration,
    SparkTTSPreTrainedModel,
)
from .processing_spark_tts import SparkTTSProcessor


AutoConfig.register(SparkTTSBiCodecConfig.model_type, SparkTTSBiCodecConfig, exist_ok=True)
AutoConfig.register(SparkTTSConfig.model_type, SparkTTSConfig, exist_ok=True)
AutoModelForAudioTokenization.register(SparkTTSBiCodecConfig, SparkTTSBiCodecModel, exist_ok=True)
AutoModel.register(SparkTTSConfig, SparkTTSForConditionalGeneration, exist_ok=True)
AutoModelForCausalLM.register(SparkTTSConfig, SparkTTSForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(SparkTTSConfig, SparkTTSForConditionalGeneration, exist_ok=True)
AutoFeatureExtractor.register(SparkTTSConfig, SparkTTSFeatureExtractor, exist_ok=True)
AutoProcessor.register(SparkTTSConfig, SparkTTSProcessor, exist_ok=True)

# `ProcessorMixin.get_possibly_dynamic_module` resolves the `audio_tokenizer_class` recorded in
# `processor_config.json` by first looking the name up on the `transformers` module and only then
# walking the `AutoClass` registries. That walk raises before it reaches
# `MODEL_FOR_AUDIO_TOKENIZATION_MAPPING`, because `transformers.IMAGE_PROCESSOR_MAPPING` is a
# placeholder object with no `_extra_content`, so the name has to be resolvable by the first lookup.
transformers.SparkTTSBiCodecModel = SparkTTSBiCodecModel


__all__ = [
    "SparkTTSBiCodecConfig",
    "SparkTTSBiCodecModel",
    "SparkTTSBiCodecPreTrainedModel",
    "SparkTTSConfig",
    "SparkTTSFeatureExtractor",
    "SparkTTSForConditionalGeneration",
    "SparkTTSPreTrainedModel",
    "SparkTTSProcessor",
]
