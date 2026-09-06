from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForTextToWaveform,
    AutoProcessor,
)
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source
# path and looks it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time
# (i.e. while `.modeling_cosyvoice_v3` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["cosyvoice-v3"] = "CosyVoiceV3Config"

from .configuration_cosyvoice_v3 import CosyVoiceV3Config
from .feature_extraction_cosyvoice_v3 import CosyVoiceV3FeatureExtractor
from .generation_cosyvoice_v3 import CosyVoiceV3GenerationMixin
from .modeling_cosyvoice_v3 import (
    CosyVoiceV3AdaLayerNormFinal,
    CosyVoiceV3CausalConv1d,
    CosyVoiceV3CausalConv1dDownsample,
    CosyVoiceV3CausalConv1dUpsample,
    CosyVoiceV3CausalConvPositionEmbedding,
    CosyVoiceV3ConditionalCFM,
    CosyVoiceV3ConditionalDecoder,
    CosyVoiceV3DecoderLayer,
    CosyVoiceV3F0Predictor,
    CosyVoiceV3FlowModel,
    CosyVoiceV3ForConditionalGeneration,
    CosyVoiceV3HiFTGenerator,
    CosyVoiceV3InputEmbedding,
    CosyVoiceV3Output,
    CosyVoiceV3PreTrainedModel,
    CosyVoiceV3ResBlock,
    CosyVoiceV3RotaryEmbedding,
    CosyVoiceV3SineGen,
    CosyVoiceV3SourceModule,
    CosyVoiceV3SpeechTokenLM,
    CosyVoiceV3TimestepEmbedding,
    build_chunk_mask,
)
from .processing_cosyvoice_v3 import SPECIAL_TOKENS, CosyVoiceV3Processor


AutoConfig.register(CosyVoiceV3Config.model_type, CosyVoiceV3Config, exist_ok=True)
AutoModel.register(CosyVoiceV3Config, CosyVoiceV3ForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(CosyVoiceV3Config, CosyVoiceV3ForConditionalGeneration, exist_ok=True)
AutoFeatureExtractor.register(CosyVoiceV3Config, CosyVoiceV3FeatureExtractor, exist_ok=True)
AutoProcessor.register(CosyVoiceV3Config, CosyVoiceV3Processor, exist_ok=True)


__all__ = [
    "SPECIAL_TOKENS",
    "CosyVoiceV3AdaLayerNormFinal",
    "CosyVoiceV3CausalConv1d",
    "CosyVoiceV3CausalConv1dDownsample",
    "CosyVoiceV3CausalConv1dUpsample",
    "CosyVoiceV3CausalConvPositionEmbedding",
    "CosyVoiceV3ConditionalCFM",
    "CosyVoiceV3ConditionalDecoder",
    "CosyVoiceV3Config",
    "CosyVoiceV3DecoderLayer",
    "CosyVoiceV3F0Predictor",
    "CosyVoiceV3FeatureExtractor",
    "CosyVoiceV3FlowModel",
    "CosyVoiceV3ForConditionalGeneration",
    "CosyVoiceV3GenerationMixin",
    "CosyVoiceV3HiFTGenerator",
    "CosyVoiceV3InputEmbedding",
    "CosyVoiceV3Output",
    "CosyVoiceV3PreTrainedModel",
    "CosyVoiceV3Processor",
    "CosyVoiceV3ResBlock",
    "CosyVoiceV3RotaryEmbedding",
    "CosyVoiceV3SineGen",
    "CosyVoiceV3SourceModule",
    "CosyVoiceV3SpeechTokenLM",
    "CosyVoiceV3TimestepEmbedding",
    "build_chunk_mask",
]
