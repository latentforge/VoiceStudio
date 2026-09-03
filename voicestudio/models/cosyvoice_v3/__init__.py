from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForTextToWaveform,
    AutoProcessor,
)

from .configuration_cosyvoice_v3 import CosyVoiceV3Config
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
from .processing_cosyvoice_v3 import (
    SPECIAL_TOKENS,
    CosyVoiceV3FeatureExtractor,
    CosyVoiceV3Processor,
)


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
