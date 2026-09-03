from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForTextToWaveform,
    AutoProcessor,
)

from .configuration_cosyvoice_v2 import CosyVoiceV2Config
from .generation_cosyvoice_v2 import CosyVoiceV2GenerationMixin
from .modeling_cosyvoice_v2 import (
    CosyVoiceV2CausalBlock1D,
    CosyVoiceV2CausalConv1d,
    CosyVoiceV2CausalResnetBlock1D,
    CosyVoiceV2ConditionalCFM,
    CosyVoiceV2ConditionalDecoder,
    CosyVoiceV2FlowModel,
    CosyVoiceV2ForConditionalGeneration,
    CosyVoiceV2HiFTGenerator,
    CosyVoiceV2Output,
    CosyVoiceV2PreLookaheadLayer,
    CosyVoiceV2PreTrainedModel,
    CosyVoiceV2SineGen,
    CosyVoiceV2SourceModule,
    CosyVoiceV2SpeechTokenLM,
    CosyVoiceV2Upsample1D,
    CosyVoiceV2UpsampleEncoder,
)
from .processing_cosyvoice_v2 import CosyVoiceV2FeatureExtractor, CosyVoiceV2Processor


AutoConfig.register(CosyVoiceV2Config.model_type, CosyVoiceV2Config, exist_ok=True)
AutoModel.register(CosyVoiceV2Config, CosyVoiceV2ForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(CosyVoiceV2Config, CosyVoiceV2ForConditionalGeneration, exist_ok=True)
AutoFeatureExtractor.register(CosyVoiceV2Config, CosyVoiceV2FeatureExtractor, exist_ok=True)
AutoProcessor.register(CosyVoiceV2Config, CosyVoiceV2Processor, exist_ok=True)


__all__ = [
    "CosyVoiceV2CausalBlock1D",
    "CosyVoiceV2CausalConv1d",
    "CosyVoiceV2CausalResnetBlock1D",
    "CosyVoiceV2ConditionalCFM",
    "CosyVoiceV2ConditionalDecoder",
    "CosyVoiceV2Config",
    "CosyVoiceV2FeatureExtractor",
    "CosyVoiceV2FlowModel",
    "CosyVoiceV2ForConditionalGeneration",
    "CosyVoiceV2GenerationMixin",
    "CosyVoiceV2HiFTGenerator",
    "CosyVoiceV2Output",
    "CosyVoiceV2PreLookaheadLayer",
    "CosyVoiceV2PreTrainedModel",
    "CosyVoiceV2Processor",
    "CosyVoiceV2SineGen",
    "CosyVoiceV2SourceModule",
    "CosyVoiceV2SpeechTokenLM",
    "CosyVoiceV2Upsample1D",
    "CosyVoiceV2UpsampleEncoder",
]
