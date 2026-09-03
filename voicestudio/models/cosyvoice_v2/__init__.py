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
# (i.e. while `.modeling_cosyvoice_v2` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["cosyvoice-v2"] = "CosyVoiceV2Config"

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
    CosyVoiceV2SpeechTokenizer,
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
    "CosyVoiceV2SpeechTokenizer",
    "CosyVoiceV2Upsample1D",
    "CosyVoiceV2UpsampleEncoder",
]
