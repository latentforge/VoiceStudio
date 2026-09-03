from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForTextToSpectrogram,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source
# path and looks it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time
# (i.e. while `.modeling_prompt_tts_pp` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["prompt-tts-pp"] = "PromptTTSPPConfig"

from .configuration_prompt_tts_pp import PromptTTSPPBigVGanConfig, PromptTTSPPConfig
from .feature_extraction_prompt_tts_pp import PromptTTSPPFeatureExtractor
from .modeling_prompt_tts_pp import (
    PromptTTSPPBigVGan,
    PromptTTSPPForConditionalGeneration,
    PromptTTSPPModel,
    PromptTTSPPPreTrainedModel,
)
from .processing_prompt_tts_pp import PromptTTSPPProcessor
from .tokenization_prompt_tts_pp import PromptTTSPPTokenizer


AutoConfig.register(PromptTTSPPConfig.model_type, PromptTTSPPConfig, exist_ok=True)
AutoConfig.register(PromptTTSPPBigVGanConfig.model_type, PromptTTSPPBigVGanConfig, exist_ok=True)
AutoModel.register(PromptTTSPPConfig, PromptTTSPPForConditionalGeneration, exist_ok=True)
AutoModel.register(PromptTTSPPBigVGanConfig, PromptTTSPPBigVGan, exist_ok=True)
AutoModelForTextToSpectrogram.register(PromptTTSPPConfig, PromptTTSPPForConditionalGeneration, exist_ok=True)
AutoFeatureExtractor.register(PromptTTSPPConfig, PromptTTSPPFeatureExtractor, exist_ok=True)
AutoProcessor.register(PromptTTSPPConfig, PromptTTSPPProcessor, exist_ok=True)
AutoTokenizer.register(PromptTTSPPConfig, PromptTTSPPTokenizer, exist_ok=True)


__all__ = [
    "PromptTTSPPBigVGan",
    "PromptTTSPPBigVGanConfig",
    "PromptTTSPPConfig",
    "PromptTTSPPFeatureExtractor",
    "PromptTTSPPForConditionalGeneration",
    "PromptTTSPPModel",
    "PromptTTSPPPreTrainedModel",
    "PromptTTSPPProcessor",
    "PromptTTSPPTokenizer",
]
