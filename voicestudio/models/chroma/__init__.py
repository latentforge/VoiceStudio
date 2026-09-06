from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source
# path and looks it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time
# (i.e. while `.modeling_chroma` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["chroma"] = "ChromaConfig"

from .configuration_chroma import ChromaBackboneConfig, ChromaConfig, ChromaDecoderConfig
from .generation_chroma import ChromaGenerateOutput, ChromaGenerationMixin
from .modeling_chroma import (
    ChromaBackboneForCausalLM,
    ChromaDecoderForCausalLM,
    ChromaForConditionalGeneration,
    ChromaLlamaModel,
    ChromaPreTrainedModel,
)
from .processing_chroma import ChromaProcessor


AutoConfig.register(ChromaConfig.model_type, ChromaConfig, exist_ok=True)
AutoModel.register(ChromaConfig, ChromaForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(ChromaConfig, ChromaForConditionalGeneration, exist_ok=True)
AutoProcessor.register(ChromaConfig, ChromaProcessor, exist_ok=True)


__all__ = [
    "ChromaBackboneConfig",
    "ChromaBackboneForCausalLM",
    "ChromaConfig",
    "ChromaDecoderConfig",
    "ChromaDecoderForCausalLM",
    "ChromaForConditionalGeneration",
    "ChromaGenerateOutput",
    "ChromaGenerationMixin",
    "ChromaLlamaModel",
    "ChromaPreTrainedModel",
    "ChromaProcessor",
]
