from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoModelForTextToWaveform, AutoProcessor
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source
# path and looks it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time
# (i.e. while `.modeling_vox_instruct` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["vox-instruct"] = "VoxInstructConfig"

from .configuration_vox_instruct import VoxInstructARConfig, VoxInstructConfig, VoxInstructNARConfig
from .feature_extraction_vox_instruct import VoxInstructFeatureExtractor
from .generation_vox_instruct import VoxInstructGenerateOutput, VoxInstructGenerationMixin
from .modeling_vox_instruct import (
    VoxInstructARForCausalLM,
    VoxInstructForConditionalGeneration,
    VoxInstructNARModel,
    VoxInstructPreTrainedModel,
    VoxInstructTextEncoder,
)
from .processing_vox_instruct import VoxInstructProcessor
from .tokenization_vox_instruct import VoxInstructSemanticTokenizerModel


AutoConfig.register(VoxInstructConfig.model_type, VoxInstructConfig, exist_ok=True)
AutoModel.register(VoxInstructConfig, VoxInstructForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(VoxInstructConfig, VoxInstructForConditionalGeneration, exist_ok=True)
AutoFeatureExtractor.register(VoxInstructConfig, VoxInstructFeatureExtractor, exist_ok=True)
AutoProcessor.register(VoxInstructConfig, VoxInstructProcessor, exist_ok=True)


__all__ = [
    "VoxInstructARConfig",
    "VoxInstructNARConfig",
    "VoxInstructConfig",
    "VoxInstructFeatureExtractor",
    "VoxInstructProcessor",
    "VoxInstructSemanticTokenizerModel",
    "VoxInstructTextEncoder",
    "VoxInstructARForCausalLM",
    "VoxInstructNARModel",
    "VoxInstructForConditionalGeneration",
    "VoxInstructPreTrainedModel",
    "VoxInstructGenerationMixin",
    "VoxInstructGenerateOutput",
]
