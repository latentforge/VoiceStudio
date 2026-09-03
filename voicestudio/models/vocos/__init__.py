from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source path, rewrites
# the underscores of a name missing from `CONFIG_MAPPING_NAMES` into hyphens, and looks the result up in
# `HARDCODED_CONFIG_FOR_MODELS` at decoration time, i.e. while `.modeling_vocos` below is being imported.
HARDCODED_CONFIG_FOR_MODELS["vocos"] = "VocosConfig"

from .configuration_vocos import VocosConfig
from .feature_extraction_vocos import VocosFeatureExtractor
from .modeling_vocos import VocosModel, VocosOutput, VocosPreTrainedModel


AutoConfig.register(VocosConfig.model_type, VocosConfig, exist_ok=True)
AutoModel.register(VocosConfig, VocosModel, exist_ok=True)
AutoFeatureExtractor.register(VocosConfig, VocosFeatureExtractor, exist_ok=True)


__all__ = [
    "VocosConfig",
    "VocosFeatureExtractor",
    "VocosModel",
    "VocosOutput",
    "VocosPreTrainedModel",
]
