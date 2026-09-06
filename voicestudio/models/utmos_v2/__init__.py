from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source path, rewrites
# the underscores of a name missing from `CONFIG_MAPPING_NAMES` into hyphens, and looks the result up in
# `HARDCODED_CONFIG_FOR_MODELS` at decoration time, i.e. while `.modeling_utmos_v2` below is being imported.
HARDCODED_CONFIG_FOR_MODELS["utmos-v2"] = "UTMOSv2Config"

from .configuration_utmos_v2 import UTMOSv2Config
from .feature_extraction_utmos_v2 import DOMAINS, UTMOSv2FeatureExtractor
from .modeling_utmos_v2 import (
    UTMOSv2ForAudioClassification,
    UTMOSv2Model,
    UTMOSv2Output,
    UTMOSv2PreTrainedModel,
)


AutoConfig.register(UTMOSv2Config.model_type, UTMOSv2Config, exist_ok=True)
AutoModel.register(UTMOSv2Config, UTMOSv2Model, exist_ok=True)
AutoFeatureExtractor.register(UTMOSv2Config, UTMOSv2FeatureExtractor, exist_ok=True)


__all__ = [
    "DOMAINS",
    "UTMOSv2Config",
    "UTMOSv2FeatureExtractor",
    "UTMOSv2ForAudioClassification",
    "UTMOSv2Model",
    "UTMOSv2Output",
    "UTMOSv2PreTrainedModel",
]
