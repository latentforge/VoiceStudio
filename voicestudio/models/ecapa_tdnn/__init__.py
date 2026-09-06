from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source path, rewrites
# the underscores of a name missing from `CONFIG_MAPPING_NAMES` into hyphens, and looks the result up in
# `HARDCODED_CONFIG_FOR_MODELS` at decoration time, i.e. while `.modeling_ecapa_tdnn` below is being imported.
HARDCODED_CONFIG_FOR_MODELS["ecapa-tdnn"] = "EcapaTdnnConfig"

from .configuration_ecapa_tdnn import EcapaTdnnConfig
from .feature_extraction_ecapa_tdnn import EcapaTdnnFeatureExtractor
from .modeling_ecapa_tdnn import (
    EcapaTdnnForXVector,
    EcapaTdnnModel,
    EcapaTdnnOutput,
    EcapaTdnnPreTrainedModel,
)


AutoConfig.register(EcapaTdnnConfig.model_type, EcapaTdnnConfig, exist_ok=True)
AutoModel.register(EcapaTdnnConfig, EcapaTdnnModel, exist_ok=True)
AutoFeatureExtractor.register(EcapaTdnnConfig, EcapaTdnnFeatureExtractor, exist_ok=True)


__all__ = [
    "EcapaTdnnConfig",
    "EcapaTdnnFeatureExtractor",
    "EcapaTdnnForXVector",
    "EcapaTdnnModel",
    "EcapaTdnnOutput",
    "EcapaTdnnPreTrainedModel",
]
