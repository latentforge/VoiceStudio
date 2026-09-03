from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source path, rewrites
# the underscores of a name missing from `CONFIG_MAPPING_NAMES` into hyphens, and looks the result up in
# `HARDCODED_CONFIG_FOR_MODELS` at decoration time, i.e. while `.modeling_bigvgan` below is being imported.
HARDCODED_CONFIG_FOR_MODELS["bigvgan"] = "BigVGANConfig"

from .configuration_bigvgan import BigVGANConfig
from .feature_extraction_bigvgan import BigVGANFeatureExtractor
from .modeling_bigvgan import (
    BigVGANAmpBlock,
    BigVGANAmpLayer,
    BigVGANModel,
    BigVGANOutput,
    BigVGANPreTrainedModel,
    BigVGANSnakeActivation,
    build_anti_alias_filter,
    dynamic_range_compression,
    mel_spectrogram,
)


AutoConfig.register(BigVGANConfig.model_type, BigVGANConfig, exist_ok=True)
AutoModel.register(BigVGANConfig, BigVGANModel, exist_ok=True)
AutoFeatureExtractor.register(BigVGANConfig, BigVGANFeatureExtractor, exist_ok=True)


__all__ = [
    "BigVGANAmpBlock",
    "BigVGANAmpLayer",
    "BigVGANConfig",
    "BigVGANFeatureExtractor",
    "BigVGANModel",
    "BigVGANOutput",
    "BigVGANPreTrainedModel",
    "BigVGANSnakeActivation",
    "build_anti_alias_filter",
    "dynamic_range_compression",
    "mel_spectrogram",
]
