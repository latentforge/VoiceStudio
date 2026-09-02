from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source path and looks it
# up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time (i.e. while `.modeling_omnivoice`
# below is being imported), neither of which knows about voicestudio-only models; must run before that import or
# its "Config not found" fallback warning already fired.
HARDCODED_CONFIG_FOR_MODELS["ommivoice"] = "OmniVoiceConfig"

from .configuration_omnivoice import OmniVoiceConfig
from .generation_omnivoice import OmniVoiceGenerationConfig, OmniVoiceGenerationMixin
from .modeling_omnivoice import (
    OmniVoiceForConditionalGeneration,
    OmniVoiceModel,
    OmniVoicePreTrainedModel,
)
from .processing_omnivoice import OmniVoiceDurationEstimator, OmniVoiceProcessor


AutoConfig.register(OmniVoiceConfig.model_type, OmniVoiceConfig, exist_ok=True)
AutoModel.register(OmniVoiceConfig, OmniVoiceForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(OmniVoiceConfig, OmniVoiceForConditionalGeneration, exist_ok=True)
AutoProcessor.register(OmniVoiceConfig, OmniVoiceProcessor, exist_ok=True)


__all__ = [
    "OmniVoiceConfig",
    "OmniVoiceDurationEstimator",
    "OmniVoiceForConditionalGeneration",
    "OmniVoiceGenerationConfig",
    "OmniVoiceGenerationMixin",
    "OmniVoiceModel",
    "OmniVoicePreTrainedModel",
    "OmniVoiceProcessor",
]
