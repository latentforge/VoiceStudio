from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source
# path and looks it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time
# (i.e. while `.modeling_higgs_tts3` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["higgs-tts3"] = "HiggsTTS3Config"

from .configuration_higgs_tts3 import HiggsTTS3AudioEncoderConfig, HiggsTTS3Config
from .modeling_higgs_tts3 import (
    HiggsTTS3ForConditionalGeneration,
    HiggsTTS3Model,
    HiggsTTS3PreTrainedModel,
)
from .processing_higgs_tts3 import HiggsTTS3Processor


AutoConfig.register(HiggsTTS3Config.model_type, HiggsTTS3Config, exist_ok=True)
# Real checkpoints report model_type "higgs_multimodal_qwen3", not "higgs_tts3"; alias it.
CONFIG_MAPPING.register("higgs_multimodal_qwen3", HiggsTTS3Config, exist_ok=True)
AutoModel.register(HiggsTTS3Config, HiggsTTS3ForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(HiggsTTS3Config, HiggsTTS3ForConditionalGeneration, exist_ok=True)
AutoProcessor.register(HiggsTTS3Config, HiggsTTS3Processor, exist_ok=True)


__all__ = [
    "HiggsTTS3AudioEncoderConfig",
    "HiggsTTS3Config",
    "HiggsTTS3ForConditionalGeneration",
    "HiggsTTS3Model",
    "HiggsTTS3PreTrainedModel",
    "HiggsTTS3Processor",
]
