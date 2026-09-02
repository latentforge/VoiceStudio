from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source path and looks
# it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time, i.e. while `.modeling_dia2`
# below is being imported, and neither mapping knows about voicestudio-only models.
HARDCODED_CONFIG_FOR_MODELS["dia2"] = "Dia2Config"

from .configuration_dia2 import Dia2Config, Dia2DepthDecoderConfig
from .generation_dia2 import Dia2GenerationMixin, Dia2ScriptEntry, Dia2TextStateMachine
from .modeling_dia2 import (
    Dia2BackboneModel,
    Dia2DepthDecoderForCausalLM,
    Dia2DepthDecoderModel,
    Dia2ForConditionalGeneration,
    Dia2PreTrainedModel,
)
from .processing_dia2 import Dia2Processor


AutoConfig.register(Dia2Config.model_type, Dia2Config, exist_ok=True)
AutoConfig.register(Dia2DepthDecoderConfig.model_type, Dia2DepthDecoderConfig, exist_ok=True)
AutoModel.register(Dia2Config, Dia2ForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(Dia2Config, Dia2ForConditionalGeneration, exist_ok=True)
AutoProcessor.register(Dia2Config, Dia2Processor, exist_ok=True)


__all__ = [
    "Dia2BackboneModel",
    "Dia2Config",
    "Dia2DepthDecoderConfig",
    "Dia2DepthDecoderForCausalLM",
    "Dia2DepthDecoderModel",
    "Dia2ForConditionalGeneration",
    "Dia2GenerationMixin",
    "Dia2PreTrainedModel",
    "Dia2Processor",
    "Dia2ScriptEntry",
    "Dia2TextStateMachine",
]
