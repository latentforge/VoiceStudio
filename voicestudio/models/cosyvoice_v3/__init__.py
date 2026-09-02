from transformers import AutoConfig, AutoProcessor
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.models.auto.modeling_auto import AutoModel

from .configuration_cosyvoice_v3 import (
    CosyVoiceV3Config,
    CosyVoiceV3FlowConfig,
    CosyVoiceV3HiftConfig,
    CosyVoiceV3LLMConfig,
)
from .modeling_cosyvoice_v3 import (
    CosyVoiceV3DiT,
    CosyVoiceV3DiTBlock,
    CosyVoiceV3FlowMatchingModel,
    CosyVoiceV3ForConditionalGeneration,
    CosyVoiceV3HiFTGenerator,
    CosyVoiceV3LLM,
    CosyVoiceV3Model,
)
from .processing_cosyvoice_v3 import CosyVoiceV3Processor
from .weight_conversion import build_hift_weight_conversion_mapping, build_llm_weight_conversion_mapping


AutoConfig.register("cosyvoice_v3", CosyVoiceV3Config)
AutoModel.register(CosyVoiceV3Config, CosyVoiceV3ForConditionalGeneration)
AutoProcessor.register(CosyVoiceV3Config, CosyVoiceV3Processor)

register_checkpoint_conversion_mapping("CosyVoiceV3LLM", build_llm_weight_conversion_mapping(), overwrite=True)
register_checkpoint_conversion_mapping("CosyVoiceV3FlowMatchingModel", [], overwrite=True)
register_checkpoint_conversion_mapping(
    "CosyVoiceV3HiFTGenerator", build_hift_weight_conversion_mapping(), overwrite=True
)


__all__ = [
    "CosyVoiceV3Config",
    "CosyVoiceV3LLMConfig",
    "CosyVoiceV3FlowConfig",
    "CosyVoiceV3HiftConfig",
    "CosyVoiceV3ForConditionalGeneration",
    "CosyVoiceV3Model",
    "CosyVoiceV3LLM",
    "CosyVoiceV3FlowMatchingModel",
    "CosyVoiceV3DiT",
    "CosyVoiceV3DiTBlock",
    "CosyVoiceV3HiFTGenerator",
    "CosyVoiceV3Processor",
]
