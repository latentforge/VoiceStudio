from transformers import AutoConfig, AutoProcessor
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.models.auto.modeling_auto import AutoModel

from .configuration_cosyvoice_v2 import CosyVoiceV2Config, CosyVoiceV2FlowConfig, CosyVoiceV2LLMConfig
from .modeling_cosyvoice_v2 import (
    CosyVoiceV2CausalConditionalDecoder,
    CosyVoiceV2FlowMatchingModel,
    CosyVoiceV2ForConditionalGeneration,
    CosyVoiceV2LLM,
    CosyVoiceV2LLMOutput,
    CosyVoiceV2Model,
    CosyVoiceV2PreLookaheadLayer,
    CosyVoiceV2Upsample1D,
    CosyVoiceV2UpsampleConformerEncoder,
)
from .processing_cosyvoice_v2 import CosyVoiceV2Processor
from .weight_conversion import build_flow_weight_conversion_mapping, build_llm_weight_conversion_mapping


AutoConfig.register("cosyvoice_v2", CosyVoiceV2Config)
AutoModel.register(CosyVoiceV2Config, CosyVoiceV2ForConditionalGeneration)
AutoProcessor.register(CosyVoiceV2Config, CosyVoiceV2Processor)
register_checkpoint_conversion_mapping(
    "CosyVoiceV2LLM", build_llm_weight_conversion_mapping(), overwrite=True
)
register_checkpoint_conversion_mapping(
    "CosyVoiceV2FlowMatchingModel", build_flow_weight_conversion_mapping(), overwrite=True
)


__all__ = [
    "CosyVoiceV2Config",
    "CosyVoiceV2LLMConfig",
    "CosyVoiceV2FlowConfig",
    "CosyVoiceV2ForConditionalGeneration",
    "CosyVoiceV2Model",
    "CosyVoiceV2LLM",
    "CosyVoiceV2LLMOutput",
    "CosyVoiceV2FlowMatchingModel",
    "CosyVoiceV2CausalConditionalDecoder",
    "CosyVoiceV2PreLookaheadLayer",
    "CosyVoiceV2Upsample1D",
    "CosyVoiceV2UpsampleConformerEncoder",
    "CosyVoiceV2Processor",
]
