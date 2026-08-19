from transformers import AutoConfig, AutoProcessor
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.models.auto.modeling_auto import AutoModel

from .configuration_cosyvoice_v1 import (
    CosyVoiceV1Config,
    CosyVoiceV1FlowConfig,
    CosyVoiceV1HiftConfig,
    CosyVoiceV1LLMConfig,
    CosyVoiceV1TextEncoderConfig,
)
from .modeling_cosyvoice_v1 import (
    CosyVoiceV1ConditionalCFM,
    CosyVoiceV1ConditionalDecoder,
    CosyVoiceV1FlowMatchingModel,
    CosyVoiceV1ForConditionalGeneration,
    CosyVoiceV1HiFTGenerator,
    CosyVoiceV1InterpolateRegulator,
    CosyVoiceV1LLM,
    CosyVoiceV1LLMOutput,
    CosyVoiceV1Model,
    CosyVoiceV1PreTrainedModel,
    CosyVoiceV1RelPositionEncoder,
    CosyVoiceV1TextEncoder,
)
from .processing_cosyvoice_v1 import CosyVoiceV1Processor
from .weight_conversion import (
    build_flow_weight_conversion_mapping,
    build_hift_weight_conversion_mapping,
    build_llm_weight_conversion_mapping,
)


AutoConfig.register("cosyvoice_v1", CosyVoiceV1Config)
AutoModel.register(CosyVoiceV1Config, CosyVoiceV1ForConditionalGeneration)
AutoProcessor.register(CosyVoiceV1Config, CosyVoiceV1Processor)
register_checkpoint_conversion_mapping(
    "CosyVoiceV1LLM", build_llm_weight_conversion_mapping(), overwrite=True
)
register_checkpoint_conversion_mapping(
    "CosyVoiceV1FlowMatchingModel", build_flow_weight_conversion_mapping(), overwrite=True
)
register_checkpoint_conversion_mapping(
    "CosyVoiceV1HiFTGenerator", build_hift_weight_conversion_mapping(), overwrite=True
)


__all__ = [
    "CosyVoiceV1Config",
    "CosyVoiceV1TextEncoderConfig",
    "CosyVoiceV1LLMConfig",
    "CosyVoiceV1FlowConfig",
    "CosyVoiceV1HiftConfig",
    "CosyVoiceV1ForConditionalGeneration",
    "CosyVoiceV1Model",
    "CosyVoiceV1LLM",
    "CosyVoiceV1LLMOutput",
    "CosyVoiceV1FlowMatchingModel",
    "CosyVoiceV1HiFTGenerator",
    "CosyVoiceV1PreTrainedModel",
    "CosyVoiceV1TextEncoder",
    "CosyVoiceV1RelPositionEncoder",
    "CosyVoiceV1ConditionalDecoder",
    "CosyVoiceV1ConditionalCFM",
    "CosyVoiceV1InterpolateRegulator",
    "CosyVoiceV1Processor",
]
