from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor

from .configuration_chroma import ChromaBackboneConfig, ChromaConfig, ChromaDecoderConfig
from .generation_chroma import ChromaGenerateOutput, ChromaGenerationMixin
from .modeling_chroma import (
    ChromaBackboneForCausalLM,
    ChromaDecoderForCausalLM,
    ChromaForConditionalGeneration,
    ChromaLlamaModel,
    ChromaPreTrainedModel,
)
from .processing_chroma import ChromaProcessor


AutoConfig.register(ChromaConfig.model_type, ChromaConfig, exist_ok=True)
AutoModel.register(ChromaConfig, ChromaForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(ChromaConfig, ChromaForConditionalGeneration, exist_ok=True)
AutoProcessor.register(ChromaConfig, ChromaProcessor, exist_ok=True)


__all__ = [
    "ChromaBackboneConfig",
    "ChromaBackboneForCausalLM",
    "ChromaConfig",
    "ChromaDecoderConfig",
    "ChromaDecoderForCausalLM",
    "ChromaForConditionalGeneration",
    "ChromaGenerateOutput",
    "ChromaGenerationMixin",
    "ChromaLlamaModel",
    "ChromaPreTrainedModel",
    "ChromaProcessor",
]
