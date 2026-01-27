from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_chroma import (
    ChromaBackboneConfig,
    ChromaDecoderConfig,
    ChromaConfig,
)
from .modeling_chroma import (
    ChromaPreTrainedModel,
    ChromaBackboneForCausalLM,
    ChromaDecoderForCausalLM,
    ChromaForConditionalGeneration,
)
from .processing_chroma import (
    ChromaAudioKwargs,
    ChromaProcessorKwargs,
    ChromaProcessor,
)

AutoConfig.register("chroma", ChromaConfig)
AutoConfig.register("chroma_backbone", ChromaBackboneConfig)
AutoConfig.register("chroma_decoder", ChromaDecoderConfig)

AutoModel.register(ChromaConfig, ChromaForConditionalGeneration)
AutoModel.register(ChromaBackboneConfig, ChromaBackboneForCausalLM)
AutoModel.register(ChromaDecoderConfig, ChromaDecoderForCausalLM)

AutoProcessor.register(ChromaConfig, ChromaProcessor)
