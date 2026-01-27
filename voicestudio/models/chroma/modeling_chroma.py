try:
    from ..._chroma.modeling_chroma import (
        ChromaPreTrainedModel,
        ChromaBackboneForCausalLM,
        ChromaDecoderForCausalLM,
        ChromaForConditionalGeneration,
    )
except ImportError:
    from voicestudio._chroma.modeling_chroma import (
        ChromaPreTrainedModel,
        ChromaBackboneForCausalLM,
        ChromaDecoderForCausalLM,
        ChromaForConditionalGeneration,
    )
