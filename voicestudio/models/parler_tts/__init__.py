from transformers import AutoConfig, AutoModel
from transformers.conversion_mapping import register_checkpoint_conversion_mapping

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .modeling_parler_tts import (
    ParlerTTSForCausalLM,
    ParlerTTSForConditionalGeneration,
    ParlerTTSLogitsProcessor,
    ParlerTTSModel,
    ParlerTTSPreTrainedModel,
)
from .processing_parler_tts import ParlerTTSProcessor
from .weight_conversion import build_dac_weight_conversion_mapping


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoConfig.register("parler_tts_decoder", ParlerTTSDecoderConfig)
AutoModel.register(ParlerTTSDecoderConfig, ParlerTTSForCausalLM)
AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
register_checkpoint_conversion_mapping(
    "parler_tts", build_dac_weight_conversion_mapping(), overwrite=True
)


__all__ = [
    "ParlerTTSConfig",
    "ParlerTTSDecoderConfig",
    "ParlerTTSForCausalLM",
    "ParlerTTSForConditionalGeneration",
    "ParlerTTSLogitsProcessor",
    "ParlerTTSModel",
    "ParlerTTSPreTrainedModel",
    "ParlerTTSProcessor",
]
