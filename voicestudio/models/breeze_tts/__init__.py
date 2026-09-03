from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3_tts_tokenizer_multi_codebook import Qwen3TTSTokenizerMultiCodebookConfig
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source
# path and looks it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time
# (i.e. while `.modeling_breeze_tts` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["breeze-tts"] = "BreezeTTSConfig"

from .configuration_breeze_tts import BreezeTTSConfig, BreezeTTSDepthDecoderConfig
from .generation_breeze_tts import (
    BreezeTTSGenerateOutput,
    BreezeTTSGenerationMixin,
    GeneratedTokenRepetitionPenaltyLogitsProcessor,
)
from .modeling_breeze_tts import (
    BreezeTTSBackboneModel,
    BreezeTTSDepthDecoderForCausalLM,
    BreezeTTSDepthDecoderModel,
    BreezeTTSForConditionalGeneration,
    BreezeTTSOutputWithPast,
    BreezeTTSPreTrainedModel,
)
from .processing_breeze_tts import BreezeTTSProcessor


AutoConfig.register(BreezeTTSConfig.model_type, BreezeTTSConfig, exist_ok=True)
AutoConfig.register(BreezeTTSDepthDecoderConfig.model_type, BreezeTTSDepthDecoderConfig, exist_ok=True)
# Real checkpoints report model_type "breeze", "breeze_depth_decoder_model" and, for the bundled audio
# tokenizer, "qwen3_tts_tokenizer_12hz"; alias all three.
CONFIG_MAPPING.register("breeze", BreezeTTSConfig, exist_ok=True)
CONFIG_MAPPING.register("breeze_depth_decoder_model", BreezeTTSDepthDecoderConfig, exist_ok=True)
CONFIG_MAPPING.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerMultiCodebookConfig, exist_ok=True)
AutoModel.register(BreezeTTSConfig, BreezeTTSForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(BreezeTTSConfig, BreezeTTSForConditionalGeneration, exist_ok=True)
AutoProcessor.register(BreezeTTSConfig, BreezeTTSProcessor, exist_ok=True)


__all__ = [
    "BreezeTTSBackboneModel",
    "BreezeTTSConfig",
    "BreezeTTSDepthDecoderConfig",
    "BreezeTTSDepthDecoderForCausalLM",
    "BreezeTTSDepthDecoderModel",
    "BreezeTTSForConditionalGeneration",
    "BreezeTTSGenerateOutput",
    "BreezeTTSGenerationMixin",
    "BreezeTTSOutputWithPast",
    "BreezeTTSPreTrainedModel",
    "BreezeTTSProcessor",
    "GeneratedTokenRepetitionPenaltyLogitsProcessor",
]
