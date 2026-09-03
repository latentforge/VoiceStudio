from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source path, rewrites
# the underscores of a name missing from `CONFIG_MAPPING_NAMES` into hyphens, and looks the result up in
# `HARDCODED_CONFIG_FOR_MODELS` at decoration time, i.e. while `.modeling_parler_tts` below is being imported.
HARDCODED_CONFIG_FOR_MODELS["parler-tts"] = "ParlerTTSConfig"

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .generation_parler_tts import ParlerTTSStreamer
from .modeling_parler_tts import ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration
from .processing_parler_tts import ParlerTTSProcessor


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoModel.register(ParlerTTSDecoderConfig, ParlerTTSForCausalLM)
AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
AutoModelForTextToWaveform.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration, exist_ok=True)
AutoProcessor.register(ParlerTTSConfig, ParlerTTSProcessor)
