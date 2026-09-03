from transformers import AutoConfig, AutoModel, AutoModelForTextToWaveform, AutoProcessor

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .generation_parler_tts import ParlerTTSStreamer
from .modeling_parler_tts import ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration
from .processing_parler_tts import ParlerTTSProcessor


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoModel.register(ParlerTTSDecoderConfig, ParlerTTSForCausalLM)
AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
AutoModelForTextToWaveform.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration, exist_ok=True)
AutoProcessor.register(ParlerTTSConfig, ParlerTTSProcessor)
