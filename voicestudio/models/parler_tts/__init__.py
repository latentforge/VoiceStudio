from transformers import AutoConfig, AutoModel

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .generation_parler_tts import ParlerTTSStreamer
from .modeling_parler_tts import ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoModel.register(ParlerTTSDecoderConfig, ParlerTTSForCausalLM)
AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
