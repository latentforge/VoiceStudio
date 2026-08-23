from transformers import AutoConfig, AutoModel

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .modeling_parler_tts import ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration
from .streamer import ParlerTTSStreamer


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoModel.register(ParlerTTSDecoderConfig, ParlerTTSForCausalLM)
AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
