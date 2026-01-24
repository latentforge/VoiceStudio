from transformers import AutoConfig, AutoModel

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .modeling_parler_tts import ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration
try:
    from ..._parler_tts import ParlerTTSStreamer
except ImportError:
    from voicestudio._parler_tts import ParlerTTSStreamer


AutoConfig.register("parler_tts", ParlerTTSConfig)
AutoModel.register(ParlerTTSDecoderConfig, ParlerTTSForCausalLM)
AutoModel.register(ParlerTTSConfig, ParlerTTSForConditionalGeneration)
