from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForTextToWaveform,
    AutoProcessor,
)

from .configuration_cosyvoice_v1 import CosyVoiceV1Config
from .generation_cosyvoice_v1 import (
    CosyVoiceV1GenerationMixin,
    fade_in_out,
    nucleus_sampling,
    random_sampling,
    repetition_aware_sampling,
)
from .modeling_cosyvoice_v1 import (
    CosyVoiceV1Attention,
    CosyVoiceV1ConditionalCFM,
    CosyVoiceV1ConditionalDecoder,
    CosyVoiceV1Encoder,
    CosyVoiceV1FlowModel,
    CosyVoiceV1ForConditionalGeneration,
    CosyVoiceV1HiFTGenerator,
    CosyVoiceV1LabelSmoothingLoss,
    CosyVoiceV1Output,
    CosyVoiceV1PreTrainedModel,
    CosyVoiceV1SpeechTokenLM,
    CosyVoiceV1VocoderOutput,
    build_speech_token_labels,
)
from .processing_cosyvoice_v1 import CosyVoiceV1FeatureExtractor, CosyVoiceV1Processor
from .tokenization_cosyvoice_v1 import CosyVoiceV1Tokenizer


AutoConfig.register(CosyVoiceV1Config.model_type, CosyVoiceV1Config, exist_ok=True)
AutoModel.register(CosyVoiceV1Config, CosyVoiceV1ForConditionalGeneration, exist_ok=True)
AutoModelForTextToWaveform.register(CosyVoiceV1Config, CosyVoiceV1ForConditionalGeneration, exist_ok=True)
AutoFeatureExtractor.register(CosyVoiceV1Config, CosyVoiceV1FeatureExtractor, exist_ok=True)
AutoProcessor.register(CosyVoiceV1Config, CosyVoiceV1Processor, exist_ok=True)


__all__ = [
    "CosyVoiceV1Attention",
    "CosyVoiceV1ConditionalCFM",
    "CosyVoiceV1ConditionalDecoder",
    "CosyVoiceV1Config",
    "CosyVoiceV1Encoder",
    "CosyVoiceV1FeatureExtractor",
    "CosyVoiceV1FlowModel",
    "CosyVoiceV1ForConditionalGeneration",
    "CosyVoiceV1GenerationMixin",
    "CosyVoiceV1HiFTGenerator",
    "CosyVoiceV1LabelSmoothingLoss",
    "CosyVoiceV1Output",
    "CosyVoiceV1PreTrainedModel",
    "CosyVoiceV1Processor",
    "CosyVoiceV1SpeechTokenLM",
    "CosyVoiceV1Tokenizer",
    "CosyVoiceV1VocoderOutput",
    "build_speech_token_labels",
    "fade_in_out",
    "nucleus_sampling",
    "random_sampling",
    "repetition_aware_sampling",
]
