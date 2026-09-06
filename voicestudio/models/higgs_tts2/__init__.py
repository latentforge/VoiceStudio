from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_higgs_tts2 import HiggsTTS2Config
from .processing_higgs_tts2 import HiggsTTS2Processor
from .tokenization_higgs_tts2 import (
    HiggsTTS2TokenizerModel,
    HiggsTTS2TokenizerConfig
)
from .modeling_higgs_tts2 import (
    HiggsTTS2ForConditionalGeneration,
    HiggsTTS2PreTrainedModel,
    HiggsTTS2Model
)


AutoConfig.register(HiggsTTS2Config.model_type, HiggsTTS2Config, exist_ok=True)
AutoModel.register(HiggsTTS2Config, HiggsTTS2ForConditionalGeneration, exist_ok=True)
AutoProcessor.register(HiggsTTS2Config, HiggsTTS2Processor, exist_ok=True)
