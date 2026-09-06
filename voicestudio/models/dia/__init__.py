from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig
from .feature_extraction_dia import DiaFeatureExtractor
from .modeling_dia import DiaForConditionalGeneration, DiaModel, DiaPreTrainedModel
from .processing_dia import DiaProcessor
from .tokenization_dia import DiaTokenizer


AutoConfig.register(DiaConfig.model_type, DiaConfig, exist_ok=True)
AutoModel.register(DiaConfig, DiaForConditionalGeneration, exist_ok=True)
AutoProcessor.register(DiaConfig, DiaProcessor, exist_ok=True)
