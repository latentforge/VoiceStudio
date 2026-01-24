from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_higgs_audio import HiggsAudioConfig, HiggsAudioEncoderConfig, HiggsAudioGenerationConfig
from .modeling_higgs_audio import HiggsAudioModel, HiggsAudioForConditionalGeneration
from .processing_higgs_audio import HiggsAudioProcessor


AutoConfig.register("higgs_audio_encoder", HiggsAudioEncoderConfig)
AutoConfig.register("higgs_audio", HiggsAudioConfig)
#AutoConfig.register("higgs_audio_generation", HiggsAudioGenerationConfig)
AutoModel.register(HiggsAudioConfig, HiggsAudioModel)
AutoModel.register(HiggsAudioConfig, HiggsAudioForConditionalGeneration)
AutoProcessor.register(HiggsAudioConfig, HiggsAudioProcessor)
