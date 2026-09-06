# coding=utf-8
# Copyright 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du) and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForTextToWaveform,
    AutoProcessor,
)
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

# `@auto_docstring` derives the model name from a decorated class's `.../models/<name>/` source
# path and looks it up in `CONFIG_MAPPING_NAMES`/`HARDCODED_CONFIG_FOR_MODELS` at decoration time
# (i.e. while `.modeling_cosyvoice_v1` below is being imported), neither of which knows about
# voicestudio-only models; must run before that import or its "Config not found" fallback warning
# already fired.
HARDCODED_CONFIG_FOR_MODELS["cosyvoice-v1"] = "CosyVoiceV1Config"

from .configuration_cosyvoice_v1 import CosyVoiceV1Config
from .feature_extraction_cosyvoice_v1 import CosyVoiceV1FeatureExtractor
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
    CosyVoiceV1SpeakerEncoder,
    CosyVoiceV1SpeechTokenLM,
    CosyVoiceV1SpeechTokenizer,
    CosyVoiceV1VocoderOutput,
    build_speech_token_labels,
)
from .processing_cosyvoice_v1 import CosyVoiceV1Processor
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
    "CosyVoiceV1SpeakerEncoder",
    "CosyVoiceV1SpeechTokenLM",
    "CosyVoiceV1SpeechTokenizer",
    "CosyVoiceV1Tokenizer",
    "CosyVoiceV1VocoderOutput",
    "build_speech_token_labels",
    "fade_in_out",
    "nucleus_sampling",
    "random_sampling",
    "repetition_aware_sampling",
]
