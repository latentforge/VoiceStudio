# Copyright 2024 Boson AI. All rights reserved.
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
#
# Modifications by LatentForge:
# - Add HiggsAudioGenerationConfig
from transformers.generation import GenerationConfig

try:
    from ..._boson.model.higgs_audio.configuration_higgs_audio import HiggsAudioEncoderConfig, HiggsAudioConfig
except ImportError:
    from voicestudio._boson.model.higgs_audio.configuration_higgs_audio import HiggsAudioEncoderConfig, HiggsAudioConfig


class HiggsAudioGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        # Extract custom kwargs before calling parent
        self.audio_out_bos_token_id = kwargs.pop('audio_out_bos_token_id', None)
        self.audio_eos_token_id = kwargs.pop('audio_eos_token_id', None)
        self.ras_win_len = kwargs.pop('ras_win_len', None)
        self.ras_win_max_num_repeat = kwargs.pop('ras_win_max_num_repeat', 2)
        self.seed = kwargs.pop('seed', None)
        self.tokenizer_length = kwargs.pop('tokenizer_length', None)

        # Call parent with remaining kwargs
        super().__init__(**kwargs)
