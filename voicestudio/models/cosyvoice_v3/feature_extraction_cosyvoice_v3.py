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
"""Feature extractor class for CosyVoice v3."""

from ..cosyvoice_v2.feature_extraction_cosyvoice_v2 import CosyVoiceV2FeatureExtractor


class CosyVoiceV3FeatureExtractor(CosyVoiceV2FeatureExtractor):
    r"""
    Constructs a CosyVoice v3 feature extractor, which is v2's unchanged: the flow matching model of
    both versions is conditioned on the same 24 kHz, 80 bin log mel spectrogram.

    Args:
        kwargs:
            Forwarded to [`CosyVoiceV2FeatureExtractor`].
    """


__all__ = ["CosyVoiceV3FeatureExtractor"]
