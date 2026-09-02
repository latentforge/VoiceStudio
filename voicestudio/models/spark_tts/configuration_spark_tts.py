# Copyright (c) 2025 SparkAudio
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

"""SparkTTS model configuration"""

from typing import Dict, List, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SparkTTSConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SparkTTSForConditionalGeneration`]. It is used to
    instantiate a SparkTTS model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SparkTTS
    [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        bicodec_path (`str`, *optional*, defaults to `"BiCodec"`):
            Path to the BiCodec audio tokenizer model within the pretrained model directory.
        llm_path (`str`, *optional*, defaults to `"LLM"`):
            Path to the LLM (Qwen2.5) model within the pretrained model directory.
        wav2vec2_path (`str`, *optional*, defaults to `"wav2vec2-large-xlsr-53"`):
            Path to the Wav2Vec2 feature extractor model within the pretrained model directory.
        sample_rate (`int`, *optional*, defaults to 16000):
            The audio sampling rate in Hz.
        highpass_cutoff_freq (`int`, *optional*, defaults to 40):
            High-pass filter cutoff frequency in Hz.
        segment_duration (`float`, *optional*, defaults to 2.4):
            Duration in seconds of audio segments for training.
        max_val_duration (`float`, *optional*, defaults to 12.0):
            Maximum duration in seconds for validation audio.
        latent_hop_length (`int`, *optional*, defaults to 320):
            Hop length for the latent representation in the BiCodec model.
        ref_segment_duration (`float`, *optional*, defaults to 6.0):
            Duration in seconds of the reference audio segment used for speaker embedding extraction.
        volume_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the volume of input audio.
        audio_tokenizer_config (`Dict`, *optional*):
            Configuration dictionary for the BiCodec audio tokenizer. If not provided, default configuration will be used.
        max_new_tokens (`int`, *optional*, defaults to 3000):
            Maximum number of new tokens to generate during inference.
        temperature (`float`, *optional*, defaults to 0.8):
            The value used to modulate the next token probabilities during generation.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering during generation.
        top_p (`float`, *optional*, defaults to 0.95):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.

    Example:

    ```python
    >>> from voicestudio.models.spark_tts import SparkTTSConfig, SparkTTSForConditionalGeneration

    >>> # Initializing a SparkTTS configuration
    >>> configuration = SparkTTSConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = SparkTTSForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "spark_tts"

    def __init__(
        self,
        bicodec_path: str = "BiCodec",
        llm_path: str = "LLM",
        wav2vec2_path: str = "wav2vec2-large-xlsr-53",
        sample_rate: int = 16000,
        highpass_cutoff_freq: int = 40,
        segment_duration: float = 2.4,
        max_val_duration: float = 12.0,
        latent_hop_length: int = 320,
        ref_segment_duration: float = 6.0,
        volume_normalize: bool = True,
        audio_tokenizer_config: Optional[Dict] = None,
        max_new_tokens: int = 3000,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Sub-model paths
        self.bicodec_path = bicodec_path
        self.llm_path = llm_path
        self.wav2vec2_path = wav2vec2_path

        # Audio processing parameters
        self.sample_rate = sample_rate
        self.highpass_cutoff_freq = highpass_cutoff_freq
        self.segment_duration = segment_duration
        self.max_val_duration = max_val_duration
        self.latent_hop_length = latent_hop_length
        self.ref_segment_duration = ref_segment_duration
        self.volume_normalize = volume_normalize

        # BiCodec audio tokenizer configuration
        if audio_tokenizer_config is None:
            audio_tokenizer_config = self._get_default_audio_tokenizer_config()
        self.audio_tokenizer_config = audio_tokenizer_config

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def _get_default_audio_tokenizer_config(self) -> Dict:
        """Get default BiCodec audio tokenizer configuration from original config.yaml"""
        return {
            "mel_params": {
                "sample_rate": 16000,
                "n_fft": 1024,
                "win_length": 640,
                "hop_length": 320,
                "mel_fmin": 10,
                "mel_fmax": None,
                "num_mels": 128,
            },
            "encoder": {
                "input_channels": 1024,
                "vocos_dim": 384,
                "vocos_intermediate_dim": 2048,
                "vocos_num_layers": 12,
                "out_channels": 1024,
                "sample_ratios": [1, 1],
            },
            "decoder": {
                "input_channel": 1024,
                "channels": 1536,
                "rates": [8, 5, 4, 2],
                "kernel_sizes": [16, 11, 8, 4],
            },
            "quantizer": {
                "input_dim": 1024,
                "codebook_size": 8192,
                "codebook_dim": 8,
                "commitment": 0.25,
                "codebook_loss_weight": 2.0,
                "use_l2_normlize": True,
                "threshold_ema_dead_code": 0.2,
            },
            "speaker_encoder": {
                "input_dim": 128,
                "out_dim": 1024,
                "latent_dim": 128,
                "token_num": 32,
                "fsq_levels": [4, 4, 4, 4, 4, 4],
                "fsq_num_quantizers": 1,
            },
            "prenet": {
                "input_channels": 1024,
                "vocos_dim": 384,
                "vocos_intermediate_dim": 2048,
                "vocos_num_layers": 12,
                "out_channels": 1024,
                "condition_dim": 1024,
                "sample_ratios": [1, 1],
                "use_tanh_at_final": False,
            },
            "postnet": {
                "input_channels": 1024,
                "vocos_dim": 384,
                "vocos_intermediate_dim": 2048,
                "vocos_num_layers": 6,
                "out_channels": 1024,
                "use_tanh_at_final": False,
            },
        }
