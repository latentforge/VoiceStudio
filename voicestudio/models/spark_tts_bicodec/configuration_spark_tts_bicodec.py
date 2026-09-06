# Copyright 2025 SparkAudio, Xinsheng Wang and the LatentForge team. All rights reserved.
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
"""Configuration class for Spark-TTS BiCodec."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig


class SparkTTSBiCodecConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SparkTTSBiCodecModel`]. It is used to
    instantiate the BiCodec audio tokenizer of Spark-TTS according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to the BiCodec
    shipped in [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B).

    BiCodec splits speech into a time-varying semantic stream, quantized by a single factorized vector quantizer over
    self-supervised speech features, and a time-invariant global stream of speaker tokens, quantized by a finite
    scalar quantizer over a perceiver-resampled ECAPA-TDNN speaker embedding.

    Args:
        semantic_model_config (`Union[dict, PreTrainedConfig]`, *optional*):
            Configuration of the self-supervised model whose hidden states feed the semantic encoder. Defaults to a
            [`Wav2Vec2Config`] matching `facebook/wav2vec2-large-xlsr-53`.
        semantic_hidden_layers (`list[int]`, *optional*, defaults to `[11, 14, 16]`):
            Indices into the semantic model's `hidden_states` tuple whose average forms the semantic encoder input.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sample rate, in Hz, of the waveform this codec encodes and reconstructs.
        hop_length (`int`, *optional*, defaults to 320):
            Number of waveform samples per semantic token.
        hidden_size (`int`, *optional*, defaults to 1024):
            Width of the latent passed between the semantic encoder, the quantizer, the prenet and the wave generator.
            Also the width of the speaker embedding used to condition the prenet.
        vocos_dim (`int`, *optional*, defaults to 384):
            Hidden width of every ConvNeXt backbone in the semantic encoder, the prenet and the postnet.
        vocos_intermediate_dim (`int`, *optional*, defaults to 2048):
            Pointwise-convolution width inside each ConvNeXt block.
        encoder_num_layers (`int`, *optional*, defaults to 12):
            Number of ConvNeXt blocks in the semantic encoder's main backbone.
        encoder_sample_ratios (`list[int]`, *optional*, defaults to `[1, 1]`):
            Downsampling factor applied by each of the semantic encoder's resampling stages.
        prenet_num_layers (`int`, *optional*, defaults to 12):
            Number of ConvNeXt blocks in the prenet's main backbone.
        prenet_sample_ratios (`list[int]`, *optional*, defaults to `[1, 1]`):
            Upsampling factor applied by each of the prenet's resampling stages.
        prenet_use_tanh_at_final (`bool`, *optional*, defaults to `False`):
            Whether the prenet squashes its output through a hyperbolic tangent.
        postnet_num_layers (`int`, *optional*, defaults to 6):
            Number of ConvNeXt blocks in the postnet's main backbone.
        postnet_sample_ratios (`list[int]`, *optional*, defaults to `[1, 1]`):
            Upsampling factor applied by each of the postnet's resampling stages.
        postnet_use_tanh_at_final (`bool`, *optional*, defaults to `False`):
            Whether the postnet squashes its output through a hyperbolic tangent.
        resampling_num_layers (`int`, *optional*, defaults to 2):
            Number of ConvNeXt blocks in the backbone that follows each resampling stage.
        layer_scale_init_value (`float`, *optional*):
            Initial value of the per-channel scale in each ConvNeXt block. Defaults to `1 / num_layers` of the
            backbone the block belongs to.
        codebook_size (`int`, *optional*, defaults to 8192):
            Number of entries in the semantic codebook.
        codebook_dim (`int`, *optional*, defaults to 8):
            Width of the low-dimensional space the semantic latent is projected into before the codebook lookup.
        commitment_weight (`float`, *optional*, defaults to 0.25):
            Weight of the commitment term of the semantic quantizer's training loss.
        codebook_loss_weight (`float`, *optional*, defaults to 2.0):
            Weight of the codebook term of the semantic quantizer's training loss.
        codebook_ema_decay (`float`, *optional*, defaults to 0.99):
            Decay of the exponential moving average tracking how often each semantic code is used.
        threshold_ema_dead_code (`float`, *optional*, defaults to 0.2):
            Moving-average usage below which a semantic code counts as inactive.
        vq_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the semantic quantizer term in the loss [`SparkTTSBiCodecModel`] returns.
        feature_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the postnet term, the error of the postnet's prediction of the self-supervised features, in the
            loss [`SparkTTSBiCodecModel`] returns.
        mel_loss_weight (`float`, *optional*, defaults to 15.0):
            Weight of the multi-resolution log-mel reconstruction term in the loss [`SparkTTSBiCodecModel`] returns.
        mel_loss_window_lengths (`list[int]`, *optional*, defaults to `[32, 64, 128, 256, 512, 1024, 2048]`):
            Fourier transform size of each resolution of the reconstruction term. The hop size of a resolution is a
            quarter of its window.
        mel_loss_num_mel_bins (`list[int]`, *optional*, defaults to `[5, 10, 20, 40, 80, 160, 320]`):
            Number of mel filters of each resolution of the reconstruction term.
        mel_loss_clamp_eps (`float`, *optional*, defaults to 1e-05):
            Floor applied to the mel magnitudes before the logarithm of the reconstruction term.
        speaker_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight of the speaker term, the error between the unquantized and the quantized speaker embedding, in the
            loss [`SparkTTSBiCodecModel`] returns.
        d_vector_train_start (`int`, *optional*, defaults to 1000):
            Training step from which the prenet and the wave generator are conditioned on the quantized speaker
            embedding rather than on the unquantized one, and from which the speaker term enters the loss.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel filters in the reference spectrogram fed to the speaker encoder.
        speaker_encoder_channels (`int`, *optional*, defaults to 512):
            Channel width of the ECAPA-TDNN speaker encoder.
        speaker_encoder_mfa_dim (`int`, *optional*, defaults to 1536):
            Output width of the ECAPA-TDNN multi-layer feature aggregation convolution.
        speaker_encoder_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the dilated convolutions inside each SE-Res2Block.
        speaker_encoder_dilations (`list[int]`, *optional*, defaults to `[2, 3, 4]`):
            Dilation of each SE-Res2Block. Their number sets how many SE-Res2Blocks the encoder stacks, and each one
            pads by its own dilation so that the time axis is preserved.
        speaker_encoder_res2net_scale (`int`, *optional*, defaults to 8):
            Number of Res2Net splits inside each SE-Res2Block.
        speaker_encoder_se_bottleneck_dim (`int`, *optional*, defaults to 128):
            Bottleneck width of the squeeze-and-excitation projection inside each SE-Res2Block.
        speaker_encoder_attention_bottleneck_dim (`int`, *optional*, defaults to 128):
            Bottleneck width of the attentive statistics pooling layer.
        speaker_latent_dim (`int`, *optional*, defaults to 128):
            Width of each perceiver latent, i.e. of the vectors the global quantizer operates on.
        num_speaker_tokens (`int`, *optional*, defaults to 32):
            Number of global tokens emitted per utterance.
        fsq_levels (`list[int]`, *optional*, defaults to `[4, 4, 4, 4, 4, 4]`):
            Number of quantization levels per dimension of the global finite scalar quantizer. Their product is the
            size of the global codebook.
        fsq_num_quantizers (`int`, *optional*, defaults to 1):
            Number of residual stages of the global finite scalar quantizer.
        perceiver_num_layers (`int`, *optional*, defaults to 2):
            Number of cross-attention blocks in the perceiver resampler.
        perceiver_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the perceiver resampler.
        perceiver_head_dim (`int`, *optional*, defaults to 64):
            Width of each perceiver attention head.
        perceiver_ffn_multiplier (`int`, *optional*, defaults to 4):
            Multiplier setting the width of the perceiver feed-forward network.
        wave_generator_hidden_size (`int`, *optional*, defaults to 1536):
            Channel width the wave generator starts from, halved by every upsampling block.
        upsample_rates (`list[int]`, *optional*, defaults to `[8, 5, 4, 2]`):
            Stride of each wave generator upsampling block. Their product must equal `hop_length`.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[16, 11, 8, 4]`):
            Transposed-convolution kernel size of each wave generator upsampling block.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer used for all weight matrices.

    Example:

    ```python
    >>> from voicestudio.models.spark_tts_bicodec import SparkTTSBiCodecConfig, SparkTTSBiCodecModel

    >>> configuration = SparkTTSBiCodecConfig()

    >>> model = SparkTTSBiCodecModel(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "spark_tts_bicodec"
    sub_configs = {"semantic_model_config": AutoConfig}

    def __init__(
        self,
        semantic_model_config: dict | PreTrainedConfig | None = None,
        semantic_hidden_layers: list[int] = [11, 14, 16],
        sampling_rate: int = 16000,
        hop_length: int = 320,
        hidden_size: int = 1024,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 2048,
        encoder_num_layers: int = 12,
        encoder_sample_ratios: list[int] = [1, 1],
        prenet_num_layers: int = 12,
        prenet_sample_ratios: list[int] = [1, 1],
        prenet_use_tanh_at_final: bool = False,
        postnet_num_layers: int = 6,
        postnet_sample_ratios: list[int] = [1, 1],
        postnet_use_tanh_at_final: bool = False,
        resampling_num_layers: int = 2,
        layer_scale_init_value: float | None = None,
        codebook_size: int = 8192,
        codebook_dim: int = 8,
        commitment_weight: float = 0.25,
        codebook_loss_weight: float = 2.0,
        codebook_ema_decay: float = 0.99,
        threshold_ema_dead_code: float = 0.2,
        vq_loss_weight: float = 1.0,
        feature_loss_weight: float = 1.0,
        mel_loss_weight: float = 15.0,
        mel_loss_window_lengths: list[int] = [32, 64, 128, 256, 512, 1024, 2048],
        mel_loss_num_mel_bins: list[int] = [5, 10, 20, 40, 80, 160, 320],
        mel_loss_clamp_eps: float = 1e-5,
        speaker_loss_weight: float = 1.0,
        d_vector_train_start: int = 1000,
        num_mel_bins: int = 128,
        speaker_encoder_channels: int = 512,
        speaker_encoder_mfa_dim: int = 1536,
        speaker_encoder_kernel_size: int = 3,
        speaker_encoder_dilations: list[int] = [2, 3, 4],
        speaker_encoder_res2net_scale: int = 8,
        speaker_encoder_se_bottleneck_dim: int = 128,
        speaker_encoder_attention_bottleneck_dim: int = 128,
        speaker_latent_dim: int = 128,
        num_speaker_tokens: int = 32,
        fsq_levels: list[int] = [4, 4, 4, 4, 4, 4],
        fsq_num_quantizers: int = 1,
        perceiver_num_layers: int = 2,
        perceiver_num_attention_heads: int = 8,
        perceiver_head_dim: int = 64,
        perceiver_ffn_multiplier: int = 4,
        wave_generator_hidden_size: int = 1536,
        upsample_rates: list[int] = [8, 5, 4, 2],
        upsample_kernel_sizes: list[int] = [16, 11, 8, 4],
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if isinstance(semantic_model_config, dict):
            semantic_model_config["model_type"] = semantic_model_config.get("model_type", "wav2vec2")
            semantic_model_config = CONFIG_MAPPING[semantic_model_config["model_type"]](**semantic_model_config)
        elif semantic_model_config is None:
            semantic_model_config = CONFIG_MAPPING["wav2vec2"](
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
                do_stable_layer_norm=True,
                feat_extract_norm="layer",
                conv_bias=True,
            )
        self.semantic_model_config = semantic_model_config

        self.semantic_hidden_layers = semantic_hidden_layers
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.encoder_num_layers = encoder_num_layers
        self.encoder_sample_ratios = encoder_sample_ratios
        self.prenet_num_layers = prenet_num_layers
        self.prenet_sample_ratios = prenet_sample_ratios
        self.prenet_use_tanh_at_final = prenet_use_tanh_at_final
        self.postnet_num_layers = postnet_num_layers
        self.postnet_sample_ratios = postnet_sample_ratios
        self.postnet_use_tanh_at_final = postnet_use_tanh_at_final
        self.resampling_num_layers = resampling_num_layers
        self.layer_scale_init_value = layer_scale_init_value
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.codebook_ema_decay = codebook_ema_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq_loss_weight = vq_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.mel_loss_window_lengths = mel_loss_window_lengths
        self.mel_loss_num_mel_bins = mel_loss_num_mel_bins
        self.mel_loss_clamp_eps = mel_loss_clamp_eps
        self.speaker_loss_weight = speaker_loss_weight
        self.d_vector_train_start = d_vector_train_start
        self.num_mel_bins = num_mel_bins
        self.speaker_encoder_channels = speaker_encoder_channels
        self.speaker_encoder_mfa_dim = speaker_encoder_mfa_dim
        self.speaker_encoder_kernel_size = speaker_encoder_kernel_size
        self.speaker_encoder_dilations = speaker_encoder_dilations
        self.speaker_encoder_res2net_scale = speaker_encoder_res2net_scale
        self.speaker_encoder_se_bottleneck_dim = speaker_encoder_se_bottleneck_dim
        self.speaker_encoder_attention_bottleneck_dim = speaker_encoder_attention_bottleneck_dim
        self.speaker_latent_dim = speaker_latent_dim
        self.num_speaker_tokens = num_speaker_tokens
        self.fsq_levels = fsq_levels
        self.fsq_num_quantizers = fsq_num_quantizers
        self.perceiver_num_layers = perceiver_num_layers
        self.perceiver_num_attention_heads = perceiver_num_attention_heads
        self.perceiver_head_dim = perceiver_head_dim
        self.perceiver_ffn_multiplier = perceiver_ffn_multiplier
        self.wave_generator_hidden_size = wave_generator_hidden_size
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def quantization_levels(self) -> list[int]:
        """`fsq_levels` under the name [`Xcodec2FiniteScalarQuantization`] reads its levels by."""
        return self.fsq_levels

    @property
    def global_codebook_size(self) -> int:
        """Number of distinct global tokens one residual stage of the finite scalar quantizer can emit."""
        size = 1
        for level in self.fsq_levels:
            size *= level
        return size


__all__ = ["SparkTTSBiCodecConfig"]
