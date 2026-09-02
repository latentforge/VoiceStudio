"""Configuration class for Spark-TTS."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


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
        semantic_model_config (`Union[dict, Wav2Vec2Config]`, *optional*):
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
        postnet_num_layers (`int`, *optional*, defaults to 6):
            Number of ConvNeXt blocks in the postnet's main backbone.
        postnet_sample_ratios (`list[int]`, *optional*, defaults to `[1, 1]`):
            Upsampling factor applied by each of the postnet's resampling stages.
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
    >>> from voicestudio.models.spark_tts import SparkTTSBiCodecConfig, SparkTTSBiCodecModel

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
        postnet_num_layers: int = 6,
        postnet_sample_ratios: list[int] = [1, 1],
        resampling_num_layers: int = 2,
        layer_scale_init_value: float | None = None,
        codebook_size: int = 8192,
        codebook_dim: int = 8,
        commitment_weight: float = 0.25,
        codebook_loss_weight: float = 2.0,
        codebook_ema_decay: float = 0.99,
        threshold_ema_dead_code: float = 0.2,
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
        self.postnet_num_layers = postnet_num_layers
        self.postnet_sample_ratios = postnet_sample_ratios
        self.resampling_num_layers = resampling_num_layers
        self.layer_scale_init_value = layer_scale_init_value
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.codebook_ema_decay = codebook_ema_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
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
    def global_codebook_size(self) -> int:
        size = 1
        for level in self.fsq_levels:
            size *= level
        return size


class SparkTTSConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SparkTTSForConditionalGeneration`]. It is used
    to instantiate a Spark-TTS model according to the specified arguments, defining a [`Qwen2Model`] backbone whose
    vocabulary is extended with BiCodec semantic and global tokens. Instantiating a configuration with the defaults
    will yield a similar configuration to that of
    [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B).

    Args:
        text_config ([`Qwen2Config`], *optional*):
            Configuration of the [`Qwen2Model`] backbone. Its `vocab_size` covers ordinary text tokens as well as the
            BiCodec semantic and global tokens and the task/style control tokens.
        audio_tokenizer_config ([`SparkTTSBiCodecConfig`], *optional*):
            Configuration of the BiCodec audio tokenizer that turns reference audio into global tokens and turns
            generated semantic tokens back into a waveform.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sample rate, in Hz, of the waveform produced by the audio tokenizer.
        ref_segment_duration (`float`, *optional*, defaults to 6.0):
            Duration, in seconds, of the reference clip the speaker encoder derives global tokens from.
        volume_normalize (`bool`, *optional*, defaults to `True`):
            Whether reference audio is volume-normalized before it is encoded.

    Example:

    ```python
    >>> from voicestudio.models.spark_tts import SparkTTSConfig, SparkTTSForConditionalGeneration

    >>> configuration = SparkTTSConfig()

    >>> model = SparkTTSForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "spark_tts"
    sub_configs = {"text_config": Qwen2Config, "audio_tokenizer_config": SparkTTSBiCodecConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: Qwen2Config | dict | None = None,
        audio_tokenizer_config: SparkTTSBiCodecConfig | dict | None = None,
        sampling_rate: int = 16000,
        ref_segment_duration: float = 6.0,
        volume_normalize: bool = True,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        elif text_config is None:
            text_config = Qwen2Config(
                vocab_size=166000,
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=24,
                num_attention_heads=14,
                num_key_value_heads=2,
                tie_word_embeddings=True,
            )
        self.text_config = text_config

        if isinstance(audio_tokenizer_config, dict):
            audio_tokenizer_config = SparkTTSBiCodecConfig(**audio_tokenizer_config)
        elif audio_tokenizer_config is None:
            audio_tokenizer_config = SparkTTSBiCodecConfig()
        self.audio_tokenizer_config = audio_tokenizer_config

        self.sampling_rate = sampling_rate
        self.ref_segment_duration = ref_segment_duration
        self.volume_normalize = volume_normalize
        # `tie_weights` reads this off the top-level config, not off `text_config`.
        kwargs.setdefault("tie_word_embeddings", text_config.tie_word_embeddings)
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PreTrainedConfig:
        del decoder
        return self.text_config

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def initializer_range(self) -> float:
        return self.text_config.initializer_range


__all__ = ["SparkTTSBiCodecConfig", "SparkTTSConfig"]
