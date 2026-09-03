"""Configuration class for PromptTTS++."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.bert.configuration_bert import BertConfig


class PromptTTSPPConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PromptTTSPPForConditionalGeneration`]. It is
    used to instantiate a PromptTTS++ model according to the specified arguments, defining a conformer phoneme
    encoder, a mixture-density-network variance adaptor with a frame prior network, a global-style-token reference
    encoder, a BERT prompt encoder with its own style mixture density network, and a denoising diffusion mel
    decoder.

    Args:
        vocab_size (`int`, *optional*, defaults to 90):
            Size of the phoneme vocabulary, the number of entries in [`PromptTTSPPTokenizer`]'s symbol table.
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the phoneme embedding, the conformer encoder and the frame-level conditioning fed to
            the diffusion decoder.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel filterbank channels of the spectrogram the model predicts and of the reference
            spectrogram the style encoder reads.
        scale_phoneme_embedding (`bool`, *optional*, defaults to `False`):
            Whether to scale the phoneme embedding by `sqrt(hidden_size)`.
        encoder_layers (`int`, *optional*, defaults to 4):
            Number of conformer blocks in the phoneme encoder.
        encoder_num_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads of each conformer block.
        encoder_linear_units (`int`, *optional*, defaults to 1024):
            Number of channels in the hidden layer of a conformer block's position-wise convolutions.
        encoder_kernel_size (`int`, *optional*, defaults to 7):
            Kernel size of the depthwise convolution in a conformer block's convolution module.
        positionwise_conv_kernel_size (`int`, *optional*, defaults to 9):
            Kernel size of a conformer block's position-wise convolutions.
        encoder_dropout_rate (`float`, *optional*, defaults to 0.2):
            Dropout rate of the conformer blocks.
        encoder_positional_dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the encoder input and to its relative positional embedding.
        encoder_attention_dropout_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate of the conformer attention weights.
        encoder_normalize_before (`bool`, *optional*, defaults to `True`):
            Whether a conformer block normalizes its input before each sub-layer rather than its output after it.
        encoder_concat_after (`bool`, *optional*, defaults to `False`):
            Whether a conformer block concatenates the attention input and output and projects them, rather than
            adding the attention output to the residual.
        use_macaron_style_in_conformer (`bool`, *optional*, defaults to `True`):
            Whether a conformer block runs a second, half-weighted feed-forward module before attention.
        use_cnn_in_conformer (`bool`, *optional*, defaults to `True`):
            Whether a conformer block contains a convolution module.
        convolution_bias (`bool`, *optional*, defaults to `True`):
            Whether the convolutions of the conformer convolution module use a bias.
        rel_pos_type (`str`, *optional*, defaults to `"legacy"`):
            Relative positional encoding variant, `"legacy"` or `"new"`. Both carry identical parameters, so a
            checkpoint trained with one loads without complaint into the other and only its output differs. The
            released checkpoint uses `"legacy"`.
        max_source_positions (`int`, *optional*, defaults to 5000):
            Number of positions the relative positional encoding is precomputed for.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation of the conformer convolution module. `"silu"` is the swish of the original implementation.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            Epsilon of the conformer layer normalizations.
        variance_layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon of the layer normalizations in the variance adaptor and the frame prior network.
        duration_predictor_layers (`int`, *optional*, defaults to 2):
            Number of convolution layers before the duration predictor's mixture density network.
        duration_predictor_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the duration predictor's convolutions.
        duration_predictor_dropout (`float`, *optional*, defaults to 0.5):
            Dropout rate of the duration predictor.
        duration_predictor_num_gaussians (`int`, *optional*, defaults to 4):
            Number of mixture components of the duration predictor's mixture density network.
        stop_gradient_from_duration_predictor (`bool`, *optional*, defaults to `True`):
            Whether to detach the encoder output before the duration predictor.
        pitch_predictor_layers (`int`, *optional*, defaults to 5):
            Number of convolution layers of the pitch predictor, which predicts log continuous f0 and voicing
            jointly.
        pitch_predictor_kernel_size (`int`, *optional*, defaults to 5):
            Kernel size of the pitch predictor's convolutions.
        pitch_predictor_dropout (`float`, *optional*, defaults to 0.5):
            Dropout rate of the pitch predictor.
        stop_gradient_from_pitch_predictor (`bool`, *optional*, defaults to `False`):
            Whether to detach the frame-level features before the pitch predictor.
        pitch_embed_kernel_size (`int`, *optional*, defaults to 1):
            Kernel size of the convolution embedding log continuous f0 back into the frame-level features.
        use_energy_predictor (`bool`, *optional*, defaults to `False`):
            Whether the variance adaptor carries an energy branch. The released checkpoint has none.
        energy_predictor_layers (`int`, *optional*, defaults to 2):
            Number of convolution layers of the energy predictor.
        energy_predictor_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the energy predictor's convolutions.
        energy_predictor_dropout (`float`, *optional*, defaults to 0.5):
            Dropout rate of the energy predictor.
        stop_gradient_from_energy_predictor (`bool`, *optional*, defaults to `False`):
            Whether to detach the frame-level features before the energy predictor.
        energy_embed_kernel_size (`int`, *optional*, defaults to 1):
            Kernel size of the convolution embedding energy back into the frame-level features.
        frame_prior_layers (`int`, *optional*, defaults to 6):
            Number of convolution layers of the frame prior network that follows length regulation.
        frame_prior_kernel_size (`int`, *optional*, defaults to 17):
            Kernel size of the frame prior network's convolutions.
        frame_prior_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate of the frame prior network.
        frame_prior_positional_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied after the frame prior network's absolute positional encoding.
        gst_tokens (`int`, *optional*, defaults to 10):
            Number of global style tokens.
        gst_token_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the style embedding produced by the style token layer.
        gst_heads (`int`, *optional*, defaults to 4):
            Number of attention heads of the style token layer.
        reference_encoder_conv_layers (`int`, *optional*, defaults to 6):
            Number of 2D convolution layers of the reference encoder.
        reference_encoder_conv_channels (`list[int]`, *optional*, defaults to `[128, 128, 256, 256, 512, 512]`):
            Output channels of each reference encoder convolution layer.
        reference_encoder_conv_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the reference encoder's convolutions.
        reference_encoder_conv_stride (`int`, *optional*, defaults to 2):
            Stride of the reference encoder's convolutions.
        reference_encoder_gru_layers (`int`, *optional*, defaults to 1):
            Number of GRU layers summarizing the reference encoder's convolution output.
        reference_encoder_gru_units (`int`, *optional*, defaults to 256):
            Hidden size of the reference encoder's GRU.
        prompt_encoder_config ([`BertConfig`], *optional*):
            Configuration of the [`BertModel`] encoding the natural language style prompt. Defaults to the
            `bert-base-uncased` architecture the released checkpoint was trained with.
        prompt_adapter_hidden_size (`int`, *optional*, defaults to 512):
            Hidden size of the multilayer perceptron mapping the prompt encoder's pooled output to a style
            embedding.
        freeze_prompt_encoder (`bool`, *optional*, defaults to `True`):
            Whether to freeze the prompt encoder except for the attention of its last layer, as training does.
        use_style_mdn (`bool`, *optional*, defaults to `True`):
            Whether a mixture density network predicts the style embedding from the prompt embedding. When
            `False`, the prompt embedding is used directly and its loss becomes a mean squared error against the
            reference style embedding.
        style_num_gaussians (`int`, *optional*, defaults to 10):
            Number of mixture components of the style mixture density network predicting the style embedding from
            the prompt embedding.
        num_diffusion_steps (`int`, *optional*, defaults to 100):
            Number of steps of the decoder's diffusion process.
        beta_schedule (`str`, *optional*, defaults to `"linear"`):
            Noise schedule of the diffusion decoder, `"linear"` or `"cosine"`.
        beta_start (`float`, *optional*, defaults to 0.0001):
            First beta of the linear noise schedule.
        beta_end (`float`, *optional*, defaults to 0.06):
            Last beta of the linear noise schedule.
        cosine_beta_shift (`float`, *optional*, defaults to 0.008):
            Offset of the cosine noise schedule.
        diffusion_norm_scale (`float`, *optional*, defaults to 6.0):
            Divisor scaling the mel spectrogram into the diffusion decoder's value range. When `None`, the
            spectrogram is rescaled from `[diffusion_min_value, diffusion_max_value]` to `[-1, 1]` instead.
        diffusion_min_value (`float`, *optional*, defaults to 0.0):
            Lower end of the spectrogram range used when `diffusion_norm_scale` is `None`.
        diffusion_max_value (`float`, *optional*, defaults to 20.0):
            Upper end of the spectrogram range used when `diffusion_norm_scale` is `None`.
        denoiser_layers (`int`, *optional*, defaults to 20):
            Number of residual blocks of the denoiser.
        denoiser_channels (`int`, *optional*, defaults to 256):
            Number of channels of the denoiser's residual blocks.
        denoiser_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the denoiser's dilated convolutions.
        denoiser_dilation_cycle_length (`int`, *optional*, defaults to 4):
            Length of the denoiser's dilation cycle, whose `i`-th block uses dilation
            `2 ** (i % denoiser_dilation_cycle_length)`.
        normalize_style_embedding (`bool`, *optional*, defaults to `True`):
            Whether style and prompt embeddings are normalized to unit norm.
        disable_mdn_autocast (`bool`, *optional*, defaults to `True`):
            Whether the mixture density networks and their losses run in full precision under autocast.
        spectrogram_loss_scale (`float`, *optional*, defaults to 8.0):
            Divisor of the diffusion loss term.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer of the linear and convolution weights.

    Example:

    ```python
    >>> from voicestudio.models.prompt_tts_pp import PromptTTSPPConfig, PromptTTSPPForConditionalGeneration

    >>> configuration = PromptTTSPPConfig()

    >>> model = PromptTTSPPForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "prompt_tts_pp"
    sub_configs = {"prompt_encoder_config": BertConfig}
    attribute_map = {
        "num_hidden_layers": "encoder_layers",
        "num_attention_heads": "encoder_num_attention_heads",
    }

    def __init__(
        self,
        vocab_size: int = 90,
        hidden_size: int = 256,
        num_mel_bins: int = 80,
        scale_phoneme_embedding: bool = False,
        encoder_layers: int = 4,
        encoder_num_attention_heads: int = 2,
        encoder_linear_units: int = 1024,
        encoder_kernel_size: int = 7,
        positionwise_conv_kernel_size: int = 9,
        encoder_dropout_rate: float = 0.2,
        encoder_positional_dropout_rate: float = 0.1,
        encoder_attention_dropout_rate: float = 0.0,
        encoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        convolution_bias: bool = True,
        rel_pos_type: str = "legacy",
        max_source_positions: int = 5000,
        hidden_act: str = "silu",
        layer_norm_eps: float = 1e-12,
        variance_layer_norm_eps: float = 1e-5,
        duration_predictor_layers: int = 2,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout: float = 0.5,
        duration_predictor_num_gaussians: int = 4,
        stop_gradient_from_duration_predictor: bool = True,
        pitch_predictor_layers: int = 5,
        pitch_predictor_kernel_size: int = 5,
        pitch_predictor_dropout: float = 0.5,
        stop_gradient_from_pitch_predictor: bool = False,
        pitch_embed_kernel_size: int = 1,
        use_energy_predictor: bool = False,
        energy_predictor_layers: int = 2,
        energy_predictor_kernel_size: int = 3,
        energy_predictor_dropout: float = 0.5,
        stop_gradient_from_energy_predictor: bool = False,
        energy_embed_kernel_size: int = 1,
        frame_prior_layers: int = 6,
        frame_prior_kernel_size: int = 17,
        frame_prior_dropout: float = 0.1,
        frame_prior_positional_dropout: float = 0.1,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        reference_encoder_conv_layers: int = 6,
        reference_encoder_conv_channels: list[int] | None = None,
        reference_encoder_conv_kernel_size: int = 3,
        reference_encoder_conv_stride: int = 2,
        reference_encoder_gru_layers: int = 1,
        reference_encoder_gru_units: int = 256,
        prompt_encoder_config: BertConfig | dict | None = None,
        prompt_adapter_hidden_size: int = 512,
        freeze_prompt_encoder: bool = True,
        use_style_mdn: bool = True,
        style_num_gaussians: int = 10,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.06,
        cosine_beta_shift: float = 0.008,
        diffusion_norm_scale: float | None = 6.0,
        diffusion_min_value: float = 0.0,
        diffusion_max_value: float = 20.0,
        denoiser_layers: int = 20,
        denoiser_channels: int = 256,
        denoiser_kernel_size: int = 3,
        denoiser_dilation_cycle_length: int = 4,
        normalize_style_embedding: bool = True,
        disable_mdn_autocast: bool = True,
        spectrogram_loss_scale: float = 8.0,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if rel_pos_type not in ("legacy", "new"):
            raise ValueError(f"rel_pos_type must be 'legacy' or 'new', got {rel_pos_type}.")
        if beta_schedule not in ("linear", "cosine"):
            raise ValueError(f"beta_schedule must be 'linear' or 'cosine', got {beta_schedule}.")
        if positionwise_conv_kernel_size % 2 == 0:
            raise ValueError(
                f"positionwise_conv_kernel_size must be odd, got {positionwise_conv_kernel_size}."
            )
        if encoder_kernel_size % 2 == 0:
            raise ValueError(f"encoder_kernel_size must be odd, got {encoder_kernel_size}.")

        if isinstance(prompt_encoder_config, dict):
            prompt_encoder_config = BertConfig(**prompt_encoder_config)
        elif prompt_encoder_config is None:
            prompt_encoder_config = BertConfig()
        self.prompt_encoder_config = prompt_encoder_config

        if reference_encoder_conv_channels is None:
            reference_encoder_conv_channels = [128, 128, 256, 256, 512, 512]
        if len(reference_encoder_conv_channels) != reference_encoder_conv_layers:
            raise ValueError(
                "reference_encoder_conv_channels must hold one entry per reference encoder convolution layer, "
                f"got {len(reference_encoder_conv_channels)} entries for "
                f"{reference_encoder_conv_layers} layers."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_mel_bins = num_mel_bins
        self.scale_phoneme_embedding = scale_phoneme_embedding
        self.encoder_layers = encoder_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_linear_units = encoder_linear_units
        self.encoder_kernel_size = encoder_kernel_size
        self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
        self.encoder_dropout_rate = encoder_dropout_rate
        self.encoder_positional_dropout_rate = encoder_positional_dropout_rate
        self.encoder_attention_dropout_rate = encoder_attention_dropout_rate
        self.encoder_normalize_before = encoder_normalize_before
        self.encoder_concat_after = encoder_concat_after
        self.use_macaron_style_in_conformer = use_macaron_style_in_conformer
        self.use_cnn_in_conformer = use_cnn_in_conformer
        self.convolution_bias = convolution_bias
        self.rel_pos_type = rel_pos_type
        self.max_source_positions = max_source_positions
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.variance_layer_norm_eps = variance_layer_norm_eps
        self.duration_predictor_layers = duration_predictor_layers
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_dropout = duration_predictor_dropout
        self.duration_predictor_num_gaussians = duration_predictor_num_gaussians
        self.stop_gradient_from_duration_predictor = stop_gradient_from_duration_predictor
        self.pitch_predictor_layers = pitch_predictor_layers
        self.pitch_predictor_kernel_size = pitch_predictor_kernel_size
        self.pitch_predictor_dropout = pitch_predictor_dropout
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.pitch_embed_kernel_size = pitch_embed_kernel_size
        self.use_energy_predictor = use_energy_predictor
        self.energy_predictor_layers = energy_predictor_layers
        self.energy_predictor_kernel_size = energy_predictor_kernel_size
        self.energy_predictor_dropout = energy_predictor_dropout
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.energy_embed_kernel_size = energy_embed_kernel_size
        self.frame_prior_layers = frame_prior_layers
        self.frame_prior_kernel_size = frame_prior_kernel_size
        self.frame_prior_dropout = frame_prior_dropout
        self.frame_prior_positional_dropout = frame_prior_positional_dropout
        self.gst_tokens = gst_tokens
        self.gst_token_dim = gst_token_dim
        self.gst_heads = gst_heads
        self.reference_encoder_conv_layers = reference_encoder_conv_layers
        self.reference_encoder_conv_channels = reference_encoder_conv_channels
        self.reference_encoder_conv_kernel_size = reference_encoder_conv_kernel_size
        self.reference_encoder_conv_stride = reference_encoder_conv_stride
        self.reference_encoder_gru_layers = reference_encoder_gru_layers
        self.reference_encoder_gru_units = reference_encoder_gru_units
        self.prompt_adapter_hidden_size = prompt_adapter_hidden_size
        self.freeze_prompt_encoder = freeze_prompt_encoder
        self.use_style_mdn = use_style_mdn
        self.style_num_gaussians = style_num_gaussians
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cosine_beta_shift = cosine_beta_shift
        self.diffusion_norm_scale = diffusion_norm_scale
        self.diffusion_min_value = diffusion_min_value
        self.diffusion_max_value = diffusion_max_value
        self.denoiser_layers = denoiser_layers
        self.denoiser_channels = denoiser_channels
        self.denoiser_kernel_size = denoiser_kernel_size
        self.denoiser_dilation_cycle_length = denoiser_dilation_cycle_length
        self.normalize_style_embedding = normalize_style_embedding
        self.disable_mdn_autocast = disable_mdn_autocast
        self.spectrogram_loss_scale = spectrogram_loss_scale
        self.initializer_range = initializer_range

        self.encoder_config = {
            "num_attention_heads": encoder_num_attention_heads,
            "layers": encoder_layers,
            "kernel_size": encoder_kernel_size,
            "attention_dropout_rate": encoder_attention_dropout_rate,
            "dropout_rate": encoder_dropout_rate,
            "positional_dropout_rate": encoder_positional_dropout_rate,
            "linear_units": encoder_linear_units,
            "normalize_before": encoder_normalize_before,
            "concat_after": encoder_concat_after,
            "activation": hidden_act,
        }

        super().__init__(**kwargs)


class PromptTTSPPBigVGanConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PromptTTSPPBigVGan`]. It is used to
    instantiate the f0 aware BigVGAN vocoder PromptTTS++ synthesizes with, whose upsampling stack is excited by a
    harmonic source signal built from the fundamental frequency the acoustic model predicts.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            Number of mel filterbank channels of the input spectrogram.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate, in Hz, of the generated waveform.
        harmonic_num (`int`, *optional*, defaults to 8):
            Number of overtones the source module adds above the fundamental frequency.
        sine_amplitude (`float`, *optional*, defaults to 0.1):
            Amplitude of the sine waveforms of the source signal.
        noise_std (`float`, *optional*, defaults to 0.003):
            Standard deviation of the Gaussian noise added to the voiced part of the source signal. The unvoiced
            part gets `sine_amplitude / 3` instead.
        voiced_threshold (`float`, *optional*, defaults to 0.0):
            Fundamental frequency, in Hz, above which a frame counts as voiced.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            Number of channels the spectrogram is projected to before the first upsampling layer.
        upsample_rates (`list[int]`, *optional*, defaults to `[6, 5, 4, 2]`):
            Upsampling factor of each transposed convolution. Their product is the number of waveform samples per
            spectrogram frame.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[12, 10, 8, 4]`):
            Kernel size of each transposed convolution.
        resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            Kernel size of each of the parallel residual blocks that follow an upsampling layer.
        resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            Dilation of each convolution of a residual block, one list per entry of `resblock_kernel_sizes`.
        anti_alias_ratio (`int`, *optional*, defaults to 2):
            Factor the activation's input is upsampled by before the snake nonlinearity and downsampled by after
            it, which keeps the harmonics the nonlinearity creates below the Nyquist frequency.
        anti_alias_kernel_size (`int`, *optional*, defaults to 12):
            Kernel size of the Kaiser windowed sinc filter of the anti aliasing resampling.
        resblock_type (`str`, *optional*, defaults to `"1"`):
            Which residual block to build after each upsampling layer, `"1"` for the block whose every dilated
            convolution is followed by an undilated one, or `"2"` for the block that holds the dilated
            convolutions alone.
        activation (`str`, *optional*, defaults to `"snake"`):
            Periodic nonlinearity of the residual blocks, `"snake"` for `x + sin(alpha * x) ** 2 / alpha` or
            `"snakebeta"` for `x + sin(alpha * x) ** 2 / beta`.
        snake_logscale (`bool`, *optional*, defaults to `True`):
            Whether `alpha` is stored as its logarithm, so that the value the nonlinearity uses is its
            exponential.
        use_tanh_at_final (`bool`, *optional*, defaults to `True`):
            Whether the waveform is bounded by a hyperbolic tangent. `False` clamps it to `[-1, 1]` instead.
        use_bias_at_final (`bool`, *optional*, defaults to `True`):
            Whether the output convolution has a bias.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer of the convolution weights.

    Example:

    ```python
    >>> from voicestudio.models.prompt_tts_pp import PromptTTSPPBigVGan, PromptTTSPPBigVGanConfig

    >>> configuration = PromptTTSPPBigVGanConfig()

    >>> model = PromptTTSPPBigVGan(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "prompt_tts_pp_big_vgan"

    def __init__(
        self,
        model_in_dim: int = 80,
        sampling_rate: int = 24000,
        harmonic_num: int = 8,
        sine_amplitude: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        upsample_initial_channel: int = 512,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        anti_alias_ratio: int = 2,
        anti_alias_kernel_size: int = 12,
        resblock_type: str = "1",
        activation: str = "snake",
        snake_logscale: bool = True,
        use_tanh_at_final: bool = True,
        use_bias_at_final: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if upsample_rates is None:
            upsample_rates = [6, 5, 4, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [12, 10, 8, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError(
                "upsample_rates and upsample_kernel_sizes must hold one entry per upsampling layer, got "
                f"{len(upsample_rates)} and {len(upsample_kernel_sizes)} entries."
            )
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError(
                "resblock_kernel_sizes and resblock_dilation_sizes must hold one entry per residual block, got "
                f"{len(resblock_kernel_sizes)} and {len(resblock_dilation_sizes)} entries."
            )

        self.model_in_dim = model_in_dim
        self.sampling_rate = sampling_rate
        self.harmonic_num = harmonic_num
        self.sine_amplitude = sine_amplitude
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.anti_alias_ratio = anti_alias_ratio
        self.anti_alias_kernel_size = anti_alias_kernel_size
        self.resblock_type = resblock_type
        self.activation = activation
        self.snake_logscale = snake_logscale
        self.use_tanh_at_final = use_tanh_at_final
        self.use_bias_at_final = use_bias_at_final
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


__all__ = ["PromptTTSPPBigVGanConfig", "PromptTTSPPConfig"]
