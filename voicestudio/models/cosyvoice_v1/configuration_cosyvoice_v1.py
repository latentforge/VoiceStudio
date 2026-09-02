"""Configuration class for CosyVoice v1."""

from transformers.configuration_utils import PreTrainedConfig


class CosyVoiceV1TextEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV1TextEncoder`]. It configures the
    relative-position Conformer text encoder that turns text token embeddings into the conditioning sequence
    consumed by the speech-token language model.

    Args:
        input_size (`int`, *optional*, defaults to 512):
            Dimensionality of the input text token embedding.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder's hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of Conformer encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of relative-position attention heads.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the feed-forward layers.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability applied to hidden states and positional embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied to attention weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by layer normalization layers.
    """

    model_type = "cosyvoice_v1_text_encoder"
    base_config_key = "text_encoder_config"

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 1024,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        super().__init__(**kwargs)


class CosyVoiceV1LLMConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV1LLM`]. It configures the
    autoregressive speech-token language model: a relative-position Conformer/Transformer decoder that predicts
    discrete speech tokens conditioned on text-encoder output, a speaker embedding, and previously generated
    speech tokens.

    Args:
        llm_input_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the language model's input embeddings.
        llm_output_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the language model's hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 14):
            Number of Transformer decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of relative-position attention heads.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the feed-forward layers.
        text_token_size (`int`, *optional*, defaults to 51866):
            Vocabulary size of the text tokenizer.
        speech_token_size (`int`, *optional*, defaults to 4096):
            Vocabulary size of the discrete speech tokenizer. One extra id above this value is reserved for the
            end-of-speech token.
        spk_embed_dim (`int`, *optional*, defaults to 192):
            Dimensionality of the incoming x-vector speaker embedding.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability applied to hidden states and positional embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied to attention weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by layer normalization layers.
        length_normalized_loss (`bool`, *optional*, defaults to `True`):
            Whether the cross-entropy loss is normalized by sequence length (`True`) or by batch size.
        label_smoothing (`float`, *optional*, defaults to 0.0):
            Label smoothing applied to the speech-token cross-entropy loss.
    """

    model_type = "cosyvoice_v1_llm"
    base_config_key = "llm_config"

    def __init__(
        self,
        llm_input_size: int = 1024,
        llm_output_size: int = 1024,
        num_hidden_layers: int = 14,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        text_token_size: int = 51866,
        speech_token_size: int = 4096,
        spk_embed_dim: int = 192,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        length_normalized_loss: bool = True,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.text_token_size = text_token_size
        self.speech_token_size = speech_token_size
        self.spk_embed_dim = spk_embed_dim
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.length_normalized_loss = length_normalized_loss
        self.label_smoothing = label_smoothing
        super().__init__(**kwargs)


class CosyVoiceV1FlowConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV1FlowMatchingModel`]. It
    configures the conditional-flow-matching decoder that turns a sequence of discrete speech tokens into a mel
    spectrogram, conditioned on a speaker embedding.

    Args:
        input_size (`int`, *optional*, defaults to 512):
            Dimensionality of the speech-token embedding fed to the Conformer encoder.
        output_size (`int`, *optional*, defaults to 80):
            Number of mel channels the flow decoder predicts.
        spk_embed_dim (`int`, *optional*, defaults to 192):
            Dimensionality of the incoming x-vector speaker embedding.
        vocab_size (`int`, *optional*, defaults to 4096):
            Vocabulary size of the discrete speech tokenizer.
        input_frame_rate (`int`, *optional*, defaults to 50):
            Frame rate, in Hz, of the input speech token sequence.
        encoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the Conformer encoder's hidden representations.
        encoder_num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of Conformer encoder layers.
        encoder_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of relative-position attention heads in the encoder.
        encoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder's feed-forward layers.
        decoder_channels (`list[int]`, *optional*, defaults to `[256, 256]`):
            Channel width of each down/up-sampling stage of the [`CosyVoiceV1ConditionalDecoder`] U-Net estimator.
        decoder_attention_head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of each attention head inside the estimator's transformer blocks.
        decoder_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads inside the estimator's transformer blocks.
        decoder_n_blocks (`int`, *optional*, defaults to 4):
            Number of transformer blocks per down/up-sampling stage.
        decoder_num_mid_blocks (`int`, *optional*, defaults to 12):
            Number of resnet+transformer blocks in the estimator's bottleneck stage.
        decoder_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability inside the estimator's transformer blocks.
        sigma_min (`float`, *optional*, defaults to 1e-06):
            Minimum noise scale of the conditional-flow-matching probability path.
        t_scheduler (`str`, *optional*, defaults to `"cosine"`):
            Timestep reparameterization used by the Euler ODE solver, `"cosine"` or `"linear"`.
        training_cfg_rate (`float`, *optional*, defaults to 0.2):
            Probability of dropping the conditioning (speaker embedding, prompt mel, encoder output) during
            training, enabling classifier-free guidance at inference.
        inference_cfg_rate (`float`, *optional*, defaults to 0.7):
            Classifier-free guidance strength used by the Euler ODE solver at inference.
    """

    model_type = "cosyvoice_v1_flow"
    base_config_key = "flow_config"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        encoder_hidden_size: int = 512,
        encoder_num_hidden_layers: int = 6,
        encoder_num_attention_heads: int = 8,
        encoder_intermediate_size: int = 2048,
        decoder_channels: list[int] = [256, 256],
        decoder_attention_head_dim: int = 64,
        decoder_num_heads: int = 8,
        decoder_n_blocks: int = 4,
        decoder_num_mid_blocks: int = 12,
        decoder_dropout: float = 0.0,
        sigma_min: float = 1e-6,
        t_scheduler: str = "cosine",
        training_cfg_rate: float = 0.2,
        inference_cfg_rate: float = 0.7,
        **kwargs,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.spk_embed_dim = spk_embed_dim
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        self.decoder_channels = decoder_channels
        self.decoder_attention_head_dim = decoder_attention_head_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_n_blocks = decoder_n_blocks
        self.decoder_num_mid_blocks = decoder_num_mid_blocks
        self.decoder_dropout = decoder_dropout
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate
        super().__init__(**kwargs)


class CosyVoiceV1HiftConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV1HiFTGenerator`]. It configures
    the neural-source-filter/ISTFT vocoder that renders a mel spectrogram into a waveform.

    Args:
        in_channels (`int`, *optional*, defaults to 80):
            Number of mel channels the vocoder consumes.
        base_channels (`int`, *optional*, defaults to 512):
            Number of channels after the input convolution.
        nb_harmonics (`int`, *optional*, defaults to 8):
            Number of harmonic overtones generated by the neural source filter.
        sampling_rate (`int`, *optional*, defaults to 22050):
            Output waveform sample rate, in Hz.
        nsf_alpha (`float`, *optional*, defaults to 0.1):
            Amplitude of the neural-source-filter sine excitation.
        nsf_sigma (`float`, *optional*, defaults to 0.003):
            Standard deviation of the neural-source-filter additive noise.
        nsf_voiced_threshold (`float`, *optional*, defaults to 10):
            F0 threshold, in Hz, below which a frame is treated as unvoiced.
        upsample_rates (`list[int]`, *optional*, defaults to `[8, 8]`):
            Upsampling factor of each transposed-convolution stage.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[16, 16]`):
            Kernel size of each transposed-convolution stage.
        istft_n_fft (`int`, *optional*, defaults to 16):
            FFT size of the final inverse-STFT synthesis stage.
        istft_hop_len (`int`, *optional*, defaults to 4):
            Hop length of the final inverse-STFT synthesis stage.
        resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            Kernel size of each residual block.
        resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            Dilation schedule of each residual block.
        source_resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[7, 11]`):
            Kernel size of each residual block applied to the downsampled source excitation.
        source_resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5]]`):
            Dilation schedule of each source-excitation residual block.
        lrelu_slope (`float`, *optional*, defaults to 0.1):
            Negative slope of the leaky ReLU activations.
        audio_limit (`float`, *optional*, defaults to 0.99):
            Absolute value the output waveform is clamped to.
        f0_predictor_num_layers (`int`, *optional*, defaults to 5):
            Number of convolutional layers in the F0 predictor.
    """

    model_type = "cosyvoice_v1_hift"
    base_config_key = "hift_config"

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 22050,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: list[int] = [8, 8],
        upsample_kernel_sizes: list[int] = [16, 16],
        istft_n_fft: int = 16,
        istft_hop_len: int = 4,
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: list[int] = [7, 11],
        source_resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor_num_layers: int = 5,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.nsf_alpha = nsf_alpha
        self.nsf_sigma = nsf_sigma
        self.nsf_voiced_threshold = nsf_voiced_threshold
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.istft_n_fft = istft_n_fft
        self.istft_hop_len = istft_hop_len
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.source_resblock_kernel_sizes = source_resblock_kernel_sizes
        self.source_resblock_dilation_sizes = source_resblock_dilation_sizes
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.f0_predictor_num_layers = f0_predictor_num_layers
        super().__init__(**kwargs)


class CosyVoiceV1Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV1ForConditionalGeneration`]. It is
    used to instantiate a CosyVoice v1 model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a configuration close to that of the
    `FunAudioLLM/CosyVoice-300M` checkpoint.

    Args:
        text_encoder_config (`CosyVoiceV1TextEncoderConfig`, *optional*):
            Configuration for the text-encoder sub-model.
        llm_config (`CosyVoiceV1LLMConfig`, *optional*):
            Configuration for the speech-token language-model sub-model.
        flow_config (`CosyVoiceV1FlowConfig`, *optional*):
            Configuration for the conditional-flow-matching decoder sub-model.
        hift_config (`CosyVoiceV1HiftConfig`, *optional*):
            Configuration for the NSF/ISTFT vocoder sub-model.
        sample_rate (`int`, *optional*, defaults to 22050):
            Output waveform sample rate, in Hz.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing weight matrices.
    """

    model_type = "cosyvoice_v1"
    sub_configs = {
        "text_encoder_config": CosyVoiceV1TextEncoderConfig,
        "llm_config": CosyVoiceV1LLMConfig,
        "flow_config": CosyVoiceV1FlowConfig,
        "hift_config": CosyVoiceV1HiftConfig,
    }

    def __init__(
        self,
        text_encoder_config: dict | None = None,
        llm_config: dict | None = None,
        flow_config: dict | None = None,
        hift_config: dict | None = None,
        sample_rate: int = 22050,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.text_encoder_config = CosyVoiceV1TextEncoderConfig(**(text_encoder_config or {}))
        self.llm_config = CosyVoiceV1LLMConfig(**(llm_config or {}))
        self.flow_config = CosyVoiceV1FlowConfig(**(flow_config or {}))
        self.hift_config = CosyVoiceV1HiftConfig(**(hift_config or {}))
        self.sample_rate = sample_rate
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


__all__ = [
    "CosyVoiceV1Config",
    "CosyVoiceV1TextEncoderConfig",
    "CosyVoiceV1LLMConfig",
    "CosyVoiceV1FlowConfig",
    "CosyVoiceV1HiftConfig",
]
