"""Configuration class for CosyVoice v1."""

from transformers.configuration_utils import PretrainedConfig


class CosyVoiceV1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`CosyVoiceV1ForConditionalGeneration`]. It is used to instantiate the three networks that make up
    CosyVoice v1: an autoregressive text-to-speech-token language model, a conditional flow matching
    model that turns speech tokens into a mel spectrogram, and a HiFTNet vocoder that turns the mel
    spectrogram into a waveform.

    Instantiating a configuration with the defaults yields the geometry of the released
    `FunAudioLLM/CosyVoice-300M` checkpoint.

    Args:
        sample_rate (`int`, *optional*, defaults to 22050):
            Sampling rate of the waveform produced by the vocoder.
        text_vocab_size (`int`, *optional*, defaults to 51866):
            Size of the text token vocabulary, matching the multilingual Whisper tokenizer.
        speech_vocab_size (`int`, *optional*, defaults to 4096):
            Number of supervised semantic speech tokens. The language model head predicts
            `speech_vocab_size + 1` classes, the extra class being the end of speech token.
        speaker_embedding_dim (`int`, *optional*, defaults to 192):
            Dimension of the utterance level speaker embedding fed to both the language model and the
            flow matching model.
        text_encoder_input_size (`int`, *optional*, defaults to 512):
            Dimension of the text embedding table, which is the input of the text encoder.
        text_encoder_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size of the text encoder.
        text_encoder_num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the text encoder.
        text_encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Inner dimension of the text encoder feed forward layers.
        text_encoder_num_layers (`int`, *optional*, defaults to 6):
            Number of text encoder layers.
        text_encoder_hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation of the text encoder feed forward layers.
        text_encoder_dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied after the input projection and after each text encoder sublayer.
        text_encoder_positional_dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied to the scaled inputs and to the relative positional embeddings.
        text_encoder_attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout applied to the text encoder attention probabilities.
        text_encoder_chunk_size (`int`, *optional*, defaults to 1):
            Static chunk size of the text encoder attention mask. A value of 1 makes the text encoder
            fully causal, 0 makes it bidirectional.
        lm_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size of the autoregressive language model.
        lm_num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the language model.
        lm_ffn_dim (`int`, *optional*, defaults to 4096):
            Inner dimension of the language model feed forward layers.
        lm_num_layers (`int`, *optional*, defaults to 14):
            Number of language model layers.
        lm_hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation of the language model feed forward layers.
        lm_dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied after the input projection and after each language model sublayer.
        lm_positional_dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied to the scaled inputs and to the relative positional embeddings.
        lm_attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout applied to the language model attention probabilities.
        lm_chunk_size (`int`, *optional*, defaults to 1):
            Static chunk size of the language model attention mask. A value of 1 makes it causal.
        lm_input_projection_activation (`bool`, *optional*, defaults to `True`):
            Whether the language model input projection ends with a ReLU, as the legacy linear input
            layer of the original implementation does.
        max_source_positions (`int`, *optional*, defaults to 5000):
            Length used to precompute the relative positional embedding table.
        label_smoothing (`float`, *optional*, defaults to 0.0):
            Label smoothing of the speech token cross entropy loss.
        length_normalized_loss (`bool`, *optional*, defaults to `True`):
            Whether the speech token loss is divided by the number of unmasked targets instead of the
            batch size.
        flow_input_size (`int`, *optional*, defaults to 512):
            Dimension of the speech token embedding table of the flow matching model.
        flow_output_size (`int`, *optional*, defaults to 80):
            Number of mel bins produced by the flow matching model.
        flow_encoder_hidden_size (`int`, *optional*, defaults to 512):
            Hidden size of the flow matching encoder.
        flow_encoder_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the flow matching encoder.
        flow_encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Inner dimension of the flow matching encoder feed forward layers.
        flow_encoder_num_layers (`int`, *optional*, defaults to 6):
            Number of flow matching encoder layers.
        flow_encoder_hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation of the flow matching encoder feed forward layers.
        flow_encoder_dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied after the input projection and after each flow encoder sublayer.
        flow_encoder_positional_dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied to the scaled inputs and to the relative positional embeddings.
        flow_encoder_attention_dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied to the flow matching encoder attention probabilities.
        flow_encoder_chunk_size (`int`, *optional*, defaults to 0):
            Static chunk size of the flow matching encoder attention mask. 0 makes it bidirectional.
        flow_input_frame_rate (`int`, *optional*, defaults to 50):
            Number of speech tokens per second.
        length_regulator_sampling_ratios (`list[int]`, *optional*, defaults to `[1, 1, 1, 1]`):
            One convolution, group norm and Mish block is created per entry.
        estimator_in_channels (`int`, *optional*, defaults to 320):
            Number of channels entering the flow matching estimator, that is the noisy mel, the encoder
            output, the speaker embedding and the conditioning mel stacked on the channel axis.
        estimator_out_channels (`int`, *optional*, defaults to 80):
            Number of mel bins predicted by the flow matching estimator.
        estimator_channels (`list[int]`, *optional*, defaults to `[256, 256]`):
            Channels of the estimator down blocks, mirrored for the up blocks.
        estimator_num_blocks (`int`, *optional*, defaults to 4):
            Number of transformer blocks inside every down, mid and up block.
        estimator_num_mid_blocks (`int`, *optional*, defaults to 12):
            Number of estimator mid blocks.
        estimator_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the estimator transformer blocks.
        estimator_head_dim (`int`, *optional*, defaults to 64):
            Dimension of every estimator attention head.
        estimator_dropout (`float`, *optional*, defaults to 0.0):
            Dropout of the estimator transformer blocks.
        estimator_ffn_mult (`int`, *optional*, defaults to 4):
            Feed forward expansion factor of the estimator transformer blocks.
        estimator_group_norm_groups (`int`, *optional*, defaults to 8):
            Number of groups of the estimator convolutional blocks group norm.
        sigma_min (`float`, *optional*, defaults to 1e-06):
            Minimum noise level of the optimal transport conditional flow matching path.
        t_scheduler (`str`, *optional*, defaults to `"cosine"`):
            Timestep schedule of the Euler solver, either `"cosine"` or linear.
        training_cfg_rate (`float`, *optional*, defaults to 0.2):
            Probability of dropping the conditioning while computing the flow matching loss.
        inference_cfg_rate (`float`, *optional*, defaults to 0.7):
            Classifier free guidance scale used by the Euler solver.
        num_flow_inference_steps (`int`, *optional*, defaults to 10):
            Number of Euler steps taken when decoding speech tokens into a mel spectrogram.
        vocoder_in_channels (`int`, *optional*, defaults to 80):
            Number of mel bins entering the vocoder.
        vocoder_base_channels (`int`, *optional*, defaults to 512):
            Channels of the first vocoder convolution.
        vocoder_num_harmonics (`int`, *optional*, defaults to 8):
            Number of harmonics above f0 generated by the neural source filter.
        vocoder_source_amplitude (`float`, *optional*, defaults to 0.1):
            Amplitude of the sine excitation.
        vocoder_source_noise_std (`float`, *optional*, defaults to 0.003):
            Standard deviation of the noise added to the sine excitation on voiced frames.
        vocoder_voiced_threshold (`float`, *optional*, defaults to 10):
            f0 above which a frame counts as voiced.
        vocoder_upsample_rates (`list[int]`, *optional*, defaults to `[8, 8]`):
            Stride of every vocoder transposed convolution.
        vocoder_upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[16, 16]`):
            Kernel size of every vocoder transposed convolution.
        vocoder_istft_n_fft (`int`, *optional*, defaults to 16):
            Window size of the inverse short time Fourier transform head.
        vocoder_istft_hop_length (`int`, *optional*, defaults to 4):
            Hop size of the inverse short time Fourier transform head.
        vocoder_resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            Kernel sizes of the vocoder residual blocks.
        vocoder_resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            Dilations of the vocoder residual blocks.
        vocoder_source_resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[7, 11]`):
            Kernel sizes of the residual blocks applied to the downsampled excitation.
        vocoder_source_resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5]]`):
            Dilations of the residual blocks applied to the downsampled excitation.
        vocoder_leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            Negative slope of the vocoder leaky ReLU.
        vocoder_audio_limit (`float`, *optional*, defaults to 0.99):
            Absolute value the generated waveform is clamped to.
        f0_predictor_hidden_size (`int`, *optional*, defaults to 512):
            Channels of the convolutional f0 predictor.
        vocoder_mel_loss_n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform of the mel spectrogram the vocoder is regressed onto.
        vocoder_mel_loss_hop_length (`int`, *optional*, defaults to 256):
            Hop of that mel spectrogram.
        vocoder_mel_loss_win_length (`int`, *optional*, defaults to 1024):
            Analysis window of that mel spectrogram.
        vocoder_mel_loss_num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel bins of that mel spectrogram.
        vocoder_mel_loss_fmin (`float`, *optional*, defaults to 0.0):
            Lowest frequency of its mel filter bank.
        vocoder_mel_loss_fmax (`float`, *optional*):
            Highest frequency of its mel filter bank. `None` means half the sampling rate, which is
            what the vocoder objective uses and what sets it apart from the mel the model consumes.
        vocoder_mel_loss_coeff (`float`, *optional*, defaults to 45.0):
            Weight of the mel reconstruction term of the vocoder objective.
        speaker_encoder_num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel bins of the kaldi filter bank the speaker encoder consumes.
        speaker_encoder_front_end_channels (`int`, *optional*, defaults to 32):
            Channels of the two dimensional convolutional front end of the speaker encoder.
        speaker_encoder_front_end_num_blocks (`list[int]`, *optional*, defaults to `[2, 2]`):
            Number of residual blocks in each of the two stages of that front end.
        speaker_encoder_init_channels (`int`, *optional*, defaults to 128):
            Channels the first time delay layer of the speaker encoder projects to.
        speaker_encoder_growth_rate (`int`, *optional*, defaults to 32):
            Channels every dense layer of the speaker encoder appends to its input.
        speaker_encoder_bottleneck_size (`int`, *optional*, defaults to 4):
            Multiple of `speaker_encoder_growth_rate` giving the bottleneck width of a dense layer.
        speaker_encoder_num_layers (`list[int]`, *optional*, defaults to `[12, 24, 16]`):
            Number of dense layers in each speaker encoder block.
        speaker_encoder_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 3, 3]`):
            Kernel size of the context aware masking convolution of each block.
        speaker_encoder_dilations (`list[int]`, *optional*, defaults to `[1, 2, 2]`):
            Dilation of that convolution in each block.
        speaker_encoder_segment_length (`int`, *optional*, defaults to 100):
            Frames per segment of the average pooled context a context aware masking layer adds to
            the utterance level context.
        speaker_encoder_reduction (`int`, *optional*, defaults to 2):
            Factor the context aware masking bottleneck divides the channels by.
        speech_tokenizer_num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel bins of the log mel spectrogram the speech tokenizer consumes.
        speech_tokenizer_hidden_size (`int`, *optional*, defaults to 1280):
            Hidden size of the speech tokenizer encoder.
        speech_tokenizer_num_heads (`int`, *optional*, defaults to 20):
            Number of attention heads in the speech tokenizer encoder.
        speech_tokenizer_ffn_dim (`int`, *optional*, defaults to 5120):
            Inner dimension of the speech tokenizer feed forward layers.
        speech_tokenizer_num_layers (`int`, *optional*, defaults to 6):
            Number of speech tokenizer encoder layers.
        speech_tokenizer_conv_stride (`int`, *optional*, defaults to 1):
            Stride of the first of the two convolutions the speech tokenizer opens with. The second
            always strides by two, so the token rate is the mel frame rate divided by twice this.
        speech_tokenizer_max_source_positions (`int`, *optional*, defaults to 1500):
            Length of the learned position table added to the speech tokenizer encoder input. `None`
            means the encoder carries no position table.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
    """

    model_type = "cosyvoice_v1"

    def __init__(
        self,
        sample_rate: int = 22050,
        text_vocab_size: int = 51866,
        speech_vocab_size: int = 4096,
        speaker_embedding_dim: int = 192,
        text_encoder_input_size: int = 512,
        text_encoder_hidden_size: int = 1024,
        text_encoder_num_heads: int = 16,
        text_encoder_ffn_dim: int = 4096,
        text_encoder_num_layers: int = 6,
        text_encoder_hidden_act: str = "silu",
        text_encoder_dropout: float = 0.1,
        text_encoder_positional_dropout: float = 0.1,
        text_encoder_attention_dropout: float = 0.0,
        text_encoder_chunk_size: int = 1,
        lm_hidden_size: int = 1024,
        lm_num_heads: int = 16,
        lm_ffn_dim: int = 4096,
        lm_num_layers: int = 14,
        lm_hidden_act: str = "relu",
        lm_dropout: float = 0.1,
        lm_positional_dropout: float = 0.1,
        lm_attention_dropout: float = 0.0,
        lm_chunk_size: int = 1,
        lm_input_projection_activation: bool = True,
        max_source_positions: int = 5000,
        label_smoothing: float = 0.0,
        length_normalized_loss: bool = True,
        flow_input_size: int = 512,
        flow_output_size: int = 80,
        flow_encoder_hidden_size: int = 512,
        flow_encoder_num_heads: int = 8,
        flow_encoder_ffn_dim: int = 2048,
        flow_encoder_num_layers: int = 6,
        flow_encoder_hidden_act: str = "silu",
        flow_encoder_dropout: float = 0.1,
        flow_encoder_positional_dropout: float = 0.1,
        flow_encoder_attention_dropout: float = 0.1,
        flow_encoder_chunk_size: int = 0,
        flow_input_frame_rate: int = 50,
        length_regulator_sampling_ratios: list[int] | None = None,
        estimator_in_channels: int = 320,
        estimator_out_channels: int = 80,
        estimator_channels: list[int] | None = None,
        estimator_num_blocks: int = 4,
        estimator_num_mid_blocks: int = 12,
        estimator_num_heads: int = 8,
        estimator_head_dim: int = 64,
        estimator_dropout: float = 0.0,
        estimator_ffn_mult: int = 4,
        estimator_group_norm_groups: int = 8,
        sigma_min: float = 1e-06,
        t_scheduler: str = "cosine",
        training_cfg_rate: float = 0.2,
        inference_cfg_rate: float = 0.7,
        num_flow_inference_steps: int = 10,
        vocoder_in_channels: int = 80,
        vocoder_base_channels: int = 512,
        vocoder_num_harmonics: int = 8,
        vocoder_source_amplitude: float = 0.1,
        vocoder_source_noise_std: float = 0.003,
        vocoder_voiced_threshold: float = 10,
        vocoder_upsample_rates: list[int] | None = None,
        vocoder_upsample_kernel_sizes: list[int] | None = None,
        vocoder_istft_n_fft: int = 16,
        vocoder_istft_hop_length: int = 4,
        vocoder_resblock_kernel_sizes: list[int] | None = None,
        vocoder_resblock_dilation_sizes: list[list[int]] | None = None,
        vocoder_source_resblock_kernel_sizes: list[int] | None = None,
        vocoder_source_resblock_dilation_sizes: list[list[int]] | None = None,
        vocoder_leaky_relu_slope: float = 0.1,
        vocoder_audio_limit: float = 0.99,
        f0_predictor_hidden_size: int = 512,
        vocoder_mel_loss_n_fft: int = 1024,
        vocoder_mel_loss_hop_length: int = 256,
        vocoder_mel_loss_win_length: int = 1024,
        vocoder_mel_loss_num_mel_bins: int = 80,
        vocoder_mel_loss_fmin: float = 0.0,
        vocoder_mel_loss_fmax: float | None = None,
        vocoder_mel_loss_coeff: float = 45.0,
        speaker_encoder_num_mel_bins: int = 80,
        speaker_encoder_front_end_channels: int = 32,
        speaker_encoder_front_end_num_blocks: list[int] | None = None,
        speaker_encoder_init_channels: int = 128,
        speaker_encoder_growth_rate: int = 32,
        speaker_encoder_bottleneck_size: int = 4,
        speaker_encoder_num_layers: list[int] | None = None,
        speaker_encoder_kernel_sizes: list[int] | None = None,
        speaker_encoder_dilations: list[int] | None = None,
        speaker_encoder_segment_length: int = 100,
        speaker_encoder_reduction: int = 2,
        speech_tokenizer_num_mel_bins: int = 128,
        speech_tokenizer_hidden_size: int = 1280,
        speech_tokenizer_num_heads: int = 20,
        speech_tokenizer_ffn_dim: int = 5120,
        speech_tokenizer_num_layers: int = 6,
        speech_tokenizer_conv_stride: int = 1,
        speech_tokenizer_max_source_positions: int | None = 1500,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.text_vocab_size = text_vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.speaker_embedding_dim = speaker_embedding_dim

        self.text_encoder_input_size = text_encoder_input_size
        self.text_encoder_hidden_size = text_encoder_hidden_size
        self.text_encoder_num_heads = text_encoder_num_heads
        self.text_encoder_ffn_dim = text_encoder_ffn_dim
        self.text_encoder_num_layers = text_encoder_num_layers
        self.text_encoder_hidden_act = text_encoder_hidden_act
        self.text_encoder_dropout = text_encoder_dropout
        self.text_encoder_positional_dropout = text_encoder_positional_dropout
        self.text_encoder_attention_dropout = text_encoder_attention_dropout
        self.text_encoder_chunk_size = text_encoder_chunk_size

        self.lm_hidden_size = lm_hidden_size
        self.lm_num_heads = lm_num_heads
        self.lm_ffn_dim = lm_ffn_dim
        self.lm_num_layers = lm_num_layers
        self.lm_hidden_act = lm_hidden_act
        self.lm_dropout = lm_dropout
        self.lm_positional_dropout = lm_positional_dropout
        self.lm_attention_dropout = lm_attention_dropout
        self.lm_chunk_size = lm_chunk_size
        self.lm_input_projection_activation = lm_input_projection_activation
        self.max_source_positions = max_source_positions
        self.label_smoothing = label_smoothing
        self.length_normalized_loss = length_normalized_loss

        self.flow_input_size = flow_input_size
        self.flow_output_size = flow_output_size
        self.flow_encoder_hidden_size = flow_encoder_hidden_size
        self.flow_encoder_num_heads = flow_encoder_num_heads
        self.flow_encoder_ffn_dim = flow_encoder_ffn_dim
        self.flow_encoder_num_layers = flow_encoder_num_layers
        self.flow_encoder_hidden_act = flow_encoder_hidden_act
        self.flow_encoder_dropout = flow_encoder_dropout
        self.flow_encoder_positional_dropout = flow_encoder_positional_dropout
        self.flow_encoder_attention_dropout = flow_encoder_attention_dropout
        self.flow_encoder_chunk_size = flow_encoder_chunk_size
        self.flow_input_frame_rate = flow_input_frame_rate
        self.length_regulator_sampling_ratios = (
            [1, 1, 1, 1] if length_regulator_sampling_ratios is None else length_regulator_sampling_ratios
        )

        self.estimator_in_channels = estimator_in_channels
        self.estimator_out_channels = estimator_out_channels
        self.estimator_channels = [256, 256] if estimator_channels is None else estimator_channels
        self.estimator_num_blocks = estimator_num_blocks
        self.estimator_num_mid_blocks = estimator_num_mid_blocks
        self.estimator_num_heads = estimator_num_heads
        self.estimator_head_dim = estimator_head_dim
        self.estimator_dropout = estimator_dropout
        self.estimator_ffn_mult = estimator_ffn_mult
        self.estimator_group_norm_groups = estimator_group_norm_groups
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate
        self.num_flow_inference_steps = num_flow_inference_steps

        self.vocoder_in_channels = vocoder_in_channels
        self.vocoder_base_channels = vocoder_base_channels
        self.vocoder_num_harmonics = vocoder_num_harmonics
        self.vocoder_source_amplitude = vocoder_source_amplitude
        self.vocoder_source_noise_std = vocoder_source_noise_std
        self.vocoder_voiced_threshold = vocoder_voiced_threshold
        self.vocoder_upsample_rates = [8, 8] if vocoder_upsample_rates is None else vocoder_upsample_rates
        self.vocoder_upsample_kernel_sizes = (
            [16, 16] if vocoder_upsample_kernel_sizes is None else vocoder_upsample_kernel_sizes
        )
        self.vocoder_istft_n_fft = vocoder_istft_n_fft
        self.vocoder_istft_hop_length = vocoder_istft_hop_length
        self.vocoder_resblock_kernel_sizes = (
            [3, 7, 11] if vocoder_resblock_kernel_sizes is None else vocoder_resblock_kernel_sizes
        )
        self.vocoder_resblock_dilation_sizes = (
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            if vocoder_resblock_dilation_sizes is None
            else vocoder_resblock_dilation_sizes
        )
        self.vocoder_source_resblock_kernel_sizes = (
            [7, 11] if vocoder_source_resblock_kernel_sizes is None else vocoder_source_resblock_kernel_sizes
        )
        self.vocoder_source_resblock_dilation_sizes = (
            [[1, 3, 5], [1, 3, 5]]
            if vocoder_source_resblock_dilation_sizes is None
            else vocoder_source_resblock_dilation_sizes
        )
        self.vocoder_leaky_relu_slope = vocoder_leaky_relu_slope
        self.vocoder_audio_limit = vocoder_audio_limit
        self.f0_predictor_hidden_size = f0_predictor_hidden_size
        self.vocoder_mel_loss_n_fft = vocoder_mel_loss_n_fft
        self.vocoder_mel_loss_hop_length = vocoder_mel_loss_hop_length
        self.vocoder_mel_loss_win_length = vocoder_mel_loss_win_length
        self.vocoder_mel_loss_num_mel_bins = vocoder_mel_loss_num_mel_bins
        self.vocoder_mel_loss_fmin = vocoder_mel_loss_fmin
        self.vocoder_mel_loss_fmax = vocoder_mel_loss_fmax
        self.vocoder_mel_loss_coeff = vocoder_mel_loss_coeff
        self.speaker_encoder_num_mel_bins = speaker_encoder_num_mel_bins
        self.speaker_encoder_front_end_channels = speaker_encoder_front_end_channels
        self.speaker_encoder_front_end_num_blocks = (
            [2, 2] if speaker_encoder_front_end_num_blocks is None else speaker_encoder_front_end_num_blocks
        )
        self.speaker_encoder_init_channels = speaker_encoder_init_channels
        self.speaker_encoder_growth_rate = speaker_encoder_growth_rate
        self.speaker_encoder_bottleneck_size = speaker_encoder_bottleneck_size
        self.speaker_encoder_num_layers = (
            [12, 24, 16] if speaker_encoder_num_layers is None else speaker_encoder_num_layers
        )
        self.speaker_encoder_kernel_sizes = (
            [3, 3, 3] if speaker_encoder_kernel_sizes is None else speaker_encoder_kernel_sizes
        )
        self.speaker_encoder_dilations = (
            [1, 2, 2] if speaker_encoder_dilations is None else speaker_encoder_dilations
        )
        self.speaker_encoder_segment_length = speaker_encoder_segment_length
        self.speaker_encoder_reduction = speaker_encoder_reduction
        self.speech_tokenizer_num_mel_bins = speech_tokenizer_num_mel_bins
        self.speech_tokenizer_hidden_size = speech_tokenizer_hidden_size
        self.speech_tokenizer_num_heads = speech_tokenizer_num_heads
        self.speech_tokenizer_ffn_dim = speech_tokenizer_ffn_dim
        self.speech_tokenizer_num_layers = speech_tokenizer_num_layers
        self.speech_tokenizer_conv_stride = speech_tokenizer_conv_stride
        self.speech_tokenizer_max_source_positions = speech_tokenizer_max_source_positions

        self.initializer_range = initializer_range
        super().__init__(**kwargs)


__all__ = ["CosyVoiceV1Config"]
