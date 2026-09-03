"""Configuration class for Vocos."""

from transformers.configuration_utils import PreTrainedConfig


class VocosConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VocosModel`], the ConvNeXt backbone plus
    inverse STFT head vocoder. Instantiating a configuration with the defaults will yield a configuration matching
    the [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) checkpoint.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        feature_extractor_type (`str`, *optional*, defaults to `"mel"`):
            Which front end produces the backbone input, `"mel"` for a log mel spectrogram or `"encodec"` for the
            sum of the EnCodec codebook embeddings of a frame. `"encodec"` gives the model a learnable codebook
            table and expects `bandwidth_id` on every call.
        input_channels (`int`, *optional*, defaults to 100):
            Number of channels the backbone consumes, the number of mel filterbank channels for the `"mel"` front
            end and the EnCodec embedding dimensionality for the `"encodec"` one.
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the ConvNeXt backbone.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the pointwise expansion inside each ConvNeXt block.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of ConvNeXt blocks.
        layer_scale_init_value (`float`, *optional*):
            Initial value of the per-channel layer scale of each ConvNeXt block. Defaults to
            `1 / num_hidden_layers`.
        adanorm_num_embeddings (`int`, *optional*):
            Number of conditioning classes of the adaptive layer normalizations. `None` builds plain layer
            normalizations and makes the model unconditional.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform the head inverts.
        hop_length (`int`, *optional*, defaults to 256):
            Distance in waveform samples between neighbouring spectrogram frames.
        padding (`str`, *optional*, defaults to `"center"`):
            How the head's frames are laid over the waveform, `"center"` for the padding of
            `torch.istft(center=True)` or `"same"` for the convolution style padding of `(n_fft - hop_length) // 2`
            samples on each side.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate, in Hz, of the waveform the model produces.
        num_quantizers (`int`, *optional*, defaults to 16):
            Number of EnCodec codebooks the codebook table holds, the number reached at the widest entry of
            `bandwidths`. Ignored by the `"mel"` front end.
        codebook_size (`int`, *optional*, defaults to 1024):
            Number of entries of a single EnCodec codebook. Ignored by the `"mel"` front end.
        bandwidths (`list[float]`, *optional*, defaults to `[1.5, 3.0, 6.0, 12.0]`):
            EnCodec bandwidths, in kbps, the conditioning classes stand for. `bandwidth_id` indexes this list.
            Ignored by the `"mel"` front end.
        mel_loss_coeff (`float`, *optional*, defaults to 45.0):
            Weight of the mel spectrogram reconstruction loss.
        mel_loss_n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform of the loss mel spectrogram. It is independent of `n_fft`.
        mel_loss_hop_length (`int`, *optional*, defaults to 256):
            Hop length of the loss mel spectrogram. It is independent of `hop_length`.
        mel_loss_num_mel_bins (`int`, *optional*, defaults to 100):
            Number of mel filterbank channels of the loss mel spectrogram. It is independent of `input_channels`.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the layer normalizations.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer used for the convolution and linear weights.

    Example:

    ```python
    >>> from voicestudio.models.vocos import VocosConfig, VocosModel

    >>> configuration = VocosConfig()
    >>> model = VocosModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "vocos"

    def __init__(
        self,
        feature_extractor_type: str = "mel",
        input_channels: int = 100,
        hidden_size: int = 512,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 8,
        layer_scale_init_value: float | None = None,
        adanorm_num_embeddings: int | None = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        padding: str = "center",
        sampling_rate: int = 24000,
        num_quantizers: int = 16,
        codebook_size: int = 1024,
        bandwidths: list[float] | None = None,
        mel_loss_coeff: float = 45.0,
        mel_loss_n_fft: int = 1024,
        mel_loss_hop_length: int = 256,
        mel_loss_num_mel_bins: int = 100,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if feature_extractor_type not in ("mel", "encodec"):
            raise ValueError(
                f"`feature_extractor_type` must be one of 'mel' or 'encodec', got {feature_extractor_type}."
            )
        if padding not in ("center", "same"):
            raise ValueError(f"`padding` must be one of 'center' or 'same', got {padding}.")

        self.feature_extractor_type = feature_extractor_type
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_scale_init_value = layer_scale_init_value
        self.adanorm_num_embeddings = adanorm_num_embeddings
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.padding = padding
        self.sampling_rate = sampling_rate
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.bandwidths = [1.5, 3.0, 6.0, 12.0] if bandwidths is None else list(bandwidths)
        self.mel_loss_coeff = mel_loss_coeff
        self.mel_loss_n_fft = mel_loss_n_fft
        self.mel_loss_hop_length = mel_loss_hop_length
        self.mel_loss_num_mel_bins = mel_loss_num_mel_bins
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


__all__ = ["VocosConfig"]
