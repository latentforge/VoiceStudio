# MIT License
#
# Copyright (c) 2024 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Configuration class for BigVGAN."""

from transformers.configuration_utils import PreTrainedConfig


class BigVGANConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BigVGANModel`], the anti aliased multi
    periodicity vocoder. Instantiating a configuration with the defaults will yield a configuration matching the
    [nvidia/bigvgan_v2_24khz_100band_256x](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)
    checkpoint.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 100):
            Number of mel filterbank channels of the input spectrogram.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate, in Hz, of the generated waveform.
        upsample_initial_channel (`int`, *optional*, defaults to 1536):
            Number of channels the spectrogram is projected to before the first upsampling layer. Every layer
            halves it.
        upsample_rates (`list[int]`, *optional*, defaults to `[4, 4, 2, 2, 2, 2]`):
            Upsampling factor of each transposed convolution. Their product is the number of waveform samples per
            spectrogram frame.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[8, 8, 4, 4, 4, 4]`):
            Kernel size of each transposed convolution.
        resblock_type (`str`, *optional*, defaults to `"1"`):
            Which residual block to build after each upsampling layer, `"1"` for the block whose every dilated
            convolution is followed by an undilated one, or `"2"` for the block that holds the dilated
            convolutions alone.
        resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            Kernel size of each of the parallel residual blocks that follow an upsampling layer.
        resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            Dilation of each layer of a residual block, one list per entry of `resblock_kernel_sizes`.
        activation (`str`, *optional*, defaults to `"snakebeta"`):
            Periodic nonlinearity of the residual blocks, `"snake"` for `x + sin(alpha * x) ** 2 / alpha` or
            `"snakebeta"` for `x + sin(alpha * x) ** 2 / beta`, whose magnitude is controlled by a second learned
            parameter.
        snake_logscale (`bool`, *optional*, defaults to `True`):
            Whether `alpha` and `beta` are stored as their logarithms, so that the values the nonlinearity uses
            are their exponentials.
        anti_alias_ratio (`int`, *optional*, defaults to 2):
            Factor the activation's input is upsampled by before the periodic nonlinearity and downsampled by
            after it, which keeps the harmonics the nonlinearity creates below the Nyquist frequency.
        anti_alias_kernel_size (`int`, *optional*, defaults to 12):
            Kernel size of the Kaiser windowed sinc filter of the anti aliasing resampling.
        use_tanh_at_final (`bool`, *optional*, defaults to `False`):
            Whether the waveform is bounded by a hyperbolic tangent. `False` clamps it to `[-1, 1]` instead.
        use_bias_at_final (`bool`, *optional*, defaults to `False`):
            Whether the output convolution has a bias.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform of the mel spectrogram the reconstruction loss is measured over when
            `use_multiscale_mel_loss` is `False`.
        hop_length (`int`, *optional*, defaults to 256):
            Distance in waveform samples between neighbouring frames of that spectrogram. It equals the product of
            `upsample_rates`.
        win_length (`int`, *optional*, defaults to 1024):
            Width in waveform samples of one analysis window of that spectrogram.
        mel_fmin (`float`, *optional*, defaults to 0.0):
            Lowest frequency, in Hz, of the mel filterbank of that spectrogram.
        mel_fmax (`float`, *optional*):
            Highest frequency, in Hz, of the mel filterbank the model consumes. `None` means half the sampling
            rate.
        mel_loss_fmax (`float`, *optional*):
            Highest frequency, in Hz, of the mel filterbank the single scale reconstruction loss is measured over.
            It is independent of `mel_fmax`, and `None` means half the sampling rate.
        use_multiscale_mel_loss (`bool`, *optional*, defaults to `True`):
            Whether the reconstruction loss sums the mel spectrogram distances of `mel_loss_window_lengths`
            resolutions, rather than measuring the single spectrogram of `n_fft`, `hop_length` and `win_length`.
        mel_loss_coeff (`float`, *optional*, defaults to 15.0):
            Weight of the mel spectrogram reconstruction loss.
        mel_loss_window_lengths (`list[int]`, *optional*, defaults to `[32, 64, 128, 256, 512, 1024, 2048]`):
            Window length, which is also the Fourier transform size, of each resolution of the multi scale
            reconstruction loss. Each hops by a quarter of its own length.
        mel_loss_num_mel_bins (`list[int]`, *optional*, defaults to `[5, 10, 20, 40, 80, 160, 320]`):
            Number of mel filterbank channels of each resolution of the multi scale reconstruction loss.
        mel_loss_clamp_eps (`float`, *optional*, defaults to 1e-05):
            Smallest value a mel spectrogram of the reconstruction loss is clipped to before its logarithm is
            taken.
        initializer_range (`float`, *optional*, defaults to 0.01):
            Standard deviation of the normal initializer of the convolution weights.

    Example:

    ```python
    >>> from voicestudio.models.bigvgan import BigVGANConfig, BigVGANModel

    >>> configuration = BigVGANConfig()

    >>> model = BigVGANModel(configuration)

    >>> configuration = model.config
    ```
    """

    model_type = "bigvgan"

    def __init__(
        self,
        model_in_dim: int = 100,
        sampling_rate: int = 24000,
        upsample_initial_channel: int = 1536,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        resblock_type: str = "1",
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        activation: str = "snakebeta",
        snake_logscale: bool = True,
        anti_alias_ratio: int = 2,
        anti_alias_kernel_size: int = 12,
        use_tanh_at_final: bool = False,
        use_bias_at_final: bool = False,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_fmin: float = 0.0,
        mel_fmax: float | None = None,
        mel_loss_fmax: float | None = None,
        use_multiscale_mel_loss: bool = True,
        mel_loss_coeff: float = 15.0,
        mel_loss_window_lengths: list[int] | None = None,
        mel_loss_num_mel_bins: list[int] | None = None,
        mel_loss_clamp_eps: float = 1e-5,
        initializer_range: float = 0.01,
        **kwargs,
    ):
        if upsample_rates is None:
            upsample_rates = [4, 4, 2, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [8, 8, 4, 4, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if mel_loss_window_lengths is None:
            mel_loss_window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
        if mel_loss_num_mel_bins is None:
            mel_loss_num_mel_bins = [5, 10, 20, 40, 80, 160, 320]

        if activation not in ("snake", "snakebeta"):
            raise ValueError(f"`activation` must be one of 'snake' or 'snakebeta', got {activation}.")
        if resblock_type not in ("1", "2"):
            raise ValueError(f"`resblock_type` must be one of '1' or '2', got {resblock_type}.")
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
        if len(mel_loss_window_lengths) != len(mel_loss_num_mel_bins):
            raise ValueError(
                "mel_loss_window_lengths and mel_loss_num_mel_bins must hold one entry per resolution, got "
                f"{len(mel_loss_window_lengths)} and {len(mel_loss_num_mel_bins)} entries."
            )

        self.model_in_dim = model_in_dim
        self.sampling_rate = sampling_rate
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_type = resblock_type
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.activation = activation
        self.snake_logscale = snake_logscale
        self.anti_alias_ratio = anti_alias_ratio
        self.anti_alias_kernel_size = anti_alias_kernel_size
        self.use_tanh_at_final = use_tanh_at_final
        self.use_bias_at_final = use_bias_at_final
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.mel_loss_fmax = mel_loss_fmax
        self.use_multiscale_mel_loss = use_multiscale_mel_loss
        self.mel_loss_coeff = mel_loss_coeff
        self.mel_loss_window_lengths = mel_loss_window_lengths
        self.mel_loss_num_mel_bins = mel_loss_num_mel_bins
        self.mel_loss_clamp_eps = mel_loss_clamp_eps
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


__all__ = ["BigVGANConfig"]
