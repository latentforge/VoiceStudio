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
"""PyTorch BigVGAN model."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from transformers import initialization as init
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring

from .configuration_bigvgan import BigVGANConfig


_MEL_FILTER_CACHE = {}
_WINDOW_CACHE = {}


@dataclass
@auto_docstring(custom_intro="Output of [`BigVGANModel`].")
class BigVGANOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Mel spectrogram reconstruction loss, weighted by `config.mel_loss_coeff`.
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        Generated waveform.
    """

    loss: torch.FloatTensor | None = None
    audio_values: torch.FloatTensor | None = None


def build_anti_alias_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    r"""
    Builds the Kaiser windowed sinc lowpass filter of the anti aliased activation.

    Args:
        cutoff (`float`):
            Cutoff frequency, as a fraction of the sample rate.
        half_width (`float`):
            Width of the transition band, as a fraction of the sample rate.
        kernel_size (`int`):
            Length of the filter.

    Returns:
        `torch.Tensor` of shape `(1, 1, kernel_size)`: The filter, normalized to unit sum.
    """
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    taps = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (taps / taps.sum()).view(1, 1, kernel_size)


def dynamic_range_compression(mel_spectrogram: torch.Tensor, clip_value: float = 1e-5) -> torch.Tensor:
    r"""
    Args:
        mel_spectrogram (`torch.Tensor`):
            Magnitude mel spectrogram to compress.
        clip_value (`float`, *optional*, defaults to 1e-05):
            Smallest value the input is clipped to first.

    Returns:
        `torch.Tensor`: The element-wise logarithm of the clipped input.
    """
    return torch.log(torch.clamp(mel_spectrogram, min=clip_value))


class BigVGANSnakeActivation(nn.Module):
    r"""
    Constructs the anti aliased periodic activation of BigVGAN, which upsamples its input, applies the snake
    nonlinearity, and lowpass filters the result back down, so that the harmonics the nonlinearity creates stay
    below the Nyquist frequency. The resampling filters of both directions are the same Kaiser windowed sinc.

    Args:
        config ([`BigVGANConfig`]):
            Model configuration.
        channels (`int`):
            Number of channels, each of which owns its own `alpha`, and its own `beta` when
            `config.activation` is `"snakebeta"`.
    """

    def __init__(self, config: BigVGANConfig, channels: int):
        super().__init__()
        self.ratio = config.anti_alias_ratio
        self.kernel_size = config.anti_alias_kernel_size
        self.logscale = config.snake_logscale
        self.alpha = nn.Parameter(torch.empty(channels))
        self.beta = nn.Parameter(torch.empty(channels)) if config.activation == "snakebeta" else None
        self.filter = nn.Buffer(self.build_filter(), persistent=False)

        self.upsample_pad = self.kernel_size // self.ratio - 1
        self.upsample_trim_left = self.upsample_pad * self.ratio + (self.kernel_size - self.ratio) // 2
        self.upsample_trim_right = self.upsample_pad * self.ratio + (self.kernel_size - self.ratio + 1) // 2
        self.downsample_pad_left = self.kernel_size // 2 - int(self.kernel_size % 2 == 0)
        self.downsample_pad_right = self.kernel_size // 2

    def build_filter(self) -> torch.Tensor:
        """
        Returns:
            `torch.Tensor` of shape `(1, 1, kernel_size)`: The resampling filter of both directions.
        """
        return build_anti_alias_filter(0.5 / self.ratio, 0.6 / self.ratio, self.kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Activation input.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, sequence_length)`: The activated input.
        """
        channels = hidden_states.shape[1]
        taps = self.filter.to(hidden_states.dtype).expand(channels, -1, -1)

        hidden_states = F.pad(hidden_states, (self.upsample_pad, self.upsample_pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(hidden_states, taps, stride=self.ratio, groups=channels)
        hidden_states = hidden_states[..., self.upsample_trim_left : -self.upsample_trim_right]

        alpha = self.alpha.view(1, -1, 1)
        beta = alpha if self.beta is None else self.beta.view(1, -1, 1)
        if self.logscale:
            alpha = alpha.exp()
            beta = beta.exp()
        hidden_states = hidden_states + (1.0 / (beta + 1e-9)) * (hidden_states * alpha).sin().pow(2)

        hidden_states = F.pad(
            hidden_states, (self.downsample_pad_left, self.downsample_pad_right), mode="replicate"
        )
        return F.conv1d(hidden_states, taps, stride=self.ratio, groups=channels)


class BigVGANAmpLayer(nn.Module):
    r"""
    Constructs one residual layer of an anti aliased multi periodicity block, a dilated convolution preceded by a
    snake activation, followed under `config.resblock_type` `"1"` by an undilated convolution and a second
    activation.

    Args:
        config ([`BigVGANConfig`]):
            Model configuration.
        channels (`int`):
            Number of channels of the layer.
        kernel_size (`int`):
            Kernel size of the convolutions.
        dilation (`int`):
            Dilation of the first convolution.
    """

    activation_class = BigVGANSnakeActivation

    def __init__(self, config: BigVGANConfig, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size * dilation - dilation) // 2,
            dilation=dilation,
        )
        self.activation1 = self.activation_class(config, channels)
        if config.resblock_type == "1":
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, dilation=1)
            self.activation2 = self.activation_class(config, channels)
        else:
            self.conv2 = None
            self.activation2 = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Layer input.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, sequence_length)`: The layer output.
        """
        residual = self.conv1(self.activation1(hidden_states))
        if self.conv2 is not None:
            residual = self.conv2(self.activation2(residual))
        return hidden_states + residual

    def apply_weight_norm(self):
        """Reparameterizes the layer's convolutions by weight and direction, as training does."""
        weight_norm = nn.utils.parametrizations.weight_norm
        weight_norm(self.conv1)
        if self.conv2 is not None:
            weight_norm(self.conv2)

    def remove_weight_norm(self):
        """Folds the weight norm reparameterization of the layer's convolutions back into plain weights."""
        nn.utils.parametrize.remove_parametrizations(self.conv1, "weight")
        if self.conv2 is not None:
            nn.utils.parametrize.remove_parametrizations(self.conv2, "weight")


class BigVGANAmpBlock(GradientCheckpointingLayer):
    r"""
    Constructs an anti aliased multi periodicity block, the stack of residual layers that follows one upsampling
    layer for a single kernel size.

    Args:
        config ([`BigVGANConfig`]):
            Model configuration.
        channels (`int`):
            Number of channels of the block.
        kernel_size (`int`):
            Kernel size of the block's convolutions.
        dilations (`list[int]`):
            Dilation of each layer of the block.
    """

    layer_class = BigVGANAmpLayer

    def __init__(self, config: BigVGANConfig, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [self.layer_class(config, channels, kernel_size, dilation) for dilation in dilations]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Block input.

        Returns:
            `torch.Tensor` of shape `(batch_size, channels, sequence_length)`: The block output.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def apply_weight_norm(self):
        """Reparameterizes the block's convolutions by weight and direction, as training does."""
        for layer in self.layers:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        """Folds the weight norm reparameterization of the block's convolutions back into plain weights."""
        for layer in self.layers:
            layer.remove_weight_norm()


def mel_spectrogram(
    waveform: torch.Tensor,
    sampling_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    num_mel_bins: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    centered: bool = False,
) -> torch.Tensor:
    r"""
    Computes a magnitude mel spectrogram over a Slaney scaled and Slaney normalized filterbank, in either of the
    two framings BigVGAN uses: the uncentered one of the spectrogram the vocoder consumes, whose waveform is
    reflection padded by `(n_fft - hop_length) // 2` samples on each side first and whose magnitude carries a
    small offset under the square root, or the centered one of the multi scale reconstruction loss.

    Args:
        waveform (`torch.Tensor` of shape `(batch_size, num_samples)`):
            Waveform at `sampling_rate`.
        sampling_rate (`int`):
            Sampling rate, in Hz, of the waveform.
        n_fft (`int`):
            Size of the Fourier transform.
        hop_length (`int`):
            Distance in waveform samples between neighbouring frames.
        win_length (`int`):
            Width in waveform samples of one analysis window.
        num_mel_bins (`int`):
            Number of mel filterbank channels.
        fmin (`float`, *optional*, defaults to 0.0):
            Lowest frequency, in Hz, of the filterbank.
        fmax (`float`, *optional*):
            Highest frequency, in Hz, of the filterbank. `None` means half the sampling rate.
        centered (`bool`, *optional*, defaults to `False`):
            Which framing to use.

    Returns:
        `torch.Tensor`: Magnitude mel spectrogram of shape `(batch_size, num_mel_bins, num_frames)`.
    """
    fmax = sampling_rate / 2.0 if fmax is None else fmax
    device, dtype = waveform.device, torch.float32

    window_key = (win_length, device)
    if window_key not in _WINDOW_CACHE:
        _WINDOW_CACHE[window_key] = torch.hann_window(win_length, device=device, dtype=dtype)
    filter_key = (n_fft, num_mel_bins, sampling_rate, fmin, fmax, device)
    if filter_key not in _MEL_FILTER_CACHE:
        _MEL_FILTER_CACHE[filter_key] = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mel_bins,
            sample_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        ).to(device=device, dtype=dtype)

    waveform = waveform.float()
    if not centered:
        padding = (n_fft - hop_length) // 2
        waveform = F.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spectrogram = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=_WINDOW_CACHE[window_key],
        center=centered,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    if centered:
        magnitude = spectrogram.abs()
    else:
        magnitude = torch.sqrt(torch.view_as_real(spectrogram).pow(2).sum(-1) + 1e-9)
    return torch.matmul(_MEL_FILTER_CACHE[filter_key].transpose(-1, -2), magnitude)


@auto_docstring
class BigVGANPreTrainedModel(PreTrainedModel):
    config: BigVGANConfig
    base_model_prefix = "bigvgan"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BigVGANAmpBlock"]
    _supports_sdpa = False
    _supports_flash_attn = False

    def _init_weights(self, module):
        if isinstance(module, BigVGANSnakeActivation):
            fill = init.zeros_ if module.logscale else init.ones_
            fill(module.alpha)
            if module.beta is not None:
                fill(module.beta)
            init.copy_(module.filter, module.build_filter())
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        else:
            super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    BigVGAN, a vocoder that upsamples a mel spectrogram to a waveform through a stack of transposed convolutions,
    each followed by parallel residual blocks whose periodic snake activations are upsampled, applied and lowpass
    filtered back down so that the harmonics they create stay below the Nyquist frequency.
    """
)
class BigVGANModel(BigVGANPreTrainedModel):
    amp_block_class = BigVGANAmpBlock
    snake_activation_class = BigVGANSnakeActivation

    def __init__(self, config: BigVGANConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.hop_length = math.prod(config.upsample_rates)

        self.conv_pre = nn.Conv1d(
            config.model_in_dim, config.upsample_initial_channel, kernel_size=7, stride=1, padding=3
        )

        self.upsampler = nn.ModuleList()
        for index, (rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                self.build_upsample_layer(
                    config.upsample_initial_channel // (2**index),
                    config.upsample_initial_channel // (2 ** (index + 1)),
                    kernel_size,
                    rate,
                )
            )

        self.resblocks = nn.ModuleList()
        for index in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (index + 1))
            self.resblocks.append(
                nn.ModuleList(
                    [
                        self.amp_block_class(config, channels, kernel_size, dilations)
                        for kernel_size, dilations in zip(
                            config.resblock_kernel_sizes, config.resblock_dilation_sizes
                        )
                    ]
                )
            )

        self.post_activation = self.snake_activation_class(config, channels)
        self.conv_post = nn.Conv1d(
            channels, 1, kernel_size=7, stride=1, padding=3, bias=config.use_bias_at_final
        )

        self.post_init()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, weights_name="bigvgan_generator.pt", **kwargs
    ):
        r"""
        Loads a BigVGAN checkpoint, from a published repository as it stands or from a directory
        [`~weight_conversion.convert`] wrote.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                `"nvidia/bigvgan_v2_24khz_100band_256x"`, any key of `PUBLISHED_CHECKPOINTS`, or any repository id
                or directory holding one of the two layouts.
            model_args (`tuple`, *optional*):
                Positional arguments of [`~PreTrainedModel.from_pretrained`].
            weights_name (`str`, *optional*, defaults to `"bigvgan_generator.pt"`):
                Generator weight file to read out of a published repository, which the v2 ones also publish as
                `bigvgan_generator_3msteps.pt`. Ignored by a converted directory.
            kwargs (`dict`, *optional*):
                Keyword arguments of [`~PreTrainedModel.from_pretrained`].

        Returns:
            [`BigVGANModel`]: The loaded model.
        """
        from .weight_conversion import converted_checkpoint, is_published_layout

        if (
            pretrained_model_name_or_path is not None
            and kwargs.get("config") is None
            and kwargs.get("state_dict") is None
            and is_published_layout(pretrained_model_name_or_path)
        ):
            pretrained_model_name_or_path = converted_checkpoint(
                pretrained_model_name_or_path, weights_name=weights_name
            )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def mel_loss_resolutions(self) -> list[dict]:
        r"""
        Returns:
            `list[dict]`: The [`mel_spectrogram`] keyword arguments of every resolution the reconstruction loss
            is measured over.
        """
        if self.config.use_multiscale_mel_loss:
            return [
                {
                    "sampling_rate": self.config.sampling_rate,
                    "n_fft": window_length,
                    "hop_length": window_length // 4,
                    "win_length": window_length,
                    "num_mel_bins": num_mel_bins,
                    "fmin": 0.0,
                    "fmax": None,
                    "centered": True,
                }
                for window_length, num_mel_bins in zip(
                    self.config.mel_loss_window_lengths, self.config.mel_loss_num_mel_bins
                )
            ]
        return [
            {
                "sampling_rate": self.config.sampling_rate,
                "n_fft": self.config.n_fft,
                "hop_length": self.config.hop_length,
                "win_length": self.config.win_length,
                "num_mel_bins": self.config.model_in_dim,
                "fmin": self.config.mel_fmin,
                "fmax": self.config.mel_loss_fmax,
                "centered": False,
            }
        ]

    def build_upsample_layer(
        self, in_channels: int, out_channels: int, kernel_size: int, rate: int
    ) -> nn.ConvTranspose1d:
        r"""
        Builds one transposed convolution of the upsampling stack.

        Args:
            in_channels (`int`):
                Number of channels the layer consumes.
            out_channels (`int`):
                Number of channels the layer produces.
            kernel_size (`int`):
                Kernel size of the layer.
            rate (`int`):
                Stride of the layer, the factor it upsamples by.

        Returns:
            `nn.ConvTranspose1d`: The layer.
        """
        return nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=rate, padding=(kernel_size - rate) // 2
        )

    def apply_weight_norm(self):
        """Reparameterizes the vocoder's convolutions by weight and direction, as training does."""
        weight_norm = nn.utils.parametrizations.weight_norm
        weight_norm(self.conv_pre)
        for layer in self.upsampler:
            weight_norm(layer)
        for blocks in self.resblocks:
            for block in blocks:
                block.apply_weight_norm()
        weight_norm(self.conv_post)

    def remove_weight_norm(self):
        """Folds the weight norm reparameterization of the vocoder's convolutions back into plain weights."""
        nn.utils.parametrize.remove_parametrizations(self.conv_pre, "weight")
        for layer in self.upsampler:
            nn.utils.parametrize.remove_parametrizations(layer, "weight")
        for blocks in self.resblocks:
            for block in blocks:
                block.remove_weight_norm()
        nn.utils.parametrize.remove_parametrizations(self.conv_post, "weight")

    def mel_loss(self, audio_values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        r"""
        Measures the mel spectrogram distance the generator is regressed onto, at every resolution the
        configuration lists.

        Args:
            audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
                Generated waveform.
            labels (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
                Ground truth waveform.

        Returns:
            `torch.Tensor`: The distance, weighted by `config.mel_loss_coeff`.
        """
        clip_value = self.config.mel_loss_clamp_eps
        # The multi scale resolutions measure the distance between base ten logarithms, the single scale one
        # between natural logarithms.
        scale = 1.0 / math.log(10.0) if self.config.use_multiscale_mel_loss else 1.0

        loss = audio_values.new_zeros(())
        for resolution in self.mel_loss_resolutions():
            loss = loss + F.l1_loss(
                dynamic_range_compression(mel_spectrogram(audio_values, **resolution), clip_value),
                dynamic_range_compression(mel_spectrogram(labels, **resolution), clip_value),
            )
        return self.config.mel_loss_coeff * scale * loss

    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor,
        labels: torch.FloatTensor | None = None,
    ) -> BigVGANOutput:
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, model_in_dim, num_frames)`):
                Log mel spectrogram to vocode, on the scale [`BigVGANFeatureExtractor`] produces.
            labels (`torch.FloatTensor` of shape `(batch_size, num_samples)`, *optional*):
                Ground truth waveform. Given, the mel spectrogram reconstruction loss against it is returned.

        Returns:
            [`BigVGANOutput`]
        """
        hidden_states = self.conv_pre(input_features)
        for upsample, blocks in zip(self.upsampler, self.resblocks):
            hidden_states = upsample(hidden_states)
            hidden_states = sum(block(hidden_states) for block in blocks) / self.num_kernels

        hidden_states = self.post_activation(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        if self.config.use_tanh_at_final:
            audio_values = torch.tanh(hidden_states)
        else:
            audio_values = torch.clamp(hidden_states, min=-1.0, max=1.0)
        audio_values = audio_values.squeeze(1)

        loss = None
        if labels is not None:
            # The stack emits a whole number of hops, which the ground truth clip need not be a multiple of.
            num_samples = min(audio_values.shape[-1], labels.shape[-1])
            loss = self.mel_loss(audio_values[..., :num_samples], labels[..., :num_samples])

        return BigVGANOutput(loss=loss, audio_values=audio_values)


__all__ = [
    "BigVGANAmpBlock",
    "BigVGANAmpLayer",
    "BigVGANModel",
    "BigVGANOutput",
    "BigVGANPreTrainedModel",
    "BigVGANSnakeActivation",
    "build_anti_alias_filter",
    "dynamic_range_compression",
    "mel_spectrogram",
]
