# MIT License
#
# Copyright (c) 2023 Charactr Inc.
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
"""PyTorch Vocos model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from transformers import initialization as init
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring

from .configuration_vocos import VocosConfig


@dataclass
@auto_docstring(custom_intro="Output of [`VocosModel`].")
class VocosOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Mel spectrogram reconstruction loss, weighted by `config.mel_loss_coeff`.
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        Reconstructed waveform.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`):
        Backbone output the head inverts.
    """

    loss: torch.FloatTensor | None = None
    audio_values: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None


def safe_log(hidden_states: torch.Tensor, clip_value: float = 1e-7) -> torch.Tensor:
    r"""
    Args:
        hidden_states (`torch.Tensor`):
            Tensor to take the logarithm of.
        clip_value (`float`, *optional*, defaults to 1e-07):
            Smallest value the input is clipped to first.

    Returns:
        `torch.Tensor`: The element-wise logarithm of the clipped input.
    """
    return torch.log(torch.clip(hidden_states, min=clip_value))


class VocosAdaLayerNorm(nn.Module):
    r"""
    Constructs a layer normalization whose gain and bias are looked up per conditioning class.

    Args:
        config ([`VocosConfig`]):
            Model configuration.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.eps = config.layer_norm_eps
        self.dim = config.hidden_size
        self.scale = nn.Embedding(config.adanorm_num_embeddings, config.hidden_size)
        self.shift = nn.Embedding(config.adanorm_num_embeddings, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        hidden_states = F.layer_norm(hidden_states, (self.dim,), eps=self.eps)
        return hidden_states * scale.unsqueeze(-2) + shift.unsqueeze(-2)


class VocosConvNeXtBlock(GradientCheckpointingLayer):
    r"""
    Constructs a ConvNeXt block, a depthwise convolution followed by a normalization and a pointwise expansion, on
    a residual branch scaled by a learned per-channel gain.

    Args:
        config ([`VocosConfig`]):
            Model configuration.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.layer_scale_init_value = config.layer_scale_init_value or 1 / config.num_hidden_layers
        self.adanorm = config.adanorm_num_embeddings is not None
        self.dwconv = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size=7, padding=3, groups=config.hidden_size
        )
        self.norm = (
            VocosAdaLayerNorm(config) if self.adanorm else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.pwconv1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gamma = nn.Parameter(self.layer_scale_init_value * torch.ones(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, cond_embedding_id: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states, cond_embedding_id) if self.adanorm else self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        return residual + hidden_states


class VocosBackbone(nn.Module):
    r"""
    Constructs the ConvNeXt backbone, which keeps the temporal resolution of its input throughout.

    Args:
        config ([`VocosConfig`]):
            Model configuration.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.adanorm = config.adanorm_num_embeddings is not None
        self.embed = nn.Conv1d(config.input_channels, config.hidden_size, kernel_size=7, padding=3)
        self.norm = (
            VocosAdaLayerNorm(config) if self.adanorm else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.convnext = nn.ModuleList([VocosConvNeXtBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_features: torch.Tensor, bandwidth_id: torch.Tensor | None = None) -> torch.Tensor:
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, input_channels, num_frames)`):
                Features to vocode.
            bandwidth_id (`torch.LongTensor` of shape `(1,)` or `(batch_size,)`, *optional*):
                Conditioning class of the adaptive layer normalizations. Required when the model was built with
                `adanorm_num_embeddings`.

        Returns:
            `torch.Tensor`: Hidden states of shape `(batch_size, num_frames, hidden_size)`.

        Raises:
            ValueError: If the model conditions on a bandwidth and none is given.
        """
        if self.adanorm and bandwidth_id is None:
            raise ValueError("This model conditions its normalizations on a bandwidth, so `bandwidth_id` is required.")

        hidden_states = self.embed(input_features).transpose(1, 2)
        hidden_states = self.norm(hidden_states, bandwidth_id) if self.adanorm else self.norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        for block in self.convnext:
            hidden_states = block(hidden_states, bandwidth_id)
        return self.final_layer_norm(hidden_states.transpose(1, 2))


class VocosISTFTHead(nn.Module):
    r"""
    Constructs the head that reads the backbone output as a log magnitude and a phase and inverts the short time
    Fourier transform.

    Args:
        config ([`VocosConfig`]):
            Model configuration.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.padding = config.padding
        self.out = nn.Linear(config.hidden_size, config.n_fft + 2)
        self.window = nn.Buffer(torch.hann_window(config.n_fft), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`):
                Backbone output.

        Returns:
            `torch.Tensor`: Waveform of shape `(batch_size, num_samples)`.
        """
        hidden_states = self.out(hidden_states).transpose(1, 2)
        magnitude, phase = hidden_states.chunk(2, dim=1)
        # `torch.polar`, `torch.fft.irfft` and `torch.istft` have no half precision complex kernels.
        magnitude = torch.exp(magnitude.float()).clip(max=1e2)
        spectrogram = torch.polar(magnitude, phase.float())
        window = self.window.float()

        if self.padding == "center":
            return torch.istft(spectrogram, self.n_fft, self.hop_length, self.n_fft, window, center=True)

        # `torch.istft` rejects frames laid out this way because its nonzero overlap add check fails on the
        # padding, which the trimming below discards anyway.
        pad = (self.n_fft - self.hop_length) // 2
        num_frames = spectrogram.shape[-1]
        output_size = (num_frames - 1) * self.hop_length + self.n_fft
        frames = torch.fft.irfft(spectrogram, self.n_fft, dim=1, norm="backward") * window[None, :, None]
        waveform = F.fold(
            frames, output_size=(1, output_size), kernel_size=(1, self.n_fft), stride=(1, self.hop_length)
        )[:, 0, 0, pad:-pad]
        window_envelope = F.fold(
            window.square().expand(1, num_frames, -1).transpose(1, 2),
            output_size=(1, output_size),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )[0, 0, 0, pad:-pad]
        return waveform / window_envelope


class VocosEncodecFeatures(nn.Module):
    r"""
    Constructs the EnCodec front end, the table that turns a frame's codebook entries into the embedding the
    backbone consumes. It holds every codebook of every bandwidth in one matrix, offset by codebook index, so a
    frame's embedding is one gather and one sum.

    Args:
        config ([`VocosConfig`]):
            Model configuration.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.codebook_weights = nn.Parameter(
            torch.empty(config.num_quantizers * config.codebook_size, config.input_channels)
        )

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
                EnCodec codes, whose leading `num_codebooks` codebooks are the ones the bandwidth selects.

        Returns:
            `torch.Tensor`: Features of shape `(batch_size, input_channels, num_frames)`.
        """
        offsets = torch.arange(
            0, self.codebook_size * audio_codes.shape[1], self.codebook_size, device=audio_codes.device
        )
        indices = audio_codes + offsets.view(1, -1, 1)
        return F.embedding(indices, self.codebook_weights).sum(dim=1).transpose(1, 2)


class VocosMelSpectrogram(nn.Module):
    r"""
    Constructs the mel spectrogram the reconstruction loss is measured over. Its resolution is fixed and
    independent of the resolution the head synthesizes at.

    Args:
        config ([`VocosConfig`]):
            Model configuration.
    """

    def __init__(self, config: VocosConfig):
        super().__init__()
        self.n_fft = config.mel_loss_n_fft
        self.hop_length = config.mel_loss_hop_length
        self.num_mel_bins = config.mel_loss_num_mel_bins
        self.sampling_rate = config.sampling_rate
        self.window = nn.Buffer(torch.hann_window(self.n_fft), persistent=False)
        self.filters = nn.Buffer(torch.empty(self.n_fft // 2 + 1, self.num_mel_bins), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (`torch.Tensor` of shape `(batch_size, num_samples)`):
                Waveform at `config.sampling_rate`.

        Returns:
            `torch.Tensor`: Mel scaled magnitude spectrogram of shape `(batch_size, num_mel_bins, num_frames)`.
        """
        spectrogram = torch.stft(
            waveform.float(),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.float(),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.matmul(self.filters.float().transpose(-1, -2), spectrogram.abs())


@auto_docstring
class VocosPreTrainedModel(PreTrainedModel):
    config: VocosConfig
    base_model_prefix = "vocos"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VocosConvNeXtBlock"]
    # Buffers of the analysis front end and of the inverse STFT, which this model rebuilds from its configuration.
    _keys_to_ignore_on_load_unexpected = [r"feature_extractor\.mel_spec\.", r"head\.istft\.window"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, VocosAdaLayerNorm):
            init.ones_(module.scale.weight)
            init.zeros_(module.shift.weight)
        elif isinstance(module, VocosConvNeXtBlock):
            init.constant_(module.gamma, module.layer_scale_init_value)
        elif isinstance(module, VocosEncodecFeatures):
            init.normal_(module.codebook_weights, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, VocosISTFTHead):
            # Upstream applies the truncated normal initializer to the backbone alone.
            module.out.reset_parameters()
            init.copy_(module.window, torch.hann_window(module.n_fft))
        elif isinstance(module, VocosMelSpectrogram):
            init.copy_(module.window, torch.hann_window(module.n_fft))
            init.copy_(
                module.filters,
                torchaudio.functional.melscale_fbanks(
                    n_freqs=module.n_fft // 2 + 1,
                    f_min=0.0,
                    f_max=module.sampling_rate / 2.0,
                    n_mels=module.num_mel_bins,
                    sample_rate=module.sampling_rate,
                    norm=None,
                    mel_scale="htk",
                ),
            )
        else:
            super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    Vocos, a vocoder that predicts the complex short time Fourier transform coefficients of a waveform with a
    ConvNeXt backbone and inverts them in one step, rather than upsampling in the time domain.
    """
)
class VocosModel(VocosPreTrainedModel):
    def __init__(self, config: VocosConfig):
        super().__init__(config)
        self.feature_extractor = VocosEncodecFeatures(config) if config.feature_extractor_type == "encodec" else None
        self.backbone = VocosBackbone(config)
        self.head = VocosISTFTHead(config)
        self.mel_spectrogram = VocosMelSpectrogram(config)
        self.post_init()

    def codes_to_features(self, audio_codes: torch.Tensor) -> torch.Tensor:
        r"""
        Sums the codebook embeddings of every frame of a code grid into the features the backbone consumes.

        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
                EnCodec codes, whose leading `num_codebooks` codebooks are the ones the bandwidth selects.

        Returns:
            `torch.Tensor`: Features of shape `(batch_size, config.input_channels, num_frames)`.

        Raises:
            ValueError: If the model was not built with the `"encodec"` front end.
        """
        if self.feature_extractor is None:
            raise ValueError(
                f"This {self.__class__.__name__} was built with the '{self.config.feature_extractor_type}' front "
                "end, which holds no codebook table."
            )
        return self.feature_extractor(audio_codes)

    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        audio_codes: torch.LongTensor | None = None,
        bandwidth_id: torch.LongTensor | None = None,
        labels: torch.FloatTensor | None = None,
    ) -> VocosOutput:
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, input_channels, num_frames)`, *optional*):
                Features to vocode. Required unless `audio_codes` is given.
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`, *optional*):
                EnCodec codes to vocode, turned into `input_features` by
                [`~VocosModel.codes_to_features`]. Only the `"encodec"` front end accepts them.
            bandwidth_id (`torch.LongTensor` of shape `(1,)` or `(batch_size,)`, *optional*):
                Index into `config.bandwidths` of the bandwidth the codes were encoded at. Required when the model
                was built with `config.adanorm_num_embeddings`.
            labels (`torch.FloatTensor` of shape `(batch_size, num_samples)`, *optional*):
                Ground truth waveform. Given, the mel spectrogram reconstruction loss against it is returned.

        Returns:
            [`VocosOutput`]

        Raises:
            ValueError: If neither `input_features` nor `audio_codes` is given.
        """
        if input_features is None:
            if audio_codes is None:
                raise ValueError("Give either `input_features` or `audio_codes` to vocode.")
            input_features = self.codes_to_features(audio_codes)

        last_hidden_state = self.backbone(input_features, bandwidth_id)
        audio_values = self.head(last_hidden_state)

        loss = None
        if labels is not None:
            # The head emits a whole number of hops, which the ground truth clip need not be a multiple of.
            num_samples = min(audio_values.shape[-1], labels.shape[-1])
            loss = self.config.mel_loss_coeff * F.l1_loss(
                safe_log(self.mel_spectrogram(labels[..., :num_samples])),
                safe_log(self.mel_spectrogram(audio_values[..., :num_samples])),
            )

        return VocosOutput(loss=loss, audio_values=audio_values, last_hidden_state=last_hidden_state)


__all__ = [
    "VocosModel",
    "VocosOutput",
    "VocosPreTrainedModel",
]
