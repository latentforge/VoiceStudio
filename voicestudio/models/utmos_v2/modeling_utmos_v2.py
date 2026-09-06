# MIT License
#
# Copyright (c) 2024 sarulab-speech
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
"""PyTorch UTMOSv2 model."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.utils import ModelOutput, auto_docstring

from .configuration_utmos_v2 import EFFICIENTNET_V2_S_STAGES, UTMOSv2Config


@dataclass
@auto_docstring(custom_intro="Output of [`UTMOSv2Model`] and [`UTMOSv2ForAudioClassification`].")
class UTMOSv2Output(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Sum of the weighted pairwise ranking and mean squared error terms of the upstream objective.
    logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
        Predicted mean opinion score, averaged over the folds.
    fold_logits (`torch.FloatTensor` of shape `(num_folds, batch_size, num_labels)`, *optional*):
        Prediction of each fold before averaging. Returned by [`UTMOSv2ForAudioClassification`].
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        Fused features the classifier reads. Returned by [`UTMOSv2Model`].
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    fold_logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None


def drop_path(hidden_states: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    r"""
    Zeroes whole samples of a residual branch and rescales the rest so the expectation is unchanged.

    Args:
        hidden_states (`torch.Tensor`):
            Output of the residual branch, of any shape whose first dimension is the batch.
        drop_prob (`float`):
            Probability that a sample is zeroed.
        training (`bool`):
            Whether to drop at all.

    Returns:
        `torch.Tensor`: The rescaled residual branch.
    """
    if drop_prob == 0.0 or not training:
        return hidden_states
    keep_prob = 1.0 - drop_prob
    shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
    mask = hidden_states.new_empty(shape).bernoulli_(keep_prob)
    return hidden_states * mask.div_(keep_prob)


class UTMOSv2Conv2d(nn.Conv2d):
    r"""
    Convolution padded the way TensorFlow's `"SAME"` pads, which the published spectrogram encoder was ported
    from. The padding depends on the input size and is one pixel wider on the bottom and the right wherever the
    stride makes it odd, so it is applied here rather than by [`nn.Conv2d`].
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        height, width = hidden_states.shape[-2:]
        pads = []
        for size, kernel, stride, dilation in zip(
            (width, height), self.kernel_size[::-1], self.stride[::-1], self.dilation[::-1]
        ):
            total = max((math.ceil(size / stride) - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)
            pads += [total // 2, total - total // 2]
        if any(pads):
            hidden_states = F.pad(hidden_states, pads)
        return self._conv_forward(hidden_states, self.weight, self.bias)


class UTMOSv2SqueezeExcite(nn.Module):
    r"""
    Channel gate that scales every channel by a sigmoid of its spatially averaged, bottlenecked activation.

    Args:
        channels (`int`):
            Number of channels to gate.
        reduced_channels (`int`):
            Width of the bottleneck.
    """

    def __init__(self, channels: int, reduced_channels: int):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.conv_expand = nn.Conv2d(reduced_channels, channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = hidden_states.mean((2, 3), keepdim=True)
        gate = F.silu(self.conv_reduce(gate), inplace=True)
        return hidden_states * torch.sigmoid(self.conv_expand(gate))


class UTMOSv2ConvBlock(nn.Module):
    r"""
    Plain convolution, batch normalization and activation, with a residual connection where the shape allows one.
    It is the block the first stage of EfficientNetV2 is built from.

    Args:
        config ([`UTMOSv2Config`]):
            Model configuration.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Side of the convolution kernel.
        stride (`int`):
            Stride of the convolution.
        drop_path_rate (`float`):
            Stochastic depth ratio of the residual connection.
    """

    def __init__(
        self,
        config: UTMOSv2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        drop_path_rate: float,
    ):
        super().__init__()
        self.has_residual = stride == 1 and in_channels == out_channels
        self.drop_path_rate = drop_path_rate
        self.conv = UTMOSv2Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.bn(self.conv(hidden_states)), inplace=True)
        if self.has_residual:
            hidden_states = drop_path(hidden_states, self.drop_path_rate, self.training) + residual
        return hidden_states


class UTMOSv2FusedMBConv(nn.Module):
    r"""
    Inverted bottleneck whose expansion and spatial convolution are fused into one dense convolution, the block
    EfficientNetV2 substitutes for a depthwise one in its early stages.

    Args:
        config ([`UTMOSv2Config`]):
            Model configuration.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Side of the expansion kernel.
        stride (`int`):
            Stride of the expansion convolution.
        expand_ratio (`int`):
            Factor the channels are widened by inside the block.
        se_ratio (`float`):
            Bottleneck ratio of the channel gate, relative to the block input. `0.0` leaves the block ungated.
        drop_path_rate (`float`):
            Stochastic depth ratio of the residual connection.
    """

    def __init__(
        self,
        config: UTMOSv2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
        drop_path_rate: float,
    ):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.has_residual = stride == 1 and in_channels == out_channels
        self.drop_path_rate = drop_path_rate
        self.conv_expand = UTMOSv2Conv2d(in_channels, mid_channels, kernel_size, stride=stride, bias=False)
        self.bn_expand = nn.BatchNorm2d(mid_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.se = (
            UTMOSv2SqueezeExcite(mid_channels, round(mid_channels * se_ratio / expand_ratio))
            if se_ratio
            else nn.Identity()
        )
        self.conv_project = UTMOSv2Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn_project = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.bn_expand(self.conv_expand(hidden_states)), inplace=True)
        hidden_states = self.se(hidden_states)
        hidden_states = self.bn_project(self.conv_project(hidden_states))
        if self.has_residual:
            hidden_states = drop_path(hidden_states, self.drop_path_rate, self.training) + residual
        return hidden_states


class UTMOSv2MBConv(nn.Module):
    r"""
    Inverted bottleneck with a pointwise expansion, a depthwise convolution, a channel gate and a pointwise linear
    projection, the block EfficientNetV2 uses in its later stages.

    Args:
        config ([`UTMOSv2Config`]):
            Model configuration.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Side of the depthwise kernel.
        stride (`int`):
            Stride of the depthwise convolution.
        expand_ratio (`int`):
            Factor the channels are widened by inside the block.
        se_ratio (`float`):
            Bottleneck ratio of the channel gate, relative to the block input.
        drop_path_rate (`float`):
            Stochastic depth ratio of the residual connection.
    """

    def __init__(
        self,
        config: UTMOSv2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
        drop_path_rate: float,
    ):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.has_residual = stride == 1 and in_channels == out_channels
        self.drop_path_rate = drop_path_rate
        self.conv_expand = UTMOSv2Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn_expand = nn.BatchNorm2d(mid_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.conv_depthwise = UTMOSv2Conv2d(
            mid_channels, mid_channels, kernel_size, stride=stride, groups=mid_channels, bias=False
        )
        self.bn_depthwise = nn.BatchNorm2d(
            mid_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        self.se = (
            UTMOSv2SqueezeExcite(mid_channels, round(mid_channels * se_ratio / expand_ratio))
            if se_ratio
            else nn.Identity()
        )
        self.conv_project = UTMOSv2Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn_project = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.bn_expand(self.conv_expand(hidden_states)), inplace=True)
        hidden_states = F.silu(self.bn_depthwise(self.conv_depthwise(hidden_states)), inplace=True)
        hidden_states = self.se(hidden_states)
        hidden_states = self.bn_project(self.conv_project(hidden_states))
        if self.has_residual:
            hidden_states = drop_path(hidden_states, self.drop_path_rate, self.training) + residual
        return hidden_states


class UTMOSv2SpectrogramEncoder(nn.Module):
    r"""
    EfficientNetV2-S over one mel spectrogram rendered as a square three channel image, stopped before its
    classifier so that it returns a feature map rather than a vector.

    Args:
        config ([`UTMOSv2Config`]):
            Model configuration.
    """

    def __init__(self, config: UTMOSv2Config):
        super().__init__()
        blocks = [layers for _, layers, *_ in EFFICIENTNET_V2_S_STAGES]
        depth, total = 0, sum(blocks)

        self.conv_stem = UTMOSv2Conv2d(3, config.stem_channels, 3, stride=2, bias=False)
        self.bn_stem = nn.BatchNorm2d(
            config.stem_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )

        in_channels = config.stem_channels
        self.blocks = nn.ModuleList()
        for block_type, layers, kernel_size, stride, expand_ratio, out_channels, se_ratio in (
            EFFICIENTNET_V2_S_STAGES
        ):
            stage = nn.ModuleList()
            for layer in range(layers):
                arguments = (
                    config,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride if layer == 0 else 1,
                )
                rate = config.drop_path_rate * depth / total
                if block_type == "conv":
                    stage.append(UTMOSv2ConvBlock(*arguments, rate))
                elif block_type == "fused":
                    stage.append(UTMOSv2FusedMBConv(*arguments, expand_ratio, se_ratio, rate))
                else:
                    stage.append(UTMOSv2MBConv(*arguments, expand_ratio, se_ratio, rate))
                in_channels = out_channels
                depth += 1
            self.blocks.append(stage)

        self.conv_head = UTMOSv2Conv2d(in_channels, config.head_channels, 1, bias=False)
        self.bn_head = nn.BatchNorm2d(
            config.head_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            pixel_values (`torch.Tensor`):
                Spectrogram images of shape `(batch_size, 3, image_size, image_size)`.

        Returns:
            `torch.Tensor`: Feature map of shape `(batch_size, head_channels, image_size / 32, image_size / 32)`.
        """
        hidden_states = F.silu(self.bn_stem(self.conv_stem(pixel_values)), inplace=True)
        for stage in self.blocks:
            for block in stage:
                hidden_states = block(hidden_states)
        return F.silu(self.bn_head(self.conv_head(hidden_states)), inplace=True)


@auto_docstring
class UTMOSv2PreTrainedModel(PreTrainedModel):
    config: UTMOSv2Config
    base_model_prefix = "utmos_v2"
    main_input_name = "input_values"
    supports_gradient_checkpointing = False
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # Upstream inherits EfficientNet's fan-out initializer, which reads the kernel rather than the input.
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, UTMOSv2Model):
            for weights in (module.ssl_layer_weights, module.spectrogram_weights):
                init.copy_(weights, F.softmax(torch.randn(weights.shape[0]), dim=0))
        else:
            super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    One UTMOSv2 predictor, the model the authors train on a single cross-validation fold. A wav2vec 2.0 branch and
    an EfficientNetV2 branch reading mel spectrograms as images are pooled independently, concatenated with a
    one-hot vector naming the listening-test corpus to imitate, and read by a linear layer.
    """
)
class UTMOSv2Model(UTMOSv2PreTrainedModel):
    def __init__(self, config: UTMOSv2Config):
        super().__init__(config)
        self.ssl_encoder = AutoModel.from_config(config.ssl_config)
        self.ssl_layer_weights = nn.Parameter(torch.empty(config.num_ssl_hidden_states))
        self.ssl_attention = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim=config.ssl_hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                batch_first=True,
            )
            for _ in range(config.num_ssl_attention_layers)
        )

        self.spectrogram_encoders = nn.ModuleList(
            UTMOSv2SpectrogramEncoder(config) for _ in range(config.num_spectrograms)
        )
        self.spectrogram_weights = nn.Parameter(torch.empty(config.num_spectrograms))
        self.spectrogram_attention = nn.MultiheadAttention(
            embed_dim=config.head_channels * 2,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def _ssl_features(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ssl_encoder(input_values, output_hidden_states=True).hidden_states
        mixed = sum(state * weight for state, weight in zip(hidden_states, self.ssl_layer_weights))
        attended = mixed
        for attention in self.ssl_attention:
            attended, _ = attention(attended, attended, attended)
        return torch.cat([attended.mean(dim=1), mixed.max(dim=1).values], dim=1)

    def _spectrogram_features(self, input_features: torch.Tensor) -> torch.Tensor:
        num_spectrograms = len(self.spectrogram_encoders)
        maps = [
            self.spectrogram_encoders[index % num_spectrograms](input_features[:, index])
            for index in range(input_features.shape[1])
        ]
        frames = [
            sum(
                maps[frame * num_spectrograms + index] * weight
                for index, weight in enumerate(self.spectrogram_weights)
            )
            for frame in range(input_features.shape[1] // num_spectrograms)
        ]
        pooled = torch.cat(frames, dim=3)
        pooled = torch.cat([F.adaptive_avg_pool2d(pooled, (None, 1)), F.adaptive_max_pool2d(pooled, (None, 1))], dim=1)
        pooled = pooled.squeeze(3)
        attended = pooled.permute(0, 2, 1)
        attended, _ = self.spectrogram_attention(attended, attended, attended)
        return torch.cat([attended.mean(dim=1), pooled.max(dim=2).values], dim=1)

    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        input_features: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> UTMOSv2Output:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, num_samples)`):
            Waveform excerpt the self-supervised branch reads, normalized to zero mean and unit variance.
        input_features (`torch.Tensor` of shape `(batch_size, num_frames * num_spectrograms, 3, image_size, image_size)`):
            Mel spectrogram images the spectrogram branch reads, the resolutions of one excerpt kept adjacent.
        domain_ids (`torch.LongTensor` of shape `(batch_size,)`):
            Index into `DOMAINS` of the listening-test corpus each prediction should imitate.

        Returns:
            [`UTMOSv2Output`]: The prediction and the fused features behind it.
        """
        domains = F.one_hot(domain_ids, self.config.num_domains).to(input_values.dtype)
        # Upstream reads each branch's own domain input off a layer it then replaces with an identity, so the
        # slots survive as zeros in the fused features and the classifier carries weights for them.
        unused = torch.zeros_like(domains)
        features = torch.cat(
            [self._ssl_features(input_values), unused, self._spectrogram_features(input_features), unused, domains],
            dim=1,
        )
        return UTMOSv2Output(logits=self.classifier(features), last_hidden_state=features)


@auto_docstring(
    custom_intro="""
    UTMOSv2, the mean opinion score predictor that won track one of the VoiceMOS Challenge 2024, as the ensemble
    of cross-validation folds its authors publish and evaluate.
    """
)
class UTMOSv2ForAudioClassification(UTMOSv2PreTrainedModel):
    def __init__(self, config: UTMOSv2Config):
        super().__init__(config)
        self.folds = nn.ModuleList(UTMOSv2Model(config) for _ in range(config.num_folds))
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        r"""
        Loads a UTMOSv2 checkpoint, from the published repository as it stands or from a directory
        [`~weight_conversion.convert`] wrote.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                `"sarulab-speech/UTMOSv2"`, or any repository id or directory holding one of the two layouts.
            args (`tuple`, *optional*):
                Positional arguments of [`~PreTrainedModel.from_pretrained`].
            kwargs (`dict`, *optional*):
                Keyword arguments of [`~PreTrainedModel.from_pretrained`].

        Returns:
            [`UTMOSv2ForAudioClassification`]: The model.
        """
        from .weight_conversion import converted_checkpoint, is_published_layout

        if pretrained_model_name_or_path is not None and is_published_layout(pretrained_model_name_or_path):
            pretrained_model_name_or_path = converted_checkpoint(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    @auto_docstring(checkpoint="sarulab-speech/UTMOSv2")
    def forward(
        self,
        input_values: torch.Tensor,
        input_features: torch.Tensor,
        domain_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> UTMOSv2Output:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, num_samples)`):
            Waveform excerpt the self-supervised branch reads, normalized to zero mean and unit variance.
        input_features (`torch.Tensor` of shape `(batch_size, num_frames * num_spectrograms, 3, image_size, image_size)`):
            Mel spectrogram images the spectrogram branch reads, the resolutions of one excerpt kept adjacent.
        domain_ids (`torch.LongTensor` of shape `(batch_size,)`):
            Index into `DOMAINS` of the listening-test corpus each prediction should imitate.
        labels (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            Mean opinion score each waveform was rated. A batch of at least two is needed for the pairwise term
            to be anything but zero.

        Returns:
            [`UTMOSv2Output`]: The averaged prediction, the prediction of each fold, and the loss where `labels`
            were given.
        """
        fold_logits = torch.stack(
            [fold(input_values, input_features, domain_ids).logits for fold in self.folds]
        )
        logits = fold_logits.mean(dim=0)

        loss = None
        if labels is not None:
            predictions = logits.squeeze(-1)
            targets = labels.to(predictions.dtype)
            differences = (predictions.unsqueeze(1) - predictions.unsqueeze(0)) - (
                targets.unsqueeze(1) - targets.unsqueeze(0)
            )
            pairwise = F.relu(differences.abs() - self.config.pairwise_loss_margin).mean().div(2)
            loss = self.config.pairwise_loss_weight * pairwise + self.config.mse_loss_weight * F.mse_loss(
                predictions, targets
            )

        return UTMOSv2Output(loss=loss, logits=logits, fold_logits=fold_logits)


__all__ = [
    "UTMOSv2ForAudioClassification",
    "UTMOSv2Model",
    "UTMOSv2Output",
    "UTMOSv2PreTrainedModel",
]
