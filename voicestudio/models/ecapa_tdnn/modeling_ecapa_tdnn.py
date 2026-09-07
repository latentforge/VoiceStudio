# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# Copyright 2024 SpeechBrain
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
"""PyTorch ECAPA-TDNN model."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import initialization as init
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring

from .configuration_ecapa_tdnn import EcapaTdnnConfig


@dataclass
@auto_docstring(custom_intro="Output of [`EcapaTdnnModel`] and [`EcapaTdnnForXVector`].")
class EcapaTdnnOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Additive angular margin softmax loss over the speakers.
    logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`, *optional*):
        Cosine similarity between the embedding and each speaker's direction. Returned by
        [`EcapaTdnnForXVector`].
    embeddings (`torch.FloatTensor` of shape `(batch_size, xvector_output_dim)`):
        Speaker embedding, the vector to compare with cosine similarity.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    embeddings: torch.FloatTensor | None = None


class EcapaTdnnConv1d(nn.Conv1d):
    r"""
    Convolution that keeps its input length by reflecting the signal at both ends, which is how the published
    weights were trained and is not what `nn.Conv1d`'s own padding does.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Side of the convolution kernel.
        dilation (`int`, *optional*, defaults to 1):
            Dilation of the convolution.
        groups (`int`, *optional*, defaults to 1):
            Number of blocked connections from input to output channels.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, groups: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)
        self.reflect_padding = dilation * (kernel_size - 1) // 2

    def forward(self, hidden_states: torch.Tensor, reflection_index: torch.Tensor | None = None) -> torch.Tensor:
        if self.reflect_padding:
            # `F.pad` reflects at the end of the tensor it is handed, which for a padded batch is the
            # end of its longest item. Reading each item through its own reflection first is what that
            # item would have seen on its own, and is what keeps a batched result equal to a lone one.
            if reflection_index is not None:
                hidden_states = hidden_states.gather(2, reflection_index.expand(-1, hidden_states.shape[1], -1))
            hidden_states = F.pad(hidden_states, (self.reflect_padding, self.reflect_padding), mode="reflect")
        return self._conv_forward(hidden_states, self.weight, self.bias)


class EcapaTdnnTdnnBlock(nn.Module):
    r"""
    Dilated convolution, activation, batch normalization and channel dropout, the layer the network is built from.

    Args:
        config ([`EcapaTdnnConfig`]):
            Model configuration.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Side of the convolution kernel.
        dilation (`int`):
            Dilation of the convolution.
        groups (`int`, *optional*, defaults to 1):
            Number of blocked connections from input to output channels.
    """

    def __init__(
        self,
        config: EcapaTdnnConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = EcapaTdnnConv1d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)
        self.norm = nn.BatchNorm1d(out_channels, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.dropout = nn.Dropout1d(p=config.dropout)

    def forward(self, hidden_states: torch.Tensor, reflection_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.dropout(self.norm(F.relu(self.conv(hidden_states, reflection_index))))


class EcapaTdnnRes2NetBlock(nn.Module):
    r"""
    Splits the channels into `res2net_scale` groups and convolves them in a chain, each group added to the
    convolved output of the previous one, which gives one block several receptive field sizes at once.

    Args:
        config ([`EcapaTdnnConfig`]):
            Model configuration.
        channels (`int`):
            Number of input and output channels, which must divide by `res2net_scale`.
        kernel_size (`int`):
            Side of each group's convolution kernel.
        dilation (`int`):
            Dilation of each group's convolution.
    """

    def __init__(self, config: EcapaTdnnConfig, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        if channels % config.res2net_scale:
            raise ValueError(f"{channels} channels do not divide into {config.res2net_scale} groups.")
        group = channels // config.res2net_scale
        self.scale = config.res2net_scale
        self.blocks = nn.ModuleList(
            EcapaTdnnTdnnBlock(config, group, group, kernel_size, dilation) for _ in range(self.scale - 1)
        )

    def forward(self, hidden_states: torch.Tensor, reflection_index: torch.Tensor | None = None) -> torch.Tensor:
        outputs = []
        for index, group in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if index == 0:
                output = group
            elif index == 1:
                output = self.blocks[0](group, reflection_index)
            else:
                output = self.blocks[index - 1](group + output, reflection_index)
            outputs.append(output)
        return torch.cat(outputs, dim=1)


class EcapaTdnnSqueezeExcite(nn.Module):
    r"""
    Channel gate that scales every channel by a sigmoid of its time averaged, bottlenecked activation, with the
    average taken over the valid frames alone.

    Args:
        channels (`int`):
            Number of channels to gate.
        se_channels (`int`):
            Width of the bottleneck.
    """

    def __init__(self, channels: int, se_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, se_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(se_channels, channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            gate = hidden_states.mean(dim=2, keepdim=True)
        else:
            gate = (hidden_states * mask).sum(dim=2, keepdim=True) / mask.sum(dim=2, keepdim=True)
        gate = F.relu(self.conv1(gate), inplace=True)
        return torch.sigmoid(self.conv2(gate)) * hidden_states


class EcapaTdnnAttentiveStatisticsPooling(nn.Module):
    r"""
    Pools a frame sequence into the concatenation of a weighted mean and standard deviation, with a per-channel
    attention deciding the weights. Where `global_context` is set, that attention also sees the utterance
    statistics repeated alongside every frame.

    Args:
        config ([`EcapaTdnnConfig`]):
            Model configuration.
    """

    def __init__(self, config: EcapaTdnnConfig):
        super().__init__()
        self.eps = 1e-12
        self.global_context = config.global_context
        channels = config.hidden_size
        self.tdnn = EcapaTdnnTdnnBlock(
            config, channels * 3 if config.global_context else channels, config.attention_channels, 1, 1
        )
        self.conv = nn.Conv1d(config.attention_channels, channels, kernel_size=1)

    def _statistics(self, hidden_states: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (weights * hidden_states).sum(dim=2)
        variance = (weights * (hidden_states - mean.unsqueeze(2)).pow(2)).sum(dim=2)
        return mean, variance.clamp(self.eps).sqrt()

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num_frames = hidden_states.shape[-1]
        if self.global_context:
            mean, std = self._statistics(hidden_states, mask / mask.sum(dim=2, keepdim=True).float())
            attention = torch.cat(
                [
                    hidden_states,
                    mean.unsqueeze(2).repeat(1, 1, num_frames),
                    std.unsqueeze(2).repeat(1, 1, num_frames),
                ],
                dim=1,
            )
        else:
            attention = hidden_states

        attention = self.conv(torch.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)

        mean, std = self._statistics(hidden_states, attention)
        return torch.cat([mean, std], dim=1)


class EcapaTdnnSeRes2NetBlock(nn.Module):
    r"""
    The residual block of ECAPA-TDNN: a pointwise convolution, a multi-scale dilated convolution, a second
    pointwise convolution and a channel gate, added to the block input.

    Args:
        config ([`EcapaTdnnConfig`]):
            Model configuration.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Side of the multi-scale convolution kernel.
        dilation (`int`):
            Dilation of the multi-scale convolution.
    """

    def __init__(
        self,
        config: EcapaTdnnConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.tdnn1 = EcapaTdnnTdnnBlock(config, in_channels, out_channels, 1, 1)
        self.res2net_block = EcapaTdnnRes2NetBlock(config, out_channels, kernel_size, dilation)
        self.tdnn2 = EcapaTdnnTdnnBlock(config, out_channels, out_channels, 1, 1)
        self.se_block = EcapaTdnnSqueezeExcite(out_channels, config.se_channels)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        reflection_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.shortcut(hidden_states) if self.shortcut is not None else hidden_states
        hidden_states = self.tdnn1(hidden_states, reflection_index)
        hidden_states = self.res2net_block(hidden_states, reflection_index)
        hidden_states = self.tdnn2(hidden_states, reflection_index)
        hidden_states = self.se_block(hidden_states, mask)
        return hidden_states + residual


@auto_docstring
class EcapaTdnnPreTrainedModel(PreTrainedModel):
    config: EcapaTdnnConfig
    base_model_prefix = "ecapa_tdnn"
    main_input_name = "input_features"
    supports_gradient_checkpointing = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        r"""
        Loads an ECAPA-TDNN checkpoint, from a published SpeechBrain repository as it stands or from a directory
        [`~weight_conversion.convert`] wrote.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                `"speechbrain/spkrec-ecapa-voxceleb"`, or any repository id or directory holding one of the two
                layouts.
            args (`tuple`, *optional*):
                Positional arguments of [`~PreTrainedModel.from_pretrained`].
            kwargs (`dict`, *optional*):
                Keyword arguments of [`~PreTrainedModel.from_pretrained`].

        Returns:
            [`EcapaTdnnPreTrainedModel`]: The model, of whichever class this was called on.
        """
        from .weight_conversion import converted_checkpoint, is_published_layout

        if pretrained_model_name_or_path is not None and is_published_layout(pretrained_model_name_or_path):
            pretrained_model_name_or_path = converted_checkpoint(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, EcapaTdnnForXVector):
            init.xavier_uniform_(module.classifier)
            init.zeros_(module.embedding_mean)
        else:
            super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    ECAPA-TDNN, the speaker embedding network that adds channel attention, multi-scale residual blocks and
    multi-layer feature aggregation to a time delay neural network.
    """
)
class EcapaTdnnModel(EcapaTdnnPreTrainedModel):
    def __init__(self, config: EcapaTdnnConfig):
        super().__init__(config)
        channels, kernel_sizes, dilations = config.channels, config.kernel_sizes, config.dilations

        self.blocks = nn.ModuleList(
            [EcapaTdnnTdnnBlock(config, config.num_mel_bins, channels[0], kernel_sizes[0], dilations[0])]
        )
        for index in range(1, len(channels) - 1):
            self.blocks.append(
                EcapaTdnnSeRes2NetBlock(
                    config, channels[index - 1], channels[index], kernel_sizes[index], dilations[index]
                )
            )

        self.mfa = EcapaTdnnTdnnBlock(
            config, channels[-2] * (len(channels) - 2), channels[-1], kernel_sizes[-1], dilations[-1]
        )
        self.asp = EcapaTdnnAttentiveStatisticsPooling(config)
        self.asp_bn = nn.BatchNorm1d(
            channels[-1] * 2, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        self.fc = nn.Conv1d(channels[-1] * 2, config.xvector_output_dim, kernel_size=1)
        self.post_init()

    @staticmethod
    def _reflection_index(mask: torch.Tensor) -> torch.Tensor:
        r"""
        Builds the index that reads each item of a padded batch through the reflection of its own frames.

        Args:
            mask (`torch.Tensor` of shape `(batch_size, 1, num_frames)`):
                Mask marking the unpadded frames of each item.

        Returns:
            `torch.Tensor` of shape `(batch_size, 1, num_frames)`: The frame each position reads from, which is
            the position itself inside the item and its reflection beyond the item's end.
        """
        num_frames = mask.shape[-1]
        lengths = mask.sum(dim=2, keepdim=True).long()
        position = torch.arange(num_frames, device=mask.device)[None, None, :]
        period = (2 * lengths - 2).clamp(min=1)
        folded = position.remainder(period)
        return torch.minimum(folded, period - folded)

    @auto_docstring(checkpoint="speechbrain/spkrec-ecapa-voxceleb")
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> EcapaTdnnOutput:
        r"""
        input_features (`torch.Tensor` of shape `(batch_size, num_frames, num_mel_bins)`):
            Mean normalized log filterbank features, as [`EcapaTdnnFeatureExtractor`] returns them.
        attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask marking the unpadded frames of each item. Padded frames are excluded from every channel gate and
            from the pooling.

        Returns:
            [`EcapaTdnnOutput`]: The speaker embeddings.
        """
        hidden_states = input_features.transpose(1, 2)
        num_frames = hidden_states.shape[-1]
        mask = (
            attention_mask[:, None, :].to(hidden_states.dtype)
            if attention_mask is not None
            else hidden_states.new_ones(hidden_states.shape[0], 1, num_frames)
        )

        reflection_index = self._reflection_index(mask) if attention_mask is not None else None
        outputs = []
        for block in self.blocks:
            hidden_states = (
                block(hidden_states, reflection_index)
                if isinstance(block, EcapaTdnnTdnnBlock)
                else block(hidden_states, mask, reflection_index)
            )
            outputs.append(hidden_states)

        hidden_states = self.mfa(torch.cat(outputs[1:], dim=1), reflection_index)
        hidden_states = self.asp_bn(self.asp(hidden_states, mask))
        embeddings = self.fc(hidden_states.unsqueeze(2)).squeeze(2)
        return EcapaTdnnOutput(embeddings=embeddings)


@auto_docstring(
    custom_intro="""
    ECAPA-TDNN with the cosine similarity head its authors train it behind, which scores an embedding against one
    learned direction per speaker.
    """
)
class EcapaTdnnForXVector(EcapaTdnnPreTrainedModel):
    def __init__(self, config: EcapaTdnnConfig):
        super().__init__(config)
        self.ecapa_tdnn = EcapaTdnnModel(config)
        # Mean of the embeddings over the training set, which upstream subtracts before the head but not before
        # the cosine similarity of two utterances.
        self.register_buffer("embedding_mean", torch.zeros(config.xvector_output_dim))
        self.classifier = nn.Parameter(torch.empty(config.num_labels, config.xvector_output_dim))
        self.post_init()

    @auto_docstring(checkpoint="speechbrain/spkrec-ecapa-voxceleb")
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> EcapaTdnnOutput:
        r"""
        input_features (`torch.Tensor` of shape `(batch_size, num_frames, num_mel_bins)`):
            Mean normalized log filterbank features, as [`EcapaTdnnFeatureExtractor`] returns them.
        attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask marking the unpadded frames of each item.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Index of the speaker each utterance belongs to.

        Returns:
            [`EcapaTdnnOutput`]: The speaker embeddings, the per-speaker cosine similarities, and the loss where
            `labels` were given.
        """
        embeddings = self.ecapa_tdnn(input_features, attention_mask=attention_mask).embeddings
        logits = F.linear(F.normalize(embeddings - self.embedding_mean), F.normalize(self.classifier))

        loss = None
        if labels is not None:
            cosine = logits.float().clamp(-1 + 1e-7, 1 - 1e-7)
            sine = (1.0 - cosine.pow(2)).sqrt()
            margin = cosine * math.cos(self.config.margin) - sine * math.sin(self.config.margin)
            threshold = math.cos(math.pi - self.config.margin)
            offset = math.sin(math.pi - self.config.margin) * self.config.margin
            margin = torch.where(cosine > threshold, margin, cosine - offset)
            targets = F.one_hot(labels, self.config.num_labels).to(cosine.dtype)
            loss = F.cross_entropy(self.config.scale * (targets * margin + (1.0 - targets) * cosine), labels)

        return EcapaTdnnOutput(loss=loss, logits=logits, embeddings=embeddings)


__all__ = [
    "EcapaTdnnForXVector",
    "EcapaTdnnModel",
    "EcapaTdnnOutput",
    "EcapaTdnnPreTrainedModel",
]
