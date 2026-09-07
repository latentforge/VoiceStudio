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
"""Configuration class for ECAPA-TDNN."""

from transformers.configuration_utils import PreTrainedConfig


class EcapaTdnnConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`EcapaTdnnModel`], the speaker embedding
    network of "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker
    Verification". Instantiating a configuration with the defaults will yield a configuration matching the
    [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) checkpoint.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of filterbank channels the network reads, the width of its first convolution.
        channels (`list[int]`, *optional*, defaults to `[1024, 1024, 1024, 1024, 3072]`):
            Output width of each stage. The first entry is the opening dilated convolution, the last is the
            multi-layer aggregation, and the ones between are the residual blocks whose outputs the aggregation
            concatenates.
        kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            Kernel size of each stage, one per entry of `channels`.
        dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            Dilation of each stage, one per entry of `channels`. The widening dilation is what gives the residual
            blocks their different temporal contexts.
        attention_channels (`int`, *optional*, defaults to 128):
            Width of the bottleneck the attentive statistics pooling scores frames through.
        res2net_scale (`int`, *optional*, defaults to 8):
            Number of groups a residual block splits its channels into, each but the first convolved and added to
            the next, which is what makes the block multi-scale.
        se_channels (`int`, *optional*, defaults to 128):
            Width of the bottleneck of each residual block's channel gate.
        global_context (`bool`, *optional*, defaults to `True`):
            Whether the pooling attention sees the utterance mean and standard deviation alongside each frame.
        xvector_output_dim (`int`, *optional*, defaults to 192):
            Dimensionality of the speaker embedding.
        num_labels (`int`, *optional*, defaults to 7205):
            Number of speakers the classification head scores, the VoxCeleb 1 and 2 development speakers.
        margin (`float`, *optional*, defaults to 0.2):
            Angular margin added to the target speaker's angle by the training objective.
        scale (`float`, *optional*, defaults to 30.0):
            Factor the margin-adjusted cosine similarities are multiplied by before the softmax.
        dropout (`float`, *optional*, defaults to 0.0):
            Rate of channel dropout after each convolution block.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon of the batch normalizations.
        batch_norm_momentum (`float`, *optional*, defaults to 0.1):
            Momentum of the batch normalizations.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer of the linear layers.
    """

    model_type = "ecapa_tdnn"

    def __init__(
        self,
        num_mel_bins: int = 80,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
        attention_channels: int = 128,
        res2net_scale: int = 8,
        se_channels: int = 128,
        global_context: bool = True,
        xvector_output_dim: int = 192,
        num_labels: int = 7205,
        margin: float = 0.2,
        scale: float = 30.0,
        dropout: float = 0.0,
        batch_norm_eps: float = 1e-5,
        batch_norm_momentum: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.num_mel_bins = num_mel_bins
        self.channels = list(channels) if channels is not None else [1024, 1024, 1024, 1024, 3072]
        self.kernel_sizes = list(kernel_sizes) if kernel_sizes is not None else [5, 3, 3, 3, 1]
        self.dilations = list(dilations) if dilations is not None else [1, 2, 3, 4, 1]
        if not len(self.channels) == len(self.kernel_sizes) == len(self.dilations):
            raise ValueError(
                "`channels`, `kernel_sizes` and `dilations` must have the same length, got "
                f"{len(self.channels)}, {len(self.kernel_sizes)} and {len(self.dilations)}."
            )
        self.attention_channels = attention_channels
        self.res2net_scale = res2net_scale
        self.se_channels = se_channels
        self.global_context = global_context
        self.xvector_output_dim = xvector_output_dim
        self.margin = margin
        self.scale = scale
        self.dropout = dropout
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.initializer_range = initializer_range
        super().__init__(num_labels=num_labels, **kwargs)

    @property
    def hidden_size(self) -> int:
        r"""
        Returns:
            `int`: Dimensionality of the aggregated frame level features the pooling reads.
        """
        return self.channels[-1]


__all__ = ["EcapaTdnnConfig"]
