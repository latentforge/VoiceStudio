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
"""Configuration class for UTMOSv2."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config


# One entry per stage of the EfficientNetV2-S spectrogram encoder, as
# `(block_type, num_layers, kernel_size, stride, expand_ratio, out_channels, se_ratio)`.
EFFICIENTNET_V2_S_STAGES = (
    ("conv", 2, 3, 1, 1, 24, 0.0),
    ("fused", 4, 3, 2, 4, 48, 0.0),
    ("fused", 4, 3, 2, 4, 64, 0.0),
    ("mbconv", 6, 3, 2, 4, 128, 0.25),
    ("mbconv", 9, 3, 1, 6, 160, 0.25),
    ("mbconv", 15, 3, 2, 6, 256, 0.25),
)


class UTMOSv2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UTMOSv2ForAudioClassification`], the mean
    opinion score predictor that fuses a wav2vec 2.0 branch with an EfficientNetV2 branch reading mel spectrograms
    as images. Instantiating a configuration with the defaults will yield a configuration matching the
    [sarulab-speech/UTMOSv2](https://huggingface.co/sarulab-speech/UTMOSv2) checkpoint, the `fusion_stage3`
    configuration of the upstream repository.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        ssl_config (`dict` or [`Wav2Vec2Config`], *optional*):
            Configuration of the self-supervised branch. Defaults to the
            [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) architecture, whose thirteen
            hidden states the branch mixes with learned weights.
        num_folds (`int`, *optional*, defaults to 5):
            Number of independently trained predictors whose outputs are averaged. The published checkpoint holds
            the five cross-validation folds the authors ensemble; `1` builds the single predictor upstream trains.
        num_domains (`int`, *optional*, defaults to 10):
            Number of listening-test corpora the training data was drawn from. The one-hot domain vector is
            concatenated onto the fused features, so a prediction is conditioned on which corpus it is asked to
            imitate.
        num_spectrograms (`int`, *optional*, defaults to 4):
            Number of mel spectrogram resolutions, each read by its own encoder. Their outputs are mixed with
            learned weights.
        num_frames (`int`, *optional*, defaults to 2):
            Number of excerpts of the waveform the spectrogram branch reads. Their feature maps are concatenated
            along the time axis before pooling.
        stem_channels (`int`, *optional*, defaults to 24):
            Number of channels the spectrogram encoder's stem convolution produces.
        head_channels (`int`, *optional*, defaults to 1280):
            Number of channels the spectrogram encoder's final pointwise convolution produces.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of heads of the self-attention applied to the pooled features of both branches.
        attention_dropout (`float`, *optional*, defaults to 0.2):
            Dropout ratio of those self-attentions.
        num_ssl_attention_layers (`int`, *optional*, defaults to 1):
            Number of self-attention layers stacked on the mixed wav2vec 2.0 hidden states.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Maximum stochastic depth ratio of the spectrogram encoder, scaled linearly with block depth.
        batch_norm_eps (`float`, *optional*, defaults to 0.001):
            Epsilon of the spectrogram encoder's batch normalizations. The published weights were ported from
            TensorFlow, which is why this is not the PyTorch default.
        batch_norm_momentum (`float`, *optional*, defaults to 0.1):
            Momentum of those batch normalizations.
        num_labels (`int`, *optional*, defaults to 1):
            Number of scores each predictor emits.
        pairwise_loss_weight (`float`, *optional*, defaults to 0.7):
            Weight of the pairwise ranking term of the training objective.
        pairwise_loss_margin (`float`, *optional*, defaults to 0.2):
            Margin below which a pairwise score difference costs nothing.
        mse_loss_weight (`float`, *optional*, defaults to 0.2):
            Weight of the mean squared error term of the training objective.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer of the linear layers.
    """

    model_type = "utmos_v2"
    sub_configs = {"ssl_config": Wav2Vec2Config}

    def __init__(
        self,
        ssl_config=None,
        num_folds: int = 5,
        num_domains: int = 10,
        num_spectrograms: int = 4,
        num_frames: int = 2,
        stem_channels: int = 24,
        head_channels: int = 1280,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.2,
        num_ssl_attention_layers: int = 1,
        drop_path_rate: float = 0.0,
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.1,
        num_labels: int = 1,
        pairwise_loss_weight: float = 0.7,
        pairwise_loss_margin: float = 0.2,
        mse_loss_weight: float = 0.2,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if ssl_config is None:
            ssl_config = Wav2Vec2Config()
        elif isinstance(ssl_config, dict):
            ssl_config = CONFIG_MAPPING[ssl_config.pop("model_type", "wav2vec2")](**ssl_config)
        self.ssl_config = ssl_config
        self.num_folds = num_folds
        self.num_domains = num_domains
        self.num_spectrograms = num_spectrograms
        self.num_frames = num_frames
        self.stem_channels = stem_channels
        self.head_channels = head_channels
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.num_ssl_attention_layers = num_ssl_attention_layers
        self.drop_path_rate = drop_path_rate
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.pairwise_loss_weight = pairwise_loss_weight
        self.pairwise_loss_margin = pairwise_loss_margin
        self.mse_loss_weight = mse_loss_weight
        self.initializer_range = initializer_range
        super().__init__(num_labels=num_labels, **kwargs)

    @property
    def num_ssl_hidden_states(self) -> int:
        r"""
        Returns:
            `int`: Number of hidden states the self-supervised branch mixes, the embedding output plus one per layer.
        """
        return self.ssl_config.num_hidden_layers + 1

    @property
    def ssl_hidden_size(self) -> int:
        r"""
        Returns:
            `int`: Dimensionality of the self-supervised branch's hidden states.
        """
        return self.ssl_config.hidden_size

    @property
    def ssl_branch_size(self) -> int:
        r"""
        Returns:
            `int`: Dimensionality the self-supervised branch contributes to the fused features. Concatenating the
            attended mean and the unattended maximum doubles the hidden size, and a block of `num_domains` zeros
            follows it.
        """
        return self.ssl_hidden_size * 2 + self.num_domains

    @property
    def spectrogram_branch_size(self) -> int:
        r"""
        Returns:
            `int`: Dimensionality the spectrogram branch contributes to the fused features. Concatenating the
            average and maximum of the two-dimensional pooling doubles the encoder width, concatenating the
            attended mean and the unattended maximum doubles it again, and a block of `num_domains` zeros follows
            it.
        """
        return self.head_channels * 4 + self.num_domains

    @property
    def hidden_size(self) -> int:
        r"""
        Returns:
            `int`: Dimensionality of the features the classifier reads, both branches and the domain vector.
        """
        return self.ssl_branch_size + self.spectrogram_branch_size + self.num_domains


__all__ = ["UTMOSv2Config"]
