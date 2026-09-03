# Copyright 2026 Nari Labs and the LatentForge team. All rights reserved.
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
"""Configuration class for Dia2."""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import logging


logger = logging.get_logger(__name__)


@strict
class Dia2DepthDecoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Dia2DepthDecoderModel`]. It is used to
    instantiate the depth decoder that turns one backbone frame into the remaining Mimi codebooks of that frame,
    one position per codebook.

    Args:
        num_codebooks (`int`, *optional*, defaults to 32):
            Number of Mimi codebooks per audio frame. The depth decoder covers `num_codebooks - 1` positions,
            since the first codebook is predicted by the backbone.
        weights_schedule (`list[int]`, *optional*):
            Group index of each of the `num_codebooks - 1` depth positions. Positions sharing a group share the
            attention and input projection weights of that group. Defaults to a single group covering every
            position.
        backbone_hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the backbone hidden state projected into the depth decoder at every position.
        vocab_size (`int`, *optional*, defaults to 2050):
            Size of a codebook vocabulary, including the beginning-of-stream and padding ids appended after the
            codec's own entries.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the depth decoder representations.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the gated MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 3):
            Number of depth decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads. Defaults to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of an attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation applied to the gate branch of the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 32):
            Longest depth position index the rotary embedding is built for.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the key/value states of the depth positions already decoded for the current frame.
        use_rope (`bool`, *optional*, defaults to `True`):
            Whether to rotate queries and keys by the depth position index.
        use_text_embedding (`bool`, *optional*, defaults to `False`):
            Whether the first depth position also receives the two text stream embeddings of the frame.
        text_vocab_size (`int`, *optional*, defaults to 49280):
            Size of the text vocabulary, used only when `use_text_embedding` is `True`.
        text_pad_token_id (`int`, *optional*, defaults to 3):
            Id whose presence on the second text stream suppresses that stream's contribution, used only when
            `use_text_embedding` is `True`.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary holding the rotary embedding type and its `rope_theta` base wavelength.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether the attention projections carry a bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio of the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether the MLP projections carry a bias.

    Example:

    ```python
    >>> from voicestudio.models.dia2 import Dia2DepthDecoderConfig, Dia2DepthDecoderModel

    >>> configuration = Dia2DepthDecoderConfig()

    >>> model = Dia2DepthDecoderModel(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "dia2_depth_decoder"
    base_config_key = "depth_decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"codebook_size": "vocab_size"}

    num_codebooks: int = 32
    weights_schedule: list[int] | None = None
    backbone_hidden_size: int = 1024
    vocab_size: int = 2050
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 3
    num_attention_heads: int = 8
    num_key_value_heads: int | None = 8
    head_dim: int | None = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    use_rope: bool = True
    use_text_embedding: bool = False
    text_vocab_size: int = 49280
    text_pad_token_id: int | None = 3
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False

    def __post_init__(self, **kwargs):
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("`tie_word_embeddings=True` is not supported for Dia2DepthDecoderConfig")

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.weights_schedule is None:
            self.weights_schedule = [0] * (self.num_codebooks - 1)
        elif len(self.weights_schedule) != self.num_codebooks - 1:
            raise ValueError(
                f"`weights_schedule` must hold one group index per depth position, got "
                f"{len(self.weights_schedule)} for {self.num_codebooks - 1} positions."
            )
        if sorted(set(self.weights_schedule)) != list(range(max(self.weights_schedule) + 1)):
            raise ValueError(f"`weights_schedule` must use contiguous group indices from 0, got {self.weights_schedule}")

        self.tie_word_embeddings = False
        super().__post_init__(**kwargs)

    @property
    def num_weight_groups(self) -> int:
        return max(self.weights_schedule) + 1


@strict
class Dia2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Dia2ForConditionalGeneration`]. It is used to
    instantiate a Dia2 model according to the specified arguments, defining a backbone that consumes two text
    streams plus every Mimi codebook of a frame and predicts the next frame's word-advance action and first
    codebook, paired with a [`Dia2DepthDecoderModel`] that predicts the frame's remaining codebooks.

    Args:
        num_codebooks (`int`, *optional*, defaults to 32):
            Number of Mimi codebooks per audio frame.
        vocab_size (`int`, *optional*, defaults to 2050):
            Size of a codebook vocabulary, including the beginning-of-stream and padding ids appended after the
            codec's own entries.
        text_vocab_size (`int`, *optional*, defaults to 49280):
            Size of the text vocabulary shared by both text streams.
        action_vocab_size (`int`, *optional*, defaults to 2):
            Size of the action vocabulary the backbone predicts, one id to hold the current word and one to
            advance to the next.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the backbone representations.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the gated MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 30):
            Number of backbone layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of an attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation applied to the gate branch of the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 1500):
            Longest frame sequence the model can be run on.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the key/value states of the frames already decoded.
        text_low_rank_dim (`int`, *optional*):
            Dimensionality of the shared text embedding table when it is smaller than `hidden_size`. Both text
            streams project from it into `hidden_size` through their own matrix. Defaults to `hidden_size`.
        delay_pattern (`list[int]`, *optional*):
            Frame offset applied to each codebook when the aligned codes are laid out on the model's grid.
            Defaults to no delay on any codebook.
        codebook_bos_token_id (`int`, *optional*, defaults to 2048):
            Id, within a codebook's vocabulary, fed on the grid positions preceding that codebook's delay.
        codebook_pad_token_id (`int`, *optional*, defaults to 2049):
            Id, within a codebook's vocabulary, filling the grid positions past the end of the audio.
        text_pad_token_id (`int`, *optional*, defaults to 3):
            Id emitted on a text stream while it holds, and whose presence on the second stream suppresses that
            stream's contribution to the frame embedding.
        text_bos_token_id (`int`, *optional*, defaults to 1):
            Id fed on the first text stream of the very first frame.
        text_new_word_token_id (`int`, *optional*, defaults to 2):
            Id emitted on a text stream on the frame that opens a new word.
        text_zero_token_id (`int`, *optional*, defaults to 7):
            Id fed on the first text stream of the unconditional branch of classifier-free guidance.
        action_pad_token_id (`int`, *optional*, defaults to 0):
            Action id meaning the current word is held for another frame.
        action_new_word_token_id (`int`, *optional*, defaults to 1):
            Action id meaning the next word starts on the next frame.
        second_stream_ahead (`int`, *optional*, defaults to 2):
            Number of word entries by which the second text stream runs ahead of the first. Zero disables the
            second stream, which then repeats the first.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary holding the rotary embedding type and its `rope_theta` base wavelength.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether the attention projections carry a bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio of the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether the MLP projections carry a bias.
        depth_decoder_config ([`Dia2DepthDecoderConfig`], *optional*):
            Configuration of the depth decoder.
        codec_model_id (`str`, *optional*, defaults to `"kyutai/mimi"`):
            Repository id of the [`MimiModel`] that encodes waveforms to codebook ids and decodes them back.
        sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate, in Hz, of the waveform the codec produces.
        frame_rate (`float`, *optional*, defaults to 12.5):
            Number of audio frames per second the codec produces, which is also the model's step rate.

    Example:

    ```python
    >>> from voicestudio.models.dia2 import Dia2Config, Dia2ForConditionalGeneration

    >>> configuration = Dia2Config()

    >>> model = Dia2ForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "dia2"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"depth_decoder_config": Dia2DepthDecoderConfig}
    attribute_map = {"codebook_size": "vocab_size"}

    num_codebooks: int = 32
    vocab_size: int = 2050
    text_vocab_size: int = 49280
    action_vocab_size: int = 2
    hidden_size: int = 1024
    intermediate_size: int = 6144
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    head_dim: int | None = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 1500
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    text_low_rank_dim: int | None = None
    delay_pattern: list[int] | None = None
    codebook_bos_token_id: int = 2048
    codebook_pad_token_id: int = 2049
    text_pad_token_id: int = 3
    text_bos_token_id: int = 1
    text_new_word_token_id: int = 2
    text_zero_token_id: int = 7
    action_pad_token_id: int = 0
    action_new_word_token_id: int = 1
    second_stream_ahead: int = 2
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False
    depth_decoder_config: dict | PreTrainedConfig | None = None
    codec_model_id: str = "kyutai/mimi"
    sample_rate: int = 24000
    frame_rate: float = 12.5
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("`tie_word_embeddings=True` is not supported for Dia2Config")

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.delay_pattern is None:
            self.delay_pattern = [0] * self.num_codebooks
        elif len(self.delay_pattern) != self.num_codebooks:
            raise ValueError(
                f"`delay_pattern` must hold one delay per codebook, got {len(self.delay_pattern)} for "
                f"{self.num_codebooks} codebooks."
            )

        if self.depth_decoder_config is None:
            self.depth_decoder_config = Dia2DepthDecoderConfig(
                num_codebooks=self.num_codebooks,
                backbone_hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                text_vocab_size=self.text_vocab_size,
                text_pad_token_id=self.text_pad_token_id,
                rms_norm_eps=self.rms_norm_eps,
            )
        elif isinstance(self.depth_decoder_config, dict):
            self.depth_decoder_config = Dia2DepthDecoderConfig(**self.depth_decoder_config)

        self.tie_word_embeddings = False
        super().__post_init__(**kwargs)

    @property
    def num_channels(self) -> int:
        return self.num_codebooks + 2

    @property
    def max_delay(self) -> int:
        return max(self.delay_pattern) if self.delay_pattern else 0


__all__ = ["Dia2Config", "Dia2DepthDecoderConfig"]
