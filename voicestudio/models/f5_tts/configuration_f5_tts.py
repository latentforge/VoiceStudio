# MIT License
#
# Copyright (c) 2024 Yushen CHEN
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
"""Configuration class for F5-TTS."""

from typing import ClassVar

from transformers.configuration_utils import PreTrainedConfig

from ..bigvgan import BigVGANConfig
from ..vocos import VocosConfig


# The two vocoders the released checkpoints were trained against, keyed by their `model_type`.
VOCODER_CONFIGS: dict[str, type[PreTrainedConfig]] = {
    VocosConfig.model_type: VocosConfig,
    BigVGANConfig.model_type: BigVGANConfig,
}


class F5TTSConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`F5TTSForConditionalGeneration`]. It is used to
    instantiate an F5-TTS model according to the specified arguments, defining the conditional flow matching model
    and the backbone that predicts its vector field. Instantiating a configuration with the defaults will yield a
    configuration matching the F5-TTS v1 Base architecture of
    [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone (`str`, *optional*, defaults to `"dit"`):
            Which vector field backbone to build, `"dit"` for the F5-TTS diffusion transformer or `"unett"` for the
            flat UNet transformer of E2-TTS.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the backbone.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of backbone layers. Must be even when `backbone` is `"unett"`.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in each layer.
        head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of a single attention head. The attention inner dimension is
            `num_attention_heads * head_dim` and is independent of `hidden_size`.
        ff_mult (`int`, *optional*, defaults to 2):
            Expansion factor of the feed forward inner dimension.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied to the attention output projection and inside the feed forward.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout applied to the attention probabilities.
        mel_dim (`int`, *optional*, defaults to 100):
            Number of mel filterbank channels the flow is defined over.
        text_vocab_size (`int`, *optional*, defaults to 2545):
            Number of characters in the vocabulary file. The text embedding table holds `text_vocab_size + 1`
            rows, the extra one being the filler at id `0` that [`F5TTSTokenizer`] pads with.
        text_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the text embedding. Defaults to `mel_dim` when set to `None`.
        text_mask_padding (`bool`, *optional*, defaults to `True`):
            Whether filler and batch padding positions are zeroed before and after every text conv block.
        text_average_upsampling (`bool`, *optional*, defaults to `False`):
            Whether the encoded text is average upsampled to the speech length after the text conv blocks. Requires
            `text_mask_padding` to be `True`.
        text_conv_layers (`int`, *optional*, defaults to 4):
            Number of ConvNeXt V2 blocks applied to the text embedding. `0` disables the sinusoidal position
            embedding and the conv blocks altogether.
        text_conv_mult (`int`, *optional*, defaults to 2):
            Expansion factor of the pointwise layer inside each text ConvNeXt V2 block.
        text_max_positions (`int`, *optional*, defaults to 8192):
            Number of precomputed sinusoidal positions available to the text embedding.
        qk_norm (`str`, *optional*):
            Query and key normalization, `None` for none or `"rms_norm"` for per-head RMS normalization.
        pe_attn_head (`int`, *optional*):
            Number of leading attention heads that receive the rotary position embedding. `None` applies it to all
            heads.
        attn_mask_enabled (`bool`, *optional*, defaults to `False`):
            Whether the padding mask is turned into an attention mask. Disabled in every released checkpoint's
            training recipe.
        long_skip_connection (`bool`, *optional*, defaults to `False`):
            Whether the `"dit"` backbone concatenates its input embedding onto its last layer output and projects
            the result back down.
        skip_connect_type (`str`, *optional*, defaults to `"concat"`):
            How the `"unett"` backbone joins a first half layer output onto its second half counterpart, one of
            `"concat"`, `"add"` or `"none"`.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the query and key RMS normalizations.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the adaptive and feed forward layer normalizations.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            Longest speech sequence, in mel frames, the rotary position embedding is built for.
        rope_parameters (`dict`, *optional*):
            Rotary position embedding parameters. Defaults to `{"rope_type": "default", "rope_theta": 10000.0}`.
        audio_drop_prob (`float`, *optional*, defaults to 0.3):
            Probability of dropping the speech conditioning during training, for classifier free guidance.
        cond_drop_prob (`float`, *optional*, defaults to 0.2):
            Probability of dropping the speech conditioning and the text together during training.
        frac_lengths_mask (`tuple[float, float]`, *optional*, defaults to `(0.7, 1.0)`):
            Bounds of the uniform distribution the infilling span length is drawn from, as a fraction of the
            sequence length.
        sigma (`float`, *optional*, defaults to 0.0):
            Standard deviation of the conditional probability path. `0.0` gives the straight optimal transport path
            that every released checkpoint is trained with.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate, in Hz, of the waveform the mel spectrogram is computed from.
        hop_length (`int`, *optional*, defaults to 256):
            Distance in waveform samples between neighbouring mel frames.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer used for the weight matrices.
        vocoder_config (`Union[dict, `VocosConfig`, `BigVGANConfig`]`, *optional*):
            Configuration of the vocoder that turns a predicted log mel spectrogram back into a waveform,
            [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) for every released
            checkpoint whose mel front end is `"vocos"` and
            [nvidia/bigvgan_v2_24khz_100band_256x](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)
            for `F5TTS_Base_bigvgan`. A dictionary is read as the entry of [`VOCODER_CONFIGS`] its `model_type`
            names.

    Example:

    ```python
    >>> from voicestudio.models.f5_tts import F5TTSConfig, F5TTSForConditionalGeneration

    >>> configuration = F5TTSConfig()

    >>> model = F5TTSForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "f5_tts"
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {"vocoder_config": PreTrainedConfig}

    def __init__(
        self,
        backbone: str = "dit",
        hidden_size: int = 1024,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 16,
        head_dim: int = 64,
        ff_mult: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        mel_dim: int = 100,
        text_vocab_size: int = 2545,
        text_dim: int | None = 512,
        text_mask_padding: bool = True,
        text_average_upsampling: bool = False,
        text_conv_layers: int = 4,
        text_conv_mult: int = 2,
        text_max_positions: int = 8192,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = False,
        long_skip_connection: bool = False,
        skip_connect_type: str = "concat",
        rms_norm_eps: float = 1e-6,
        layer_norm_eps: float = 1e-6,
        max_position_embeddings: int = 8192,
        rope_parameters=None,
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        sigma: float = 0.0,
        sampling_rate: int = 24000,
        hop_length: int = 256,
        initializer_range: float = 0.02,
        vocoder_config: dict | PreTrainedConfig | None = None,
        **kwargs,
    ):
        if backbone not in ("dit", "unett"):
            raise ValueError(f"`backbone` must be one of 'dit' or 'unett', got {backbone}.")
        if backbone == "unett" and num_hidden_layers % 2 != 0:
            raise ValueError(f"The 'unett' backbone needs an even `num_hidden_layers`, got {num_hidden_layers}.")
        if qk_norm not in (None, "rms_norm"):
            raise ValueError(f"`qk_norm` must be one of `None` or 'rms_norm', got {qk_norm}.")
        if skip_connect_type not in ("concat", "add", "none"):
            raise ValueError(
                f"`skip_connect_type` must be one of 'concat', 'add' or 'none', got {skip_connect_type}."
            )
        if text_average_upsampling and not text_mask_padding:
            raise ValueError("`text_average_upsampling` requires `text_mask_padding` to be True.")

        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.mel_dim = mel_dim
        self.text_vocab_size = text_vocab_size
        self.text_dim = text_dim if text_dim is not None else mel_dim
        self.text_mask_padding = text_mask_padding
        self.text_average_upsampling = text_average_upsampling
        self.text_conv_layers = text_conv_layers
        self.text_conv_mult = text_conv_mult
        self.text_max_positions = text_max_positions
        self.qk_norm = qk_norm
        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled
        self.long_skip_connection = long_skip_connection
        self.skip_connect_type = skip_connect_type
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.frac_lengths_mask = tuple(frac_lengths_mask)
        self.sigma = sigma
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.initializer_range = initializer_range

        if isinstance(vocoder_config, dict):
            model_type = vocoder_config.get("model_type", VocosConfig.model_type)
            if model_type not in VOCODER_CONFIGS:
                raise ValueError(
                    f"`vocoder_config` names the model type {model_type}, which is not one of the vocoders this "
                    f"model composes, {sorted(VOCODER_CONFIGS)}."
                )
            vocoder_config = VOCODER_CONFIGS[model_type](**vocoder_config)
        elif vocoder_config is None:
            vocoder_config = VocosConfig()
        self.vocoder_config = vocoder_config

        super().__init__(**kwargs)


__all__ = ["VOCODER_CONFIGS", "F5TTSConfig"]
