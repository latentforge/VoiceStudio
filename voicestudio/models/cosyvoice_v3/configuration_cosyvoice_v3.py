"""Configuration class for CosyVoice v3."""

from ..cosyvoice_v1.configuration_cosyvoice_v1 import CosyVoiceV1Config, CosyVoiceV1HiftConfig
from ..cosyvoice_v2.configuration_cosyvoice_v2 import CosyVoiceV2Config, CosyVoiceV2FlowConfig, CosyVoiceV2LLMConfig


class CosyVoiceV3LLMConfig(CosyVoiceV2LLMConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV3LLM`]. Identical field set to
    [`CosyVoiceV2LLMConfig`]; CosyVoice v3 reuses the same Qwen2 backbone architecture, only changing how the
    start/task/fill/end-of-speech ids are placed inside the speech-token embedding table (see [`CosyVoiceV3LLM`]).
    Unlike [`CosyVoiceV2LLMConfig`], the Qwen2 dimension defaults here are overridden from bare `Qwen2Config`'s
    (7B-scale) defaults to match the real `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` checkpoint's small Qwen2-0.5B
    backbone (`CosyVoice-BlankEN/config.json`).

    Args:
        `**kwargs`:
            Keyword arguments passed to [`CosyVoiceV2LLMConfig`]; `hidden_size`, `intermediate_size`,
            `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, and `rope_parameters` default to
            the real checkpoint's Qwen2-0.5B dimensions instead of `Qwen2Config`'s much larger defaults.
    """

    model_type = "cosyvoice_v3_llm"
    base_config_key = "llm_config"

    def __init__(
        self,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        rope_parameters: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rope_parameters=rope_parameters or {"rope_theta": 1000000.0, "rope_type": "default"},
            **kwargs,
        )


class CosyVoiceV3FlowConfig(CosyVoiceV2FlowConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV3FlowMatchingModel`]. Extends
    [`CosyVoiceV2FlowConfig`] with the fields of the diffusion-transformer (DiT) estimator that CosyVoice v3 uses
    in place of the CosyVoice v1/v2 U-Net estimator. Unlike [`CosyVoiceV2FlowConfig`]'s subject model, there is no
    Conformer text encoder or length regulator; `encoder_hidden_size`/`encoder_num_hidden_layers`/
    `encoder_num_attention_heads`/`encoder_intermediate_size` are unused here.

    Args:
        dit_hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the DiT backbone.
        dit_num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of DiT blocks.
        dit_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in each DiT block.
        dit_head_dim (`int`, *optional*, defaults to 64):
            Dimensionality of each attention head.
        dit_ff_mult (`int`, *optional*, defaults to 2):
            Hidden layer size multiplier for the DiT feed-forward blocks, relative to `dit_hidden_size`.
        pre_lookahead_channels (`int`, *optional*, defaults to 1024):
            Bottleneck width of the pre-lookahead layer's hidden convolution, in
            [`CosyVoiceV3PreLookaheadLayer`].
        `**kwargs`:
            Additional keyword arguments passed to [`CosyVoiceV2FlowConfig`]. `input_size` (defaults to 80) and
            `vocab_size` (defaults to 6561) are overridden from [`CosyVoiceV1FlowConfig`]'s v1/v2 defaults (512
            and 4096): unlike v1/v2's Conformer text encoder, v3's `input_embedding` feeds the pre-lookahead
            layer directly at mel-dim width, and the real checkpoint's speech-token vocabulary is larger.
    """

    model_type = "cosyvoice_v3_flow"

    def __init__(
        self,
        dit_hidden_size: int = 1024,
        dit_num_hidden_layers: int = 22,
        dit_num_attention_heads: int = 16,
        dit_head_dim: int = 64,
        dit_ff_mult: int = 2,
        pre_lookahead_channels: int = 1024,
        input_size: int = 80,
        vocab_size: int = 6561,
        **kwargs,
    ):
        self.dit_hidden_size = dit_hidden_size
        self.dit_num_hidden_layers = dit_num_hidden_layers
        self.dit_num_attention_heads = dit_num_attention_heads
        self.dit_head_dim = dit_head_dim
        self.dit_ff_mult = dit_ff_mult
        self.pre_lookahead_channels = pre_lookahead_channels
        super().__init__(input_size=input_size, vocab_size=vocab_size, **kwargs)


class CosyVoiceV3HiftConfig(CosyVoiceV1HiftConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV3HiFTGenerator`]. Unlike
    [`CosyVoiceV1HiftConfig`]'s subject model (two `[8, 8]` transposed-convolution upsample stages), the real
    `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` checkpoint's vocoder is the original repository's
    `CausalHiFTGenerator`, with three `[8, 5, 3]` causal upsample stages and a causal F0 predictor (see
    [`CosyVoiceV3HiFTGenerator`]).

    Args:
        conv_pre_look_right (`int`, *optional*, defaults to 4):
            Number of right-context frames the causal input convolution looks ahead by (`kernel_size =
            conv_pre_look_right + 1`, right-padded instead of symmetrically padded).
        `**kwargs`:
            Additional keyword arguments passed to [`CosyVoiceV1HiftConfig`]; defaults for `upsample_rates`,
            `upsample_kernel_sizes`, `source_resblock_kernel_sizes`, `source_resblock_dilation_sizes`, and
            `sampling_rate` are overridden here to match the real checkpoint.
    """

    model_type = "cosyvoice_v3_hift"

    def __init__(
        self,
        conv_pre_look_right: int = 4,
        sampling_rate: int = 24000,
        upsample_rates: list[int] = [8, 5, 3],
        upsample_kernel_sizes: list[int] = [16, 11, 7],
        source_resblock_kernel_sizes: list[int] = [7, 7, 11],
        source_resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        **kwargs,
    ):
        self.conv_pre_look_right = conv_pre_look_right
        super().__init__(
            sampling_rate=sampling_rate,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            source_resblock_kernel_sizes=source_resblock_kernel_sizes,
            source_resblock_dilation_sizes=source_resblock_dilation_sizes,
            **kwargs,
        )


class CosyVoiceV3Config(CosyVoiceV2Config):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV3ForConditionalGeneration`]. It is
    used to instantiate a CosyVoice v3 model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a configuration close to that of the
    `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` checkpoint.

    Args:
        llm_config (`CosyVoiceV3LLMConfig`, *optional*):
            Configuration for the Qwen2-backbone speech-token language-model sub-model.
        flow_config (`CosyVoiceV3FlowConfig`, *optional*):
            Configuration for the DiT conditional-flow-matching decoder sub-model.
        hift_config (`CosyVoiceV3HiftConfig`, *optional*):
            Configuration for the NSF/ISTFT vocoder sub-model.
        sample_rate (`int`, *optional*, defaults to 24000):
            Output waveform sample rate, in Hz.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing weight matrices.
    """

    model_type = "cosyvoice_v3"
    sub_configs = {
        "llm_config": CosyVoiceV3LLMConfig,
        "flow_config": CosyVoiceV3FlowConfig,
        "hift_config": CosyVoiceV3HiftConfig,
    }

    def __init__(
        self,
        llm_config: dict | None = None,
        flow_config: dict | None = None,
        hift_config: dict | None = None,
        sample_rate: int = 24000,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.llm_config = CosyVoiceV3LLMConfig(**(llm_config or {}))
        self.flow_config = CosyVoiceV3FlowConfig(**(flow_config or {}))
        self.hift_config = self.sub_configs["hift_config"](**(hift_config or {}))
        self.sample_rate = sample_rate
        self.initializer_range = initializer_range
        super(CosyVoiceV1Config, self).__init__(**kwargs)


__all__ = ["CosyVoiceV3Config", "CosyVoiceV3LLMConfig", "CosyVoiceV3FlowConfig", "CosyVoiceV3HiftConfig"]
