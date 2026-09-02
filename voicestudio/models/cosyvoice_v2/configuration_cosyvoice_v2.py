"""Configuration class for CosyVoice v2."""

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from ..cosyvoice_v1.configuration_cosyvoice_v1 import CosyVoiceV1Config, CosyVoiceV1FlowConfig, CosyVoiceV1HiftConfig


class CosyVoiceV2LLMConfig(Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV2LLM`]. It extends
    [`Qwen2Config`] (the pretrained text LLM backbone CosyVoice v2 repurposes as a speech-token language model)
    with the extra fields needed to predict discrete speech tokens.

    Args:
        speech_token_size (`int`, *optional*, defaults to 6561):
            Vocabulary size of the discrete speech tokenizer. Three extra ids above this value are reserved for
            the start-of-sequence/task and fill tokens.
        mix_ratio (`list[int]`, *optional*, defaults to `[5, 15]`):
            Number of text tokens to number of speech tokens interleaved per chunk in bi-streaming decoding.
        length_normalized_loss (`bool`, *optional*, defaults to `True`):
            Whether the cross-entropy loss is normalized by sequence length (`True`) or by batch size.
        label_smoothing (`float`, *optional*, defaults to 0.0):
            Label smoothing applied to the speech-token cross-entropy loss.
        `**kwargs`:
            Additional keyword arguments passed to [`Qwen2Config`].
    """

    model_type = "cosyvoice_v2_llm"
    base_config_key = "llm_config"

    def __init__(
        self,
        speech_token_size: int = 6561,
        mix_ratio: list[int] = [5, 15],
        length_normalized_loss: bool = True,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        self.speech_token_size = speech_token_size
        self.mix_ratio = mix_ratio
        self.length_normalized_loss = length_normalized_loss
        self.label_smoothing = label_smoothing
        super().__init__(**kwargs)


class CosyVoiceV2FlowConfig(CosyVoiceV1FlowConfig):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV2FlowMatchingModel`]. It extends
    [`CosyVoiceV1FlowConfig`] with the causal-streaming fields CosyVoice v2's flow decoder adds.

    Args:
        token_mel_ratio (`int`, *optional*, defaults to 2):
            Number of mel frames generated per input speech token.
        pre_lookahead_len (`int`, *optional*, defaults to 3):
            Number of future speech tokens the pre-lookahead convolution is allowed to see.
        `**kwargs`:
            Additional keyword arguments passed to [`CosyVoiceV1FlowConfig`].
    """

    model_type = "cosyvoice_v2_flow"

    def __init__(self, token_mel_ratio: int = 2, pre_lookahead_len: int = 3, **kwargs):
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        super().__init__(**kwargs)


class CosyVoiceV2Config(CosyVoiceV1Config):
    r"""
    This is the configuration class to store the configuration of a [`CosyVoiceV2ForConditionalGeneration`]. It is
    used to instantiate a CosyVoice v2 model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a configuration close to that of the
    `FunAudioLLM/CosyVoice2-0.5B` checkpoint.

    Args:
        llm_config (`CosyVoiceV2LLMConfig`, *optional*):
            Configuration for the Qwen2-backbone speech-token language-model sub-model.
        flow_config (`CosyVoiceV2FlowConfig`, *optional*):
            Configuration for the causal conditional-flow-matching decoder sub-model.
        hift_config (`CosyVoiceV1HiftConfig`, *optional*):
            Configuration for the NSF/ISTFT vocoder sub-model, unchanged from CosyVoice v1.
        sample_rate (`int`, *optional*, defaults to 24000):
            Output waveform sample rate, in Hz.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing weight matrices.
    """

    model_type = "cosyvoice_v2"
    sub_configs = {
        "llm_config": CosyVoiceV2LLMConfig,
        "flow_config": CosyVoiceV2FlowConfig,
        "hift_config": CosyVoiceV1HiftConfig,
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
        kwargs.pop("text_encoder_config", None)
        self.llm_config = CosyVoiceV2LLMConfig(**(llm_config or {}))
        self.flow_config = CosyVoiceV2FlowConfig(**(flow_config or {}))
        self.hift_config = CosyVoiceV1HiftConfig(**(hift_config or {}))
        self.sample_rate = sample_rate
        self.initializer_range = initializer_range
        # Bypass CosyVoiceV1Config.__init__, which requires a text_encoder_config; v2 has no separate text
        # encoder because the Qwen2 backbone embeds text tokens directly.
        super(CosyVoiceV1Config, self).__init__(**kwargs)


__all__ = ["CosyVoiceV2Config", "CosyVoiceV2LLMConfig", "CosyVoiceV2FlowConfig"]
