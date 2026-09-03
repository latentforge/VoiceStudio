"""Configuration class for CosyVoice v3."""

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from ..cosyvoice_v2.configuration_cosyvoice_v2 import CosyVoiceV2Config
from ..f5_tts.configuration_f5_tts import F5TTSConfig


class CosyVoiceV3Config(CosyVoiceV2Config):
    r"""
    This is the configuration class to store the configuration of a
    [`CosyVoiceV3ForConditionalGeneration`]. CosyVoice v3 keeps v2's three network layout and
    replaces two of the three networks. The flow matching model drops the conformer encoder entirely,
    so a lookahead convolution and a repeat interleave carry the speech tokens straight to the mel
    frame rate and a diffusion transformer predicts the vector field. The vocoder becomes causal.

    The language model keeps v2's Qwen2 decoder but moves its start of sequence, end of speech and
    task id vectors out of a separate two entry table and into the speech token table, which grows by
    `num_speech_special_tokens` instead of by three.

    Instantiating a configuration with the defaults yields the geometry of the released
    `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` checkpoint.

    The flow encoder fields inherited from [`CosyVoiceV2Config`] are unused: v3 has no flow encoder.

    Args:
        estimator_config ([`F5TTSConfig`] or `dict`, *optional*):
            Configuration of the diffusion transformer that predicts the flow matching vector field.
            It is an [`F5TTSConfig`] because the estimator is that model's diffusion transformer at
            the same size, and `pe_attn_head` is 1 because upstream applies the rotary embedding to
            the unreshaped projection, which reaches the first attention head only.
        silent_token_ids (`list[int]`, *optional*):
            Speech token ids upstream treats as silence or breath and thins out during generation.
        max_silent_run (`int`, *optional*, defaults to 5):
            Longest run of `silent_token_ids` kept before the rest of the run is dropped.
        end_of_prompt_token_id (`int`, *optional*, defaults to 151646):
            Text token upstream requires the prompt to contain.
        vocoder_conv_pre_look_right (`int`, *optional*, defaults to 4):
            Number of future mel frames the vocoder's first convolution sees.
        source_noise_length (`int`, *optional*, defaults to 7200000):
            Length in samples of the fixed phase offsets and noise the causal sine generator draws
            once and reuses, so that consecutive chunks of a stream stay consistent.
        source_noise_seed (`int`, *optional*, defaults to 0):
            Seed those fixed tensors are drawn with. Upstream draws them from whatever global random
            state happens to be current, which makes its vocoder irreproducible across processes.
        kwargs:
            Forwarded to [`CosyVoiceV2Config`], whose defaults are overridden to the v3 geometry.
    """

    model_type = "cosyvoice_v3"
    sub_configs = {"text_config": Qwen2Config, "estimator_config": F5TTSConfig}

    def __init__(
        self,
        estimator_config: "F5TTSConfig | dict | None" = None,
        silent_token_ids: list[int] | None = None,
        max_silent_run: int = 5,
        end_of_prompt_token_id: int = 151646,
        vocoder_conv_pre_look_right: int = 4,
        source_noise_length: int = 300 * 24000,
        source_noise_seed: int = 0,
        **kwargs,
    ):
        defaults = {
            "num_speech_special_tokens": 200,
            "flow_input_size": 80,
            "pre_lookahead_channels": 1024,
            "estimator_in_channels": 320,
            "vocoder_source_resblock_kernel_sizes": [7, 7, 11],
        }
        for name, value in defaults.items():
            kwargs.setdefault(name, value)

        if estimator_config is None:
            estimator_config = F5TTSConfig(
                backbone="dit",
                hidden_size=1024,
                num_hidden_layers=22,
                num_attention_heads=16,
                head_dim=64,
                ff_mult=2,
                dropout=0.1,
                attention_dropout=0.0,
                mel_dim=80,
                text_dim=80,
                qk_norm=None,
                pe_attn_head=1,
                long_skip_connection=False,
                layer_norm_eps=1e-6,
                max_position_embeddings=8192,
            )
        elif isinstance(estimator_config, dict):
            estimator_config = F5TTSConfig(**estimator_config)
        self.estimator_config = estimator_config

        self.silent_token_ids = (
            [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
            if silent_token_ids is None
            else silent_token_ids
        )
        self.max_silent_run = max_silent_run
        self.end_of_prompt_token_id = end_of_prompt_token_id
        self.vocoder_conv_pre_look_right = vocoder_conv_pre_look_right
        self.source_noise_length = source_noise_length
        self.source_noise_seed = source_noise_seed
        super().__init__(**kwargs)

        for field in ("dtype", "torch_dtype"):
            if not hasattr(self.estimator_config, field) or not hasattr(self, field):
                continue
            try:
                setattr(self.estimator_config, field, getattr(self, field))
            except AttributeError:
                pass

        # The estimator is a plain module rather than a `PreTrainedModel`, so nothing runs the
        # attention implementation autoselection over this sub configuration and it would otherwise
        # stay `None` and dispatch to the eager path, whose softmax is computed in float32 whatever
        # the model dtype is.
        if getattr(self.estimator_config, "_attn_implementation", None) is None:
            self.estimator_config._attn_implementation = "sdpa"

    # Named apart from `PretrainedConfig`'s own `eos_token_id`, which is a settable attribute the
    # base constructor assigns and which refers to the text vocabulary rather than the speech one.
    @property
    def speech_sos_token_id(self) -> int:
        r"""
        Returns:
            `int`: Id of the start of sequence token inside the speech token table.
        """
        return self.speech_vocab_size

    @property
    def speech_eos_token_id(self) -> int:
        r"""
        Returns:
            `int`: Id of the end of speech token inside the speech token table.
        """
        return self.speech_vocab_size + 1

    @property
    def speech_task_token_id(self) -> int:
        r"""
        Returns:
            `int`: Id of the task token inside the speech token table.
        """
        return self.speech_vocab_size + 2

    @property
    def speech_fill_token_id(self) -> int:
        r"""
        Returns:
            `int`: Id of the fill token that separates the groups of a bistream sequence.
        """
        return self.speech_vocab_size + 3


__all__ = ["CosyVoiceV3Config"]
