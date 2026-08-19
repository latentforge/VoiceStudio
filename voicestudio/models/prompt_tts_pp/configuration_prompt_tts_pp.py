"""Configuration class for PromptTTS++."""

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.fastspeech2_conformer.configuration_fastspeech2_conformer import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGanConfig,
)

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class PromptTTSppPromptEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PromptTTSppPromptEncoder`]. It is used to
    instantiate a PromptTTS++ prompt encoder according to the specified arguments, defining the module that turns a
    natural-language style/speaker description into a style embedding.

    Args:
        text_config ([`BertConfig`, *optional*]):
            Configuration of the BERT model used to embed the style prompt text.
        mid_channels (`int`, *optional*, defaults to 256):
            Hidden size of the adaptor MLP that maps the pooled BERT embedding to the style embedding space.
        out_channels (`int`, *optional*, defaults to 384):
            Dimensionality of the produced style embedding. Must match the acoustic model's `hidden_size`, since the
            style embedding is added directly onto the phoneme encoder's output.
    """

    model_type = "prompt_tts_pp_prompt_encoder"
    sub_configs = {"text_config": BertConfig}

    def __init__(
        self,
        text_config: BertConfig | dict | None = None,
        mid_channels: int = 256,
        out_channels: int = 384,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = BertConfig(**text_config)
        elif text_config is None:
            text_config = BertConfig()
        self.text_config = text_config
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        super().__init__(**kwargs)


class PromptTTSppConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PromptTTSppForConditionalGeneration`]. It is
    used to instantiate a PromptTTS++ model according to the specified arguments, defining the acoustic model,
    vocoder, and prompt encoder sub-configurations.

    Args:
        model_config ([`FastSpeech2ConformerConfig`, *optional*]):
            Configuration of the FastSpeech2Conformer-based acoustic model that predicts the mel-spectrogram from
            phoneme inputs conditioned on a style embedding.
        vocoder_config ([`FastSpeech2ConformerHifiGanConfig`, *optional*]):
            Configuration of the HiFi-GAN vocoder that converts the predicted mel-spectrogram into a waveform.
        prompt_encoder_config ([`PromptTTSppPromptEncoderConfig`, *optional*]):
            Configuration of the prompt encoder that turns a natural-language style/speaker description into the
            style embedding consumed by `model_config`.

    Example:

    ```python
    >>> from voicestudio.models.prompt_tts_pp import PromptTTSppConfig, PromptTTSppForConditionalGeneration

    >>> configuration = PromptTTSppConfig()
    >>> model = PromptTTSppForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "prompt_tts_pp"
    sub_configs = {
        "model_config": FastSpeech2ConformerConfig,
        "vocoder_config": FastSpeech2ConformerHifiGanConfig,
        "prompt_encoder_config": PromptTTSppPromptEncoderConfig,
    }

    def __init__(
        self,
        model_config: FastSpeech2ConformerConfig | dict | None = None,
        vocoder_config: FastSpeech2ConformerHifiGanConfig | dict | None = None,
        prompt_encoder_config: PromptTTSppPromptEncoderConfig | dict | None = None,
        **kwargs,
    ):
        if model_config is None:
            model_config = FastSpeech2ConformerConfig()
            logger.info("model_config is None. initializing the acoustic model with default values.")
        elif isinstance(model_config, dict):
            model_config = FastSpeech2ConformerConfig(**model_config)

        if vocoder_config is None:
            vocoder_config = FastSpeech2ConformerHifiGanConfig()
            logger.info("vocoder_config is None. initializing the vocoder with default values.")
        elif isinstance(vocoder_config, dict):
            vocoder_config = FastSpeech2ConformerHifiGanConfig(**vocoder_config)

        if prompt_encoder_config is None:
            prompt_encoder_config = PromptTTSppPromptEncoderConfig(out_channels=model_config.hidden_size)
            logger.info("prompt_encoder_config is None. initializing the prompt encoder with default values.")
        elif isinstance(prompt_encoder_config, dict):
            prompt_encoder_config = PromptTTSppPromptEncoderConfig(**prompt_encoder_config)

        # The style embedding is added directly onto the acoustic model's encoder output (see
        # `PromptTTSppModel._acoustic_forward_with_style`), not passed through `FastSpeech2ConformerModel`'s own
        # `speaker_embedding` argument, so `speaker_embed_dim` (which activates that separate concatenation +
        # projection path) is left unset.
        if prompt_encoder_config.out_channels != model_config.hidden_size:
            raise ValueError(
                f"prompt_encoder_config.out_channels ({prompt_encoder_config.out_channels}) must equal "
                f"model_config.hidden_size ({model_config.hidden_size}); the style embedding is added directly "
                "onto the acoustic model's encoder output."
            )

        self.model_config = model_config
        self.vocoder_config = vocoder_config
        self.prompt_encoder_config = prompt_encoder_config
        super().__init__(**kwargs)


__all__ = ["PromptTTSppConfig", "PromptTTSppPromptEncoderConfig"]
