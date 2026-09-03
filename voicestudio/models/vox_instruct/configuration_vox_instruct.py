"""Configuration class for VoxInstruct."""

from typing import ClassVar

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.encodec.configuration_encodec import EncodecConfig
from transformers.models.hubert.configuration_hubert import HubertConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mt5.configuration_mt5 import MT5Config
from transformers.utils import logging


logger = logging.get_logger(__name__)


@strict
class VoxInstructARConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`VoxInstructARForCausalLM`]. It is used to
    instantiate the autoregressive stage of a VoxInstruct model according to the specified arguments, defining the
    model architecture. Instantiating a configuration with the defaults will yield a configuration matching the
    `configs/train_ar.yaml` of the [thuhcsi/VoxInstruct](https://github.com/thuhcsi/VoxInstruct) release.

    The token space is a single flat vocabulary laid out as `[pad] [language] [semantic] [acoustic] [bos] [eos] [eos]`,
    so `vocab_size` is `1 + num_language_ids + semantic_vocab_size + acoustic_vocab_size + 3`.

    Args:
        text_encoder_config (`Union[dict, `MT5Config`]`, *optional*):
            Configuration of the frozen instruction text encoder whose output is projected onto `hidden_size` and
            prepended to the decoder input embeddings. `google/mt5-base` in the released model.
        max_text_len (`int`, *optional*, defaults to 512):
            Number of text encoder positions prepended to every decoder sequence. Decoder logits are read back from
            this offset.
        num_language_ids (`int`, *optional*, defaults to 2):
            Number of language identity tokens reserved after the padding token.
        semantic_vocab_size (`int`, *optional*, defaults to 500):
            Number of HuBERT k-means semantic tokens.
        acoustic_vocab_size (`int`, *optional*, defaults to 1024):
            Size of a single EnCodec codebook.
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of EnCodec codebooks. The autoregressive stage models the first one only.
        num_segment_ids (`int`, *optional*, defaults to 3):
            Number of segment embeddings, one each for the text prefix, the semantic span and the acoustic span.
        text_free_guidance_ratio (`float`, *optional*, defaults to 0.1):
            Per-sample probability of zeroing the whole text encoding during training, which trains the unconditional
            branch used by classifier-free guidance.
        semantic_free_guidance_ratio (`float`, *optional*, defaults to 0.1):
            Per-sample probability of replacing the semantic span of `input_ids` with the padding token during
            training.
        use_lora (`bool`, *optional*, defaults to `True`):
            Whether the text encoder query and value projections carry LoRA adapters. They are the only trainable
            parameters of the text encoder.
        lora_rank (`int`, *optional*, defaults to 16):
            Rank of the text encoder LoRA adapters.
        lora_alpha (`int`, *optional*, defaults to 16):
            Scaling numerator of the text encoder LoRA adapters. The applied scale is `lora_alpha / lora_rank`.
        lora_dropout (`float`, *optional*, defaults to 0.1):
            Dropout applied to the LoRA branch input.
    """

    model_type = "vox_instruct_ar"
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {"text_encoder_config": MT5Config}

    vocab_size: int = 1530
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 2560
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1527
    eos_token_id: int | list[int] | None = 1528
    attention_dropout: int | float | None = 0.1

    text_encoder_config: dict | PreTrainedConfig | None = None
    max_text_len: int = 512
    num_language_ids: int = 2
    semantic_vocab_size: int = 500
    acoustic_vocab_size: int = 1024
    num_codebooks: int = 8
    num_segment_ids: int = 3
    text_free_guidance_ratio: float = 0.1
    semantic_free_guidance_ratio: float = 0.1
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    def __post_init__(self, **kwargs):
        if isinstance(self.text_encoder_config, dict):
            self.text_encoder_config = MT5Config(**self.text_encoder_config)
        elif self.text_encoder_config is None:
            self.text_encoder_config = MT5Config(
                d_model=768,
                d_ff=2048,
                num_layers=12,
                num_decoder_layers=12,
                num_heads=12,
                tie_word_embeddings=False,
            )

        super().__post_init__(**kwargs)

    @property
    def language_token_offset(self) -> int:
        """Token id of the first language identity token."""
        return 1

    @property
    def semantic_token_offset(self) -> int:
        """Token id of the first semantic token."""
        return 1 + self.num_language_ids

    @property
    def acoustic_token_offset(self) -> int:
        """Token id of the first acoustic token."""
        return 1 + self.num_language_ids + self.semantic_vocab_size

    @property
    def semantic_eos_token_id(self) -> int:
        """Token id closing the semantic span and opening the acoustic span."""
        return self.eos_token_id

    @property
    def acoustic_eos_token_id(self) -> int:
        """Token id closing the acoustic span."""
        return self.eos_token_id + 1


@strict
class VoxInstructNARConfig(VoxInstructARConfig):
    r"""
    This is the configuration class to store the configuration of a [`VoxInstructNARModel`]. It is used to instantiate
    the non-autoregressive stage of a VoxInstruct model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a configuration matching the
    `configs/train_nar.yaml` of the [thuhcsi/VoxInstruct](https://github.com/thuhcsi/VoxInstruct) release.

    The stage reads every codebook at once and attends bidirectionally, so it keeps the flat token space of
    [`VoxInstructARConfig`] on its inputs but emits `acoustic_vocab_size + 3` classes per residual codebook.

    Args:
        text_free_guidance_ratio (`float`, *optional*, defaults to 0.25):
            Per-sample probability of zeroing the whole text encoding during training.
        acoustic_free_guidance_ratio (`float`, *optional*, defaults to 0.3):
            Per-sample probability of drawing an empty acoustic prompt during training, so the stage also learns to
            predict residual codebooks without one.
        mask_strategy (`str`, *optional*, defaults to `"cosine"`):
            Schedule drawing the fraction of predictable positions that stay unmasked. `"cosine"` draws
            `cos(pi / 2 * u)` with `u` uniform, `"full"` keeps none of them.
    """

    model_type = "vox_instruct_nar"

    text_free_guidance_ratio: float = 0.25
    semantic_free_guidance_ratio: float = 0.0
    acoustic_free_guidance_ratio: float = 0.3
    mask_strategy: str = "cosine"

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        super().validate_architecture()
        if self.mask_strategy not in ("cosine", "full"):
            raise ValueError(f"`mask_strategy` must be one of 'cosine' or 'full', got {self.mask_strategy}.")


@strict
class VoxInstructConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VoxInstructForConditionalGeneration`]. It is
    used to instantiate a VoxInstruct model according to the specified arguments, defining the text encoder, the
    semantic tokenizer, the audio codec and the two codec language modelling stages.

    Args:
        ar_config (`Union[dict, `VoxInstructARConfig`]`, *optional*):
            Configuration of the autoregressive stage.
        nar_config (`Union[dict, `VoxInstructNARConfig`]`, *optional*):
            Configuration of the non-autoregressive stage.
        audio_encoder_config (`Union[dict, `EncodecConfig`]`, *optional*):
            Configuration of the acoustic tokenizer and vocoder, `facebook/encodec_24khz` in the released model.
        semantic_encoder_config (`Union[dict, `HubertConfig`]`, *optional*):
            Configuration of the semantic tokenizer backbone, `facebook/hubert-base-ls960` in the released model.
        semantic_num_clusters (`int`, *optional*, defaults to 500):
            Number of k-means centroids quantizing the semantic tokenizer features.
        semantic_feature_layer (`int`, *optional*, defaults to 9):
            One-based index of the semantic tokenizer layer whose output is quantized.
        semantic_frame_multiple (`int`, *optional*, defaults to 320):
            Waveform length is truncated to a multiple of this before semantic tokenization, matching the stride of
            the semantic tokenizer feature encoder.
        audio_bandwidth (`float`, *optional*, defaults to 6.0):
            Bandwidth in kbps selecting how many EnCodec codebooks are used. 6.0 selects 8 codebooks.

    Example:

    ```python
    >>> from voicestudio.models.vox_instruct import VoxInstructConfig, VoxInstructForConditionalGeneration

    >>> configuration = VoxInstructConfig()
    >>> model = VoxInstructForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "vox_instruct"
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {
        "ar_config": VoxInstructARConfig,
        "nar_config": VoxInstructNARConfig,
        "audio_encoder_config": EncodecConfig,
        "semantic_encoder_config": HubertConfig,
    }

    ar_config: dict | PreTrainedConfig | None = None
    nar_config: dict | PreTrainedConfig | None = None
    audio_encoder_config: dict | PreTrainedConfig | None = None
    semantic_encoder_config: dict | PreTrainedConfig | None = None
    semantic_num_clusters: int = 500
    semantic_feature_layer: int = 9
    semantic_frame_multiple: int = 320
    audio_bandwidth: float = 6.0

    def __post_init__(self, **kwargs):
        if isinstance(self.ar_config, dict):
            self.ar_config = VoxInstructARConfig(**self.ar_config)
        elif self.ar_config is None:
            self.ar_config = VoxInstructARConfig()

        if isinstance(self.nar_config, dict):
            self.nar_config = VoxInstructNARConfig(**self.nar_config)
        elif self.nar_config is None:
            self.nar_config = VoxInstructNARConfig()

        if isinstance(self.audio_encoder_config, dict):
            self.audio_encoder_config = EncodecConfig(**self.audio_encoder_config)
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = EncodecConfig()

        if isinstance(self.semantic_encoder_config, dict):
            self.semantic_encoder_config = HubertConfig(**self.semantic_encoder_config)
        elif self.semantic_encoder_config is None:
            self.semantic_encoder_config = HubertConfig()

        super().__post_init__(**kwargs)

    @property
    def num_codebooks(self) -> int:
        """Number of EnCodec codebooks both stages jointly model."""
        return self.ar_config.num_codebooks

    @property
    def sampling_rate(self) -> int:
        """Sampling rate of the waveform the audio codec consumes and emits."""
        return self.audio_encoder_config.sampling_rate

    @property
    def semantic_sampling_rate(self) -> int:
        """Sampling rate the semantic tokenizer consumes."""
        return 16000


__all__ = ["VoxInstructARConfig", "VoxInstructNARConfig", "VoxInstructConfig"]
