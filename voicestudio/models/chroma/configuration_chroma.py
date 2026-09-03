"""Configuration class for Chroma."""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mimi.configuration_mimi import MimiConfig
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


@strict
class ChromaBackboneConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`ChromaBackboneForCausalLM`]. It is used to
    instantiate the Llama decoder stack that consumes the interleaved text/audio embedding sequence and predicts
    the first Mimi codebook of every audio frame. Instantiating a configuration with the defaults will yield a
    configuration similar to the backbone of [FlashLabs/Chroma-4B](https://huggingface.co/FlashLabs/Chroma-4B).

    Args:
        audio_num_codebooks (`int`, *optional*, defaults to 8):
            Number of Mimi codebooks per audio frame. The backbone owns the embedding table for all of them and
            predicts codebook 0; [`ChromaDecoderForCausalLM`] predicts the remaining `audio_num_codebooks - 1`.
        vocab_size (`int`, *optional*, defaults to 2051):
            Size of a single codebook vocabulary, including the codebook padding and end-of-stream ids.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the decoder layers, and of the embeddings the reasoner feeds in.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads implementing grouped query attention.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length the backbone can attend over.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon of the RMS normalization layers.
        head_dim (`int`, *optional*, defaults to 64):
            Attention head dimension.
        bos_token_id (`int`, *optional*):
            Unused, the backbone consumes embeddings rather than token ids.
        eos_token_id (`int` or `list[int]`, *optional*):
            Unused, generation stops on `codebook_eos_token_id` of the parent [`ChromaConfig`].
    """

    model_type = "chroma_backbone"
    base_config_key = "backbone_config"
    default_theta = 500000.0

    audio_num_codebooks: int = 8
    vocab_size: int = 2051
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    head_dim: int | None = 64
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


@strict
class ChromaDecoderConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`ChromaDecoderForCausalLM`]. It is used to
    instantiate the small Llama decoder stack that runs once per audio frame and predicts residual codebooks
    `1 .. audio_num_codebooks - 1` from the backbone hidden state and codebook 0. Instantiating a configuration
    with the defaults will yield a configuration similar to the decoder of
    [FlashLabs/Chroma-4B](https://huggingface.co/FlashLabs/Chroma-4B).

    Args:
        audio_num_codebooks (`int`, *optional*, defaults to 8):
            Number of Mimi codebooks per audio frame. The decoder predicts `audio_num_codebooks - 1` of them.
        audio_embedding_dim (`int`, *optional*, defaults to 2048):
            Width of the codebook embedding table shared with [`ChromaBackboneForCausalLM`], which is also the
            width of the backbone hidden state spliced in at frame position 0. `projection` maps it to
            `hidden_size`.
        vocab_size (`int`, *optional*, defaults to 2051):
            Size of a single codebook vocabulary, including the codebook padding and end-of-stream ids.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the decoder layers.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads implementing grouped query attention.
        max_position_embeddings (`int`, *optional*, defaults to 33):
            Maximum intra-frame sequence length, one slot for the backbone hidden state plus one per codebook.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon of the RMS normalization layers.
        head_dim (`int`, *optional*, defaults to 128):
            Attention head dimension.
        bos_token_id (`int`, *optional*):
            Unused, the decoder is driven by codebook ids rather than text tokens.
        eos_token_id (`int` or `list[int]`, *optional*):
            Unused, the decoder always emits exactly `audio_num_codebooks - 1` tokens.
    """

    model_type = "chroma_decoder"
    base_config_key = "decoder_config"
    default_theta = 500000.0

    audio_num_codebooks: int = 8
    audio_embedding_dim: int = 2048
    vocab_size: int = 2051
    hidden_size: int = 1024
    intermediate_size: int = 8192
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int | None = 2
    max_position_embeddings: int = 33
    rms_norm_eps: float = 1e-5
    head_dim: int | None = 128
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


class ChromaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ChromaForConditionalGeneration`]. It is used
    to instantiate a Chroma model according to the specified arguments, defining a frozen Qwen2.5-Omni thinker
    reasoner, a Llama backbone, a Llama residual decoder and a Mimi codec. Instantiating a configuration with the
    defaults will yield a configuration similar to
    [FlashLabs/Chroma-4B](https://huggingface.co/FlashLabs/Chroma-4B).

    Args:
        thinker_config ([`Qwen2_5OmniThinkerConfig`], *optional*):
            Configuration of the reasoner. Its token embeddings and last hidden states are the text half of the
            interleaved sequence the backbone consumes.
        backbone_config ([`ChromaBackboneConfig`], *optional*):
            Configuration of the codebook 0 backbone.
        decoder_config ([`ChromaDecoderConfig`], *optional*):
            Configuration of the residual codebook decoder.
        codec_config ([`MimiConfig`], *optional*):
            Configuration of the [`MimiModel`] that encodes reference audio to codebook ids and decodes generated
            ids back to a waveform.
        codebook_pad_token_id (`int`, *optional*, defaults to 2050):
            Codebook id written into frames generated after a sequence has finished.
        codebook_eos_token_id (`int`, *optional*, defaults to 0):
            Codebook id that marks the end of the audio stream when it is present in every codebook of a frame.
        audio_num_codebooks (`int`, *optional*, defaults to 8):
            Number of Mimi codebooks per audio frame, propagated to the backbone, decoder and codec configs.
        text_start_token_id (`int`, *optional*, defaults to 151665):
            Reasoner token id whose embedding opens the prompt text span of the backbone prompt.
        text_end_token_id (`int`, *optional*, defaults to 151666):
            Reasoner token id whose embedding closes the prompt text span of the backbone prompt.
        im_end_token_id (`int`, *optional*, defaults to 151645):
            Reasoner token id that ends its turn. Once sampled, no further text is interleaved into the backbone.
        audio_frame_freq (`int`, *optional*, defaults to 1920):
            Number of waveform samples per Mimi frame, used to turn prompt audio lengths into frame counts.
        decoder_loss_weight (`float`, *optional*, defaults to 0.5):
            Weight `lambda` of the decoder term in the training loss
            `(1 - lambda) * backbone_loss + lambda * decoder_loss`. Upstream trains stage one at 0.5 and stage two
            at 1.0 with the backbone frozen.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the backbone and the decoder share one codebook token embedding table.

    Example:

    ```python
    >>> from voicestudio.models.chroma import ChromaConfig, ChromaForConditionalGeneration

    >>> configuration = ChromaConfig()
    >>> model = ChromaForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "chroma"
    keys_to_ignore_at_inference = ["past_key_values"]

    sub_configs = {
        "thinker_config": Qwen2_5OmniThinkerConfig,
        "codec_config": MimiConfig,
        "backbone_config": ChromaBackboneConfig,
        "decoder_config": ChromaDecoderConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        backbone_config=None,
        decoder_config=None,
        codec_config=None,
        codebook_pad_token_id: int = 2050,
        codebook_eos_token_id: int = 0,
        audio_num_codebooks: int = 8,
        text_start_token_id: int = 151665,
        text_end_token_id: int = 151666,
        im_end_token_id: int = 151645,
        audio_frame_freq: int = 1920,
        decoder_loss_weight: float = 0.5,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if isinstance(thinker_config, dict):
            thinker_config = Qwen2_5OmniThinkerConfig(**thinker_config)
        self.thinker_config = thinker_config if thinker_config is not None else Qwen2_5OmniThinkerConfig()

        if isinstance(backbone_config, dict):
            backbone_config = ChromaBackboneConfig(**backbone_config)
        self.backbone_config = (
            backbone_config
            if backbone_config is not None
            else ChromaBackboneConfig(audio_num_codebooks=audio_num_codebooks)
        )

        if isinstance(decoder_config, dict):
            decoder_config = ChromaDecoderConfig(**decoder_config)
        self.decoder_config = (
            decoder_config
            if decoder_config is not None
            else ChromaDecoderConfig(audio_num_codebooks=audio_num_codebooks)
        )

        if isinstance(codec_config, dict):
            codec_config = MimiConfig(**codec_config)
        self.codec_config = (
            codec_config
            if codec_config is not None
            else MimiConfig(num_quantizers=audio_num_codebooks, frame_rate=12.5)
        )

        self.audio_num_codebooks = audio_num_codebooks
        self.codebook_pad_token_id = codebook_pad_token_id
        self.codebook_eos_token_id = codebook_eos_token_id
        self.text_start_token_id = text_start_token_id
        self.text_end_token_id = text_end_token_id
        self.im_end_token_id = im_end_token_id
        self.audio_frame_freq = audio_frame_freq
        self.decoder_loss_weight = decoder_loss_weight
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder=None, encoder=None) -> PreTrainedConfig:
        """
        Returns the config of the autoregressive text-side stack, which for Chroma is the backbone.

        The names [`PreTrainedConfig.get_text_config`] scans for are `decoder`, `generator`, `text_config` and
        `text_encoder`; Chroma names its sub-configs `thinker_config`, `backbone_config`, `decoder_config` and
        `codec_config`, so without this override the scan finds nothing and returns the composite config itself.

        Args:
            decoder (`bool`, *optional*):
                Ignored, Chroma has a single text-side stack.
            encoder (`bool`, *optional*):
                Ignored, Chroma has a single text-side stack.

        Returns:
            [`ChromaBackboneConfig`]: The config of the decoder-only stack that `generate` steps over.
        """
        return self.backbone_config


__all__ = [
    "ChromaBackboneConfig",
    "ChromaConfig",
    "ChromaDecoderConfig",
]
