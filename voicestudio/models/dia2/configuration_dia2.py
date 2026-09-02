"""Configuration class for Dia2."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


class Dia2DepthDecoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Dia2DepthDecoderModel`]. It is used to
    instantiate the depth decoder that predicts codebooks `1 .. num_codebooks - 1` of an audio frame whose first
    codebook was already predicted by the [`Dia2BackboneModel`].

    The depth decoder runs one position per codebook, and its attention projections are not shared across those
    positions: `weights_schedule[i]` selects which of `num_weight_groups` projection sets position `i` uses.

    Args:
        num_codebooks (`int`, *optional*, defaults to 32):
            Number of audio codebooks per frame. The depth decoder covers `num_codebooks - 1` of them.
        vocab_size (`int`, *optional*, defaults to 2050):
            Size of a single codebook's vocabulary, including the codebook beginning-of-stream and padding ids.
        backbone_hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the [`Dia2BackboneModel`] hidden state projected into every depth decoder position.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the depth decoder's hidden representations.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of query heads per attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads per attention layer.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of a single attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation applied to the gate branch of the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 32):
            Maximum number of depth decoder positions, i.e. codebooks per frame.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the RMS normalization layers.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            Rotary position embedding parameters. Only used when `use_rope` is `True`.
        use_rope (`bool`, *optional*, defaults to `True`):
            Whether to rotate queries and keys by their codebook position.
        use_text_stream_embedding (`bool`, *optional*, defaults to `False`):
            Whether position 0 additionally embeds the frame's two text streams, as
            [`Dia2BackboneModel`] does.
        text_vocab_size (`int`, *optional*, defaults to 49280):
            Size of the text vocabulary. Only used when `use_text_stream_embedding` is `True`.
        text_pad_token_id (`int`, *optional*, defaults to 3):
            Id marking an empty slot of the second text stream. Only used when
            `use_text_stream_embedding` is `True`.
        weights_schedule (`list[int]`, *optional*):
            Attention projection group used by each of the `num_codebooks - 1` depth decoder positions.
            Defaults to one group per position.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether the attention projections carry a bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio of the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether the MLP projections carry a bias.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model returns the key/value states of the already decoded codebooks.
    """

    model_type = "dia2_depth_decoder"
    base_config_key = "depth_decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_codebooks: int = 32,
        vocab_size: int = 2050,
        backbone_hidden_size: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32,
        rms_norm_eps: float = 1e-6,
        rope_parameters: RopeParameters | dict | None = None,
        use_rope: bool = True,
        use_text_stream_embedding: bool = False,
        text_vocab_size: int = 49280,
        text_pad_token_id: int = 3,
        weights_schedule: list[int] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.backbone_hidden_size = backbone_hidden_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_rope = use_rope
        self.use_text_stream_embedding = use_text_stream_embedding
        self.text_vocab_size = text_vocab_size
        self.text_pad_token_id = text_pad_token_id
        self.weights_schedule = (
            list(weights_schedule) if weights_schedule is not None else list(range(num_codebooks - 1))
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        if len(self.weights_schedule) != num_codebooks - 1:
            raise ValueError(
                f"`weights_schedule` must hold one entry per depth decoder position, got "
                f"{len(self.weights_schedule)} entries for {num_codebooks - 1} positions."
            )

        self.rope_parameters = rope_parameters or {}
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", 10000.0))
        super().__init__(**kwargs)

    @property
    def num_weight_groups(self) -> int:
        return max(self.weights_schedule) + 1


class Dia2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Dia2ForConditionalGeneration`]. It is used
    to instantiate a Dia2 model, a decoder-only transformer backbone that predicts a binary word-advance action
    and the first Mimi codebook of every frame, paired with a [`Dia2DepthDecoderModel`] that predicts the
    remaining codebooks of the same frame.

    Every frame of `input_ids` carries `2 + num_codebooks` channels: a main text stream, a second text stream
    running `second_stream_ahead` words ahead of it, and one channel per audio codebook.

    Args:
        num_codebooks (`int`, *optional*, defaults to 32):
            Number of audio codebooks per frame produced by the audio tokenizer.
        vocab_size (`int`, *optional*, defaults to 2050):
            Size of a single codebook's vocabulary, including `audio_bos_token_id` and `audio_pad_token_id`.
        text_vocab_size (`int`, *optional*, defaults to 49280):
            Size of the vocabulary of both text streams.
        num_actions (`int`, *optional*, defaults to 2):
            Size of the action head's output, one entry per value of `action_pad_token_id` /
            `action_new_word_token_id`.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the backbone's hidden representations.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of backbone layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of query heads per attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads per attention layer.
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of a single attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation applied to the gate branch of the MLP.
        max_position_embeddings (`int`, *optional*, defaults to 1500):
            Maximum number of frames the model can attend over.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon of the RMS normalization layers.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            Rotary position embedding parameters.
        text_embedding_rank (`int`, *optional*):
            Width of the shared text embedding table before it is projected to `hidden_size`. Defaults to
            `hidden_size`, i.e. no low-rank factorization.
        text_bos_token_id (`int`, *optional*, defaults to 1):
            Id fed to the main text stream on the first frame.
        text_pad_token_id (`int`, *optional*, defaults to 3):
            Id fed to a text stream on frames that carry no word token.
        text_new_word_token_id (`int`, *optional*, defaults to 2):
            Id fed to a text stream on the frame that starts a new word.
        text_zero_token_id (`int`, *optional*, defaults to 7):
            Id fed to the main text stream of the unconditional branch used by classifier-free guidance.
        audio_bos_token_id (`int`, *optional*, defaults to 2048):
            Codebook id fed to a codebook channel whose delay has not elapsed yet.
        audio_pad_token_id (`int`, *optional*, defaults to 2049):
            Codebook id used to pad a codebook channel outside the generated span.
        action_pad_token_id (`int`, *optional*, defaults to 0):
            Action head class meaning "stay on the current word".
        action_new_word_token_id (`int`, *optional*, defaults to 1):
            Action head class meaning "advance to the next word".
        delay_pattern (`list[int]`, *optional*):
            Per-codebook delay, in frames, applied to the codebook grid. Defaults to no delay.
        first_word_min_start (`int`, *optional*, defaults to 3):
            Number of leading frames during which the action head may not advance to the first word.
        max_pad (`int`, *optional*, defaults to 8):
            Number of frames a word may be held before the action head is forced to advance.
        second_stream_ahead (`int`, *optional*, defaults to 2):
            Number of words the second text stream runs ahead of the main one. `0` disables it.
        depth_decoder_config ([`Dia2DepthDecoderConfig`], *optional*):
            Configuration of the depth decoder.
        audio_tokenizer_id (`str`, *optional*, defaults to `"kyutai/mimi"`):
            Repository id of the [`MimiModel`] whose codebooks this model predicts.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether the attention projections carry a bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio of the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether the MLP projections carry a bias.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model returns the key/value states of the already decoded frames.

    Example:

    ```python
    >>> from voicestudio.models.dia2 import Dia2Config, Dia2ForConditionalGeneration

    >>> configuration = Dia2Config()

    >>> model = Dia2ForConditionalGeneration(configuration)

    >>> configuration = model.config
    ```"""

    model_type = "dia2"
    sub_configs = {"depth_decoder_config": Dia2DepthDecoderConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_codebooks: int = 32,
        vocab_size: int = 2050,
        text_vocab_size: int = 49280,
        num_actions: int = 2,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1500,
        rms_norm_eps: float = 1e-6,
        rope_parameters: RopeParameters | dict | None = None,
        text_embedding_rank: int | None = None,
        text_bos_token_id: int = 1,
        text_pad_token_id: int = 3,
        text_new_word_token_id: int = 2,
        text_zero_token_id: int = 7,
        audio_bos_token_id: int = 2048,
        audio_pad_token_id: int = 2049,
        action_pad_token_id: int = 0,
        action_new_word_token_id: int = 1,
        delay_pattern: list[int] | None = None,
        first_word_min_start: int = 3,
        max_pad: int = 8,
        second_stream_ahead: int = 2,
        depth_decoder_config: Dia2DepthDecoderConfig | dict | None = None,
        audio_tokenizer_id: str = "kyutai/mimi",
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.text_embedding_rank = text_embedding_rank
        self.text_bos_token_id = text_bos_token_id
        self.text_pad_token_id = text_pad_token_id
        self.text_new_word_token_id = text_new_word_token_id
        self.text_zero_token_id = text_zero_token_id
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.action_pad_token_id = action_pad_token_id
        self.action_new_word_token_id = action_new_word_token_id
        self.delay_pattern = list(delay_pattern) if delay_pattern is not None else [0] * num_codebooks
        self.first_word_min_start = first_word_min_start
        self.max_pad = max_pad
        self.second_stream_ahead = second_stream_ahead
        self.audio_tokenizer_id = audio_tokenizer_id
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        if len(self.delay_pattern) != num_codebooks:
            raise ValueError(
                f"`delay_pattern` must hold one entry per codebook, got {len(self.delay_pattern)} entries "
                f"for {num_codebooks} codebooks."
            )

        if isinstance(depth_decoder_config, dict):
            depth_decoder_config = Dia2DepthDecoderConfig(**depth_decoder_config)
        elif depth_decoder_config is None:
            depth_decoder_config = Dia2DepthDecoderConfig(
                num_codebooks=num_codebooks,
                vocab_size=vocab_size,
                backbone_hidden_size=hidden_size,
                text_vocab_size=text_vocab_size,
                text_pad_token_id=text_pad_token_id,
                rms_norm_eps=rms_norm_eps,
            )
        self.depth_decoder_config = depth_decoder_config

        self.rope_parameters = rope_parameters or {}
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", 10000.0))
        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def num_channels(self) -> int:
        return self.num_codebooks + 2

    @property
    def codebook_size(self) -> int:
        return self.vocab_size


__all__ = ["Dia2Config", "Dia2DepthDecoderConfig"]
