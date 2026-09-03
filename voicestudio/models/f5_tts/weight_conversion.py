"""Checkpoint conversion for F5-TTS."""

import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

from ..bigvgan.weight_conversion import build_model_files as build_bigvgan_files
from ..vocos.weight_conversion import build_model_files as build_vocos_files
from .configuration_f5_tts import F5TTSConfig
from .feature_extraction_f5_tts import F5TTSFeatureExtractor
from .processing_f5_tts import F5TTSProcessor
from .tokenization_f5_tts import F5TTSTokenizer


# The `model.arch` and `model.mel_spec` blocks of the training configs `SWivid/F5-TTS` ships, one entry per
# released architecture. `text_max_positions` is not a config field upstream: the `"dit"` text embedding hard
# codes 8192 sinusoidal positions and the `"unett"` one 4096.
ARCHITECTURES = {
    "F5TTS_v1_Base": {
        "backbone": "dit",
        "hidden_size": 1024,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": True,
        "qk_norm": None,
        "text_conv_layers": 4,
        "pe_attn_head": None,
        "attn_mask_enabled": False,
        "text_max_positions": 8192,
    },
    "F5TTS_v1_Small": {
        "backbone": "dit",
        "hidden_size": 768,
        "num_hidden_layers": 18,
        "num_attention_heads": 12,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": True,
        "qk_norm": None,
        "text_conv_layers": 4,
        "pe_attn_head": None,
        "attn_mask_enabled": False,
        "text_max_positions": 8192,
    },
    "F5TTS_Base": {
        "backbone": "dit",
        "hidden_size": 1024,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": False,
        "text_conv_layers": 4,
        "pe_attn_head": 1,
        "attn_mask_enabled": False,
        "text_max_positions": 8192,
    },
    "F5TTS_Small": {
        "backbone": "dit",
        "hidden_size": 768,
        "num_hidden_layers": 18,
        "num_attention_heads": 12,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": False,
        "text_conv_layers": 4,
        "pe_attn_head": 1,
        "attn_mask_enabled": False,
        "text_max_positions": 8192,
    },
    "E2TTS_Base": {
        "backbone": "unett",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "ff_mult": 4,
        "text_dim": None,
        "text_mask_padding": False,
        "text_conv_layers": 0,
        "pe_attn_head": 1,
        "attn_mask_enabled": False,
        "text_max_positions": 4096,
    },
    "E2TTS_Small": {
        "backbone": "unett",
        "hidden_size": 768,
        "num_hidden_layers": 20,
        "num_attention_heads": 12,
        "ff_mult": 4,
        "text_dim": None,
        "text_mask_padding": False,
        "text_conv_layers": 0,
        "pe_attn_head": 1,
        "attn_mask_enabled": False,
        "text_max_positions": 4096,
    },
}

# Every checkpoint the F5-TTS authors published, keyed by the name of the directory holding it. No released file
# records `mel_spec_type`: `F5TTS_Base_bigvgan` holds the backbone alone, and the only two checkpoints holding the
# training time `mel_spec` buffers are `"vocos"` ones, so the directory name is what names the front end and with
# it the vocoder. Only `SWivid/F5-TTS` ships a vocabulary file, and the `E2TTS_Base` text embedding is a table
# over those same 2545 characters.
PUBLISHED_CHECKPOINTS = {
    "F5TTS_v1_Base": {
        "repo_id": "SWivid/F5-TTS",
        "weights_file": "F5TTS_v1_Base/model_1250000.safetensors",
        "vocab_repo_id": "SWivid/F5-TTS",
        "vocab_file": "F5TTS_v1_Base/vocab.txt",
        "mel_spec_type": "vocos",
    },
    "F5TTS_v1_Base_no_zero_init": {
        "repo_id": "SWivid/F5-TTS",
        "weights_file": "F5TTS_v1_Base_no_zero_init/model_1250000.safetensors",
        "vocab_repo_id": "SWivid/F5-TTS",
        "vocab_file": "F5TTS_v1_Base/vocab.txt",
        "mel_spec_type": "vocos",
    },
    "F5TTS_Base": {
        "repo_id": "SWivid/F5-TTS",
        "weights_file": "F5TTS_Base/model_1200000.safetensors",
        "vocab_repo_id": "SWivid/F5-TTS",
        "vocab_file": "F5TTS_Base/vocab.txt",
        "mel_spec_type": "vocos",
    },
    "F5TTS_Base_bigvgan": {
        "repo_id": "SWivid/F5-TTS",
        "weights_file": "F5TTS_Base_bigvgan/model_1250000.pt",
        "vocab_repo_id": "SWivid/F5-TTS",
        "vocab_file": "F5TTS_Base/vocab.txt",
        "mel_spec_type": "bigvgan",
    },
    "E2TTS_Base": {
        "repo_id": "SWivid/E2-TTS",
        "weights_file": "E2TTS_Base/model_1200000.safetensors",
        "vocab_repo_id": "SWivid/F5-TTS",
        "vocab_file": "F5TTS_Base/vocab.txt",
        "mel_spec_type": "vocos",
    },
}

# The checkpoint each published repository holds by default, for a caller naming the repository and no `subfolder`.
DEFAULT_CHECKPOINTS = {"SWivid/F5-TTS": "F5TTS_v1_Base", "SWivid/E2-TTS": "E2TTS_Base"}

# The vocoder each mel front end was trained against, and the function that reads its published repository.
VOCODER_SOURCES = {
    "vocos": ("charactr/vocos-mel-24khz", build_vocos_files),
    "bigvgan": ("nvidia/bigvgan_v2_24khz_100band_256x", build_bigvgan_files),
}

# Buffers the EMA wrapper and the mel front end of a `CFM` module keep in the checkpoint and that no parameter
# of this model corresponds to.
_DISCARDED_KEYS = (
    "initted",
    "step",
    "update",
    "mel_spec.mel_stft.mel_scale.fb",
    "mel_spec.mel_stft.spectrogram.window",
)

_UNET_LAYER = re.compile(r"^layers\.(\d+)\.(\d)\.")


def is_published_layout(source: str, subfolder: str | None = None) -> bool:
    r"""
    Returns whether `source` is a published F5-TTS or E2-TTS repository rather than a directory [`convert`] wrote.

    The published repositories hold one bare exponential moving average state dict per directory and no
    `config.json` at all, so the discriminator is a `config.json` declaring this model's `model_type`.
    `PreTrainedConfig.from_pretrained` draws no such distinction of its own: `cached_file` returns `None` for the
    missing file and the configuration silently falls back to its defaults.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`] or of [`DEFAULT_CHECKPOINTS`], repository id, or local directory.
        subfolder (`str`, *optional*):
            Directory inside `source` the configuration would sit in.

    Returns:
        `bool`: Whether `source` holds the published layout.
    """
    if source in PUBLISHED_CHECKPOINTS or source in DEFAULT_CHECKPOINTS:
        return True
    config_file = cached_file(
        source,
        CONFIG_NAME,
        subfolder=subfolder or "",
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if config_file is None:
        return True
    return json.loads(Path(config_file).read_text()).get("model_type") != F5TTSConfig.model_type


def resolve_checkpoint(source: str, subfolder: str | None = None) -> str:
    r"""
    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], key of [`DEFAULT_CHECKPOINTS`], or a repository id or local
            directory holding the published tree.
        subfolder (`str`, *optional*):
            Name of the directory holding the checkpoint to read, which is the name it is published under.
            Defaults to the entry of [`DEFAULT_CHECKPOINTS`] the repository names.

    Returns:
        `str`: Key of [`PUBLISHED_CHECKPOINTS`] the two name together.

    Raises:
        ValueError: If they name no published checkpoint.
    """
    name = subfolder or DEFAULT_CHECKPOINTS.get(source, source)
    if name in PUBLISHED_CHECKPOINTS:
        return name
    raise ValueError(
        f"{source} and subfolder {subfolder} name no published checkpoint. Pass `subfolder` as one of "
        f"{sorted(PUBLISHED_CHECKPOINTS)}."
    )


def resolve_file(source: str, filename: str) -> str:
    r"""
    Args:
        source (`str`):
            Repository id, or local directory holding `filename`.
        filename (`str`):
            Path of the file inside the repository or directory.

    Returns:
        `str`: Local path of the file, downloading it if `source` is a repository id.
    """
    path = Path(source) / filename
    if path.is_file():
        return str(path)
    return hf_hub_download(source, filename)


def read_text_vocab_size(vocab_file: str) -> int:
    r"""
    Args:
        vocab_file (`str`):
            Path to a vocabulary file, one token per line.

    Returns:
        `int`: Number of lines the file holds, which is the model's `text_vocab_size`.
    """
    with open(vocab_file, "r", encoding="utf-8") as vocab_handle:
        return sum(1 for _ in vocab_handle)


def resolve_architecture(checkpoint_name: str) -> str:
    r"""
    Args:
        checkpoint_name (`str`):
            Name of a published checkpoint, or of an entry of [`ARCHITECTURES`].

    Returns:
        `str`: Key of [`ARCHITECTURES`] the checkpoint was trained with.

    Raises:
        ValueError: If the name matches no known architecture.
    """
    if checkpoint_name in ARCHITECTURES:
        return checkpoint_name
    for suffix in ("_no_zero_init", "_bigvgan"):
        if checkpoint_name.endswith(suffix) and checkpoint_name[: -len(suffix)] in ARCHITECTURES:
            return checkpoint_name[: -len(suffix)]
    raise ValueError(
        f"{checkpoint_name} matches no entry of `ARCHITECTURES`. Pass `architecture` explicitly, one of "
        f"{sorted(ARCHITECTURES)}."
    )


def build_config(architecture: str, text_vocab_size: int, **overrides) -> F5TTSConfig:
    r"""
    Builds an [`F5TTSConfig`] from one of the training configurations the F5-TTS authors published.

    Args:
        architecture (`str`):
            Key of [`ARCHITECTURES`], or the name of a published checkpoint trained with one of them.
        text_vocab_size (`int`):
            Number of lines of the checkpoint's vocabulary file.
        overrides (`dict`, *optional*):
            Configuration fields overriding the architecture's own.

    Returns:
        [`F5TTSConfig`]: The equivalent VoiceStudio configuration.
    """
    fields = dict(ARCHITECTURES[resolve_architecture(architecture)])
    fields.update(overrides)
    return F5TTSConfig(text_vocab_size=text_vocab_size, **fields)


def convert_state_dict(state_dict: dict[str, torch.Tensor], config: F5TTSConfig) -> dict[str, torch.Tensor]:
    r"""
    Renames a published F5-TTS state dict onto [`F5TTSForConditionalGeneration`]'s parameter names.

    Every module of the released backbones keeps its upstream name, so the rename is the `ema_model.transformer.`
    prefix becoming `model.` plus, for the `"unett"` backbone, the flat `layers.{i}.{j}` index pair gaining the
    `layer` `ModuleList` that holds it here.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            Tensors of a published checkpoint, with or without the exponential moving average prefix.
        config ([`F5TTSConfig`]):
            Configuration built from the matching architecture.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.

    Raises:
        ValueError: If the checkpoint holds a tensor this model has no destination for.
    """
    converted: dict[str, torch.Tensor] = {}
    leftover: list[str] = []

    for key, value in state_dict.items():
        name = key.removeprefix("ema_model.")
        if name in _DISCARDED_KEYS or name == "transformer.rotary_embed.inv_freq":
            continue
        if not name.startswith("transformer."):
            leftover.append(key)
            continue

        name = name.removeprefix("transformer.")
        if config.backbone == "unett":
            name = _UNET_LAYER.sub(lambda match: f"layers.{match.group(1)}.layer.{match.group(2)}.", name)
        converted["model." + name] = value.contiguous()

    if leftover:
        raise ValueError(f"The published checkpoint holds tensors this model has no destination for: {leftover}")

    return converted


def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    r"""
    Reads a published F5-TTS checkpoint, preferring its exponential moving average weights, which every released
    `.safetensors` file holds on its own and which the `.pt` files hold beside the training weights.

    Args:
        path (`str`):
            Path to a `.safetensors` or `.pt` checkpoint.

    Returns:
        `dict[str, torch.Tensor]`: The checkpoint's tensors.
    """
    if path.endswith(".safetensors"):
        return load_file(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "ema_model_state_dict" in checkpoint:
        return checkpoint["ema_model_state_dict"]
    return checkpoint["model_state_dict"]


def build_model_files(
    source: str = "SWivid/F5-TTS",
    subfolder: str | None = None,
    vocab_file: str | None = None,
    vocoder: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[F5TTSConfig, dict[str, torch.Tensor]]:
    r"""
    Reads a published F5-TTS or E2-TTS checkpoint and returns what [`F5TTSForConditionalGeneration`] needs to load
    it, with the vocoder of the mel front end it was trained against read out of that vocoder's own repository and
    prefixed onto the same state dict.

    Args:
        source (`str`, *optional*, defaults to `"SWivid/F5-TTS"`):
            Key of [`PUBLISHED_CHECKPOINTS`], key of [`DEFAULT_CHECKPOINTS`], or a repository id or local
            directory holding the published tree.
        subfolder (`str`, *optional*):
            Name of the directory holding the checkpoint to read. Defaults to the entry of
            [`DEFAULT_CHECKPOINTS`] the repository names.
        vocab_file (`str`, *optional*):
            Local vocabulary file to read instead of the published one.
        vocoder (`str`, *optional*):
            Repository id or local directory holding the published vocoder. Defaults to the entry of
            [`VOCODER_SOURCES`] the checkpoint's mel front end names.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `tuple[F5TTSConfig, dict[str, torch.Tensor]]`: The configuration and the renamed tensors.
    """
    checkpoint_name = resolve_checkpoint(source, subfolder)
    checkpoint = PUBLISHED_CHECKPOINTS[checkpoint_name]
    repo_id = checkpoint["repo_id"] if source in PUBLISHED_CHECKPOINTS or source in DEFAULT_CHECKPOINTS else source

    checkpoint_path = resolve_file(repo_id, checkpoint["weights_file"])
    vocab_file = vocab_file or resolve_file(checkpoint["vocab_repo_id"], checkpoint["vocab_file"])

    vocoder_source, build_vocoder_files = VOCODER_SOURCES[checkpoint["mel_spec_type"]]
    vocoder_config, vocoder_state_dict = build_vocoder_files(vocoder or vocoder_source, dtype=dtype)

    config = build_config(
        checkpoint_name, text_vocab_size=read_text_vocab_size(vocab_file), vocoder_config=vocoder_config
    )
    converted = convert_state_dict(load_checkpoint(checkpoint_path), config)
    converted = {key: value.to(dtype) for key, value in converted.items()}
    converted.update({"vocoder." + key: value for key, value in vocoder_state_dict.items()})
    return config, converted


def build_processor(
    source: str = "SWivid/F5-TTS", subfolder: str | None = None, vocab_file: str | None = None
) -> F5TTSProcessor:
    r"""
    Builds the [`F5TTSProcessor`] of a published F5-TTS or E2-TTS checkpoint, over the vocabulary it was trained
    with and the mel front end its vocoder inverts.

    Args:
        source (`str`, *optional*, defaults to `"SWivid/F5-TTS"`):
            Key of [`PUBLISHED_CHECKPOINTS`], key of [`DEFAULT_CHECKPOINTS`], or a repository id or local
            directory holding the published tree.
        subfolder (`str`, *optional*):
            Name of the directory holding the checkpoint to read. Defaults to the entry of
            [`DEFAULT_CHECKPOINTS`] the repository names.
        vocab_file (`str`, *optional*):
            Local vocabulary file to read instead of the published one.

    Returns:
        [`F5TTSProcessor`]: The processor.
    """
    checkpoint_name = resolve_checkpoint(source, subfolder)
    checkpoint = PUBLISHED_CHECKPOINTS[checkpoint_name]
    vocab_file = vocab_file or resolve_file(checkpoint["vocab_repo_id"], checkpoint["vocab_file"])

    config = build_config(checkpoint_name, text_vocab_size=read_text_vocab_size(vocab_file))
    feature_extractor = F5TTSFeatureExtractor(
        feature_size=config.mel_dim,
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        mel_spec_type=checkpoint["mel_spec_type"],
    )
    return F5TTSProcessor(feature_extractor=feature_extractor, tokenizer=F5TTSTokenizer(vocab_file))


def convert(
    source: str = "SWivid/F5-TTS",
    output_dir: str = "f5-tts-converted",
    subfolder: str | None = None,
    vocab_file: str | None = None,
    vocoder: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Writes a published F5-TTS or E2-TTS checkpoint into a directory of its own, which
    [`F5TTSForConditionalGeneration.from_pretrained`] and [`F5TTSProcessor.from_pretrained`] read without
    reaching the hub for the checkpoint, the vocabulary or the vocoder again.

    Args:
        source (`str`, *optional*, defaults to `"SWivid/F5-TTS"`):
            Key of [`PUBLISHED_CHECKPOINTS`], key of [`DEFAULT_CHECKPOINTS`], or a repository id or local
            directory holding the published tree.
        output_dir (`str`, *optional*, defaults to `"f5-tts-converted"`):
            Directory the config, weights, vocabulary, processor and vocoder are written to.
        subfolder (`str`, *optional*):
            Name of the directory holding the checkpoint to read. Defaults to the entry of
            [`DEFAULT_CHECKPOINTS`] the repository names.
        vocab_file (`str`, *optional*):
            Local vocabulary file to read instead of the published one.
        vocoder (`str`, *optional*):
            Repository id or local directory holding the published vocoder. Defaults to the entry of
            [`VOCODER_SOURCES`] the checkpoint's mel front end names.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config, converted = build_model_files(
        source, subfolder=subfolder, vocab_file=vocab_file, vocoder=vocoder, dtype=dtype
    )
    build_processor(source, subfolder=subfolder, vocab_file=vocab_file).save_pretrained(output_path)
    config.save_pretrained(output_path)
    save_file(converted, str(output_path / "model.safetensors"), metadata={"format": "pt"})


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_CHECKPOINTS",
    "PUBLISHED_CHECKPOINTS",
    "VOCODER_SOURCES",
    "build_config",
    "build_model_files",
    "build_processor",
    "convert",
    "convert_state_dict",
    "is_published_layout",
    "load_checkpoint",
    "read_text_vocab_size",
    "resolve_architecture",
    "resolve_checkpoint",
    "resolve_file",
]
