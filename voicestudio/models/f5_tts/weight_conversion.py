"""Checkpoint conversion for F5-TTS."""

import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

from ...utils.checkpoint_cache import CheckpointWriter, cached_conversion, file_identity, source_identity
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

# The file each vocoder repository declares its architecture in, which is what names its revision.
_VOCODER_CONFIG_FILES = {"vocos": "config.yaml", "bigvgan": "config.json"}

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


def convert_key(key: str, config: F5TTSConfig) -> str | None:
    r"""
    Maps one key of a published F5-TTS checkpoint onto its [`F5TTSForConditionalGeneration`] name.

    Every module of the released backbones keeps its upstream name, so the rename is the `ema_model.transformer.`
    prefix becoming `model.` plus, for the `"unett"` backbone, the flat `layers.{i}.{j}` index pair gaining the
    `layer` `ModuleList` that holds it here.

    Args:
        key (`str`):
            Key of a published checkpoint, with or without the exponential moving average prefix.
        config ([`F5TTSConfig`]):
            Configuration built from the matching architecture.

    Returns:
        `str` or `None`: The corresponding key of [`F5TTSForConditionalGeneration`], or `None` for a training
        time buffer the model does not carry.

    Raises:
        ValueError: If the key has no destination in this model.
    """
    name = key.removeprefix("ema_model.")
    if name in _DISCARDED_KEYS or name == "transformer.rotary_embed.inv_freq":
        return None
    if not name.startswith("transformer."):
        raise ValueError(f"The published checkpoint holds a tensor this model has no destination for: {key}")

    name = name.removeprefix("transformer.")
    if config.backbone == "unett":
        name = _UNET_LAYER.sub(lambda match: f"layers.{match.group(1)}.layer.{match.group(2)}.", name)
    return "model." + name


def convert_state_dict(state_dict: dict[str, torch.Tensor], config: F5TTSConfig) -> dict[str, torch.Tensor]:
    r"""
    Renames a published F5-TTS state dict onto [`F5TTSForConditionalGeneration`]'s parameter names.

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
    converted = {}
    for key, value in state_dict.items():
        name = convert_key(key, config)
        if name is not None:
            converted[name] = value.contiguous()
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


def resolve_sources(source: str, subfolder: str | None = None, vocab_file: str | None = None) -> tuple[str, str, str]:
    r"""
    Resolves the three files a published F5-TTS or E2-TTS checkpoint is read out of.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], key of [`DEFAULT_CHECKPOINTS`], or a repository id or local
            directory holding the published tree.
        subfolder (`str`, *optional*):
            Name of the directory holding the checkpoint to read. Defaults to the entry of
            [`DEFAULT_CHECKPOINTS`] the repository names.
        vocab_file (`str`, *optional*):
            Local vocabulary file to read instead of the published one.

    Returns:
        `tuple[str, str, str]`: The name of the published checkpoint, the local path of its weights and the
        local path of its vocabulary.
    """
    checkpoint_name = resolve_checkpoint(source, subfolder)
    checkpoint = PUBLISHED_CHECKPOINTS[checkpoint_name]
    repo_id = checkpoint["repo_id"] if source in PUBLISHED_CHECKPOINTS or source in DEFAULT_CHECKPOINTS else source
    return (
        checkpoint_name,
        resolve_file(repo_id, checkpoint["weights_file"]),
        vocab_file or resolve_file(checkpoint["vocab_repo_id"], checkpoint["vocab_file"]),
    )


def write_checkpoint(
    source: str = "SWivid/F5-TTS",
    directory: str = "f5-tts-converted",
    subfolder: str | None = None,
    vocab_file: str | None = None,
    vocoder: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> F5TTSConfig:
    r"""
    Reads a published F5-TTS or E2-TTS checkpoint and writes what
    [`F5TTSForConditionalGeneration.from_pretrained`] reads into `directory`, with the vocoder of the mel front
    end it was trained against read out of that vocoder's own repository and prefixed onto the same weights.

    The vocoder is written out and dropped before the backbone is read, and the backbone is copied over a tensor
    at a time, so neither the whole checkpoint nor the whole composed model is ever resident.

    Args:
        source (`str`, *optional*, defaults to `"SWivid/F5-TTS"`):
            Key of [`PUBLISHED_CHECKPOINTS`], key of [`DEFAULT_CHECKPOINTS`], or a repository id or local
            directory holding the published tree.
        directory (`str`, *optional*, defaults to `"f5-tts-converted"`):
            Directory the converted config and weights are written to.
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
        [`F5TTSConfig`]: The configuration that was written.
    """
    checkpoint_name, checkpoint_path, vocab_file = resolve_sources(source, subfolder, vocab_file)
    vocoder_source, build_vocoder_files = VOCODER_SOURCES[PUBLISHED_CHECKPOINTS[checkpoint_name]["mel_spec_type"]]
    vocoder_config, vocoder_state_dict = build_vocoder_files(vocoder or vocoder_source, dtype=dtype)

    config = build_config(
        checkpoint_name, text_vocab_size=read_text_vocab_size(vocab_file), vocoder_config=vocoder_config
    )

    with CheckpointWriter(directory) as writer:
        for key in list(vocoder_state_dict):
            writer.add("vocoder." + key, vocoder_state_dict.pop(key))
        writer.flush()

        state_dict = load_checkpoint(checkpoint_path)
        for key in list(state_dict):
            name = convert_key(key, config)
            value = state_dict.pop(key)
            if name is not None:
                writer.add(name, value.to(dtype))

    config.save_pretrained(directory)
    return config


def converted_checkpoint(
    source: str = "SWivid/F5-TTS",
    subfolder: str | None = None,
    vocab_file: str | None = None,
    vocoder: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Path:
    r"""
    Returns a directory holding the converted form of a published F5-TTS or E2-TTS checkpoint, which
    [`~PreTrainedModel.from_pretrained`] reads the ordinary way, converting it the first time it is asked for
    and reusing that conversion afterwards.

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
        `Path`: The directory holding the converted checkpoint.
    """
    checkpoint_name, checkpoint_path, vocab_file = resolve_sources(source, subfolder, vocab_file)
    mel_spec_type = PUBLISHED_CHECKPOINTS[checkpoint_name]["mel_spec_type"]
    vocoder_source = vocoder or VOCODER_SOURCES[mel_spec_type][0]
    vocoder_config_file = resolve_file(vocoder_source, _VOCODER_CONFIG_FILES[mel_spec_type])
    parts = [
        checkpoint_name,
        str(dtype),
        file_identity(checkpoint_path),
        file_identity(vocab_file),
        source_identity(vocoder_source, vocoder_config_file),
    ]
    return cached_conversion(
        "f5_tts",
        parts,
        lambda directory: write_checkpoint(
            source, directory, subfolder=subfolder, vocab_file=vocab_file, vocoder=vocoder, dtype=dtype
        ),
    )


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
    reaching the hub for the checkpoint, the vocabulary or the vocoder again, for a checkpoint that is shipped
    elsewhere or kept outside the conversion cache [`converted_checkpoint`] holds.

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
    write_checkpoint(source, output_dir, subfolder=subfolder, vocab_file=vocab_file, vocoder=vocoder, dtype=dtype)
    build_processor(source, subfolder=subfolder, vocab_file=vocab_file).save_pretrained(output_dir)


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_CHECKPOINTS",
    "PUBLISHED_CHECKPOINTS",
    "VOCODER_SOURCES",
    "build_config",
    "build_processor",
    "convert",
    "convert_key",
    "convert_state_dict",
    "converted_checkpoint",
    "is_published_layout",
    "load_checkpoint",
    "read_text_vocab_size",
    "resolve_architecture",
    "resolve_checkpoint",
    "resolve_file",
    "resolve_sources",
    "write_checkpoint",
]
