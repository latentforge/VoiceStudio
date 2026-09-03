"""Checkpoint conversion for Parler-TTS."""

import io
import json
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import DacConfig, DacModel, GenerationConfig
from transformers.utils import (
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    logging,
)
from transformers.utils.hub import cached_file

from .configuration_parler_tts import ParlerTTSConfig, is_legacy_audio_encoder_config
from .processing_parler_tts import AUDIO_TOKENIZER_SUBFOLDER


# The published Parler-TTS checkpoints store the audio codec as the original
# `descript-audio-codec` module under an `audio_encoder.model.` prefix: weight-normalized
# convolutions kept as separate `weight_g`/`weight_v` tensors, and block indices numbered
# the way the upstream `DAC` module nests them. `transformers`'s `DacModel` uses flattened
# `encoder.block.<i>.res_unit<j>` names with the weight norm already folded in.
_AUDIO_ENCODER_PREFIX = "audio_encoder."
_DAC_PREFIX = f"{_AUDIO_ENCODER_PREFIX}model."

# The entry of the `descript-audio-codec` release the published codec comes from, which names the layout
# `recursively_load_weights` reads.
_DAC_VARIANT = "dac_44khz"

# Keyword arguments of [`~transformers.utils.hub.cached_file`] that select which copy of a repository is read,
# which the loaders forward from their own call.
_DOWNLOAD_KWARGS = ("revision", "token", "cache_dir", "local_files_only")

_COPIED_FILES = (
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def is_published_layout(source, subfolder: str | None = None, **kwargs) -> bool:
    r"""
    Returns whether `source` holds a published Parler-TTS checkpoint rather than one [`convert`] wrote.

    A published checkpoint carries the codec as the `descript-audio-codec` module it was trained with, described
    by the vendored `DACConfig` its `config.json` serializes as the `audio_encoder` entry, so that entry is the
    discriminator. Neither layout raises on its own: [`ParlerTTSConfig`] reads both, and the weights differ only
    in names and in a weight norm reparameterization that a load reports as missing and unexpected keys.

    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding the checkpoint.
        subfolder (`str`, *optional*):
            Directory inside `source` the configuration sits in.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`~transformers.utils.hub.cached_file`], of which `revision`, `token`,
            `cache_dir` and `local_files_only` are used.

    Returns:
        `bool`: Whether `source` holds the published layout.
    """
    config_file = resolve_file(source, CONFIG_NAME, subfolder=subfolder, **kwargs)
    if config_file is None:
        return False
    audio_encoder_config = json.loads(Path(config_file).read_text()).get("audio_encoder")
    return isinstance(audio_encoder_config, dict) and is_legacy_audio_encoder_config(audio_encoder_config)


def resolve_file(source, filename: str, subfolder: str | None = None, **kwargs) -> str | None:
    r"""
    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding `filename`.
        filename (`str`):
            Name of the file inside the repository or directory.
        subfolder (`str`, *optional*):
            Directory inside `source` the file sits in.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`~transformers.utils.hub.cached_file`], of which `revision`, `token`,
            `cache_dir` and `local_files_only` are used.

    Returns:
        `str` or `None`: Local path of the file, downloading it if `source` is a repository id, or `None` if the
        checkpoint holds no such file.
    """
    return cached_file(
        source,
        filename,
        subfolder=subfolder or "",
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        **{key: value for key, value in kwargs.items() if key in _DOWNLOAD_KWARGS},
    )


def read_config(source, subfolder: str | None = None, **kwargs) -> ParlerTTSConfig:
    r"""
    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding the checkpoint.
        subfolder (`str`, *optional*):
            Directory inside `source` the configuration sits in.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`resolve_file`].

    Returns:
        [`ParlerTTSConfig`]: The checkpoint's configuration.
    """
    return ParlerTTSConfig.from_pretrained(
        source,
        subfolder=subfolder or "",
        **{key: value for key, value in kwargs.items() if key in _DOWNLOAD_KWARGS},
    )


def read_generation_config(source, subfolder: str | None = None, **kwargs) -> GenerationConfig | None:
    r"""
    Reads the generation configuration a checkpoint ships, which decides how long a generation runs and whether
    it samples, and which `from_pretrained` cannot resolve itself for a checkpoint handed to it as tensors.

    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding the checkpoint.
        subfolder (`str`, *optional*):
            Directory inside `source` the configuration sits in.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`resolve_file`].

    Returns:
        [`GenerationConfig`] or `None`: The checkpoint's generation configuration, or `None` if it ships none.
    """
    if resolve_file(source, GENERATION_CONFIG_NAME, subfolder=subfolder, **kwargs) is None:
        return None
    return GenerationConfig.from_pretrained(
        source,
        subfolder=subfolder or "",
        **{key: value for key, value in kwargs.items() if key in _DOWNLOAD_KWARGS},
    )


def resolve_weight_files(source, subfolder: str | None = None, **kwargs) -> list[str]:
    r"""
    Resolves the safetensors files of a checkpoint, which the larger published ones shard.

    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding the checkpoint.
        subfolder (`str`, *optional*):
            Directory inside `source` the weights sit in.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`resolve_file`].

    Returns:
        `list[str]`: Local paths of the checkpoint's shards, in the order the index lists them.

    Raises:
        OSError: If the checkpoint holds no safetensors file at all.
    """
    index_file = resolve_file(source, SAFE_WEIGHTS_INDEX_NAME, subfolder=subfolder, **kwargs)
    if index_file is not None:
        shards = dict.fromkeys(json.loads(Path(index_file).read_text())["weight_map"].values())
        return [resolve_file(source, shard, subfolder=subfolder, **kwargs) for shard in shards]

    weights_file = resolve_file(source, SAFE_WEIGHTS_NAME, subfolder=subfolder, **kwargs)
    if weights_file is None:
        raise OSError(f"{source} holds neither {SAFE_WEIGHTS_NAME} nor {SAFE_WEIGHTS_INDEX_NAME}.")
    return [weights_file]


def read_dac_state_dict(weight_files: list[str]) -> dict[str, torch.Tensor]:
    r"""
    Reads the codec's tensors out of a published checkpoint, leaving the text encoder and the decoder unread.

    Args:
        weight_files (`list[str]`):
            Local paths of the checkpoint's safetensors shards.

    Returns:
        `dict[str, torch.Tensor]`: The codec's tensors, under the names the `descript-audio-codec` module gives
        them.

    Raises:
        ValueError: If the checkpoint holds no codec weights.
    """
    state_dict = {}
    for path in weight_files:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(_DAC_PREFIX):
                    state_dict[key[len(_DAC_PREFIX) :]] = handle.get_tensor(key)

    if not state_dict:
        raise ValueError(f"No `{_DAC_PREFIX}` weights found in {weight_files}.")
    return state_dict


def fold_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Merges every `weight_g`/`weight_v` pair of a published checkpoint's codec into the single `weight` tensor
    [`DacModel`] declares.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The codec's tensors, as [`read_dac_state_dict`] returns them.

    Returns:
        `dict[str, torch.Tensor]`: The same tensors with the weight norm reparameterization folded in.
    """
    folded = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_v"):
            continue
        if key.endswith(".weight_g"):
            direction = state_dict[f"{key[: -len('.weight_g')]}.weight_v"]
            norm = direction.pow(2).sum(dim=tuple(range(1, direction.dim())), keepdim=True).sqrt()
            folded[f"{key[: -len('.weight_g')]}.weight"] = value * direction / norm
        else:
            folded[key] = value
    return folded


def build_dac_model(config: DacConfig, state_dict: dict[str, torch.Tensor]) -> DacModel:
    r"""
    Builds the [`DacModel`] a published Parler-TTS checkpoint's codec weights describe.

    The names are the `descript-audio-codec` module's own, so they are mapped by the converter `transformers`
    ships for that layout, which shape checks every tensor against the module it lands in. That check is what
    ties the configuration read from `config.json`, which declares no architecture hyperparameter, to the
    checkpoint it describes.

    Args:
        config ([`DacConfig`]):
            Configuration of the codec.
        state_dict (`dict[str, torch.Tensor]`):
            The codec's tensors, as [`read_dac_state_dict`] returns them.

    Returns:
        [`DacModel`]: The codec, holding the checkpoint's weights.

    Raises:
        ValueError: If the checkpoint holds a different number of codec tensors than the codec has.
    """
    model = DacModel(config)
    state_dict = fold_weight_norm(state_dict)
    if len(state_dict) != len(model.state_dict()):
        raise ValueError(
            f"The published checkpoint holds {len(state_dict)} codec tensors, against the "
            f"{len(model.state_dict())} of the codec its configuration describes."
        )

    # `transformers.models.dac.convert_dac_checkpoint` is a conversion script rather than library code: importing
    # it raises the library's verbosity to INFO, and it reports the tensors it mapped nowhere on stdout instead of
    # returning them, which the count above already accounts for.
    verbosity = logging.get_verbosity()
    try:
        from transformers.models.dac.convert_dac_checkpoint import recursively_load_weights

        logging.set_verbosity_error()
        with redirect_stdout(io.StringIO()):
            recursively_load_weights(state_dict, model, _DAC_VARIANT)
    finally:
        logging.set_verbosity(verbosity)

    return model


def build_model_files(
    source, subfolder: str | None = None, config: ParlerTTSConfig | None = None, **kwargs
) -> tuple[ParlerTTSConfig, dict[str, torch.Tensor]]:
    r"""
    Reads a published Parler-TTS checkpoint and returns what [`ParlerTTSForConditionalGeneration`] needs to load
    it, with the codec rewritten into [`DacModel`]'s weight layout under the same `audio_encoder` prefix.

    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding the published checkpoint.
        subfolder (`str`, *optional*):
            Directory inside `source` the checkpoint sits in.
        config ([`ParlerTTSConfig`], *optional*):
            Configuration to load the checkpoint into. Defaults to the one its `config.json` describes.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`resolve_file`].

    Returns:
        `tuple[ParlerTTSConfig, dict[str, torch.Tensor]]`: The configuration and the converted tensors.
    """
    if config is None:
        config = read_config(source, subfolder=subfolder, **kwargs)

    weight_files = resolve_weight_files(source, subfolder=subfolder, **kwargs)
    dac_model = build_dac_model(config.audio_encoder, read_dac_state_dict(weight_files))

    converted = {}
    for path in weight_files:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                if not key.startswith(_AUDIO_ENCODER_PREFIX):
                    converted[key] = handle.get_tensor(key)
    converted.update(
        {_AUDIO_ENCODER_PREFIX + key: value for key, value in dac_model.state_dict().items()}
    )
    return config, converted


def build_audio_tokenizer(source, subfolder: str | None = None, **kwargs) -> DacModel:
    r"""
    Reads the codec of a published Parler-TTS checkpoint on its own, for [`ParlerTTSProcessor`] to decode with.

    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding the published checkpoint.
        subfolder (`str`, *optional*):
            Directory inside `source` the checkpoint sits in.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`resolve_file`].

    Returns:
        [`DacModel`]: The codec, in evaluation mode.
    """
    config = read_config(source, subfolder=subfolder, **kwargs)
    weight_files = resolve_weight_files(source, subfolder=subfolder, **kwargs)
    return build_dac_model(config.audio_encoder, read_dac_state_dict(weight_files)).eval()


def convert(checkpoint_path, output_dir):
    """
    Writes a published Parler-TTS checkpoint into a directory of its own, which
    [`ParlerTTSForConditionalGeneration.from_pretrained`] and [`ParlerTTSProcessor.from_pretrained`] read without
    converting the codec again, for a checkpoint that is loaded many times or shipped elsewhere.

    Args:
        checkpoint_path (`str`):
            A Hugging Face repo id or a local directory holding the published checkpoint.
        output_dir (`str`):
            Directory the converted checkpoint is written to.

    Returns:
        `str`: The `output_dir` that was written.
    """
    source = Path(checkpoint_path)
    if not source.is_dir():
        source = Path(snapshot_download(checkpoint_path))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    config, converted = build_model_files(source)

    save_file(
        {key: value.contiguous() for key, value in converted.items()},
        target / SAFE_WEIGHTS_NAME,
        metadata={"format": "pt"},
    )
    config.save_pretrained(target)
    for name in _COPIED_FILES:
        if (source / name).is_file():
            shutil.copy(source / name, target / name)

    dac_model = DacModel(config.audio_encoder)
    dac_model.load_state_dict(
        {
            key[len(_AUDIO_ENCODER_PREFIX) :]: value
            for key, value in converted.items()
            if key.startswith(_AUDIO_ENCODER_PREFIX)
        }
    )
    dac_model.save_pretrained(target / AUDIO_TOKENIZER_SUBFOLDER)

    return str(target)


__all__ = [
    "build_audio_tokenizer",
    "build_dac_model",
    "fold_weight_norm",
    "build_model_files",
    "convert",
    "is_published_layout",
    "read_config",
    "read_dac_state_dict",
    "read_generation_config",
    "resolve_file",
    "resolve_weight_files",
]
