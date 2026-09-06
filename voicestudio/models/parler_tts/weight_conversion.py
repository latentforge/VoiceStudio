"""Checkpoint conversion for Parler-TTS."""

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import DacConfig, DacModel
from transformers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from transformers.utils.hub import cached_file

from ...utils.checkpoint_cache import CheckpointWriter, cached_conversion, source_identity
from .configuration_parler_tts import ParlerTTSConfig, is_legacy_audio_encoder_config
from .processing_parler_tts import AUDIO_TOKENIZER_SUBFOLDER


# The published Parler-TTS checkpoints bundle the audio codec as the original `descript-audio-codec`
# module under an `audio_encoder.model.` prefix: weight-normalized convolutions kept as separate
# `weight_g`/`weight_v` tensors, and block indices numbered the way the upstream `DAC` module nests
# them. `transformers`'s `DacModel` uses flattened `encoder.block.<i>.res_unit<j>` names with the
# weight norm already folded in.
_AUDIO_ENCODER_PREFIX = "audio_encoder."
_DAC_PREFIX = f"{_AUDIO_ENCODER_PREFIX}model."

# The codec's quantizer, the one part of the module both layouts name alike.
_QUANTIZER_PREFIX = "quantizer."

# Repository publishing the codec Parler-TTS is trained against, in [`DacModel`]'s own layout.
_CODEC_REPO = "descript/dac_44khz"

# Fields of [`DacConfig`] that fix the codec's shape and bandwidth, which the checkpoint's own configuration
# and `_CODEC_REPO`'s have to agree on for the two to be the same codec.
_CODEC_FIELDS = (
    "encoder_hidden_size",
    "downsampling_ratios",
    "decoder_hidden_size",
    "upsampling_ratios",
    "hidden_size",
    "n_codebooks",
    "codebook_size",
    "codebook_dim",
    "sampling_rate",
    "hop_length",
)

# Largest difference tolerated between a quantizer tensor of the bundle and `_CODEC_REPO`'s, which is the
# float32 rounding of folding the weight norm here against `remove_weight_norm` folding it there. The 18
# projection tensors of `parler-tts/parler-tts-mini-v1` differ by at most 5.96e-08, the other 27 not at all.
_CODEC_ATOL = 1e-06

# Keyword arguments of [`~transformers.utils.hub.cached_file`] that select which copy of a repository is read,
# which the loaders forward from their own call.
_DOWNLOAD_KWARGS = ("revision", "token", "cache_dir", "local_files_only")

# Of those, the ones that carry over to a repository other than the one the caller named. A revision pins the
# checkpoint alone.
_CODEC_DOWNLOAD_KWARGS = ("token", "cache_dir", "local_files_only")

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


def read_quantizer_state_dict(weight_files: list[str]) -> dict[str, torch.Tensor]:
    r"""
    Reads the bundled codec's quantizer tensors out of a published checkpoint, leaving everything else unread.

    Args:
        weight_files (`list[str]`):
            Local paths of the checkpoint's safetensors shards.

    Returns:
        `dict[str, torch.Tensor]`: The quantizer's tensors, under the names the `descript-audio-codec` module
        gives them.

    Raises:
        ValueError: If the checkpoint bundles no codec quantizer.
    """
    prefix = _DAC_PREFIX + _QUANTIZER_PREFIX
    state_dict = {}
    for path in weight_files:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state_dict[key[len(_DAC_PREFIX) :]] = handle.get_tensor(key)

    if not state_dict:
        raise ValueError(f"No `{prefix}` weights found in {weight_files}.")
    return state_dict


def fold_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Merges every `weight_g`/`weight_v` pair of a published checkpoint's codec into the single `weight` tensor
    [`DacModel`] declares.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            Tensors of the bundled codec, as [`read_quantizer_state_dict`] returns them.

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


def _as_list(value):
    """Returns a configuration field's value as a list if it is a tuple or a list, and unchanged otherwise."""
    return list(value) if isinstance(value, (list, tuple)) else value


def load_codec(config: DacConfig, quantizer_state_dict: dict[str, torch.Tensor], **kwargs) -> DacModel:
    r"""
    Loads the codec a published Parler-TTS checkpoint speaks, from the repository that publishes it in
    [`DacModel`]'s own layout.

    Parler-TTS holds its codec frozen for the whole of training and ships it unchanged, so the tensors the
    checkpoint bundles under `audio_encoder.model.` are `_CODEC_REPO`'s, spelled the way the
    `descript-audio-codec` module names them. Both halves of that are checked here rather than assumed: the
    shape and bandwidth the checkpoint's own `audio_encoder` entry describes against the repository's
    configuration, and the quantizer, whose codebooks fix what the decoder's tokens mean, tensor by tensor
    against the bundle. A checkpoint carrying a codec of its own is refused instead of being read as this one.

    Args:
        config ([`DacConfig`]):
            Configuration of the codec, as the checkpoint describes it.
        quantizer_state_dict (`dict[str, torch.Tensor]`):
            The bundled quantizer's tensors, as [`read_quantizer_state_dict`] returns them.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`~transformers.utils.hub.cached_file`], of which `token`, `cache_dir` and
            `local_files_only` are used.

    Returns:
        [`DacModel`]: The codec, holding the published weights.

    Raises:
        ValueError: If the checkpoint describes or bundles a different codec than `_CODEC_REPO` holds.
    """
    model = DacModel.from_pretrained(
        _CODEC_REPO, **{key: value for key, value in kwargs.items() if key in _CODEC_DOWNLOAD_KWARGS}
    )

    # A field holding a sequence reads back as a tuple from one configuration and as a list from the other,
    # depending on whether it was built in Python or parsed from JSON.
    described = {field: _as_list(getattr(config, field)) for field in _CODEC_FIELDS}
    holds = {field: _as_list(getattr(model.config, field)) for field in _CODEC_FIELDS}
    differing = [
        f"{field} is {described[field]!r} against {holds[field]!r}"
        for field in _CODEC_FIELDS
        if described[field] != holds[field]
    ]
    if differing:
        raise ValueError(
            f"The checkpoint describes a codec {_CODEC_REPO} does not hold: {', '.join(differing)}."
        )

    published = {key: value for key, value in model.state_dict().items() if key.startswith(_QUANTIZER_PREFIX)}
    bundled = fold_weight_norm(quantizer_state_dict)
    if set(bundled) != set(published):
        raise ValueError(
            f"The checkpoint bundles a quantizer of {sorted(set(bundled) - set(published))} that {_CODEC_REPO} "
            f"does not hold, and none of {sorted(set(published) - set(bundled))} that it does."
        )

    mismatched = sorted(
        key
        for key, value in bundled.items()
        if not torch.allclose(value.float(), published[key].float(), rtol=0, atol=_CODEC_ATOL)
    )
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} of the checkpoint's {len(bundled)} quantizer tensors differ from "
            f"{_CODEC_REPO}'s by more than {_CODEC_ATOL}, so it carries a codec of its own: {mismatched}."
        )

    return model


def write_checkpoint(
    source, directory, subfolder: str | None = None, config: ParlerTTSConfig | None = None, **kwargs
) -> None:
    r"""
    Reads a published Parler-TTS checkpoint and writes what [`ParlerTTSForConditionalGeneration.from_pretrained`]
    reads into `directory`, with the codec taken from the repository that publishes it in [`DacModel`]'s layout
    and written under the same `audio_encoder` prefix. The codec is also saved standalone under `directory`'s
    `audio_encoder` subfolder, for [`ParlerTTSProcessor`] to read with an ordinary
    [`DacModel.from_pretrained`].

    The text encoder and the decoder are copied over a tensor at a time and written out in shards, so a
    checkpoint too large to hold twice is never held once.

    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding the published checkpoint.
        directory (`str` or `os.PathLike`):
            Directory the converted checkpoint is written to.
        subfolder (`str`, *optional*):
            Directory inside `source` the checkpoint sits in.
        config ([`ParlerTTSConfig`], *optional*):
            Configuration to convert the checkpoint into. Defaults to the one its `config.json` describes.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`resolve_file`].
    """
    if config is None:
        config = read_config(source, subfolder=subfolder, **kwargs)

    target = Path(directory)
    weight_files = resolve_weight_files(source, subfolder=subfolder, **kwargs)
    with CheckpointWriter(directory) as writer:
        codec = load_codec(config.audio_encoder, read_quantizer_state_dict(weight_files), **kwargs)
        writer.update({_AUDIO_ENCODER_PREFIX + key: value for key, value in codec.state_dict().items()})
        # The codec is written out and dropped before the rest is read, so the two are never resident together.
        writer.flush()
        codec.save_pretrained(target / AUDIO_TOKENIZER_SUBFOLDER)
        del codec

        for path in weight_files:
            with safe_open(path, framework="pt") as handle:
                for key in handle.keys():
                    if not key.startswith(_AUDIO_ENCODER_PREFIX):
                        writer.add(key, handle.get_tensor(key))

    config.save_pretrained(target)
    for name in _COPIED_FILES:
        copied = resolve_file(source, name, subfolder=subfolder, **kwargs)
        if copied is not None:
            shutil.copy(copied, target / name)


def converted_checkpoint(
    source, subfolder: str | None = None, config: ParlerTTSConfig | None = None, **kwargs
) -> Path:
    r"""
    Returns a directory holding the converted form of a published Parler-TTS checkpoint, which
    [`~PreTrainedModel.from_pretrained`] reads the ordinary way, converting it the first time it is asked for
    and reusing that conversion afterwards.

    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory holding the published checkpoint.
        subfolder (`str`, *optional*):
            Directory inside `source` the checkpoint sits in.
        config ([`ParlerTTSConfig`], *optional*):
            Configuration to convert the checkpoint into. Defaults to the one its `config.json` describes.
        kwargs (`dict`, *optional*):
            Keyword arguments of [`resolve_file`].

    Returns:
        `Path`: The directory holding the converted checkpoint.
    """
    config_file = resolve_file(source, CONFIG_NAME, subfolder=subfolder, **kwargs)
    parts = [str(source), subfolder, source_identity(source, config_file), _CODEC_REPO]
    if config is not None:
        parts.append(config.to_json_string())
    return cached_conversion(
        "parler_tts",
        parts,
        lambda directory: write_checkpoint(source, directory, subfolder=subfolder, config=config, **kwargs),
    )


def convert(checkpoint_path, output_dir):
    """
    Writes a published Parler-TTS checkpoint into a directory of its own, which
    [`ParlerTTSForConditionalGeneration.from_pretrained`] and [`ParlerTTSProcessor.from_pretrained`] read without
    converting the codec again, for a checkpoint that is shipped elsewhere or kept outside the conversion cache
    [`converted_checkpoint`] holds.

    Args:
        checkpoint_path (`str`):
            A Hugging Face repo id or a local directory holding the published checkpoint.
        output_dir (`str`):
            Directory the converted checkpoint is written to.

    Returns:
        `str`: The `output_dir` that was written.
    """
    target = Path(output_dir)
    write_checkpoint(checkpoint_path, target)
    return str(target)


__all__ = [
    "convert",
    "converted_checkpoint",
    "fold_weight_norm",
    "is_published_layout",
    "load_codec",
    "read_config",
    "read_quantizer_state_dict",
    "resolve_file",
    "resolve_weight_files",
    "write_checkpoint",
]
