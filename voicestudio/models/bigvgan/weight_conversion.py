"""Checkpoint conversion for BigVGAN."""

import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

from ...utils.checkpoint_cache import CheckpointWriter, cached_conversion, source_identity
from .configuration_bigvgan import BigVGANConfig
from .feature_extraction_bigvgan import BigVGANFeatureExtractor


# Every repository NVIDIA published, keyed by the suffix of its name.
PUBLISHED_CHECKPOINTS = {
    "v2_24khz_100band_256x": "nvidia/bigvgan_v2_24khz_100band_256x",
    "v2_22khz_80band_256x": "nvidia/bigvgan_v2_22khz_80band_256x",
    "v2_22khz_80band_fmax8k_256x": "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x",
    "v2_44khz_128band_256x": "nvidia/bigvgan_v2_44khz_128band_256x",
    "v2_44khz_128band_512x": "nvidia/bigvgan_v2_44khz_128band_512x",
    "base_22khz_80band": "nvidia/bigvgan_base_22khz_80band",
    "base_24khz_100band": "nvidia/bigvgan_base_24khz_100band",
    "22khz_80band": "nvidia/bigvgan_22khz_80band",
    "24khz_100band": "nvidia/bigvgan_24khz_100band",
}

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "bigvgan_generator.pt"

# Resampling filters of the anti aliased activations, which the model rebuilds from its configuration.
_DISCARDED_SUFFIXES = (".upsample.filter", ".downsample.lowpass.filter")

_CONV_PRE = re.compile(r"^conv_pre\.(.+)$")
_CONV_POST = re.compile(r"^conv_post\.(.+)$")
_UPSAMPLE = re.compile(r"^ups\.(\d+)\.0\.(.+)$")
_RESBLOCK_CONV = re.compile(r"^resblocks\.(\d+)\.convs(1|2|)\.(\d+)\.(.+)$")
_RESBLOCK_ACTIVATION = re.compile(r"^resblocks\.(\d+)\.activations\.(\d+)\.act\.(alpha|beta)$")
_POST_ACTIVATION = re.compile(r"^activation_post\.act\.(alpha|beta)$")


def is_published_layout(source: str) -> bool:
    r"""
    Returns whether `source` is a published BigVGAN repository rather than a directory [`convert`] wrote.

    The published repositories carry the training script's own `config.json`, which names its fields `num_mels`,
    `hop_size`, `win_size`, `fmin`, `fmax` and `resblock` and declares no `model_type`, so the discriminator is a
    `config.json` declaring this model's `model_type`. `PreTrainedConfig.from_pretrained` draws no such
    distinction of its own: the unknown fields are kept as extra attributes and every field this model reads
    silently falls back to its default.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.

    Returns:
        `bool`: Whether `source` holds the published layout.
    """
    if source in PUBLISHED_CHECKPOINTS:
        return True
    config_file = cached_file(
        source,
        CONFIG_NAME,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if config_file is None:
        return True
    with open(config_file, "r", encoding="utf-8") as handle:
        return json.load(handle).get("model_type") != BigVGANConfig.model_type


def build_config(hyperparameters: dict, **overrides) -> BigVGANConfig:
    r"""
    Builds a [`BigVGANConfig`] from a BigVGAN `config.json`.

    Args:
        hyperparameters (`dict`):
            Parsed `config.json` of a BigVGAN repository.
        overrides (`dict`, *optional*):
            Configuration fields overriding the ones read from `hyperparameters`.

    Returns:
        [`BigVGANConfig`]: The equivalent VoiceStudio configuration.
    """
    fields = {
        "model_in_dim": hyperparameters["num_mels"],
        "sampling_rate": hyperparameters["sampling_rate"],
        "upsample_initial_channel": hyperparameters["upsample_initial_channel"],
        "upsample_rates": hyperparameters["upsample_rates"],
        "upsample_kernel_sizes": hyperparameters["upsample_kernel_sizes"],
        "resblock_type": str(hyperparameters["resblock"]),
        "resblock_kernel_sizes": hyperparameters["resblock_kernel_sizes"],
        "resblock_dilation_sizes": hyperparameters["resblock_dilation_sizes"],
        "activation": hyperparameters["activation"],
        "snake_logscale": hyperparameters["snake_logscale"],
        # Both default to True upstream, and the v1 repositories predate the two keys.
        "use_tanh_at_final": hyperparameters.get("use_tanh_at_final", True),
        "use_bias_at_final": hyperparameters.get("use_bias_at_final", True),
        "n_fft": hyperparameters["n_fft"],
        "hop_length": hyperparameters["hop_size"],
        "win_length": hyperparameters["win_size"],
        "mel_fmin": hyperparameters["fmin"],
        "mel_fmax": hyperparameters["fmax"],
        "mel_loss_fmax": hyperparameters["fmax_for_loss"],
        "use_multiscale_mel_loss": hyperparameters.get("use_multiscale_melloss", False),
        # The v1 repositories predate the key, and upstream falls back to HiFi-GAN's weight.
        "mel_loss_coeff": hyperparameters.get("lambda_melloss", 45.0),
    }

    fields.update(overrides)
    return BigVGANConfig(**fields)


def convert_state_dict(state_dict: dict[str, torch.Tensor], config: BigVGANConfig) -> dict[str, torch.Tensor]:
    r"""
    Renames a published BigVGAN state dict onto [`BigVGANModel`]'s parameter names and folds the weight norm
    reparameterization of every convolution back into plain weights.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The `"generator"` entry of a published `bigvgan_generator.pt`.
        config ([`BigVGANConfig`]):
            Configuration built from the matching `config.json`.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.

    Raises:
        ValueError: If a tensor has no destination in the migrated model.
    """
    num_kernels = len(config.resblock_kernel_sizes)

    folded = {}
    for key, value in state_dict.items():
        if key.endswith(_DISCARDED_SUFFIXES):
            continue
        if key.endswith(".weight_v"):
            continue
        if key.endswith(".weight_g"):
            stem = key[: -len(".weight_g")]
            direction = state_dict[f"{stem}.weight_v"]
            norm = direction.pow(2).sum(dim=tuple(range(1, direction.dim())), keepdim=True).sqrt()
            folded[f"{stem}.weight"] = value * direction / norm
        else:
            folded[key] = value

    converted = {}
    for key, value in folded.items():
        match = _CONV_PRE.match(key) or _CONV_POST.match(key)
        if match:
            converted[key] = value
            continue

        match = _UPSAMPLE.match(key)
        if match:
            converted[f"upsampler.{match.group(1)}.{match.group(2)}"] = value
            continue

        match = _RESBLOCK_CONV.match(key)
        if match:
            block, which, layer, leaf = match.groups()
            index = 1 if which in ("1", "") else 2
            block = int(block)
            converted[
                f"resblocks.{block // num_kernels}.{block % num_kernels}.layers.{layer}.conv{index}.{leaf}"
            ] = value
            continue

        match = _RESBLOCK_ACTIVATION.match(key)
        if match:
            block, activation, leaf = match.groups()
            block, activation = int(block), int(activation)
            # `AMPBlock1` holds one activation per convolution, the even ones in front of `convs1` and the odd
            # ones in front of `convs2`, while `AMPBlock2` holds a single one per layer.
            if config.resblock_type == "1":
                layer, index = activation // 2, activation % 2 + 1
            else:
                layer, index = activation, 1
            converted[
                f"resblocks.{block // num_kernels}.{block % num_kernels}.layers.{layer}.activation{index}.{leaf}"
            ] = value
            continue

        match = _POST_ACTIVATION.match(key)
        if match:
            converted[f"post_activation.{match.group(1)}"] = value
            continue

        raise ValueError(f"The published checkpoint holds a tensor this model has no destination for: {key}")

    return converted


def resolve_file(source: str, filename: str) -> str:
    r"""
    Args:
        source (`str`):
            Repository id, or local directory holding `filename`.
        filename (`str`):
            Name of the file inside the repository or directory.

    Returns:
        `str`: Local path of the file, downloading it if `source` is a repository id.
    """
    path = Path(source) / filename
    if path.is_file():
        return str(path)
    return hf_hub_download(source, filename)


def load_hyperparameters(source: str) -> dict:
    r"""
    Reads the `config.json` of a BigVGAN repository or local directory.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding the file.

    Returns:
        `dict`: The parsed configuration.
    """
    config_file = resolve_file(PUBLISHED_CHECKPOINTS.get(source, source), CONFIG_FILE)
    with open(config_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_hyperparameters_and_weights(
    source: str, weights_name: str = WEIGHTS_FILE
) -> tuple[dict, dict[str, torch.Tensor]]:
    r"""
    Reads the `config.json` and the generator weights of a BigVGAN repository or local directory.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding both files.
        weights_name (`str`, *optional*, defaults to `"bigvgan_generator.pt"`):
            Name of the weight file to read, which the v2 repositories also publish as
            `bigvgan_generator_3msteps.pt`.

    Returns:
        `tuple[dict, dict[str, torch.Tensor]]`: The parsed configuration and the generator's tensors.
    """
    hyperparameters = load_hyperparameters(source)
    weights_file = resolve_file(PUBLISHED_CHECKPOINTS.get(source, source), weights_name)
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=True)
    return hyperparameters, checkpoint["generator"]


def build_model_files(
    source: str = "v2_24khz_100band_256x",
    weights_name: str = WEIGHTS_FILE,
    dtype: torch.dtype = torch.float32,
) -> tuple[BigVGANConfig, dict[str, torch.Tensor]]:
    r"""
    Reads a published BigVGAN repository and returns what [`BigVGANModel`] needs to load it.

    Args:
        source (`str`, *optional*, defaults to `"v2_24khz_100band_256x"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.json` and a
            generator weight file.
        weights_name (`str`, *optional*, defaults to `"bigvgan_generator.pt"`):
            Name of the weight file to read.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `tuple[BigVGANConfig, dict[str, torch.Tensor]]`: The configuration and the renamed tensors.
    """
    hyperparameters, state_dict = load_hyperparameters_and_weights(source, weights_name=weights_name)
    config = build_config(hyperparameters)
    converted = convert_state_dict(state_dict, config)
    return config, {key: value.to(dtype).contiguous() for key, value in converted.items()}


def build_feature_extractor(config: BigVGANConfig) -> BigVGANFeatureExtractor:
    r"""
    Builds the [`BigVGANFeatureExtractor`] matching a configuration.

    Args:
        config ([`BigVGANConfig`]):
            Configuration of the converted model.

    Returns:
        [`BigVGANFeatureExtractor`]: The extractor.
    """
    return BigVGANFeatureExtractor(
        feature_size=config.model_in_dim,
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_fft=config.n_fft,
        fmin=config.mel_fmin,
        fmax=config.mel_fmax,
    )


def write_checkpoint(
    source: str = "v2_24khz_100band_256x",
    directory: str = "bigvgan-converted",
    weights_name: str = WEIGHTS_FILE,
    dtype: torch.dtype = torch.float32,
) -> BigVGANConfig:
    r"""
    Reads a published BigVGAN repository and writes what [`BigVGANModel.from_pretrained`] reads into `directory`.

    Args:
        source (`str`, *optional*, defaults to `"v2_24khz_100band_256x"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.json` and a
            generator weight file.
        directory (`str`, *optional*, defaults to `"bigvgan-converted"`):
            Directory the converted config and weights are written to.
        weights_name (`str`, *optional*, defaults to `"bigvgan_generator.pt"`):
            Name of the weight file to read.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        [`BigVGANConfig`]: The configuration that was written.
    """
    config, converted = build_model_files(source, weights_name=weights_name, dtype=dtype)
    with CheckpointWriter(directory) as writer:
        for key in list(converted):
            writer.add(key, converted.pop(key))
    config.save_pretrained(directory)
    return config


def converted_checkpoint(
    source: str = "v2_24khz_100band_256x",
    weights_name: str = WEIGHTS_FILE,
    dtype: torch.dtype = torch.float32,
) -> Path:
    r"""
    Returns a directory holding the converted form of a published BigVGAN repository, which
    [`~PreTrainedModel.from_pretrained`] reads the ordinary way, converting it the first time it is asked for
    and reusing that conversion afterwards.

    Args:
        source (`str`, *optional*, defaults to `"v2_24khz_100band_256x"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.json` and a
            generator weight file.
        weights_name (`str`, *optional*, defaults to `"bigvgan_generator.pt"`):
            Name of the weight file to read.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `Path`: The directory holding the converted checkpoint.
    """
    repository = PUBLISHED_CHECKPOINTS.get(source, source)
    parts = [str(source), weights_name, str(dtype), source_identity(repository, resolve_file(repository, CONFIG_FILE))]
    return cached_conversion(
        "bigvgan",
        parts,
        lambda directory: write_checkpoint(source, directory, weights_name=weights_name, dtype=dtype),
    )


def convert(
    source: str = "v2_24khz_100band_256x",
    output_dir: str = "bigvgan-converted",
    weights_name: str = WEIGHTS_FILE,
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Converts a published BigVGAN repository into a directory [`BigVGANModel.from_pretrained`] can load, for a
    checkpoint that is shipped elsewhere or kept outside the conversion cache [`converted_checkpoint`] holds.

    Args:
        source (`str`, *optional*, defaults to `"v2_24khz_100band_256x"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.json` and a
            generator weight file.
        output_dir (`str`, *optional*, defaults to `"bigvgan-converted"`):
            Directory the converted config, weights and feature extractor are written to.
        weights_name (`str`, *optional*, defaults to `"bigvgan_generator.pt"`):
            Name of the weight file to read.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    config = write_checkpoint(source, output_dir, weights_name=weights_name, dtype=dtype)
    build_feature_extractor(config).save_pretrained(output_dir)


__all__ = [
    "PUBLISHED_CHECKPOINTS",
    "build_config",
    "build_feature_extractor",
    "build_model_files",
    "convert",
    "convert_state_dict",
    "converted_checkpoint",
    "is_published_layout",
    "load_hyperparameters",
    "load_hyperparameters_and_weights",
    "resolve_file",
    "write_checkpoint",
]
