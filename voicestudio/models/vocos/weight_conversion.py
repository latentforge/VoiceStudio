"""Checkpoint conversion for Vocos."""

import json
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

from ...utils.checkpoint_cache import CheckpointWriter, cached_conversion, source_identity
from .configuration_vocos import VocosConfig
from .feature_extraction_vocos import VocosFeatureExtractor


# Repositories the Vocos authors published, keyed by the front end their backbone was trained behind.
PUBLISHED_CHECKPOINTS = {
    "mel": "charactr/vocos-mel-24khz",
    "encodec": "charactr/vocos-encodec-24khz",
}

# The `class_path` suffix of each supported front end, backbone and head of a Vocos `config.yaml`.
_FEATURE_EXTRACTORS = {"MelSpectrogramFeatures": "mel", "EncodecFeatures": "encodec"}
_BACKBONE = "VocosBackbone"
_HEAD = "ISTFTHead"

# Buffers of the analysis front end and of the inverse STFT, which the model rebuilds from its configuration.
_DISCARDED_PREFIXES = ("feature_extractor.mel_spec.", "head.istft.")


def is_published_layout(source: str) -> bool:
    r"""
    Returns whether `source` is a published Vocos repository rather than a directory [`convert`] wrote.

    The published repositories carry a `config.yaml` naming three classes and no `config.json` at all, so the
    discriminator is a `config.json` declaring this model's `model_type`. `PreTrainedConfig.from_pretrained`
    draws no such distinction of its own: `cached_file` returns `None` for the missing file and the
    configuration silently falls back to its defaults.

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
        return json.load(handle).get("model_type") != VocosConfig.model_type


def build_feature_extractor(config: VocosConfig) -> VocosFeatureExtractor:
    r"""
    Builds the [`VocosFeatureExtractor`] matching a configuration.

    Args:
        config ([`VocosConfig`]):
            Configuration of the converted model.

    Returns:
        [`VocosFeatureExtractor`]: The extractor.
    """
    return VocosFeatureExtractor(
        feature_size=config.input_channels,
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        padding=config.padding,
    )


def build_config(hyperparameters: dict, num_quantizers: int | None = None, **overrides) -> VocosConfig:
    r"""
    Builds a [`VocosConfig`] from a Vocos `config.yaml`.

    Args:
        hyperparameters (`dict`):
            Parsed `config.yaml` of a Vocos repository.
        num_quantizers (`int`, *optional*):
            Number of EnCodec codebooks the published codebook table holds. Only the `"encodec"` front end uses
            it, and [`convert`] reads it off the checkpoint itself.
        overrides (`dict`, *optional*):
            Configuration fields overriding the ones read from `hyperparameters`.

    Returns:
        [`VocosConfig`]: The equivalent VoiceStudio configuration.

    Raises:
        ValueError: If the repository holds a front end, backbone or head this model does not implement.
    """
    feature_extractor = hyperparameters["feature_extractor"]
    backbone = hyperparameters["backbone"]
    head = hyperparameters["head"]

    front_end = _FEATURE_EXTRACTORS.get(feature_extractor["class_path"].rsplit(".", 1)[-1])
    if front_end is None:
        raise ValueError(
            f"{feature_extractor['class_path']} is not one of the front ends this model implements, "
            f"{sorted(_FEATURE_EXTRACTORS)}."
        )
    if not backbone["class_path"].endswith(_BACKBONE) or not head["class_path"].endswith(_HEAD):
        raise ValueError(
            f"This model is a `{_BACKBONE}` plus `{_HEAD}` vocoder, got {backbone['class_path']} plus "
            f"{head['class_path']}."
        )

    feature_extractor_args = feature_extractor["init_args"]
    backbone_args = backbone["init_args"]
    head_args = head["init_args"]

    fields = {
        "feature_extractor_type": front_end,
        "input_channels": backbone_args["input_channels"],
        "hidden_size": backbone_args["dim"],
        "intermediate_size": backbone_args["intermediate_dim"],
        "num_hidden_layers": backbone_args["num_layers"],
        "layer_scale_init_value": backbone_args.get("layer_scale_init_value"),
        "adanorm_num_embeddings": backbone_args.get("adanorm_num_embeddings"),
        "n_fft": head_args["n_fft"],
        "hop_length": head_args["hop_length"],
        "padding": head_args.get("padding", "same"),
    }
    if front_end == "mel":
        fields["sampling_rate"] = feature_extractor_args["sample_rate"]
    else:
        fields["bandwidths"] = feature_extractor_args["bandwidths"]
        if num_quantizers is not None:
            fields["num_quantizers"] = num_quantizers

    fields.update(overrides)
    return VocosConfig(**fields)


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Renames a published Vocos state dict onto [`VocosModel`]'s parameter names.

    Every module carrying weights keeps its upstream name, so the only change is that the mel front end and
    inverse STFT buffers are dropped. `feature_extractor.codebook_weights`, which the `"encodec"` front end trains
    against and which is not a buffer, is kept.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            Tensors of a Vocos `pytorch_model.bin`.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.
    """
    return {key: value.contiguous() for key, value in state_dict.items() if not key.startswith(_DISCARDED_PREFIXES)}


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
    Reads the `config.yaml` of a Vocos repository or local directory.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding the file.

    Returns:
        `dict`: The parsed configuration.
    """
    config_file = resolve_file(PUBLISHED_CHECKPOINTS.get(source, source), "config.yaml")
    with open(config_file, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_hyperparameters_and_weights(source: str) -> tuple[dict, dict[str, torch.Tensor]]:
    r"""
    Reads the `config.yaml` and `pytorch_model.bin` of a Vocos repository or local directory.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding both files.

    Returns:
        `tuple[dict, dict[str, torch.Tensor]]`: The parsed configuration and the checkpoint's tensors.
    """
    hyperparameters = load_hyperparameters(source)
    weights_file = resolve_file(PUBLISHED_CHECKPOINTS.get(source, source), "pytorch_model.bin")
    return hyperparameters, torch.load(weights_file, map_location="cpu", weights_only=True)


def build_model_files(
    source: str = "mel", dtype: torch.dtype = torch.float32
) -> tuple[VocosConfig, dict[str, torch.Tensor]]:
    r"""
    Reads a published Vocos repository and returns what [`VocosModel`] needs to load it.

    Args:
        source (`str`, *optional*, defaults to `"mel"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.yaml` and a
            `pytorch_model.bin`.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `tuple[VocosConfig, dict[str, torch.Tensor]]`: The configuration and the renamed tensors.
    """
    hyperparameters, state_dict = load_hyperparameters_and_weights(source)
    converted = {key: value.to(dtype) for key, value in convert_state_dict(state_dict).items()}

    num_quantizers = None
    codebook_weights = converted.get("feature_extractor.codebook_weights")
    if codebook_weights is not None:
        num_quantizers = codebook_weights.shape[0] // VocosConfig().codebook_size

    return build_config(hyperparameters, num_quantizers=num_quantizers), converted


def write_checkpoint(
    source: str = "mel", directory: str = "vocos-converted", dtype: torch.dtype = torch.float32
) -> VocosConfig:
    r"""
    Reads a published Vocos repository and writes what [`VocosModel.from_pretrained`] reads into `directory`.

    Args:
        source (`str`, *optional*, defaults to `"mel"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.yaml` and a
            `pytorch_model.bin`.
        directory (`str`, *optional*, defaults to `"vocos-converted"`):
            Directory the converted config and weights are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        [`VocosConfig`]: The configuration that was written.
    """
    config, converted = build_model_files(source, dtype=dtype)
    with CheckpointWriter(directory) as writer:
        for key in list(converted):
            writer.add(key, converted.pop(key))
    config.save_pretrained(directory)
    return config


def converted_checkpoint(source: str = "mel", dtype: torch.dtype = torch.float32) -> Path:
    r"""
    Returns a directory holding the converted form of a published Vocos repository, which
    [`~PreTrainedModel.from_pretrained`] reads the ordinary way, converting it the first time it is asked for
    and reusing that conversion afterwards.

    Args:
        source (`str`, *optional*, defaults to `"mel"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.yaml` and a
            `pytorch_model.bin`.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `Path`: The directory holding the converted checkpoint.
    """
    repository = PUBLISHED_CHECKPOINTS.get(source, source)
    parts = [str(source), str(dtype), source_identity(repository, resolve_file(repository, "config.yaml"))]
    return cached_conversion("vocos", parts, lambda directory: write_checkpoint(source, directory, dtype=dtype))


def convert(source: str = "mel", output_dir: str = "vocos-converted", dtype: torch.dtype = torch.float32) -> None:
    r"""
    Converts a published Vocos repository into a directory [`VocosModel.from_pretrained`] can load, for a
    checkpoint that is shipped elsewhere or kept outside the conversion cache [`converted_checkpoint`] holds.

    Args:
        source (`str`, *optional*, defaults to `"mel"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.yaml` and a
            `pytorch_model.bin`.
        output_dir (`str`, *optional*, defaults to `"vocos-converted"`):
            Directory the converted config and weights are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    config = write_checkpoint(source, output_dir, dtype=dtype)
    if config.feature_extractor_type == "mel":
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
