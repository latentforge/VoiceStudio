"""Checkpoint conversion for Vocos."""

from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

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


def load_hyperparameters_and_weights(source: str) -> tuple[dict, dict[str, torch.Tensor]]:
    r"""
    Reads the `config.yaml` and `pytorch_model.bin` of a Vocos repository or local directory.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding both files.

    Returns:
        `tuple[dict, dict[str, torch.Tensor]]`: The parsed configuration and the checkpoint's tensors.
    """
    source = PUBLISHED_CHECKPOINTS.get(source, source)
    if Path(source).is_dir():
        config_file = str(Path(source) / "config.yaml")
        weights_file = str(Path(source) / "pytorch_model.bin")
    else:
        config_file = hf_hub_download(source, "config.yaml")
        weights_file = hf_hub_download(source, "pytorch_model.bin")

    with open(config_file, "r", encoding="utf-8") as handle:
        hyperparameters = yaml.safe_load(handle)
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


def convert(source: str = "mel", output_dir: str = "vocos-converted", dtype: torch.dtype = torch.float32) -> None:
    r"""
    Converts a published Vocos repository into a directory [`VocosModel.from_pretrained`] can load.

    Args:
        source (`str`, *optional*, defaults to `"mel"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding a `config.yaml` and a
            `pytorch_model.bin`.
        output_dir (`str`, *optional*, defaults to `"vocos-converted"`):
            Directory the converted config and weights are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config, converted = build_model_files(source, dtype=dtype)
    config.save_pretrained(output_path)
    save_file(converted, str(output_path / "model.safetensors"), metadata={"format": "pt"})

    if config.feature_extractor_type == "mel":
        VocosFeatureExtractor(
            feature_size=config.input_channels,
            sampling_rate=config.sampling_rate,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            padding=config.padding,
        ).save_pretrained(output_path)


__all__ = [
    "PUBLISHED_CHECKPOINTS",
    "build_config",
    "build_model_files",
    "convert",
    "convert_state_dict",
    "load_hyperparameters_and_weights",
]
