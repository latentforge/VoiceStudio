# MIT License
#
# Copyright (c) 2024 sarulab-speech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Checkpoint conversion for UTMOSv2."""

import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

from ...utils.checkpoint_cache import CheckpointWriter, cached_conversion, source_identity
from .configuration_utmos_v2 import EFFICIENTNET_V2_S_STAGES, UTMOSv2Config
from .feature_extraction_utmos_v2 import UTMOSv2FeatureExtractor


PUBLISHED_CHECKPOINT = "sarulab-speech/UTMOSv2"

# The seed behind the published weights, which is part of every file name in the repository.
PUBLISHED_SEED = 42

# Leaf module of an EfficientNetV2 block under its timm name, keyed by the kind of block the stage is built from.
_BLOCK_NAMES = {
    "conv": {"conv": "conv", "bn1": "bn"},
    "fused": {
        "conv_exp": "conv_expand",
        "bn1": "bn_expand",
        "conv_pwl": "conv_project",
        "bn2": "bn_project",
    },
    "mbconv": {
        "conv_pw": "conv_expand",
        "bn1": "bn_expand",
        "conv_dw": "conv_depthwise",
        "bn2": "bn_depthwise",
        "conv_pwl": "conv_project",
        "bn3": "bn_project",
    },
}

# Leaf module of an EfficientNetV2 stem and head under its timm name.
_ENCODER_NAMES = {"conv_stem": "conv_stem", "bn1": "bn_stem", "conv_head": "conv_head", "bn2": "bn_head"}

# Weight normalization of the wav2vec 2.0 positional convolution, under the names `torch.nn.utils.weight_norm`
# gave its two factors before `torch.nn.utils.parametrizations` took over.
_WEIGHT_NORM_NAMES = {"weight_g": "parametrizations.weight.original0", "weight_v": "parametrizations.weight.original1"}

_BLOCK_KEY = re.compile(r"^spec_long\.backbones\.(\d+)\.blocks\.(\d+)\.(\d+)\.(.+)$")
_ENCODER_KEY = re.compile(r"^spec_long\.backbones\.(\d+)\.([^.]+)\.(.+)$")


def is_published_layout(source: str) -> bool:
    r"""
    Returns whether `source` is the published UTMOSv2 repository rather than a directory [`convert`] wrote.

    The published repository carries one loose `.pth` per fold and no `config.json` at all, so the discriminator
    is a `config.json` declaring this model's `model_type`. `PreTrainedConfig.from_pretrained` draws no such
    distinction of its own: `cached_file` returns `None` for the missing file and the configuration silently
    falls back to its defaults.

    Args:
        source (`str`):
            Repository id or local directory.

    Returns:
        `bool`: Whether `source` holds the published layout.
    """
    config_file = cached_file(
        source,
        CONFIG_NAME,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if config_file is None:
        return True
    with open(config_file, "r", encoding="utf-8") as handle:
        return json.load(handle).get("model_type") != UTMOSv2Config.model_type


def fold_file(source: str, fold: int, seed: int = PUBLISHED_SEED) -> str:
    r"""
    Args:
        source (`str`):
            Repository id or local directory holding the published layout.
        fold (`int`):
            Which cross-validation fold to read.
        seed (`int`, *optional*, defaults to 42):
            Seed the fold was trained under, which its file name carries.

    Returns:
        `str`: Local path of that fold's weights.
    """
    name = f"fold{fold}_s{seed}_best_model.pth"
    local = Path(source) / name
    return str(local) if local.exists() else hf_hub_download(source, name)


def convert_key(key: str) -> str:
    r"""
    Renames one key of an upstream fold checkpoint to the name [`UTMOSv2Model`] holds the same tensor under.

    Args:
        key (`str`):
            Key of the upstream `SSLMultiSpecExtModelV2` state dict.

    Returns:
        `str`: The corresponding key of a [`UTMOSv2Model`] state dict.

    Raises:
        ValueError: If the key belongs to no module this model has.
    """
    if key == "fc.weight" or key == "fc.bias":
        return key.replace("fc.", "classifier.")
    if key == "ssl.weights":
        return "ssl_layer_weights"
    if key == "spec_long.weights":
        return "spectrogram_weights"
    if key.startswith("ssl.attn."):
        return key.replace("ssl.attn.", "ssl_attention.")
    if key.startswith("spec_long.attn."):
        return key.replace("spec_long.attn.", "spectrogram_attention.")
    if key.startswith("ssl.encoder.model."):
        rest = key[len("ssl.encoder.model.") :]
        for old, new in _WEIGHT_NORM_NAMES.items():
            if rest.endswith(f".{old}"):
                rest = f"{rest[: -len(old)]}{new}"
        return f"ssl_encoder.{rest}"

    block = _BLOCK_KEY.match(key)
    if block:
        encoder, stage, layer, rest = block.groups()
        leaf, _, tail = rest.partition(".")
        if leaf == "se":
            return f"spectrogram_encoders.{encoder}.blocks.{stage}.{layer}.{rest}"
        names = _BLOCK_NAMES[EFFICIENTNET_V2_S_STAGES[int(stage)][0]]
        return f"spectrogram_encoders.{encoder}.blocks.{stage}.{layer}.{names[leaf]}.{tail}"

    encoder_key = _ENCODER_KEY.match(key)
    if encoder_key:
        encoder, leaf, tail = encoder_key.groups()
        return f"spectrogram_encoders.{encoder}.{_ENCODER_NAMES[leaf]}.{tail}"

    raise ValueError(f"Unknown UTMOSv2 checkpoint key: {key}")


def build_config(num_folds: int = 5) -> UTMOSv2Config:
    r"""
    Args:
        num_folds (`int`, *optional*, defaults to 5):
            Number of folds the converted checkpoint holds.

    Returns:
        [`UTMOSv2Config`]: Configuration of the published `fusion_stage3` model.
    """
    return UTMOSv2Config(num_folds=num_folds)


def write_checkpoint(
    source: str = PUBLISHED_CHECKPOINT,
    directory: str = "utmos-v2-converted",
    num_folds: int = 5,
    seed: int = PUBLISHED_SEED,
    dtype: torch.dtype = torch.float32,
) -> UTMOSv2Config:
    r"""
    Reads the published UTMOSv2 repository and writes what [`UTMOSv2ForAudioClassification.from_pretrained`]
    reads into `directory`, a fold at a time so that no more than one fold is held at once.

    Args:
        source (`str`, *optional*, defaults to `"sarulab-speech/UTMOSv2"`):
            Repository id or local directory holding the published layout.
        directory (`str`, *optional*, defaults to `"utmos-v2-converted"`):
            Directory the converted config, feature extractor and weights are written to.
        num_folds (`int`, *optional*, defaults to 5):
            Number of folds to convert, counting from zero.
        seed (`int`, *optional*, defaults to 42):
            Seed the folds were trained under, which their file names carry.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        [`UTMOSv2Config`]: The configuration that was written.
    """
    config = build_config(num_folds)
    with CheckpointWriter(directory) as writer:
        for fold in range(num_folds):
            state_dict = torch.load(fold_file(source, fold, seed), map_location="cpu", weights_only=True)
            for key in list(state_dict):
                writer.add(f"folds.{fold}.{convert_key(key)}", state_dict.pop(key).to(dtype))
    config.save_pretrained(directory)
    UTMOSv2FeatureExtractor().save_pretrained(directory)
    return config


def converted_checkpoint(
    source: str = PUBLISHED_CHECKPOINT,
    num_folds: int = 5,
    seed: int = PUBLISHED_SEED,
    dtype: torch.dtype = torch.float32,
) -> Path:
    r"""
    Returns a directory holding the converted form of the published UTMOSv2 repository, which
    [`~PreTrainedModel.from_pretrained`] reads the ordinary way, converting it the first time it is asked for and
    reusing that conversion afterwards.

    Args:
        source (`str`, *optional*, defaults to `"sarulab-speech/UTMOSv2"`):
            Repository id or local directory holding the published layout.
        num_folds (`int`, *optional*, defaults to 5):
            Number of folds to convert, counting from zero.
        seed (`int`, *optional*, defaults to 42):
            Seed the folds were trained under, which their file names carry.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `Path`: The directory holding the converted checkpoint.
    """
    parts = [str(source), num_folds, seed, str(dtype), source_identity(source, fold_file(source, 0, seed))]
    return cached_conversion(
        "utmos_v2",
        parts,
        lambda directory: write_checkpoint(source, directory, num_folds=num_folds, seed=seed, dtype=dtype),
    )


def convert(
    source: str = PUBLISHED_CHECKPOINT,
    output_dir: str = "utmos-v2-converted",
    num_folds: int = 5,
    seed: int = PUBLISHED_SEED,
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Converts the published UTMOSv2 repository into a directory
    [`UTMOSv2ForAudioClassification.from_pretrained`] can load, for a checkpoint that is to be shipped elsewhere
    or kept outside the conversion cache [`converted_checkpoint`] holds.

    Args:
        source (`str`, *optional*, defaults to `"sarulab-speech/UTMOSv2"`):
            Repository id or local directory holding the published layout.
        output_dir (`str`, *optional*, defaults to `"utmos-v2-converted"`):
            Directory the converted config, feature extractor and weights are written to.
        num_folds (`int`, *optional*, defaults to 5):
            Number of folds to convert, counting from zero.
        seed (`int`, *optional*, defaults to 42):
            Seed the folds were trained under, which their file names carry.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    write_checkpoint(source, output_dir, num_folds=num_folds, seed=seed, dtype=dtype)


__all__ = [
    "PUBLISHED_CHECKPOINT",
    "PUBLISHED_SEED",
    "build_config",
    "convert",
    "convert_key",
    "converted_checkpoint",
    "fold_file",
    "is_published_layout",
    "write_checkpoint",
]
