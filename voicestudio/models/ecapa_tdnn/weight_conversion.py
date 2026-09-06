# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# Copyright 2024 SpeechBrain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checkpoint conversion for ECAPA-TDNN."""

import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

from ...utils.checkpoint_cache import CheckpointWriter, cached_conversion, source_identity
from .configuration_ecapa_tdnn import EcapaTdnnConfig
from .feature_extraction_ecapa_tdnn import EcapaTdnnFeatureExtractor


PUBLISHED_CHECKPOINT = "speechbrain/spkrec-ecapa-voxceleb"

EMBEDDING_FILE = "embedding_model.ckpt"
CLASSIFIER_FILE = "classifier.ckpt"
NORMALIZATION_FILE = "mean_var_norm_emb.ckpt"
LABEL_FILE = "label_encoder.txt"

# SpeechBrain wraps every convolution and normalization in a module of its own, so each carries the wrapper's
# name as well as its own. These collapse the pair down to the single module this port holds.
_RENAMES = (
    (re.compile(r"\.conv\.conv\."), ".conv."),
    (re.compile(r"\.norm\.norm\."), ".norm."),
    (re.compile(r"\.se_block\.(conv[12])\.conv\."), r".se_block.\1."),
    (re.compile(r"\.shortcut\.conv\."), ".shortcut."),
    (re.compile(r"^asp_bn\.norm\."), "asp_bn."),
    (re.compile(r"^fc\.conv\."), "fc."),
)

_LABEL_LINE = re.compile(r"^'(?P<label>.+)' => (?P<index>\d+)$")


def is_published_layout(source: str) -> bool:
    r"""
    Returns whether `source` is a published SpeechBrain repository rather than a directory [`convert`] wrote.

    A published repository does carry a `config.json`, but it holds SpeechBrain's own one-key interface
    declaration rather than a `transformers` configuration, so the discriminator is a `config.json` declaring
    this model's `model_type`. `PreTrainedConfig.from_pretrained` draws no such distinction of its own: it
    absorbs the foreign schema as extra attributes and falls back to its defaults for everything else.

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
        return json.load(handle).get("model_type") != EcapaTdnnConfig.model_type


def resolve_file(source: str, filename: str) -> str:
    r"""
    Args:
        source (`str`):
            Repository id or local directory holding the published layout.
        filename (`str`):
            Name of the file to read out of it.

    Returns:
        `str`: Local path of that file.
    """
    local = Path(source) / filename
    return str(local) if local.exists() else hf_hub_download(source, filename)


def convert_key(key: str) -> str:
    r"""
    Renames one key of a published `embedding_model.ckpt` to the name [`EcapaTdnnModel`] holds the same tensor
    under.

    Args:
        key (`str`):
            Key of the upstream `ECAPA_TDNN` state dict.

    Returns:
        `str`: The corresponding key of an [`EcapaTdnnModel`] state dict.
    """
    for pattern, replacement in _RENAMES:
        key = pattern.sub(replacement, key)
    return key


def read_labels(source: str) -> dict[int, str]:
    r"""
    Args:
        source (`str`):
            Repository id or local directory holding the published layout.

    Returns:
        `dict[int, str]`: Speaker identifier of each classifier output, by index.
    """
    labels = {}
    with open(resolve_file(source, LABEL_FILE), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            # A rule of equals signs closes the table; what follows is SpeechBrain's own encoder state, whose
            # `'starting_index' => 0` line reads as a label for speaker 0.
            if line.startswith("=="):
                break
            match = _LABEL_LINE.match(line)
            if match:
                labels[int(match.group("index"))] = match.group("label")
    return labels


def build_config(state_dict: dict[str, torch.Tensor], num_labels: int, labels: dict[int, str]) -> EcapaTdnnConfig:
    r"""
    Reads the shape of the published weights into a configuration, since the published `hyperparams.yaml` is a
    SpeechBrain recipe rather than a schema anything here can load.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The renamed embedding network weights.
        num_labels (`int`):
            Number of speakers the classification head scores.
        labels (`dict[int, str]`):
            Speaker identifier of each of those outputs.

    Returns:
        [`EcapaTdnnConfig`]: The configuration those weights load into.

    Raises:
        ValueError: If the weights hold no residual block, so that the stage layout cannot be read off them.
    """
    stem = state_dict["blocks.0.conv.weight"]
    blocks = sorted({int(key.split(".")[1]) for key in state_dict if key.startswith("blocks.")})
    if len(blocks) < 2:
        raise ValueError(f"Expected at least one residual block, found {len(blocks)} stages.")

    channels = [stem.shape[0]]
    kernel_sizes = [stem.shape[2]]
    for index in blocks[1:]:
        channels.append(state_dict[f"blocks.{index}.tdnn1.conv.weight"].shape[0])
        kernel_sizes.append(state_dict[f"blocks.{index}.res2net_block.blocks.0.conv.weight"].shape[2])
    channels.append(state_dict["mfa.conv.weight"].shape[0])
    kernel_sizes.append(state_dict["mfa.conv.weight"].shape[2])

    attention_in = state_dict["asp.tdnn.conv.weight"].shape[1]
    return EcapaTdnnConfig(
        num_mel_bins=stem.shape[1],
        channels=channels,
        kernel_sizes=kernel_sizes,
        # The dilations widen the residual blocks' receptive fields without changing any weight's shape, so they
        # are the one stage parameter the published weights do not carry.
        dilations=EcapaTdnnConfig().dilations[: len(channels)],
        attention_channels=state_dict["asp.tdnn.conv.weight"].shape[0],
        res2net_scale=len(
            [key for key in state_dict if re.match(r"^blocks\.1\.res2net_block\.blocks\.\d+\.conv\.weight$", key)]
        )
        + 1,
        se_channels=state_dict["blocks.1.se_block.conv1.weight"].shape[0],
        global_context=attention_in == channels[-1] * 3,
        xvector_output_dim=state_dict["fc.weight"].shape[0],
        num_labels=num_labels,
        id2label={index: label for index, label in sorted(labels.items())},
        label2id={label: index for index, label in labels.items()},
    )


def build_feature_extractor(source: str = PUBLISHED_CHECKPOINT) -> EcapaTdnnFeatureExtractor:
    r"""
    Args:
        source (`str`, *optional*, defaults to `"speechbrain/spkrec-ecapa-voxceleb"`):
            Repository id or local directory holding the published layout.

    Returns:
        [`EcapaTdnnFeatureExtractor`]: An extractor matching the filterbank width the checkpoint expects.
    """
    state_dict = torch.load(resolve_file(source, EMBEDDING_FILE), map_location="meta", weights_only=True)
    return EcapaTdnnFeatureExtractor(feature_size=state_dict["blocks.0.conv.conv.weight"].shape[1])


def write_checkpoint(
    source: str = PUBLISHED_CHECKPOINT,
    directory: str = "ecapa-tdnn-converted",
    dtype: torch.dtype = torch.float32,
) -> EcapaTdnnConfig:
    r"""
    Reads a published SpeechBrain repository and writes what [`EcapaTdnnForXVector.from_pretrained`] reads into
    `directory`.

    Args:
        source (`str`, *optional*, defaults to `"speechbrain/spkrec-ecapa-voxceleb"`):
            Repository id or local directory holding the published layout.
        directory (`str`, *optional*, defaults to `"ecapa-tdnn-converted"`):
            Directory the converted config, feature extractor and weights are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        [`EcapaTdnnConfig`]: The configuration that was written.
    """
    embedding = torch.load(resolve_file(source, EMBEDDING_FILE), map_location="cpu", weights_only=True)
    embedding = {convert_key(key): value.to(dtype) for key, value in embedding.items()}
    classifier = torch.load(resolve_file(source, CLASSIFIER_FILE), map_location="cpu", weights_only=True)
    normalization = torch.load(resolve_file(source, NORMALIZATION_FILE), map_location="cpu", weights_only=True)

    labels = read_labels(source)
    config = build_config(embedding, classifier["weight"].shape[0], labels)

    with CheckpointWriter(directory) as writer:
        for key in list(embedding):
            writer.add(f"ecapa_tdnn.{key}", embedding.pop(key))
        writer.add("classifier", classifier["weight"].to(dtype))
        writer.add("embedding_mean", normalization["glob_mean"].to(dtype))

    config.save_pretrained(directory)
    build_feature_extractor(source).save_pretrained(directory)
    return config


def converted_checkpoint(source: str = PUBLISHED_CHECKPOINT, dtype: torch.dtype = torch.float32) -> Path:
    r"""
    Returns a directory holding the converted form of a published SpeechBrain repository, which
    [`~PreTrainedModel.from_pretrained`] reads the ordinary way, converting it the first time it is asked for and
    reusing that conversion afterwards.

    Args:
        source (`str`, *optional*, defaults to `"speechbrain/spkrec-ecapa-voxceleb"`):
            Repository id or local directory holding the published layout.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `Path`: The directory holding the converted checkpoint.
    """
    parts = [str(source), str(dtype), source_identity(source, resolve_file(source, EMBEDDING_FILE))]
    return cached_conversion(
        "ecapa_tdnn", parts, lambda directory: write_checkpoint(source, directory, dtype=dtype)
    )


def convert(
    source: str = PUBLISHED_CHECKPOINT,
    output_dir: str = "ecapa-tdnn-converted",
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Converts a published SpeechBrain repository into a directory [`EcapaTdnnForXVector.from_pretrained`] can
    load, for a checkpoint that is to be shipped elsewhere or kept outside the conversion cache
    [`converted_checkpoint`] holds.

    Args:
        source (`str`, *optional*, defaults to `"speechbrain/spkrec-ecapa-voxceleb"`):
            Repository id or local directory holding the published layout.
        output_dir (`str`, *optional*, defaults to `"ecapa-tdnn-converted"`):
            Directory the converted config, feature extractor and weights are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    write_checkpoint(source, output_dir, dtype=dtype)


__all__ = [
    "CLASSIFIER_FILE",
    "EMBEDDING_FILE",
    "LABEL_FILE",
    "NORMALIZATION_FILE",
    "PUBLISHED_CHECKPOINT",
    "build_config",
    "build_feature_extractor",
    "convert",
    "convert_key",
    "converted_checkpoint",
    "is_published_layout",
    "read_labels",
    "resolve_file",
]
