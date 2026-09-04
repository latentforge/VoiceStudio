"""Checkpoint conversion for PromptTTS++."""

import json
import pickle
import re
import types
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

from ...utils.checkpoint_cache import CheckpointWriter, cached_conversion, file_identity
from .configuration_prompt_tts_pp import PromptTTSPPBigVGanConfig, PromptTTSPPConfig
from .feature_extraction_prompt_tts_pp import PromptTTSPPFeatureExtractor
from .processing_prompt_tts_pp import PromptTTSPPProcessor
from .tokenization_prompt_tts_pp import PromptTTSPPTokenizer


# The Space that ships the only public PromptTTS++ weights, and the three files of it the loaders read. No model
# repository holds them, so the Space is the repository id `from_pretrained` takes.
DEFAULT_REPO_ID = "line-corporation/promptttspp"
DEFAULT_MODEL_FILE = "pretrained_model/checkpoint/proposed/last.ckpt"
DEFAULT_VOCODER_FILE = "pretrained_model/checkpoint/bigvgan_f0_full/last.ckpt"
DEFAULT_STATS_FILE = "pretrained_model/checkpoint/stats.yaml"

PROMPT_ENCODER_ID = "bert-base-uncased"

# Directory of a converted checkpoint the vocoder is written to, which is where [`PromptTTSPPBigVGan`] reads it
# back from.
VOCODER_SUBFOLDER = "vocoder"

_ENCODER_RENAMES = {
    "norm_ff_macaron": "ff_macaron_layer_norm",
    "norm_mha": "self_attn_layer_norm",
    "norm_conv": "conv_layer_norm",
    "norm_final": "final_layer_norm",
    "norm_ff": "ff_layer_norm",
    "feed_forward.w_1": "feed_forward.conv1",
    "feed_forward.w_2": "feed_forward.conv2",
    "feed_forward_macaron.w_1": "feed_forward_macaron.conv1",
    "feed_forward_macaron.w_2": "feed_forward_macaron.conv2",
}

_MODEL_PREFIX_RENAMES = {
    "phoneme_emb.emb.": "model.phoneme_embedding.",
    "encoder.encoder.encoders.": "model.encoder.layers.",
    "encoder.encoder.after_norm.": "model.encoder.after_norm.",
    "variance_adaptor.pitch_emb.": "model.variance_adaptor.pitch_embed.",
    "variance_adaptor.": "model.variance_adaptor.",
    "reference_encoder.ref_enc.": "model.style_encoder.reference_encoder.",
    "reference_encoder.stl.gst_embs": "model.style_encoder.style_tokens",
    "reference_encoder.stl.mha.": "model.style_encoder.attention.",
    "prompt_encoder.bert.model.": "model.prompt_encoder.bert.",
    "prompt_encoder.adaptor.": "model.prompt_encoder.adapter.",
    "style_mdn.": "model.style_mdn.",
    "decoder.": "decoder.",
}

_VOCODER_PREFIX_RENAMES = {
    "m_source.l_linear.": "source.linear.",
    "upsamples.": "upsampler.",
    "mrfs.": "resblocks.",
    "act_post.act.alpha": "post_activation.alpha",
}


class _Placeholder:
    """Stands in for a class the checkpoint pickles that this package does not depend on."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    """Unpickler that resolves globals it cannot import to [`_Placeholder`]."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (_Placeholder,), {})


def load_upstream_checkpoint(path: str, key: str) -> dict[str, torch.Tensor]:
    r"""
    Reads one state dict out of a checkpoint the upstream trainer wrote.

    The trainer saves the optimizer state next to the weights, and that state pickles the Hydra configuration
    objects of the training run, so the tensors are read through an unpickler that stands those objects in.

    Args:
        path (`str`):
            Path of the `.ckpt` file.
        key (`str`):
            Entry of the checkpoint holding the state dict, `"model"` or `"generator"`.

    Returns:
        `dict[str, torch.Tensor]`: The state dict.
    """
    pickle_module = types.ModuleType("prompt_tts_pp_pickle")
    pickle_module.Unpickler = _Unpickler
    pickle_module.load = lambda file, **kwargs: _Unpickler(file, **kwargs).load()
    checkpoint = torch.load(path, map_location="cpu", pickle_module=pickle_module, weights_only=False)
    return checkpoint[key]


def is_published_layout(source, model_type: str) -> bool:
    r"""
    Returns whether `source` is the published Space rather than a directory [`convert`] wrote.

    The Space holds the upstream trainer's own `.ckpt` files under `pretrained_model/checkpoint/` and no
    `config.json` anywhere, so the discriminator is a `config.json` declaring the caller's own `model_type`.
    `PreTrainedConfig.from_pretrained` draws no such distinction of its own: `cached_file` returns `None` for
    the missing file and the configuration silently falls back to its defaults.

    Args:
        source (`str` or `os.PathLike`):
            [`DEFAULT_REPO_ID`], or a repository id or local directory.
        model_type (`str`):
            `model_type` of the configuration the caller loads, which a converted directory declares.

    Returns:
        `bool`: Whether `source` holds the published layout.
    """
    if str(source) == DEFAULT_REPO_ID:
        return True
    config_file = cached_file(
        source,
        CONFIG_NAME,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if config_file is None:
        return True
    return json.loads(Path(config_file).read_text()).get("model_type") != model_type


def resolve_file(source, filename: str) -> str:
    r"""
    Args:
        source (`str` or `os.PathLike`):
            Repository id of the Space, or a local directory holding a copy of its tree.
        filename (`str`):
            Path of the file inside the Space or the directory.

    Returns:
        `str`: Local path of the file, downloading it from the Space if `source` is a repository id.
    """
    path = Path(source) / filename
    if path.is_file():
        return str(path)
    return hf_hub_download(str(source), filename, repo_type="space")


def build_config(rel_pos_type: str = "legacy") -> PromptTTSPPConfig:
    r"""
    Builds the [`PromptTTSPPConfig`] of the released checkpoint.

    Args:
        rel_pos_type (`str`, *optional*, defaults to `"legacy"`):
            Relative positional encoding variant. The demo configuration the released checkpoint ships with sets
            `"legacy"`, while the training configuration of the paper sets `"new"`. The two hold identical
            parameters, so the wrong one still loads and only degrades the output.

    Returns:
        [`PromptTTSPPConfig`]: The configuration.
    """
    return PromptTTSPPConfig(rel_pos_type=rel_pos_type)


def build_vocoder_config() -> PromptTTSPPBigVGanConfig:
    r"""
    Builds the [`PromptTTSPPBigVGanConfig`] of the released vocoder checkpoint.

    Returns:
        [`PromptTTSPPBigVGanConfig`]: The configuration.
    """
    return PromptTTSPPBigVGanConfig()


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Renames the acoustic model's tensors onto [`PromptTTSPPForConditionalGeneration`].

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The `"model"` entry of an upstream checkpoint.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.

    Raises:
        ValueError: If a tensor has no destination in the migrated model.
    """
    converted = {}
    for key, value in state_dict.items():
        name = key
        for old, new in _MODEL_PREFIX_RENAMES.items():
            if name.startswith(old):
                name = new + name[len(old) :]
                break
        else:
            raise ValueError(f"The published checkpoint holds a tensor this model has no destination for: {key}")

        if name.startswith("model.encoder.layers."):
            for old, new in _ENCODER_RENAMES.items():
                name = name.replace(f".{old}.", f".{new}.")

        # The variance adaptor and the frame prior network normalize over the channel dimension of a
        # `(batch_size, channels, sequence_length)` tensor with parameters shaped to broadcast against it, which
        # `nn.LayerNorm` over the transposed tensor expresses with flat parameters.
        if name.endswith(".gamma"):
            name = name[: -len(".gamma")] + ".weight"
            value = value.reshape(-1)
        elif name.endswith(".beta"):
            name = name[: -len(".beta")] + ".bias"
            value = value.reshape(-1)

        converted[name] = value
    return converted


def convert_vocoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Renames the vocoder's tensors onto [`PromptTTSPPBigVGan`] and folds their weight norm reparameterization
    back into plain weights.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The `"generator"` entry of an upstream vocoder checkpoint.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.
    """
    folded = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_g"):
            stem = key[: -len(".weight_g")]
            direction = state_dict[f"{stem}.weight_v"]
            norm = direction.pow(2).sum(dim=tuple(range(1, direction.dim())), keepdim=True).sqrt()
            folded[f"{stem}.weight"] = value * direction / norm
        elif key.endswith(".weight_v"):
            continue
        elif key.endswith(".filter"):
            # The anti aliasing filters are a deterministic function of the configuration.
            continue
        else:
            folded[key] = value

    converted = {}
    for key, value in folded.items():
        name = key
        for old, new in _VOCODER_PREFIX_RENAMES.items():
            if name.startswith(old):
                name = new + name[len(old) :]
                break
        name = re.sub(r"\.act(\d)\.act\.alpha$", r".activation\1.alpha", name)
        if name.endswith(".alpha"):
            # The snake activation keeps one `alpha` per channel, which upstream shapes to broadcast against a
            # `(batch_size, channels, sequence_length)` tensor.
            value = value.reshape(-1)
        converted[name] = value
    return converted


def build_model_files(
    source=DEFAULT_REPO_ID, rel_pos_type: str = "legacy", dtype: torch.dtype = torch.float32
) -> tuple[PromptTTSPPConfig, dict[str, torch.Tensor]]:
    r"""
    Reads the released acoustic model out of the Space and returns what
    [`PromptTTSPPForConditionalGeneration`] needs to load it.

    Args:
        source (`str` or `os.PathLike`, *optional*, defaults to [`DEFAULT_REPO_ID`]):
            Repository id of the Space, or a local directory holding a copy of its tree.
        rel_pos_type (`str`, *optional*, defaults to `"legacy"`):
            Relative positional encoding variant of the checkpoint, see [`build_config`].
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `tuple[PromptTTSPPConfig, dict[str, torch.Tensor]]`: The configuration and the renamed tensors.
    """
    config = build_config(rel_pos_type=rel_pos_type)
    converted = convert_state_dict(load_upstream_checkpoint(resolve_file(source, DEFAULT_MODEL_FILE), "model"))
    return config, {key: value.to(dtype).contiguous() for key, value in converted.items()}


def build_vocoder_files(
    source=DEFAULT_REPO_ID, dtype: torch.dtype = torch.float32
) -> tuple[PromptTTSPPBigVGanConfig, dict[str, torch.Tensor]]:
    r"""
    Reads the released f0 aware vocoder out of the Space and returns what [`PromptTTSPPBigVGan`] needs to load
    it.

    Args:
        source (`str` or `os.PathLike`, *optional*, defaults to [`DEFAULT_REPO_ID`]):
            Repository id of the Space, or a local directory holding a copy of its tree.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `tuple[PromptTTSPPBigVGanConfig, dict[str, torch.Tensor]]`: The configuration and the renamed tensors.
    """
    config = build_vocoder_config()
    converted = convert_vocoder_state_dict(
        load_upstream_checkpoint(resolve_file(source, DEFAULT_VOCODER_FILE), "generator")
    )
    return config, {key: value.to(dtype).contiguous() for key, value in converted.items()}


def build_processor(source=DEFAULT_REPO_ID, phonemize: bool = True) -> PromptTTSPPProcessor:
    r"""
    Builds the [`PromptTTSPPProcessor`] of the released checkpoint, over the mel spectrogram statistics of the
    training set the Space ships beside the weights.

    Args:
        source (`str` or `os.PathLike`, *optional*, defaults to [`DEFAULT_REPO_ID`]):
            Repository id of the Space, or a local directory holding a copy of its tree.
        phonemize (`bool`, *optional*, defaults to `True`):
            Whether the phoneme tokenizer runs grapheme to phoneme conversion, which needs the `g2p_en` backend.
            Pass `False` to hand it a whitespace separated phoneme sequence instead.

    Returns:
        [`PromptTTSPPProcessor`]: The processor.
    """
    stats = yaml.safe_load(Path(resolve_file(source, DEFAULT_STATS_FILE)).read_text())
    return PromptTTSPPProcessor(
        feature_extractor=PromptTTSPPFeatureExtractor(mel_mean=stats["mean"], mel_std=stats["std"]),
        tokenizer=PromptTTSPPTokenizer(phonemize=phonemize),
        prompt_tokenizer=AutoTokenizer.from_pretrained(PROMPT_ENCODER_ID),
    )


def write_checkpoint(
    source=DEFAULT_REPO_ID,
    directory: str = "prompt_tts_pp",
    rel_pos_type: str = "legacy",
    dtype: torch.dtype = torch.float32,
) -> PromptTTSPPConfig:
    r"""
    Reads the released acoustic model out of the Space and writes what
    [`PromptTTSPPForConditionalGeneration.from_pretrained`] reads into `directory`.

    Args:
        source (`str` or `os.PathLike`, *optional*, defaults to [`DEFAULT_REPO_ID`]):
            Repository id of the Space, or a local directory holding a copy of its tree.
        directory (`str`, *optional*, defaults to `"prompt_tts_pp"`):
            Directory the converted config and weights are written to.
        rel_pos_type (`str`, *optional*, defaults to `"legacy"`):
            Relative positional encoding variant of the checkpoint, see [`build_config`].
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        [`PromptTTSPPConfig`]: The configuration that was written.
    """
    config, converted = build_model_files(source, rel_pos_type=rel_pos_type, dtype=dtype)
    with CheckpointWriter(directory) as writer:
        for key in list(converted):
            writer.add(key, converted.pop(key))
    config.save_pretrained(directory)
    return config


def write_vocoder_checkpoint(
    source=DEFAULT_REPO_ID, directory: str = "prompt_tts_pp/vocoder", dtype: torch.dtype = torch.float32
) -> PromptTTSPPBigVGanConfig:
    r"""
    Reads the released f0 aware vocoder out of the Space and writes what
    [`PromptTTSPPBigVGan.from_pretrained`] reads into `directory`.

    Args:
        source (`str` or `os.PathLike`, *optional*, defaults to [`DEFAULT_REPO_ID`]):
            Repository id of the Space, or a local directory holding a copy of its tree.
        directory (`str`, *optional*, defaults to `"prompt_tts_pp/vocoder"`):
            Directory the converted config and weights are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        [`PromptTTSPPBigVGanConfig`]: The configuration that was written.
    """
    config, converted = build_vocoder_files(source, dtype=dtype)
    with CheckpointWriter(directory) as writer:
        for key in list(converted):
            writer.add(key, converted.pop(key))
    config.save_pretrained(directory)
    return config


def converted_checkpoint(
    source=DEFAULT_REPO_ID, rel_pos_type: str = "legacy", dtype: torch.dtype = torch.float32
) -> Path:
    r"""
    Returns a directory holding the converted form of the released acoustic model, with the released vocoder in
    its [`VOCODER_SUBFOLDER`] subdirectory, which [`~PreTrainedModel.from_pretrained`] reads the ordinary way,
    converting them the first time either is asked for and reusing that conversion afterwards.

    Both networks come out of one Space and are converted in one pass, since the pass drops the Space's
    checkpoints from the `huggingface_hub` cache once it has read them.

    Args:
        source (`str` or `os.PathLike`, *optional*, defaults to [`DEFAULT_REPO_ID`]):
            Repository id of the Space, or a local directory holding a copy of its tree.
        rel_pos_type (`str`, *optional*, defaults to `"legacy"`):
            Relative positional encoding variant of the checkpoint, see [`build_config`].
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `Path`: The directory holding the converted checkpoint.
    """
    # A downloaded file is named by the revision every file of the Space shares, so the smallest of them names
    # the two checkpoints as well and both are read inside the conversion. A local copy is named by the file
    # itself and downloads nothing, so there it is the checkpoint that is named.
    identity_file = DEFAULT_MODEL_FILE if Path(source).is_dir() else DEFAULT_STATS_FILE
    parts = [str(source), rel_pos_type, str(dtype), file_identity(resolve_file(source, identity_file))]

    def write(directory) -> None:
        write_checkpoint(source, directory, rel_pos_type=rel_pos_type, dtype=dtype)
        write_vocoder_checkpoint(source, Path(directory) / VOCODER_SUBFOLDER, dtype=dtype)

    return cached_conversion("prompt_tts_pp", parts, write)


def converted_vocoder_checkpoint(source=DEFAULT_REPO_ID, dtype: torch.dtype = torch.float32) -> Path:
    r"""
    Returns a directory holding the converted form of the released f0 aware vocoder, which
    [`~PreTrainedModel.from_pretrained`] reads the ordinary way. It is the [`VOCODER_SUBFOLDER`] subdirectory of
    the conversion [`converted_checkpoint`] holds, which covers both released networks.

    Args:
        source (`str` or `os.PathLike`, *optional*, defaults to [`DEFAULT_REPO_ID`]):
            Repository id of the Space, or a local directory holding a copy of its tree.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `Path`: The directory holding the converted vocoder.
    """
    return converted_checkpoint(source, dtype=dtype) / VOCODER_SUBFOLDER


def convert(
    source=DEFAULT_REPO_ID,
    output_dir: str = "prompt_tts_pp",
    rel_pos_type: str = "legacy",
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Writes the released PromptTTS++ checkpoints into a directory of its own, with the vocoder in its `vocoder`
    subdirectory, which [`PromptTTSPPForConditionalGeneration.from_pretrained`],
    [`PromptTTSPPBigVGan.from_pretrained`] and [`PromptTTSPPProcessor.from_pretrained`] read without reaching
    the Space again, for a checkpoint that is shipped elsewhere or kept outside the conversion cache
    [`converted_checkpoint`] holds.

    Args:
        source (`str` or `os.PathLike`, *optional*, defaults to [`DEFAULT_REPO_ID`]):
            Repository id of the Space, or a local directory holding a copy of its tree.
        output_dir (`str`, *optional*, defaults to `"prompt_tts_pp"`):
            Directory the converted config, weights, tokenizers and processor files are written to.
        rel_pos_type (`str`, *optional*, defaults to `"legacy"`):
            Relative positional encoding variant of the checkpoint, see [`build_config`].
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    output_path = Path(output_dir)
    write_checkpoint(source, output_path, rel_pos_type=rel_pos_type, dtype=dtype)
    write_vocoder_checkpoint(source, output_path / VOCODER_SUBFOLDER, dtype=dtype)
    build_processor(source).save_pretrained(output_path)


__all__ = [
    "DEFAULT_MODEL_FILE",
    "DEFAULT_REPO_ID",
    "DEFAULT_STATS_FILE",
    "DEFAULT_VOCODER_FILE",
    "VOCODER_SUBFOLDER",
    "build_config",
    "build_model_files",
    "build_processor",
    "build_vocoder_config",
    "build_vocoder_files",
    "convert",
    "convert_state_dict",
    "convert_vocoder_state_dict",
    "converted_checkpoint",
    "converted_vocoder_checkpoint",
    "is_published_layout",
    "load_upstream_checkpoint",
    "resolve_file",
    "write_checkpoint",
    "write_vocoder_checkpoint",
]
