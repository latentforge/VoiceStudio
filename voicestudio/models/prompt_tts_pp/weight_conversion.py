"""Checkpoint conversion for PromptTTS++."""

import pickle
import re
import types
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .configuration_prompt_tts_pp import PromptTTSPPBigVGanConfig, PromptTTSPPConfig
from .feature_extraction_prompt_tts_pp import PromptTTSPPFeatureExtractor
from .processing_prompt_tts_pp import PromptTTSPPProcessor
from .tokenization_prompt_tts_pp import PromptTTSPPTokenizer


# The Space that ships the only public PromptTTS++ weights, and the three files of it the conversion reads.
DEFAULT_REPO_ID = "line-corporation/promptttspp"
DEFAULT_MODEL_FILE = "pretrained_model/checkpoint/proposed/last.ckpt"
DEFAULT_VOCODER_FILE = "pretrained_model/checkpoint/bigvgan_f0_full/last.ckpt"
DEFAULT_STATS_FILE = "pretrained_model/checkpoint/stats.yaml"

PROMPT_ENCODER_ID = "bert-base-uncased"

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


def convert(
    checkpoint_path: str | None = None,
    output_dir: str = "prompt_tts_pp",
    vocoder_checkpoint_path: str | None = None,
    stats_path: str | None = None,
    rel_pos_type: str = "legacy",
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Converts the released PromptTTS++ checkpoints into a directory
    [`PromptTTSPPForConditionalGeneration.from_pretrained`] and [`PromptTTSPPProcessor.from_pretrained`] can
    load, with the vocoder written to its own `vocoder` subdirectory.

    Args:
        checkpoint_path (`str`, *optional*):
            Path of the acoustic model's `last.ckpt`. Defaults to the one bundled in the
            `line-corporation/promptttspp` Space.
        output_dir (`str`, *optional*, defaults to `"prompt_tts_pp"`):
            Directory the converted config, weights, tokenizers and processor files are written to.
        vocoder_checkpoint_path (`str`, *optional*):
            Path of the vocoder's `last.ckpt`. Defaults to the one bundled in the same Space.
        stats_path (`str`, *optional*):
            Path of the `stats.yaml` holding the mel spectrogram statistics of the training set. Defaults to the
            one bundled in the same Space.
        rel_pos_type (`str`, *optional*, defaults to `"legacy"`):
            Relative positional encoding variant of the checkpoint, see [`build_config`].
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(DEFAULT_REPO_ID, DEFAULT_MODEL_FILE, repo_type="space")
    if vocoder_checkpoint_path is None:
        vocoder_checkpoint_path = hf_hub_download(DEFAULT_REPO_ID, DEFAULT_VOCODER_FILE, repo_type="space")
    if stats_path is None:
        stats_path = hf_hub_download(DEFAULT_REPO_ID, DEFAULT_STATS_FILE, repo_type="space")

    output_path = Path(output_dir)
    vocoder_path = output_path / "vocoder"
    output_path.mkdir(parents=True, exist_ok=True)
    vocoder_path.mkdir(parents=True, exist_ok=True)

    stats = yaml.safe_load(Path(stats_path).read_text())

    config = build_config(rel_pos_type=rel_pos_type)
    converted = convert_state_dict(load_upstream_checkpoint(checkpoint_path, "model"))
    converted = {key: value.to(dtype).contiguous() for key, value in converted.items()}
    config.save_pretrained(output_path)
    save_file(converted, str(output_path / "model.safetensors"), metadata={"format": "pt"})

    vocoder_config = build_vocoder_config()
    vocoder_converted = convert_vocoder_state_dict(load_upstream_checkpoint(vocoder_checkpoint_path, "generator"))
    vocoder_converted = {key: value.to(dtype).contiguous() for key, value in vocoder_converted.items()}
    vocoder_config.save_pretrained(vocoder_path)
    save_file(vocoder_converted, str(vocoder_path / "model.safetensors"), metadata={"format": "pt"})

    processor = PromptTTSPPProcessor(
        feature_extractor=PromptTTSPPFeatureExtractor(mel_mean=stats["mean"], mel_std=stats["std"]),
        tokenizer=PromptTTSPPTokenizer(),
        prompt_tokenizer=AutoTokenizer.from_pretrained(PROMPT_ENCODER_ID),
    )
    processor.save_pretrained(output_path)


__all__ = [
    "build_config",
    "build_vocoder_config",
    "convert",
    "convert_state_dict",
    "convert_vocoder_state_dict",
    "load_upstream_checkpoint",
]
