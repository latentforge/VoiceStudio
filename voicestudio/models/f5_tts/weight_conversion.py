"""Checkpoint conversion for F5-TTS."""

import re
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from .configuration_f5_tts import F5TTSConfig, F5TTSVocosConfig
from .feature_extraction_f5_tts import F5TTSFeatureExtractor
from .processing_f5_tts import VOCODER_SUBFOLDER, F5TTSProcessor
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

# Repository, weight file and vocabulary file of every checkpoint the F5-TTS authors published, keyed by the
# architecture it was trained with. `F5TTS_Base_bigvgan` is the only one whose mel front end is `"bigvgan"`.
PUBLISHED_CHECKPOINTS = {
    "F5TTS_v1_Base": ("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors", "F5TTS_v1_Base/vocab.txt"),
    "F5TTS_v1_Base_no_zero_init": (
        "SWivid/F5-TTS",
        "F5TTS_v1_Base_no_zero_init/model_1250000.safetensors",
        "F5TTS_v1_Base/vocab.txt",
    ),
    "F5TTS_Base": ("SWivid/F5-TTS", "F5TTS_Base/model_1200000.safetensors", "F5TTS_Base/vocab.txt"),
    "F5TTS_Base_bigvgan": ("SWivid/F5-TTS", "F5TTS_Base_bigvgan/model_1250000.pt", "F5TTS_Base/vocab.txt"),
    "E2TTS_Base": ("SWivid/E2-TTS", "E2TTS_Base/model_1200000.safetensors", "F5TTS_Base/vocab.txt"),
}

VOCOS_REPO_ID = "charactr/vocos-mel-24khz"

# Buffers the EMA wrapper and the mel front end of a `CFM` module keep in the checkpoint and that no parameter
# of this model corresponds to.
_DISCARDED_KEYS = (
    "initted",
    "step",
    "update",
    "mel_spec.mel_stft.mel_scale.fb",
    "mel_spec.mel_stft.spectrogram.window",
)

_UNET_LAYER = re.compile(r"^layers\.(\d+)\.(\d)\.")


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


def build_vocoder_config(hyperparameters: dict) -> F5TTSVocosConfig:
    r"""
    Builds an [`F5TTSVocosConfig`] from a Vocos `config.yaml`.

    Args:
        hyperparameters (`dict`):
            Parsed `config.yaml` of a Vocos repository.

    Returns:
        [`F5TTSVocosConfig`]: The equivalent VoiceStudio configuration.

    Raises:
        ValueError: If the vocoder is not the mel spectrogram conditioned inverse STFT one F5-TTS uses.
    """
    backbone = hyperparameters["backbone"]
    head = hyperparameters["head"]
    if not backbone["class_path"].endswith("VocosBackbone") or not head["class_path"].endswith("ISTFTHead"):
        raise ValueError(
            "F5-TTS decodes mel spectrograms with a `VocosBackbone` plus `ISTFTHead` vocoder, got "
            f"{backbone['class_path']} plus {head['class_path']}."
        )
    if head["init_args"].get("padding", "center") != "center":
        raise ValueError(
            "The inverse STFT of this model pads its frames the way `torch.istft(center=True)` does, which the "
            f"Vocos `padding` setting {head['init_args']['padding']} does not match."
        )

    backbone_args = backbone["init_args"]
    head_args = head["init_args"]
    return F5TTSVocosConfig(
        mel_dim=backbone_args["input_channels"],
        hidden_size=backbone_args["dim"],
        intermediate_size=backbone_args["intermediate_dim"],
        num_hidden_layers=backbone_args["num_layers"],
        layer_scale_init_value=backbone_args.get("layer_scale_init_value"),
        n_fft=head_args["n_fft"],
        hop_length=head_args["hop_length"],
        sampling_rate=hyperparameters["feature_extractor"]["init_args"]["sample_rate"],
    )


def convert_state_dict(state_dict: dict[str, torch.Tensor], config: F5TTSConfig) -> dict[str, torch.Tensor]:
    r"""
    Renames a published F5-TTS state dict onto [`F5TTSForConditionalGeneration`]'s parameter names.

    Every module of the released backbones keeps its upstream name, so the rename is the `ema_model.transformer.`
    prefix becoming `model.` plus, for the `"unett"` backbone, the flat `layers.{i}.{j}` index pair gaining the
    `layer` `ModuleList` that holds it here.

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
    converted: dict[str, torch.Tensor] = {}
    leftover: list[str] = []

    for key, value in state_dict.items():
        name = key.removeprefix("ema_model.")
        if name in _DISCARDED_KEYS or name == "transformer.rotary_embed.inv_freq":
            continue
        if not name.startswith("transformer."):
            leftover.append(key)
            continue

        name = name.removeprefix("transformer.")
        if config.backbone == "unett":
            name = _UNET_LAYER.sub(lambda match: f"layers.{match.group(1)}.layer.{match.group(2)}.", name)
        converted["model." + name] = value.contiguous()

    if leftover:
        raise ValueError(f"The published checkpoint holds tensors this model has no destination for: {leftover}")

    return converted


def convert_vocoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Renames a published Vocos state dict onto [`F5TTSVocosModel`]'s parameter names.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            Tensors of a Vocos `pytorch_model.bin`.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors, without the mel front end and inverse STFT window
        buffers, which this model rebuilds from its configuration.
    """
    return {
        key: value.contiguous()
        for key, value in state_dict.items()
        if not key.startswith("feature_extractor.") and key != "head.istft.window"
    }


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


def convert(
    checkpoint_name: str = "F5TTS_v1_Base",
    output_dir: str = "f5-tts-converted",
    architecture: str | None = None,
    checkpoint_path: str | None = None,
    vocab_file: str | None = None,
    vocoder_path: str | None = None,
    mel_spec_type: str = "vocos",
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Converts a published F5-TTS or E2-TTS checkpoint into a directory
    [`F5TTSForConditionalGeneration.from_pretrained`] and [`F5TTSProcessor.from_pretrained`] can load, bundling
    the Vocos vocoder into its `vocoder` subfolder.

    Args:
        checkpoint_name (`str`, *optional*, defaults to `"F5TTS_v1_Base"`):
            Key of [`PUBLISHED_CHECKPOINTS`] to download, ignored when `checkpoint_path` is given.
        output_dir (`str`, *optional*, defaults to `"f5-tts-converted"`):
            Directory the converted config, weights, vocabulary, processor and vocoder are written to.
        architecture (`str`, *optional*):
            Key of [`ARCHITECTURES`] the checkpoint was trained with. Defaults to the one `checkpoint_name`
            names.
        checkpoint_path (`str`, *optional*):
            Local `.safetensors` or `.pt` checkpoint to convert instead of downloading one.
        vocab_file (`str`, *optional*):
            Local vocabulary file to use instead of downloading the published one.
        vocoder_path (`str`, *optional*):
            Local directory holding a Vocos `config.yaml` and `pytorch_model.bin`. Defaults to downloading
            [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz).
        mel_spec_type (`str`, *optional*, defaults to `"vocos"`):
            Mel front end the checkpoint was trained against, `"vocos"` or `"bigvgan"`.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None or vocab_file is None:
        repo_id, weights_name, vocab_name = PUBLISHED_CHECKPOINTS[checkpoint_name]
        checkpoint_path = checkpoint_path or hf_hub_download(repo_id, weights_name)
        vocab_file = vocab_file or hf_hub_download(repo_id, vocab_name)

    with open(vocab_file, "r", encoding="utf-8") as vocab_handle:
        text_vocab_size = sum(1 for _ in vocab_handle)

    config = build_config(architecture or checkpoint_name, text_vocab_size=text_vocab_size)
    converted = convert_state_dict(load_checkpoint(checkpoint_path), config)
    converted = {key: value.to(dtype) for key, value in converted.items()}

    feature_extractor = F5TTSFeatureExtractor(
        feature_size=config.mel_dim,
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        mel_spec_type=mel_spec_type,
    )
    processor = F5TTSProcessor(feature_extractor=feature_extractor, tokenizer=F5TTSTokenizer(vocab_file))
    processor.save_pretrained(output_path)
    config.save_pretrained(output_path)
    save_file(converted, str(output_path / "model.safetensors"), metadata={"format": "pt"})

    if vocoder_path is None:
        vocoder_config_file = hf_hub_download(VOCOS_REPO_ID, "config.yaml")
        vocoder_weights_file = hf_hub_download(VOCOS_REPO_ID, "pytorch_model.bin")
    else:
        vocoder_config_file = str(Path(vocoder_path) / "config.yaml")
        vocoder_weights_file = str(Path(vocoder_path) / "pytorch_model.bin")

    with open(vocoder_config_file, "r", encoding="utf-8") as config_handle:
        vocoder_config = build_vocoder_config(yaml.safe_load(config_handle))
    vocoder_state_dict = convert_vocoder_state_dict(
        torch.load(vocoder_weights_file, map_location="cpu", weights_only=True)
    )

    vocoder_path = output_path / VOCODER_SUBFOLDER
    vocoder_path.mkdir(parents=True, exist_ok=True)
    vocoder_config.save_pretrained(vocoder_path)
    save_file(
        {key: value.to(dtype) for key, value in vocoder_state_dict.items()},
        str(vocoder_path / "model.safetensors"),
        metadata={"format": "pt"},
    )


__all__ = [
    "ARCHITECTURES",
    "PUBLISHED_CHECKPOINTS",
    "build_config",
    "build_vocoder_config",
    "convert",
    "convert_state_dict",
    "convert_vocoder_state_dict",
    "load_checkpoint",
    "resolve_architecture",
]
