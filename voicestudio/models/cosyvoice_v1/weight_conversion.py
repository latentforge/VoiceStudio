"""Checkpoint conversion for CosyVoice v1."""

import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer

from .configuration_cosyvoice_v1 import CosyVoiceV1Config
from .modeling_cosyvoice_v1 import CosyVoiceV1ForConditionalGeneration
from .processing_cosyvoice_v1 import CosyVoiceV1FeatureExtractor, CosyVoiceV1Processor


# The v1 repositories the CosyVoice authors published. `CosyVoice-300M` is the base model;
# `-SFT` and `-Instruct` add a `spk2info.pt` holding the built in speakers.
PUBLISHED_CHECKPOINTS = {
    "base": "FunAudioLLM/CosyVoice-300M",
    "sft": "FunAudioLLM/CosyVoice-300M-SFT",
    "instruct": "FunAudioLLM/CosyVoice-300M-Instruct",
}

# The three files of a released directory this conversion reads, one per network.
CHECKPOINT_FILES = ("llm.pt", "flow.pt", "hift.pt")

# Upstream tokenizes text with `whisper.tokenizer.get_tokenizer(multilingual=True,
# num_languages=100, language='en', task='transcribe')`, whose 51866 entry vocabulary is the one
# `openai/whisper-large-v3` ships.
TEXT_TOKENIZER_ID = "openai/whisper-large-v3"

# `BaseEncoder` names its input projection `embed.out`, its blocks `encoders` and its closing
# norm `after_norm`; `TransformerEncoderLayer` names its two norms `norm1`/`norm2` while
# `ConformerEncoderLayer` names the same two `norm_mha`/`norm_ff`.
ENCODER_RULES = (
    (r"^(.*)\.embed\.out\.0\.", r"\1.input_projection.proj."),
    (r"^(.*)\.embed\.out\.1\.", r"\1.input_projection.layer_norm."),
    (r"^(.*)\.encoders\.(\d+)\.norm_mha\.", r"\1.layers.\2.self_attn_layer_norm."),
    (r"^(.*)\.encoders\.(\d+)\.norm_ff\.", r"\1.layers.\2.final_layer_norm."),
    (r"^(.*)\.encoders\.(\d+)\.norm1\.", r"\1.layers.\2.self_attn_layer_norm."),
    (r"^(.*)\.encoders\.(\d+)\.norm2\.", r"\1.layers.\2.final_layer_norm."),
    (r"^(.*)\.encoders\.(\d+)\.", r"\1.layers.\2."),
    (r"^(.*)\.after_norm\.", r"\1.layer_norm."),
)

# The vocoder was trained with the pre-parametrization spelling of weight norm.
WEIGHT_NORM_RULES = (
    (r"\.weight_g$", ".parametrizations.weight.original0"),
    (r"\.weight_v$", ".parametrizations.weight.original1"),
)


def rename(key: str, rules: tuple[tuple[str, str], ...]) -> str:
    r"""
    Applies the first matching rename rule to a state dict key.

    Args:
        key (`str`):
            Key of the released state dict.
        rules (`tuple`):
            Pairs of regular expression and replacement.

    Returns:
        `str`: The renamed key.
    """
    for pattern, replacement in rules:
        renamed, count = re.subn(pattern, replacement, key)
        if count:
            return renamed
    return key


def build_config(**overrides) -> CosyVoiceV1Config:
    r"""
    Builds the [`CosyVoiceV1Config`] of the released 300M checkpoints.

    Every released v1 directory ships the same `cosyvoice.yaml`, so the geometry is the class
    defaults and only the overrides a caller passes change it.

    Args:
        overrides (`dict`, *optional*):
            Configuration fields overriding the released geometry.

    Returns:
        [`CosyVoiceV1Config`]: The configuration.
    """
    return CosyVoiceV1Config(**overrides)


def convert_state_dict(
    llm_state_dict: dict[str, torch.Tensor],
    flow_state_dict: dict[str, torch.Tensor],
    hift_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    r"""
    Renames the three released state dicts onto [`CosyVoiceV1ForConditionalGeneration`].

    Args:
        llm_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `llm.pt`.
        flow_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `flow.pt`.
        hift_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `hift.pt`, stored either as the bare generator or as a full `HiFiGan`
            module whose generator keys carry a `generator.` prefix.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.
    """
    converted = {}
    for key, value in llm_state_dict.items():
        converted[f"llm.{rename(key, ENCODER_RULES)}"] = value.contiguous()
    for key, value in flow_state_dict.items():
        converted[f"flow.{rename(key, ENCODER_RULES)}"] = value.contiguous()
    for key, value in hift_state_dict.items():
        key = key.removeprefix("generator.")
        converted[f"hift.{rename(key, WEIGHT_NORM_RULES)}"] = value.contiguous()
    return converted


def load_upstream_checkpoints(source: str) -> tuple[dict[str, torch.Tensor], ...]:
    r"""
    Reads `llm.pt`, `flow.pt` and `hift.pt` out of a released CosyVoice v1 directory.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory holding the three
            files.

    Returns:
        `tuple[dict[str, torch.Tensor], ...]`: The language model, flow and vocoder tensors.
    """
    source = PUBLISHED_CHECKPOINTS.get(source, source)
    if not Path(source).is_dir():
        source = snapshot_download(source, allow_patterns=list(CHECKPOINT_FILES))
    return tuple(
        torch.load(Path(source) / name, map_location="cpu", weights_only=True) for name in CHECKPOINT_FILES
    )


def build_model_files(
    source: str = "base", dtype: torch.dtype = torch.float32
) -> tuple[CosyVoiceV1Config, dict[str, torch.Tensor]]:
    r"""
    Reads a released CosyVoice v1 directory and returns what
    [`CosyVoiceV1ForConditionalGeneration`] needs to load it.

    Args:
        source (`str`, *optional*, defaults to `"base"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `tuple[CosyVoiceV1Config, dict[str, torch.Tensor]]`: The configuration and the renamed
        tensors.

    Raises:
        RuntimeError: If the renamed tensors do not cover the model exactly.
    """
    config = build_config()
    converted = convert_state_dict(*load_upstream_checkpoints(source))

    model = CosyVoiceV1ForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    missing = [key for key in missing if key not in dict(model.named_buffers())]
    if missing or unexpected:
        raise RuntimeError(
            f"The conversion does not cover the model exactly: missing={missing}, unexpected={unexpected}."
        )

    return config, {key: value.to(dtype) for key, value in converted.items()}


def convert(
    source: str = "base",
    output_dir: str = "cosyvoice-v1-converted",
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Converts a released CosyVoice v1 directory into one
    [`CosyVoiceV1ForConditionalGeneration.from_pretrained`] and
    [`CosyVoiceV1Processor.from_pretrained`] can load.

    The speech tokenizer and the speaker encoder are not converted: upstream publishes them as
    ONNX graphs only, and the saved processor leaves their paths unset for the caller to fill in.

    Args:
        source (`str`, *optional*, defaults to `"base"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.
        output_dir (`str`, *optional*, defaults to `"cosyvoice-v1-converted"`):
            Directory the converted config, weights and processor files are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config, converted = build_model_files(source, dtype=dtype)
    config.save_pretrained(output_path)
    save_file(converted, str(output_path / "model.safetensors"), metadata={"format": "pt"})

    processor = CosyVoiceV1Processor(
        feature_extractor=CosyVoiceV1FeatureExtractor(
            feature_size=config.flow_output_size, mel_sampling_rate=config.sample_rate
        ),
        tokenizer=WhisperTokenizer.from_pretrained(TEXT_TOKENIZER_ID, language="en", task="transcribe"),
    )
    processor.save_pretrained(output_path)


__all__ = [
    "CHECKPOINT_FILES",
    "ENCODER_RULES",
    "PUBLISHED_CHECKPOINTS",
    "TEXT_TOKENIZER_ID",
    "WEIGHT_NORM_RULES",
    "build_config",
    "build_model_files",
    "convert",
    "convert_state_dict",
    "load_upstream_checkpoints",
    "rename",
]
