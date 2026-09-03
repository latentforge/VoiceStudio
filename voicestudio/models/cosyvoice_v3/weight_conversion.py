"""Checkpoint conversion for CosyVoice v3."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoTokenizer

from ..cosyvoice_v1.weight_conversion import WEIGHT_NORM_RULES, rename
from ..cosyvoice_v2.weight_conversion import (
    DROPPED_PREFIXES as LANGUAGE_MODEL_DROPPED_PREFIXES,
    LANGUAGE_MODEL_RULES,
    TEXT_MODEL_SUBDIR,
    convert_llm_state_dict,
)
from .configuration_cosyvoice_v3 import CosyVoiceV3Config
from .modeling_cosyvoice_v3 import CosyVoiceV3ForConditionalGeneration
from .processing_cosyvoice_v3 import CosyVoiceV3FeatureExtractor, CosyVoiceV3Processor


PUBLISHED_CHECKPOINTS = {"base": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"}

CHECKPOINT_FILES = ("llm.pt", "flow.pt", "hift.pt")

# x-transformers registers the rotary frequency table as a persistent buffer, so the released flow
# weights carry it. The migrated estimator derives its own from the configuration.
FLOW_DROPPED_PREFIXES = ("decoder.estimator.rotary_embed.",)


def convert_flow_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Renames `flow.pt` onto [`CosyVoiceV3FlowModel`].

    v3's flow matching model has no encoder, so none of the encoder rename rules apply and the only
    change is dropping the rotary bookkeeping buffer.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            Tensors of `flow.pt`.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.
    """
    return {
        f"flow.{key}": value.contiguous()
        for key, value in state_dict.items()
        if not key.startswith(FLOW_DROPPED_PREFIXES)
    }


def convert_state_dict(
    llm_state_dict: dict[str, torch.Tensor],
    flow_state_dict: dict[str, torch.Tensor],
    hift_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    r"""
    Renames the three released state dicts onto [`CosyVoiceV3ForConditionalGeneration`].

    Args:
        llm_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `llm.pt`.
        flow_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `flow.pt`.
        hift_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `hift.pt`.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.
    """
    converted = convert_llm_state_dict(llm_state_dict)
    converted.update(convert_flow_state_dict(flow_state_dict))
    for key, value in hift_state_dict.items():
        key = key.removeprefix("generator.")
        converted[f"hift.{rename(key, WEIGHT_NORM_RULES)}"] = value.contiguous()
    return converted


def build_config(source: str, **overrides) -> CosyVoiceV3Config:
    r"""
    Builds the [`CosyVoiceV3Config`] of a released CosyVoice v3 directory.

    Args:
        source (`str`):
            Local directory of the released checkpoint.
        overrides (`dict`, *optional*):
            Configuration fields overriding the released geometry.

    Returns:
        [`CosyVoiceV3Config`]: The configuration.
    """
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    text_config = Qwen2Config.from_pretrained(str(Path(source) / TEXT_MODEL_SUBDIR))
    return CosyVoiceV3Config(text_config=text_config, **overrides)


def load_upstream_checkpoints(source: str) -> tuple[str, tuple[dict[str, torch.Tensor], ...]]:
    r"""
    Reads `llm.pt`, `flow.pt` and `hift.pt` out of a released CosyVoice v3 directory.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.

    Returns:
        `tuple`: The resolved local directory and the three state dicts.
    """
    source = PUBLISHED_CHECKPOINTS.get(source, source)
    if not Path(source).is_dir():
        source = snapshot_download(
            source, allow_patterns=list(CHECKPOINT_FILES) + [f"{TEXT_MODEL_SUBDIR}/*"]
        )
    tensors = tuple(
        torch.load(Path(source) / name, map_location="cpu", weights_only=True) for name in CHECKPOINT_FILES
    )
    return source, tensors


def build_model_files(
    source: str = "base", dtype: torch.dtype = torch.float32
) -> tuple[CosyVoiceV3Config, dict[str, torch.Tensor]]:
    r"""
    Reads a released CosyVoice v3 directory and returns what
    [`CosyVoiceV3ForConditionalGeneration`] needs to load it.

    Args:
        source (`str`, *optional*, defaults to `"base"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `tuple[CosyVoiceV3Config, dict[str, torch.Tensor]]`: The configuration and the renamed
        tensors.

    Raises:
        RuntimeError: If the renamed tensors do not cover the model exactly.
    """
    source, tensors = load_upstream_checkpoints(source)
    config = build_config(source)
    converted = convert_state_dict(*tensors)

    model = CosyVoiceV3ForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    missing = [key for key in missing if key not in dict(model.named_buffers())]
    if missing or unexpected:
        raise RuntimeError(
            f"The conversion does not cover the model exactly: missing={missing}, unexpected={unexpected}."
        )

    return config, {key: value.to(dtype) for key, value in converted.items()}


def convert(
    source: str = "base",
    output_dir: str = "cosyvoice-v3-converted",
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Converts a released CosyVoice v3 directory into one
    [`CosyVoiceV3ForConditionalGeneration.from_pretrained`] and
    [`CosyVoiceV3Processor.from_pretrained`] can load.

    The speech tokenizer and the speaker encoder are not converted: upstream publishes them as ONNX
    graphs only, and the saved processor leaves their paths unset for the caller to fill in.

    Args:
        source (`str`, *optional*, defaults to `"base"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.
        output_dir (`str`, *optional*, defaults to `"cosyvoice-v3-converted"`):
            Directory the converted config, weights and processor files are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved = PUBLISHED_CHECKPOINTS.get(source, source)
    config, converted = build_model_files(source, dtype=dtype)
    config.save_pretrained(output_path)
    save_file(converted, str(output_path / "model.safetensors"), metadata={"format": "pt"})

    tokenizer = AutoTokenizer.from_pretrained(str(Path(resolved) / TEXT_MODEL_SUBDIR))
    CosyVoiceV3Processor.add_special_tokens(tokenizer)
    processor = CosyVoiceV3Processor(
        feature_extractor=CosyVoiceV3FeatureExtractor(
            feature_size=config.flow_output_size, sampling_rate=config.sample_rate,
            mel_sampling_rate=config.sample_rate,
        ),
        tokenizer=tokenizer,
        token_mel_ratio=config.token_mel_ratio,
    )
    processor.save_pretrained(output_path)


__all__ = [
    "CHECKPOINT_FILES",
    "FLOW_DROPPED_PREFIXES",
    "LANGUAGE_MODEL_DROPPED_PREFIXES",
    "LANGUAGE_MODEL_RULES",
    "PUBLISHED_CHECKPOINTS",
    "TEXT_MODEL_SUBDIR",
    "build_config",
    "build_model_files",
    "convert",
    "convert_flow_state_dict",
    "convert_state_dict",
    "load_upstream_checkpoints",
]
