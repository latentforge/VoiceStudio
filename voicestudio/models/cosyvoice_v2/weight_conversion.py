"""Checkpoint conversion for CosyVoice v2."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoTokenizer

from ..cosyvoice_v1.weight_conversion import ENCODER_RULES, WEIGHT_NORM_RULES, rename
from .configuration_cosyvoice_v2 import CosyVoiceV2Config
from .modeling_cosyvoice_v2 import CosyVoiceV2ForConditionalGeneration
from .processing_cosyvoice_v2 import CosyVoiceV2FeatureExtractor, CosyVoiceV2Processor


PUBLISHED_CHECKPOINTS = {"base": "FunAudioLLM/CosyVoice2-0.5B"}

# The three networks, plus the Qwen2 directory the language model is built from.
CHECKPOINT_FILES = ("llm.pt", "flow.pt", "hift.pt")
TEXT_MODEL_SUBDIR = "CosyVoice-BlankEN"

# `UpsampleConformerEncoder` runs a second, shorter stack after the upsampling layer, under its own
# input projection. Those rules have to come before the ones shared with v1, whose greedy prefix
# would otherwise swallow the `up_` and rename `up_embed` as if it were `embed`.
UPSAMPLE_ENCODER_RULES = (
    (r"^(.*)\.up_embed\.out\.0\.", r"\1.up_input_projection.proj."),
    (r"^(.*)\.up_embed\.out\.1\.", r"\1.up_input_projection.layer_norm."),
    (r"^(.*)\.up_encoders\.(\d+)\.norm_mha\.", r"\1.up_layers.\2.self_attn_layer_norm."),
    (r"^(.*)\.up_encoders\.(\d+)\.norm_ff\.", r"\1.up_layers.\2.final_layer_norm."),
    (r"^(.*)\.up_encoders\.(\d+)\.", r"\1.up_layers.\2."),
) + ENCODER_RULES

# `Qwen2Encoder` wraps a `Qwen2ForCausalLM`, whose head is tied to its embedding table and is never
# used by CosyVoice, so only the decoder underneath it is carried over.
LANGUAGE_MODEL_RULES = ((r"^llm\.model\.model\.", "model."),)

# The only source tensors the conversion drops. `Qwen2Config.tie_word_embeddings` is true, so the
# head holds the same values as `model.embed_tokens.weight`.
DROPPED_PREFIXES = ("llm.model.lm_head.",)


def convert_llm_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Renames `llm.pt` onto [`CosyVoiceV2SpeechTokenLM`].

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            Tensors of `llm.pt`.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.
    """
    converted = {}
    for key, value in state_dict.items():
        if key.startswith(DROPPED_PREFIXES):
            continue
        converted[f"llm.{rename(key, LANGUAGE_MODEL_RULES)}"] = value.contiguous()
    return converted


def convert_state_dict(
    llm_state_dict: dict[str, torch.Tensor],
    flow_state_dict: dict[str, torch.Tensor],
    hift_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    r"""
    Renames the three released state dicts onto [`CosyVoiceV2ForConditionalGeneration`].

    Args:
        llm_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `llm.pt`.
        flow_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `flow.pt`.
        hift_state_dict (`dict[str, torch.Tensor]`):
            Tensors of `hift.pt`, stored either as the bare generator or as a full `HiFiGan` module
            whose generator keys carry a `generator.` prefix.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.
    """
    converted = convert_llm_state_dict(llm_state_dict)
    for key, value in flow_state_dict.items():
        converted[f"flow.{rename(key, UPSAMPLE_ENCODER_RULES)}"] = value.contiguous()
    for key, value in hift_state_dict.items():
        key = key.removeprefix("generator.")
        converted[f"hift.{rename(key, WEIGHT_NORM_RULES)}"] = value.contiguous()
    return converted


def build_config(source: str, **overrides) -> CosyVoiceV2Config:
    r"""
    Builds the [`CosyVoiceV2Config`] of a released CosyVoice v2 directory.

    The geometry of the three CosyVoice networks is the class defaults; the Qwen2 sub configuration
    is read from the `CosyVoice-BlankEN` directory the checkpoint ships.

    Args:
        source (`str`):
            Local directory of the released checkpoint.
        overrides (`dict`, *optional*):
            Configuration fields overriding the released geometry.

    Returns:
        [`CosyVoiceV2Config`]: The configuration.
    """
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    text_config = Qwen2Config.from_pretrained(str(Path(source) / TEXT_MODEL_SUBDIR))
    return CosyVoiceV2Config(text_config=text_config, **overrides)


def load_upstream_checkpoints(source: str) -> tuple[str, tuple[dict[str, torch.Tensor], ...]]:
    r"""
    Reads `llm.pt`, `flow.pt` and `hift.pt` out of a released CosyVoice v2 directory.

    Args:
        source (`str`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.

    Returns:
        `tuple`: The resolved local directory and the language model, flow and vocoder tensors.
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
) -> tuple[CosyVoiceV2Config, dict[str, torch.Tensor]]:
    r"""
    Reads a released CosyVoice v2 directory and returns what
    [`CosyVoiceV2ForConditionalGeneration`] needs to load it.

    Args:
        source (`str`, *optional*, defaults to `"base"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.

    Returns:
        `tuple[CosyVoiceV2Config, dict[str, torch.Tensor]]`: The configuration and the renamed
        tensors.

    Raises:
        RuntimeError: If the renamed tensors do not cover the model exactly.
    """
    source, tensors = load_upstream_checkpoints(source)
    config = build_config(source)
    converted = convert_state_dict(*tensors)

    model = CosyVoiceV2ForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    missing = [key for key in missing if key not in dict(model.named_buffers())]
    if missing or unexpected:
        raise RuntimeError(
            f"The conversion does not cover the model exactly: missing={missing}, unexpected={unexpected}."
        )

    return config, {key: value.to(dtype) for key, value in converted.items()}


def convert(
    source: str = "base",
    output_dir: str = "cosyvoice-v2-converted",
    dtype: torch.dtype = torch.float32,
) -> None:
    r"""
    Converts a released CosyVoice v2 directory into one
    [`CosyVoiceV2ForConditionalGeneration.from_pretrained`] and
    [`CosyVoiceV2Processor.from_pretrained`] can load.

    The speech tokenizer and the speaker encoder are not converted: upstream publishes them as ONNX
    graphs only, and the saved processor leaves their paths unset for the caller to fill in.

    Args:
        source (`str`, *optional*, defaults to `"base"`):
            Key of [`PUBLISHED_CHECKPOINTS`], repository id, or local directory.
        output_dir (`str`, *optional*, defaults to `"cosyvoice-v2-converted"`):
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
    CosyVoiceV2Processor.add_special_tokens(tokenizer)
    processor = CosyVoiceV2Processor(
        feature_extractor=CosyVoiceV2FeatureExtractor(
            feature_size=config.flow_output_size, sampling_rate=config.sample_rate,
            mel_sampling_rate=config.sample_rate,
        ),
        tokenizer=tokenizer,
        token_mel_ratio=config.token_mel_ratio,
    )
    processor.save_pretrained(output_path)


__all__ = [
    "CHECKPOINT_FILES",
    "DROPPED_PREFIXES",
    "LANGUAGE_MODEL_RULES",
    "PUBLISHED_CHECKPOINTS",
    "TEXT_MODEL_SUBDIR",
    "UPSAMPLE_ENCODER_RULES",
    "build_config",
    "build_model_files",
    "convert",
    "convert_llm_state_dict",
    "convert_state_dict",
    "load_upstream_checkpoints",
]
