"""Checkpoint conversion for Dia2."""

import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.encodec.feature_extraction_encodec import EncodecFeatureExtractor
from transformers.models.mimi.modeling_mimi import MimiModel
from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

from .configuration_dia2 import Dia2Config, Dia2DepthDecoderConfig
from .processing_dia2 import Dia2Processor


# Files the published `nari-labs/Dia2-*` repos ship alongside the weights that the converted checkpoint keeps
# verbatim so the tokenizer of the output directory is byte-for-byte the published one.
_TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

_BACKBONE_LAYER_RENAMES = {
    "pre_norm.weight": "input_layernorm.weight",
    "post_norm.weight": "post_attention_layernorm.weight",
    "attn.q_proj.weight": "self_attn.q_proj.weight",
    "attn.k_proj.weight": "self_attn.k_proj.weight",
    "attn.v_proj.weight": "self_attn.v_proj.weight",
    "attn.o_proj.weight": "self_attn.o_proj.weight",
    "attn.q_norm.weight": "self_attn.q_norm.weight",
    "attn.k_norm.weight": "self_attn.k_norm.weight",
}

_DEPTH_LAYER_RENAMES = {
    "pre_norm.weight": "input_layernorm.weight",
    "post_norm.weight": "post_attention_layernorm.weight",
    "self_attention.q_norm.weight": "self_attn.q_norm.weight",
    "self_attention.k_norm.weight": "self_attn.k_norm.weight",
}


def is_published_layout(source: str) -> bool:
    r"""
    Returns whether `source` is a published Dia2 repository rather than a directory [`convert`] wrote.

    The published repositories carry the training runtime's own `config.json`, a `data`/`model`/`runtime` tree
    that declares no `model_type`, so the discriminator is a `config.json` declaring this model's `model_type`.
    `PreTrainedConfig.from_pretrained` draws no such distinction of its own: the three sections are kept as extra
    attributes and every field this model reads silently falls back to its default.

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
    return json.loads(Path(config_file).read_text()).get("model_type") != Dia2Config.model_type


def build_pretrained_config(source: str) -> Dia2Config:
    r"""
    Builds the [`Dia2Config`] of a published Dia2 repository.

    Args:
        source (`str`):
            A `nari-labs/Dia2-*` repository id, or a local directory holding the published `config.json` and
            tokenizer files.

    Returns:
        [`Dia2Config`]: The equivalent VoiceStudio configuration.
    """
    if Path(source).is_dir():
        config_file = Path(source) / "config.json"
    else:
        config_file = Path(hf_hub_download(source, "config.json"))
    tokenizer = AutoTokenizer.from_pretrained(source)
    # `0` is the id of this tokenizer's `bos_token`, which Dia2's runtime falls back to `1` on.
    return build_config(json.loads(config_file.read_text()), text_bos_token_id=tokenizer.bos_token_id or 1)


def build_processor(source: str, config: Dia2Config) -> Dia2Processor:
    r"""
    Builds the [`Dia2Processor`] of a published Dia2 repository.

    Args:
        source (`str`):
            A `nari-labs/Dia2-*` repository id, or a local directory holding the published tokenizer files.
        config ([`Dia2Config`]):
            Configuration naming the codec the processor tokenizes audio with.

    Returns:
        [`Dia2Processor`]: The processor.
    """
    return Dia2Processor(
        feature_extractor=EncodecFeatureExtractor.from_pretrained(config.codec_model_id),
        tokenizer=AutoTokenizer.from_pretrained(source),
        audio_tokenizer=MimiModel.from_pretrained(config.codec_model_id),
    )


def build_config(upstream_config: dict, text_bos_token_id: int = 1) -> Dia2Config:
    r"""
    Builds a [`Dia2Config`] from a published Dia2 `config.json`.

    Args:
        upstream_config (`dict`):
            Parsed `config.json` of a `nari-labs/Dia2-*` repository.
        text_bos_token_id (`int`, *optional*, defaults to 1):
            Id fed on the first text stream of the very first frame.

    Returns:
        [`Dia2Config`]: The equivalent VoiceStudio configuration.

    Raises:
        ValueError: If the checkpoint uses a RoPE timescale or an MLP activation pair that the `transformers`
            rotary embedding and [`LlamaMLP`] cannot express, or a depth decoder head count the fused upstream
            projection cannot express.
    """
    data = upstream_config["data"]
    model = upstream_config["model"]
    runtime = upstream_config["runtime"]
    decoder = model["decoder"]
    depformer = model["depformer"]

    if model.get("rope_min_timescale", 1) != 1:
        raise ValueError(
            "`transformers`' rotary embedding derives its inverse frequencies from a single base, which only "
            f"matches Dia2 when rope_min_timescale is 1; got {model['rope_min_timescale']}."
        )

    if depformer["kv_heads"] != depformer["gqa_query_heads"]:
        raise ValueError(
            "Dia2's depth decoder stores one fused query/key/value projection sized for its query heads, so it "
            f"cannot express {depformer['kv_heads']} key/value heads against "
            f"{depformer['gqa_query_heads']} query heads."
        )

    num_codebooks = data["channels"] - 2
    rms_norm_eps = model.get("normalization_layer_epsilon", 1e-5)
    rope_parameters = {"rope_type": "default", "rope_theta": float(model.get("rope_max_timescale", 10000.0))}

    depth_decoder_config = Dia2DepthDecoderConfig(
        num_codebooks=num_codebooks,
        weights_schedule=runtime["weights_schedule"],
        backbone_hidden_size=decoder["n_embd"],
        vocab_size=data["audio_vocab_size"],
        hidden_size=depformer["n_embd"],
        intermediate_size=depformer["n_hidden"],
        num_hidden_layers=depformer["n_layer"],
        num_attention_heads=depformer["gqa_query_heads"],
        num_key_value_heads=depformer["kv_heads"],
        head_dim=depformer["gqa_head_dim"],
        hidden_act=_resolve_activation(depformer.get("mlp_activations", ["silu", "linear"])),
        max_position_embeddings=num_codebooks,
        rms_norm_eps=rms_norm_eps,
        use_rope=depformer.get("apply_rope", True),
        use_text_embedding=depformer.get("text_embedding", True),
        text_vocab_size=data["text_vocab_size"],
        text_pad_token_id=data["text_pad_token_id"],
        rope_parameters=dict(rope_parameters),
    )

    return Dia2Config(
        num_codebooks=num_codebooks,
        vocab_size=data["audio_vocab_size"],
        text_vocab_size=data["text_vocab_size"],
        action_vocab_size=data["action_vocab_size"],
        hidden_size=decoder["n_embd"],
        intermediate_size=decoder["n_hidden"],
        num_hidden_layers=decoder["n_layer"],
        num_attention_heads=decoder["gqa_query_heads"],
        num_key_value_heads=decoder["kv_heads"],
        head_dim=decoder["gqa_head_dim"],
        hidden_act=_resolve_activation(model.get("linear", {}).get("mlp_activations", ["silu", "linear"])),
        max_position_embeddings=runtime.get("max_context_steps", 1500),
        rms_norm_eps=rms_norm_eps,
        text_low_rank_dim=decoder.get("low_rank_dim"),
        delay_pattern=data["delay_pattern"],
        codebook_bos_token_id=data.get("audio_bos_token_id", data["audio_vocab_size"] - 2),
        codebook_pad_token_id=data.get("audio_pad_token_id", data["audio_vocab_size"] - 1),
        text_pad_token_id=data["text_pad_token_id"],
        text_bos_token_id=text_bos_token_id,
        text_new_word_token_id=data["text_new_word_token_id"],
        text_zero_token_id=data.get("text_zero_token_id", 7),
        action_pad_token_id=data["action_pad_token_id"],
        action_new_word_token_id=data["action_new_word_token_id"],
        second_stream_ahead=data.get("second_stream_ahead", 0),
        rope_parameters=dict(rope_parameters),
        depth_decoder_config=depth_decoder_config,
        codec_model_id=(upstream_config.get("assets") or {}).get("mimi") or "kyutai/mimi",
    )


def _resolve_activation(activations: list[str]) -> str:
    if len(activations) != 2 or activations[1] != "linear":
        raise ValueError(
            f"Dia2's MLP gates one activated branch with one linear branch; got activations {activations}."
        )
    return activations[0]


def convert_state_dict(state_dict: dict[str, torch.Tensor], config: Dia2Config) -> dict[str, torch.Tensor]:
    r"""
    Renames a published Dia2 state dict onto [`Dia2ForConditionalGeneration`]'s parameter names.

    The backbone and depth decoder MLPs are stored as one fused `wi` projection and are split into the
    `gate_proj`/`up_proj` pair [`LlamaMLP`] expects; the depth decoder's fused `in_proj` is split the same way
    into `q_proj`/`k_proj`/`v_proj`. The per-codebook embedding tables and output heads are stacked into the
    single offset-indexed table and the single stacked head this model uses.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            Tensors of a published `model.safetensors`.
        config ([`Dia2Config`]):
            Configuration built from the matching `config.json`.

    Returns:
        `dict[str, torch.Tensor]`: The renamed tensors.

    Raises:
        ValueError: If a tensor of the published checkpoint has no destination in this model.
    """
    depth_config = config.depth_decoder_config
    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def take(key: str) -> torch.Tensor:
        consumed.add(key)
        return state_dict[key]

    converted["backbone_model.embed_tokens.embed_text_tokens.weight"] = take(
        "transformer.text_embed.embedding.weight"
    )
    converted["backbone_model.embed_tokens.text_stream_proj.weight"] = take(
        "transformer.text_embed.main_proj.weight"
    )
    converted["backbone_model.embed_tokens.second_text_stream_proj.weight"] = take(
        "transformer.text_embed.second_proj.weight"
    )
    converted["backbone_model.embed_tokens.embed_audio_tokens.weight"] = torch.cat(
        [take(f"transformer.audio_embeds.{index}.weight") for index in range(config.num_codebooks)], dim=0
    )
    converted["backbone_model.norm.weight"] = take("transformer.norm.weight")
    converted["action_head.weight"] = take("transformer.action_head.weight")
    converted["lm_head.weight"] = take("transformer.cb0_head.weight")

    for layer in range(config.num_hidden_layers):
        source = f"transformer.layers.{layer}."
        target = f"backbone_model.layers.{layer}."
        for suffix, renamed in _BACKBONE_LAYER_RENAMES.items():
            converted[target + renamed] = take(source + suffix)
        gate, up = take(source + "mlp.wi.weight").chunk(2, dim=0)
        converted[target + "mlp.gate_proj.weight"] = gate
        converted[target + "mlp.up_proj.weight"] = up
        converted[target + "mlp.down_proj.weight"] = take(source + "mlp.wo.weight")

    converted["depth_decoder.model.embed_tokens.weight"] = torch.cat(
        [take(f"depformer.audio_embeds.{index}.weight") for index in range(config.num_codebooks - 1)], dim=0
    )
    converted["depth_decoder.model.norm.weight"] = take("depformer.norm.weight")
    converted["depth_decoder.codebooks_head.weight"] = torch.stack(
        [take(f"depformer.logits.{index}.weight").T for index in range(config.num_codebooks - 1)], dim=0
    )
    for group in range(depth_config.num_weight_groups):
        converted[f"depth_decoder.model.inputs_embeds_projector.{group}.weight"] = take(
            f"depformer.depformer_in.{group}.weight"
        )

    if depth_config.use_text_embedding:
        converted["depth_decoder.model.embed_text_tokens.weight"] = take("depformer.text_embed.embedding.weight")
        converted["depth_decoder.model.text_stream_proj.weight"] = take("depformer.text_embed.main_proj.weight")
        converted["depth_decoder.model.second_text_stream_proj.weight"] = take(
            "depformer.text_embed.second_proj.weight"
        )

    for layer in range(depth_config.num_hidden_layers):
        source = f"depformer.layers.{layer}."
        target = f"depth_decoder.model.layers.{layer}."
        for suffix, renamed in _DEPTH_LAYER_RENAMES.items():
            converted[target + renamed] = take(source + suffix)
        gate, up = take(source + "mlp.wi.weight").chunk(2, dim=0)
        converted[target + "mlp.gate_proj.weight"] = gate
        converted[target + "mlp.up_proj.weight"] = up
        converted[target + "mlp.down_proj.weight"] = take(source + "mlp.wo.weight")
        for group in range(depth_config.num_weight_groups):
            query, key, value = take(source + f"self_attention.in_proj.{group}.weight").chunk(3, dim=0)
            converted[target + f"self_attn.q_proj.{group}.weight"] = query
            converted[target + f"self_attn.k_proj.{group}.weight"] = key
            converted[target + f"self_attn.v_proj.{group}.weight"] = value
            converted[target + f"self_attn.o_proj.{group}.weight"] = take(
                source + f"self_attention.out_proj.{group}.weight"
            )

    leftover = sorted(set(state_dict) - consumed)
    if leftover:
        raise ValueError(f"The published checkpoint holds tensors this model has no destination for: {leftover}")

    return {key: value.contiguous() for key, value in converted.items()}


def convert(checkpoint_path: str, output_dir: str, dtype: torch.dtype = torch.float32) -> None:
    r"""
    Converts a published Dia2 checkpoint into a directory
    [`Dia2ForConditionalGeneration.from_pretrained`] and [`Dia2Processor.from_pretrained`] can load.

    Args:
        checkpoint_path (`str`):
            A `nari-labs/Dia2-*` repository id, or a local directory holding `config.json` and
            `model.safetensors`.
        output_dir (`str`):
            Directory the converted config, weights, tokenizer and processor files are written to.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype the converted weights are cast to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source = Path(checkpoint_path)

    if source.is_dir():
        weights_file = source / "model.safetensors"
        tokenizer_files = {name: source / name for name in _TOKENIZER_FILES if (source / name).exists()}
    else:
        weights_file = Path(hf_hub_download(checkpoint_path, "model.safetensors"))
        tokenizer_files = {}
        for name in _TOKENIZER_FILES:
            try:
                tokenizer_files[name] = Path(hf_hub_download(checkpoint_path, name))
            except Exception:
                continue

    config = build_pretrained_config(checkpoint_path)
    converted = convert_state_dict(load_file(str(weights_file)), config)
    converted = {key: value.to(dtype) for key, value in converted.items()}

    build_processor(checkpoint_path, config).save_pretrained(output_path)
    # A tokenizer save/load round trip drops this tokenizer's added `[S1]`/`[S2]` and sound-effect tokens, so
    # the published files are copied over the ones `save_pretrained` just wrote.
    for name, path in tokenizer_files.items():
        shutil.copyfile(path, output_path / name)

    config.save_pretrained(output_path)
    save_file(converted, str(output_path / "model.safetensors"), metadata={"format": "pt"})


__all__ = [
    "build_config",
    "build_pretrained_config",
    "build_processor",
    "convert",
    "convert_state_dict",
    "is_published_layout",
]
