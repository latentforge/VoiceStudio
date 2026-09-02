"""Checkpoint conversion for CosyVoice v2.

The original CosyVoice repo ships three independent PyTorch `state_dict` checkpoints (`llm.pt`, `flow.pt`,
`hift.pt`) saved directly from the reference implementation rather than in `transformers`-native module layout.
This module builds the `WeightRenaming` rules that translate `llm.pt` and `flow.pt` into the corresponding
[`CosyVoiceV2LLM`] and [`CosyVoiceV2FlowMatchingModel`] module layouts (`hift.pt` reuses
[`CosyVoiceV1HiFTGenerator`] unchanged, see `cosyvoice_v1.weight_conversion`).
"""

import re

from transformers.core_model_loading import WeightRenaming


def build_llm_weight_conversion_mapping(prefix: str = "") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate an original CosyVoice v2 `llm.pt` state dict into
    [`CosyVoiceV2LLM`]'s module layout. The original `Qwen2Encoder` wraps a full `Qwen2ForCausalLM` under
    `llm.model`, so the checkpoint's Qwen2 backbone lives at `llm.model.model.*` and its (unused here, and not
    tied to the input embedding) text `lm_head` at `llm.model.lm_head.*`; [`CosyVoiceV2LLM`] holds the bare
    `Qwen2Model` backbone directly at `llm.*`. `llm_embedding`, `llm_decoder`, and `speech_embedding` are already
    named identically in both.

    Args:
        prefix (`str`, *optional*, defaults to `""`):
            Prefix under which the LLM submodule's weights live in a composite checkpoint's state dict.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    p = re.escape(prefix)
    return [
        WeightRenaming(rf"{p}llm\.model\.model\.(.+)", rf"{prefix}llm.\1"),
        WeightRenaming(rf"{p}llm\.model\.lm_head\.(weight|bias)", rf"{prefix}llm.lm_head.\1"),
    ]


def _upsample_conformer_stack_renaming(
    old_prefix: str, new_prefix: str, old_layers_name: str, new_layers_name: str
) -> list[WeightRenaming]:
    """Renaming rules for one Conformer sub-stack (either the token-rate `encoders` stack or the mel-rate
    `up_encoders` stack) of the original `UpsampleConformerEncoder`, following the same WeNet/ESPnet naming
    translation as CosyVoice v1's `_rel_position_encoder_renaming`: `norm_mha`/`norm_ff` pre-norms and
    `feed_forward.w_1`/`w_2` become `self_attn_layer_norm`/`norm_ff` and
    `feed_forward.intermediate_dense`/`output_dense`; `self_attn.*` keys are already named identically."""
    op = re.escape(old_prefix)
    return [
        WeightRenaming(
            rf"{op}\.{old_layers_name}\.(\d+)\.norm_mha\.(weight|bias)",
            rf"{new_prefix}.{new_layers_name}.\1.self_attn_layer_norm.\2",
        ),
        WeightRenaming(
            rf"{op}\.{old_layers_name}\.(\d+)\.norm_ff\.(weight|bias)",
            rf"{new_prefix}.{new_layers_name}.\1.norm_ff.\2",
        ),
        WeightRenaming(
            rf"{op}\.{old_layers_name}\.(\d+)\.feed_forward\.w_1\.(weight|bias)",
            rf"{new_prefix}.{new_layers_name}.\1.feed_forward.intermediate_dense.\2",
        ),
        WeightRenaming(
            rf"{op}\.{old_layers_name}\.(\d+)\.feed_forward\.w_2\.(weight|bias)",
            rf"{new_prefix}.{new_layers_name}.\1.feed_forward.output_dense.\2",
        ),
        WeightRenaming(
            rf"{op}\.{old_layers_name}\.(\d+)\.self_attn\.(.+)", rf"{new_prefix}.{new_layers_name}.\1.self_attn.\2"
        ),
    ]


def build_flow_weight_conversion_mapping(prefix: str = "") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate an original CosyVoice v2 `flow.pt` state dict's
    `UpsampleConformerEncoder` (`encoder.*`) into [`CosyVoiceV2UpsampleConformerEncoder`]'s module layout.
    `input_embedding`, `spk_embed_affine_layer`, `encoder_proj`, `encoder.pre_lookahead_layer.*`,
    `encoder.up_layer.conv.*`, and `encoder.after_norm.*` are already named identically; `decoder.*` (the
    flow-matching estimator) is intentionally left unmapped here, see the CosyVoice v1 flow decoder note.

    Args:
        prefix (`str`, *optional*, defaults to `""`):
            Prefix under which the flow-matching submodule's weights live in a composite checkpoint's state dict.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    ep = f"{prefix}encoder"
    rules = [
        WeightRenaming(rf"{re.escape(ep)}\.embed\.out\.(\d+)\.(weight|bias)", rf"{ep}.embed.\1.\2"),
        WeightRenaming(rf"{re.escape(ep)}\.up_embed\.out\.(\d+)\.(weight|bias)", rf"{ep}.up_embed.\1.\2"),
    ]
    rules += _upsample_conformer_stack_renaming(ep, ep, "encoders", "layers")
    rules += _upsample_conformer_stack_renaming(ep, ep, "up_encoders", "up_layers")
    return rules


__all__ = ["build_llm_weight_conversion_mapping", "build_flow_weight_conversion_mapping"]
