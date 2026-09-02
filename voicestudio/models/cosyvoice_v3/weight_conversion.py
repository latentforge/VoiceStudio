"""Checkpoint conversion for CosyVoice v3.

The original CosyVoice repo ships three independent PyTorch `state_dict` checkpoints (`llm.pt`, `flow.pt`,
`hift.pt`) saved directly from the reference implementation rather than in `transformers`-native module layout.
This module builds the `WeightRenaming` rules that translate `llm.pt` into [`CosyVoiceV3LLM`]'s module layout
(`flow.pt`'s `pre_lookahead_layer.*`/`decoder.estimator.*`/`input_embedding`/`spk_embed_affine_layer` are already
named identically to [`CosyVoiceV3FlowMatchingModel`]). [`CosyVoiceV3HiFTGenerator`] reuses the same module
attribute names (`conv_pre`/`ups`/`source_downs`/`source_resblocks`/`resblocks`/`conv_post`/`f0_predictor`) as
[`CosyVoiceV1HiFTGenerator`] even though several of those attributes hold architecturally different (causal)
submodules for v3's real `CausalHiFTGenerator` checkpoint, so `cosyvoice_v1.weight_conversion`'s
`build_hift_weight_conversion_mapping` (translating classic `weight_norm`'s `weight_g`/`weight_v` into the
`parametrizations.weight` layout) applies unchanged here too.
"""

import re

from transformers.core_model_loading import WeightRenaming

from ..cosyvoice_v1.weight_conversion import build_hift_weight_conversion_mapping


def build_llm_weight_conversion_mapping(prefix: str = "") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate an original CosyVoice v3 `llm.pt` state dict into
    [`CosyVoiceV3LLM`]'s module layout. The original `Qwen2Encoder` wraps a full `Qwen2ForCausalLM` under
    `llm.model`, so the checkpoint's Qwen2 backbone lives at `llm.model.model.*` and its unused `lm_head` at
    `llm.model.lm_head.*` (dropped via `_keys_to_ignore_on_load_unexpected`); [`CosyVoiceV3LLM`] holds the bare
    `Qwen2Model` backbone directly at `llm.*`. `llm_decoder` and `speech_embedding` are already named identically
    in both (CosyVoice v3 has no separate `llm_embedding`; the sos/task/fill ids live inside `speech_embedding`).

    Args:
        prefix (`str`, *optional*, defaults to `""`):
            Prefix under which the LLM submodule's weights live in a composite checkpoint's state dict.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    p = re.escape(prefix)
    return [WeightRenaming(rf"{p}llm\.model\.model\.(.+)", rf"{prefix}llm.\1")]


__all__ = ["build_llm_weight_conversion_mapping", "build_hift_weight_conversion_mapping"]
