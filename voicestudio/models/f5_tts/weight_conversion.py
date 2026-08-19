"""Checkpoint conversion for F5-TTS."""

from transformers.core_model_loading import WeightRenaming


def _both(old_prefix: str, new_prefix: str) -> list[WeightRenaming]:
    """`WeightRenaming` only resolves a single `\\1`-style backreference per rule, so `weight`/`bias` (and any
    other multi-way suffix) must be listed as separate literal rules rather than as an alternation group."""
    return [
        WeightRenaming(rf"{old_prefix}\.weight", rf"{new_prefix}.weight"),
        WeightRenaming(rf"{old_prefix}\.bias", rf"{new_prefix}.bias"),
    ]


def build_f5_tts_weight_conversion_mapping(prefix: str = "model.") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate the original F5-TTS repo's `CFM`/`DiT` state dict (the format
    the `SWivid/F5-TTS` checkpoints are saved in, e.g. `F5TTS_v1_Base/model_1250000.safetensors`) into the module
    layout of [`F5TTSModel`].

    The original checkpoint stores an `ema_model.transformer.*`-prefixed EMA copy of the weights (the one actually
    used at inference time by the reference repo) alongside a non-EMA `transformer.*` copy and optimizer/EMA
    bookkeeping tensors (`initted`, `step`, `ema_model.initted`, `ema_model.step`, `ema_model.online_model.*`); only
    the EMA transformer weights are mapped here; everything else is left unmapped and dropped as unexpected.

    Args:
        prefix (`str`, *optional*, defaults to `"model."`):
            Prefix under which the F5-TTS backbone's weights live in [`F5TTSForConditionalGeneration`] (its
            `base_model_prefix` is `"model"`); pass `""` when loading directly into [`F5TTSModel`] instead.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    p = prefix
    o = r"ema_model\.transformer"
    rules = []
    rules += _both(rf"{o}\.input_embed\.conv_pos_embed\.conv1d\.(\d+)", rf"{p}input_embed.conv_pos_embed.block.\1")
    rules += _both(rf"{o}\.input_embed\.proj", rf"{p}input_embed.proj")
    rules += _both(rf"{o}\.time_embed\.time_mlp\.0", rf"{p}time_embed.mlp.0")
    rules += _both(rf"{o}\.time_embed\.time_mlp\.2", rf"{p}time_embed.mlp.2")
    rules.append(
        WeightRenaming(rf"{o}\.text_embed\.text_embed\.weight", rf"{p}text_embed.text_embed.weight")
    )
    for block_name in ("dwconv", "norm", "pwconv1", "pwconv2"):
        rules += _both(
            rf"{o}\.text_embed\.text_blocks\.(\d+)\.{block_name}", rf"{p}text_embed.text_blocks.\1.{block_name}"
        )
    for grn_param in ("gamma", "beta"):
        rules.append(
            WeightRenaming(
                rf"{o}\.text_embed\.text_blocks\.(\d+)\.grn\.{grn_param}",
                rf"{p}text_embed.text_blocks.\1.grn.{grn_param}",
            )
        )
    rules += _both(rf"{o}\.transformer_blocks\.(\d+)\.attn_norm\.linear", rf"{p}layers.\1.attn_norm.linear")
    for proj_name in ("to_q", "to_k", "to_v"):
        rules += _both(rf"{o}\.transformer_blocks\.(\d+)\.attn\.{proj_name}", rf"{p}layers.\1.attn.{proj_name}")
    rules += _both(rf"{o}\.transformer_blocks\.(\d+)\.attn\.to_out\.0", rf"{p}layers.\1.attn.to_out")
    rules.append(
        WeightRenaming(rf"{o}\.transformer_blocks\.(\d+)\.attn\.q_norm\.weight", rf"{p}layers.\1.attn.q_norm.weight")
    )
    rules.append(
        WeightRenaming(rf"{o}\.transformer_blocks\.(\d+)\.attn\.k_norm\.weight", rf"{p}layers.\1.attn.k_norm.weight")
    )
    rules += _both(rf"{o}\.transformer_blocks\.(\d+)\.ff\.ff\.0\.0", rf"{p}layers.\1.ff.net.0")
    rules += _both(rf"{o}\.transformer_blocks\.(\d+)\.ff\.ff\.2", rf"{p}layers.\1.ff.net.3")
    rules.append(WeightRenaming(rf"{o}\.long_skip_connection\.weight", rf"{p}long_skip_connection.weight"))
    rules += _both(rf"{o}\.norm_out\.linear", rf"{p}norm_out.linear")
    rules += _both(rf"{o}\.proj_out", rf"{p}proj_out")
    return rules


__all__ = ["build_f5_tts_weight_conversion_mapping"]
