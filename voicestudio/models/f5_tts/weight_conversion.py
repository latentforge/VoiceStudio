"""Checkpoint conversion for F5-TTS."""

from transformers.core_model_loading import WeightRenaming


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
    return [
        WeightRenaming(
            r"ema_model\.transformer\.input_embed\.conv_pos_embed\.conv1d\.(\d+)\.(weight|bias)",
            rf"{p}input_embed.conv_pos_embed.block.\1.\2",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.input_embed\.proj\.(weight|bias)",
            rf"{p}input_embed.proj.\1",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.time_embed\.time_mlp\.0\.(weight|bias)",
            rf"{p}time_embed.mlp.0.\1",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.time_embed\.time_mlp\.2\.(weight|bias)",
            rf"{p}time_embed.mlp.2.\1",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.text_embed\.text_embed\.weight",
            rf"{p}text_embed.text_embed.weight",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.text_embed\.text_blocks\.(\d+)\.(dwconv|norm|pwconv1|pwconv2)\.(weight|bias)",
            rf"{p}text_embed.text_blocks.\1.\2.\3",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.text_embed\.text_blocks\.(\d+)\.grn\.(gamma|beta)",
            rf"{p}text_embed.text_blocks.\1.grn.\2",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.transformer_blocks\.(\d+)\.attn_norm\.linear\.(weight|bias)",
            rf"{p}layers.\1.attn_norm.linear.\2",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.transformer_blocks\.(\d+)\.attn\.(to_q|to_k|to_v)\.(weight|bias)",
            rf"{p}layers.\1.attn.\2.\3",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.transformer_blocks\.(\d+)\.attn\.to_out\.0\.(weight|bias)",
            rf"{p}layers.\1.attn.to_out.\2",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.transformer_blocks\.(\d+)\.attn\.q_norm\.weight",
            rf"{p}layers.\1.attn.q_norm.weight",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.transformer_blocks\.(\d+)\.attn\.k_norm\.weight",
            rf"{p}layers.\1.attn.k_norm.weight",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.transformer_blocks\.(\d+)\.ff\.ff\.0\.0\.(weight|bias)",
            rf"{p}layers.\1.ff.net.0.\2",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.transformer_blocks\.(\d+)\.ff\.ff\.2\.(weight|bias)",
            rf"{p}layers.\1.ff.net.3.\2",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.long_skip_connection\.weight",
            rf"{p}long_skip_connection.weight",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.norm_out\.linear\.(weight|bias)",
            rf"{p}norm_out.linear.\1",
        ),
        WeightRenaming(
            r"ema_model\.transformer\.proj_out\.(weight|bias)",
            rf"{p}proj_out.\1",
        ),
    ]


__all__ = ["build_f5_tts_weight_conversion_mapping"]
