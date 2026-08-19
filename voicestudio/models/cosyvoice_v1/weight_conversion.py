"""Checkpoint conversion for CosyVoice v1.

The original CosyVoice repo ships three independent PyTorch `state_dict` checkpoints per model
(`llm.pt`, `flow.pt`, `hift.pt`), saved directly from the WeNet/ESPnet-derived reference implementation rather than
in `transformers`-native module layout. This module builds the `WeightRenaming` rules that translate each of those
checkpoints into the corresponding [`CosyVoiceV1LLM`], [`CosyVoiceV1FlowMatchingModel`], and
[`CosyVoiceV1HiFTGenerator`] module layouts.
"""

import re

from transformers.core_model_loading import WeightRenaming


def _rel_position_encoder_renaming(
    old_prefix: str, new_prefix: str, norm_mha_name: str = "norm_mha", norm_ff_name: str = "norm_ff"
) -> list[WeightRenaming]:
    """
    Renaming rules for one [`CosyVoiceV1RelPositionEncoder`] stack: the original checkpoint's WeNet/ESPnet
    `TransformerEncoder` names its input projection `embed.out.{0,1}` (`Linear` then `LayerNorm`) and its single
    feed-forward `feed_forward.{w_1,w_2}`; the module layout here names those `embed.{0,1}` and
    `feed_forward.{intermediate_dense,output_dense}` respectively (`after_norm` and every `self_attn.*` key are
    already named identically). The self-attention pre-norm and feed-forward pre-norm are named `norm_mha`/
    `norm_ff` in the text encoder and flow encoder checkpoints but `norm1`/`norm2` in the LLM checkpoint;
    `norm_mha_name`/`norm_ff_name` select which the source checkpoint uses.

    Args:
        old_prefix (`str`):
            Dotted prefix under which this encoder stack's weights live in the original checkpoint.
        new_prefix (`str`):
            Dotted prefix under which this encoder stack's weights live in the `transformers` module.
        norm_mha_name (`str`, *optional*, defaults to `"norm_mha"`):
            Name of the self-attention pre-norm layer in the original checkpoint (`"norm_mha"` or `"norm1"`).
        norm_ff_name (`str`, *optional*, defaults to `"norm_ff"`):
            Name of the feed-forward pre-norm layer in the original checkpoint (`"norm_ff"` or `"norm2"`).

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    op = re.escape(old_prefix)
    return [
        WeightRenaming(rf"{op}\.embed\.out\.(\d+)\.(weight|bias)", rf"{new_prefix}.embed.\1.\2"),
        WeightRenaming(rf"{op}\.after_norm\.(weight|bias)", rf"{new_prefix}.after_norm.\1"),
        WeightRenaming(
            rf"{op}\.encoders\.(\d+)\.{norm_mha_name}\.(weight|bias)",
            rf"{new_prefix}.layers.\1.self_attn_layer_norm.\2",
        ),
        WeightRenaming(
            rf"{op}\.encoders\.(\d+)\.{norm_ff_name}\.(weight|bias)", rf"{new_prefix}.layers.\1.norm_ff.\2"
        ),
        WeightRenaming(
            rf"{op}\.encoders\.(\d+)\.feed_forward\.w_1\.(weight|bias)",
            rf"{new_prefix}.layers.\1.feed_forward.intermediate_dense.\2",
        ),
        WeightRenaming(
            rf"{op}\.encoders\.(\d+)\.feed_forward\.w_2\.(weight|bias)",
            rf"{new_prefix}.layers.\1.feed_forward.output_dense.\2",
        ),
        WeightRenaming(
            rf"{op}\.encoders\.(\d+)\.self_attn\.(.+)", rf"{new_prefix}.layers.\1.self_attn.\2"
        ),
    ]


def build_llm_weight_conversion_mapping(prefix: str = "") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate an original CosyVoice v1 `llm.pt` state dict into
    [`CosyVoiceV1LLM`]'s module layout.

    Args:
        prefix (`str`, *optional*, defaults to `""`):
            Prefix under which the LLM submodule's weights live in a composite checkpoint's state dict.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    rules = []
    rules += _rel_position_encoder_renaming(f"{prefix}text_encoder", f"{prefix}text_encoder.encoder")
    rules += _rel_position_encoder_renaming(
        f"{prefix}llm", f"{prefix}llm", norm_mha_name="norm1", norm_ff_name="norm2"
    )
    return rules


def build_flow_weight_conversion_mapping(prefix: str = "") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate an original CosyVoice v1 `flow.pt` state dict into
    [`CosyVoiceV1FlowMatchingModel`]'s module layout.

    Args:
        prefix (`str`, *optional*, defaults to `""`):
            Prefix under which the flow-matching submodule's weights live in a composite checkpoint's state dict.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    return _rel_position_encoder_renaming(f"{prefix}encoder", f"{prefix}encoder")


def _conv_weight_norm_renaming(old_prefix: str, new_prefix: str) -> list[WeightRenaming]:
    """Renaming rules for one classic (`weight_g`/`weight_v`) weight-normalized conv layer into the
    `torch.nn.utils.parametrizations.weight_norm` layout `apply_weight_norm()`-wrapped modules use here."""
    op, npr = re.escape(old_prefix), new_prefix
    return [
        WeightRenaming(rf"{op}\.weight_g", rf"{npr}.parametrizations.weight.original0"),
        WeightRenaming(rf"{op}\.weight_v", rf"{npr}.parametrizations.weight.original1"),
        WeightRenaming(rf"{op}\.bias", rf"{npr}.bias"),
    ]


def build_hift_weight_conversion_mapping(prefix: str = "") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate an original CosyVoice v1 `hift.pt` state dict into
    [`CosyVoiceV1HiFTGenerator`]'s module layout. Every conv the reference NSF-HiFiGAN wraps in classic
    `weight_norm` (`conv_pre`, `conv_post`, `ups.*`, `f0_predictor.condnet.*`, and every `resblocks`/
    `source_resblocks` conv) is translated to the `parametrizations.weight` layout; `m_source.l_linear` and
    `source_downs.*` are plain (unwrapped) layers already named identically.

    Args:
        prefix (`str`, *optional*, defaults to `""`):
            Prefix under which the vocoder submodule's weights live in a composite checkpoint's state dict.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    p = prefix
    rules = []
    rules += _conv_weight_norm_renaming(f"{p}conv_pre", f"{p}conv_pre")
    rules += _conv_weight_norm_renaming(f"{p}conv_post", f"{p}conv_post")
    rules.append(
        WeightRenaming(rf"{re.escape(p)}ups\.(\d+)\.weight_g", rf"{p}ups.\1.parametrizations.weight.original0")
    )
    rules.append(
        WeightRenaming(rf"{re.escape(p)}ups\.(\d+)\.weight_v", rf"{p}ups.\1.parametrizations.weight.original1")
    )
    rules.append(WeightRenaming(rf"{re.escape(p)}ups\.(\d+)\.bias", rf"{p}ups.\1.bias"))
    for block in ("resblocks", "source_resblocks"):
        for conv in ("convs1", "convs2"):
            rules.append(
                WeightRenaming(
                    rf"{re.escape(p)}{block}\.(\d+)\.{conv}\.(\d+)\.weight_g",
                    rf"{p}{block}.\1.{conv}.\2.parametrizations.weight.original0",
                )
            )
            rules.append(
                WeightRenaming(
                    rf"{re.escape(p)}{block}\.(\d+)\.{conv}\.(\d+)\.weight_v",
                    rf"{p}{block}.\1.{conv}.\2.parametrizations.weight.original1",
                )
            )
            rules.append(
                WeightRenaming(
                    rf"{re.escape(p)}{block}\.(\d+)\.{conv}\.(\d+)\.bias", rf"{p}{block}.\1.{conv}.\2.bias"
                )
            )
    rules.append(
        WeightRenaming(
            rf"{re.escape(p)}f0_predictor\.condnet\.(\d+)\.weight_g",
            rf"{p}f0_predictor.condnet.\1.parametrizations.weight.original0",
        )
    )
    rules.append(
        WeightRenaming(
            rf"{re.escape(p)}f0_predictor\.condnet\.(\d+)\.weight_v",
            rf"{p}f0_predictor.condnet.\1.parametrizations.weight.original1",
        )
    )
    rules.append(
        WeightRenaming(rf"{re.escape(p)}f0_predictor\.condnet\.(\d+)\.bias", rf"{p}f0_predictor.condnet.\1.bias")
    )
    return rules
