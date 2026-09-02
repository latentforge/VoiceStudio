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
    # `WeightRenaming.rename_source_key` only substitutes a single `\1` backreference (the first capturing
    # group inside the matched source pattern) into the target pattern, then splices that target text in for
    # exactly the matched span, leaving the rest of the original key untouched. So every rule below matches only
    # up through the piece that actually gets renamed (a `weight`/`bias` leaf's parent path, never the leaf name
    # itself), letting the identically-named leaf suffix (`.weight`, `.bias`, `.linear_q.weight`, `.pos_bias_u`,
    # ...) pass through unmatched instead of being re-captured and reinserted.
    op = re.escape(old_prefix)
    return [
        WeightRenaming(rf"{op}\.embed\.out\.(\d+)", rf"{new_prefix}.embed.\1"),
        WeightRenaming(rf"{op}\.after_norm", rf"{new_prefix}.after_norm"),
        WeightRenaming(
            rf"{op}\.encoders\.(\d+)\.{norm_mha_name}", rf"{new_prefix}.layers.\1.self_attn_layer_norm"
        ),
        WeightRenaming(rf"{op}\.encoders\.(\d+)\.{norm_ff_name}", rf"{new_prefix}.layers.\1.norm_ff"),
        WeightRenaming(
            rf"{op}\.encoders\.(\d+)\.feed_forward\.w_1", rf"{new_prefix}.layers.\1.feed_forward.intermediate_dense"
        ),
        WeightRenaming(
            rf"{op}\.encoders\.(\d+)\.feed_forward\.w_2", rf"{new_prefix}.layers.\1.feed_forward.output_dense"
        ),
        WeightRenaming(rf"{op}\.encoders\.(\d+)\.self_attn", rf"{new_prefix}.layers.\1.self_attn"),
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


def _estimator_downsample_renaming(old_prefix: str, new_prefix: str) -> list[WeightRenaming]:
    """Renaming rules for one `decoder.estimator.{down,up}_blocks.N.2` down/upsample module: the original
    checkpoint's `matcha.models.components.decoder.Downsample1D`/`Upsample1D` wraps its conv in a `.conv`
    submodule (`CosyVoiceV1Downsample1D`/`CosyVoiceV1Upsample1D` here keep that layout identically), except at the
    last stage where both the original and this module use a bare, unwrapped `nn.Conv1d` and the keys already
    match."""
    op = re.escape(old_prefix)
    return [WeightRenaming(rf"{op}\.conv\.(weight|bias)", rf"{new_prefix}.conv.\1")]


def _estimator_renaming(old_prefix: str, new_prefix: str, num_down_up_stages: int) -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate an original CosyVoice v1 `decoder.estimator.*` checkpoint
    prefix (`matcha.models.components.decoder.ConditionalDecoder`/`BasicTransformerBlock`-based) into
    [`CosyVoiceV1ConditionalDecoder`]'s module layout. Every key already matches identically
    (`time_mlp.linear_{1,2}`, every `.0` `ResnetBlock1D`, `final_block`, `final_proj`, and every `.1.M`
    transformer block's `norm1`/`attn1.to_{q,k,v,out.0}`/`norm3`/`ff.net.{0.proj,2}`) except the non-final
    `down_blocks`/`up_blocks` stages' `.2` down/upsample module, which the original `Downsample1D`/`Upsample1D`
    wraps in a `.conv` submodule that this module's [`CosyVoiceV1Downsample1D`]/[`CosyVoiceV1Upsample1D`] also
    uses (the final stage in both `down_blocks` and `up_blocks` uses a bare, unwrapped `nn.Conv1d` already named
    identically, so it needs no renaming rule).

    Args:
        old_prefix (`str`):
            Dotted prefix under which the estimator's weights live in the original checkpoint.
        new_prefix (`str`):
            Dotted prefix under which the estimator's weights live in the `transformers` module.
        num_down_up_stages (`int`):
            Number of `down_blocks`/`up_blocks` stages.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    rules = []
    for block in ("down_blocks", "up_blocks"):
        for i in range(num_down_up_stages - 1):
            rules += _estimator_downsample_renaming(
                f"{old_prefix}.{block}.{i}.2", f"{new_prefix}.{block}.{i}.2"
            )
    return rules


def build_flow_weight_conversion_mapping(prefix: str = "", num_down_up_stages: int = 2) -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate an original CosyVoice v1 `flow.pt` state dict into
    [`CosyVoiceV1FlowMatchingModel`]'s module layout.

    Args:
        prefix (`str`, *optional*, defaults to `""`):
            Prefix under which the flow-matching submodule's weights live in a composite checkpoint's state dict.
        num_down_up_stages (`int`, *optional*, defaults to 2):
            Number of `decoder.estimator.down_blocks`/`up_blocks` stages (`len(decoder_channels)` in
            [`CosyVoiceV1FlowConfig`]).

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    rules = _rel_position_encoder_renaming(f"{prefix}encoder", f"{prefix}encoder")
    rules += _estimator_renaming(f"{prefix}decoder.estimator", f"{prefix}decoder.estimator", num_down_up_stages)
    return rules


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
    # `WeightRenaming` only substitutes a single `\1` backreference into the target pattern (see
    # `_rel_position_encoder_renaming`'s docstring for the same issue), so a naive `(block idx)`/`(conv idx)`
    # two-group rule silently leaves a literal `\2` in the renamed key. Since the block/conv index path is
    # identical on both sides here (only the `weight_g`/`weight_v`/`bias` leaf changes), capture the whole
    # `<block idx>.<conv>.<conv idx>` chunk as a single group and splice it back in unchanged.
    for block in ("resblocks", "source_resblocks"):
        for conv in ("convs1", "convs2"):
            idx_path = rf"(\d+\.{conv}\.\d+)"
            rules.append(
                WeightRenaming(
                    rf"{re.escape(p)}{block}\.{idx_path}\.weight_g",
                    rf"{p}{block}.\1.parametrizations.weight.original0",
                )
            )
            rules.append(
                WeightRenaming(
                    rf"{re.escape(p)}{block}\.{idx_path}\.weight_v",
                    rf"{p}{block}.\1.parametrizations.weight.original1",
                )
            )
            rules.append(
                WeightRenaming(rf"{re.escape(p)}{block}\.{idx_path}\.bias", rf"{p}{block}.\1.bias")
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
