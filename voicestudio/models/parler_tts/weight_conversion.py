"""Checkpoint conversion for Parler-TTS."""

import re

from transformers.core_model_loading import WeightRenaming


def _conv_rules(old_prefix: str, new_prefix: str) -> list[WeightRenaming]:
    """
    Renaming rules for one weight-normalized conv layer: `DacModel.apply_weight_norm()` wraps every
    conv/proj layer in `torch.nn.utils.parametrizations.weight_norm`, which stores the magnitude and
    direction factors under `parametrizations.weight.original0`/`original1` instead of the classic
    `weight_g`/`weight_v` names original Parler-TTS checkpoints were saved with. `bias` is unaffected.
    """
    return [
        WeightRenaming(rf"{old_prefix}\.weight_g", rf"{new_prefix}.parametrizations.weight.original0"),
        WeightRenaming(rf"{old_prefix}\.weight_v", rf"{new_prefix}.parametrizations.weight.original1"),
        WeightRenaming(rf"{old_prefix}\.bias", rf"{new_prefix}.bias"),
    ]


def _res_unit_renaming(old_prefix: str, new_prefix: str, block_offset: int) -> list[WeightRenaming]:
    rules = []
    for unit_idx in range(3):
        old_unit = rf"{re.escape(old_prefix)}\.{unit_idx + block_offset}\.block"
        new_unit = f"{new_prefix}.res_unit{unit_idx + 1}"
        rules += _conv_rules(rf"{old_unit}\.1", f"{new_unit}.conv1")
        rules += _conv_rules(rf"{old_unit}\.3", f"{new_unit}.conv2")
        rules.append(WeightRenaming(rf"{old_unit}\.0\.alpha", rf"{new_unit}.snake1.alpha"))
        rules.append(WeightRenaming(rf"{old_unit}\.2\.alpha", rf"{new_unit}.snake2.alpha"))
    return rules


def build_dac_weight_conversion_mapping(prefix: str = "audio_encoder.") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate a `descript-audio-codec`-style state dict (the format
    original Parler-TTS checkpoints ship their audio encoder weights in) into the module layout of the
    `transformers` `DacModel`.

    Args:
        prefix (`str`, *optional*, defaults to `"audio_encoder."`):
            Prefix under which the audio encoder submodule's weights live in the composite Parler-TTS
            checkpoint's state dict.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    p = re.escape(prefix)
    rules = []
    rules += _conv_rules(rf"{p}model\.encoder\.block\.0", f"{prefix}encoder.conv1")
    rules.append(WeightRenaming(rf"{p}model\.encoder\.block\.5\.alpha", rf"{prefix}encoder.snake1.alpha"))
    rules += _conv_rules(rf"{p}model\.encoder\.block\.6", f"{prefix}encoder.conv2")
    rules += _conv_rules(rf"{p}model\.decoder\.model\.0", f"{prefix}decoder.conv1")
    rules.append(WeightRenaming(rf"{p}model\.decoder\.model\.5\.alpha", rf"{prefix}decoder.snake1.alpha"))
    rules += _conv_rules(rf"{p}model\.decoder\.model\.6", f"{prefix}decoder.conv2")
    rules.append(
        WeightRenaming(
            rf"{p}model\.quantizer\.quantizers\.(\d+)\.codebook\.weight",
            rf"{prefix}quantizer.quantizers.\1.codebook.weight",
        )
    )
    for i in range(4):
        rules += _res_unit_renaming(f"{prefix}model.encoder.block.{i + 1}.block", f"{prefix}encoder.block.{i}", 0)
        rules.append(
            WeightRenaming(
                rf"{p}model\.encoder\.block\.{i + 1}\.block\.3\.alpha", rf"{prefix}encoder.block.{i}.snake1.alpha"
            )
        )
        rules += _conv_rules(
            rf"{p}model\.encoder\.block\.{i + 1}\.block\.4", f"{prefix}encoder.block.{i}.conv1"
        )
        rules.append(
            WeightRenaming(
                rf"{p}model\.decoder\.model\.{i + 1}\.block\.0\.alpha", rf"{prefix}decoder.block.{i}.snake1.alpha"
            )
        )
        rules += _conv_rules(
            rf"{p}model\.decoder\.model\.{i + 1}\.block\.1", f"{prefix}decoder.block.{i}.conv_t1"
        )
        rules += _res_unit_renaming(f"{prefix}model.decoder.model.{i + 1}.block", f"{prefix}decoder.block.{i}", 2)
    for i in range(9):
        rules += _conv_rules(
            rf"{p}model\.quantizer\.quantizers\.{i}\.in_proj", f"{prefix}quantizer.quantizers.{i}.in_proj"
        )
        rules += _conv_rules(
            rf"{p}model\.quantizer\.quantizers\.{i}\.out_proj", f"{prefix}quantizer.quantizers.{i}.out_proj"
        )
    return rules
