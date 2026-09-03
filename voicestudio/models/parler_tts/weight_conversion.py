"""Checkpoint conversion for Parler-TTS."""

import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import DacConfig, DacModel
from transformers.models.dac.convert_dac_checkpoint import apply_weight_norm, recursively_load_weights


# The published Parler-TTS checkpoints store the audio codec as the original
# `descript-audio-codec` module under an `audio_encoder.model.` prefix: weight-normalized
# convolutions kept as separate `weight_g`/`weight_v` tensors, and block indices numbered
# the way the upstream `DAC` module nests them. `transformers`'s `DacModel` uses flattened
# `encoder.block.<i>.res_unit<j>` names with the weight norm already folded in.
_DAC_PREFIX = "audio_encoder.model."

# Subfolder the converted DAC codec is additionally saved to, standalone, for `ParlerTTSProcessor`
# to load with a plain `DacModel.from_pretrained`.
_AUDIO_TOKENIZER_SUBFOLDER = "audio_encoder"

_COPIED_FILES = (
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _build_dac_config(legacy_config: dict) -> DacConfig:
    """
    Builds a `transformers` [`DacConfig`] from a vendored Parler-TTS `DACConfig` dict, which
    carries `num_codebooks`, `latent_dim`, `model_bitrate` and `frame_rate` and omits every
    architecture hyperparameter. `frame_rate` is a read-only property on [`DacConfig`], so the
    legacy dict cannot be passed through to it directly.

    Args:
        legacy_config (`dict`):
            The `audio_encoder` entry of a published Parler-TTS `config.json`.

    Returns:
        [`DacConfig`]: The equivalent `transformers` codec configuration.

    Raises:
        ValueError: If the resulting latent width disagrees with the checkpoint's `latent_dim`.
    """
    config = DacConfig(
        n_codebooks=legacy_config["num_codebooks"],
        codebook_size=legacy_config["codebook_size"],
        sampling_rate=legacy_config["sampling_rate"],
    )
    if config.hidden_size != legacy_config["latent_dim"]:
        raise ValueError(
            f"Derived a latent width of {config.hidden_size} from the default DAC encoder shape, but the "
            f"checkpoint declares latent_dim={legacy_config['latent_dim']}."
        )
    return config


def convert(checkpoint_path, output_dir):
    """
    Converts a published Parler-TTS checkpoint into one that loads directly with
    [`ParlerTTSForConditionalGeneration`], rewriting the vendored DAC audio codec into
    `transformers`'s own [`DacModel`] configuration and weight layout.

    Args:
        checkpoint_path (`str`):
            A Hugging Face repo id or a local directory holding the published checkpoint.
        output_dir (`str`):
            Directory the converted checkpoint is written to.

    Returns:
        `str`: The `output_dir` that was written.
    """
    source = Path(checkpoint_path)
    if not source.is_dir():
        source = Path(snapshot_download(checkpoint_path))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text())
    dac_config = _build_dac_config(config["audio_encoder"])

    state_dict = load_file(source / "model.safetensors")
    dac_state_dict = {
        key[len(_DAC_PREFIX) :]: value for key, value in state_dict.items() if key.startswith(_DAC_PREFIX)
    }
    if not dac_state_dict:
        raise ValueError(f"No `{_DAC_PREFIX}` weights found in {source / 'model.safetensors'}.")

    dac_model = DacModel(dac_config)
    apply_weight_norm(dac_model)
    recursively_load_weights(dac_state_dict, dac_model, "dac_44khz")
    dac_model.remove_weight_norm()

    converted = {key: value for key, value in state_dict.items() if not key.startswith("audio_encoder.")}
    converted.update({f"audio_encoder.{key}": value for key, value in dac_model.state_dict().items()})

    config["audio_encoder"] = dac_config.to_dict()
    config["audio_encoder"]["model_type"] = DacConfig.model_type

    save_file(
        {key: value.contiguous() for key, value in converted.items()},
        target / "model.safetensors",
        metadata={"format": "pt"},
    )
    (target / "config.json").write_text(json.dumps(config, indent=2))
    for name in _COPIED_FILES:
        if (source / name).is_file():
            shutil.copy(source / name, target / name)

    dac_model.save_pretrained(target / _AUDIO_TOKENIZER_SUBFOLDER)

    return str(target)


__all__ = ["convert"]
