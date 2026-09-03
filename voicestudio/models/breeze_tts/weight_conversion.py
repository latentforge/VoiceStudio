"""Checkpoint conversion for Breeze TTS 2."""

import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers.models.qwen3_tts_tokenizer_multi_codebook import (
    Qwen3TTSTokenizerMultiCodebookConfig,
    Qwen3TTSTokenizerMultiCodebookModel,
)
from transformers.models.qwen3_tts_tokenizer_multi_codebook.convert_qwen3_tts_tokenizer_multi_codebook_to_hf import (
    remap_keys,
)

from .modeling_breeze_tts import BreezeTTSForConditionalGeneration
from .processing_breeze_tts import AUDIO_TOKENIZER_SUBFOLDER


# Files the published repository ships alongside the weights that the converted checkpoint keeps verbatim, so
# the tokenizer and the generation defaults of the output directory are byte-for-byte the published ones.
_COPIED_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_COPIED_AUDIO_TOKENIZER_FILES = ("preprocessor_config.json",)


def convert_audio_tokenizer(checkpoint_path, output_dir, bfloat16: bool = True):
    """
    Converts the audio tokenizer bundled in a Breeze TTS 2 checkpoint to the key names
    [`Qwen3TTSTokenizerMultiCodebookModel`] expects.

    The bundled tokenizer is a raw Qwen3-TTS-Tokenizer-12Hz checkpoint, whose decoder quantizer is stored under
    `decoder.quantizer.rvq_first` and `decoder.quantizer.rvq_rest`. Loading it as-is leaves the whole decoder
    quantizer at its random initialization, so the codes a reference waveform is encoded to, and the waveform
    generated codes are decoded to, are both wrong.

    Args:
        checkpoint_path (`str` or `os.PathLike`):
            Local path of the Breeze TTS 2 checkpoint.
        output_dir (`str` or `os.PathLike`):
            Directory the converted audio tokenizer is written to, as an `audio_tokenizer` subfolder.
        bfloat16 (`bool`, *optional*, defaults to `True`):
            Whether the converted weights are saved in `bfloat16` rather than `float32`.

    Returns:
        `pathlib.Path`: the directory the converted audio tokenizer was written to.

    Raises:
        FileNotFoundError: if the checkpoint bundles no audio tokenizer.
    """
    source_dir = Path(checkpoint_path) / AUDIO_TOKENIZER_SUBFOLDER
    weights_path = source_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"'{checkpoint_path}' bundles no '{AUDIO_TOKENIZER_SUBFOLDER}/model.safetensors'.")

    target_dir = Path(output_dir) / AUDIO_TOKENIZER_SUBFOLDER
    target_dir.mkdir(parents=True, exist_ok=True)

    config = Qwen3TTSTokenizerMultiCodebookConfig.from_pretrained(source_dir)
    model = Qwen3TTSTokenizerMultiCodebookModel(config).to(torch.bfloat16 if bfloat16 else torch.float32)
    model.load_state_dict(remap_keys(load_file(str(weights_path))), strict=False)
    model.save_pretrained(str(target_dir))

    for file_name in _COPIED_AUDIO_TOKENIZER_FILES:
        source_file = source_dir / file_name
        if source_file.is_file():
            shutil.copyfile(source_file, target_dir / file_name)
    return target_dir


def convert(checkpoint_path, output_dir, push_to_hub=None, bfloat16: bool = True, max_shard_size: str = "5GB"):
    """
    Converts a published Breeze TTS 2 checkpoint to a directory [`BreezeTTSForConditionalGeneration`] and
    [`BreezeTTSProcessor`] load without remaining MISSING or UNEXPECTED keys.

    The model weights themselves are already in the expected layout and are re-saved unchanged; only the
    bundled audio tokenizer is remapped, by [`~weight_conversion.convert_audio_tokenizer`].

    Args:
        checkpoint_path (`str` or `os.PathLike`):
            Local path of the published checkpoint.
        output_dir (`str` or `os.PathLike`):
            Directory the converted checkpoint is written to.
        push_to_hub (`str`, *optional*):
            Repository id the converted checkpoint is pushed to.
        bfloat16 (`bool`, *optional*, defaults to `True`):
            Whether the converted weights are saved in `bfloat16` rather than `float32`.
        max_shard_size (`str`, *optional*, defaults to `"5GB"`):
            Largest size of a single weight shard.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if bfloat16 else torch.float32
    model = BreezeTTSForConditionalGeneration.from_pretrained(checkpoint_path, dtype=dtype)
    model.save_pretrained(str(output_path), max_shard_size=max_shard_size)

    for file_name in _COPIED_FILES:
        source_file = Path(checkpoint_path) / file_name
        if source_file.is_file():
            shutil.copyfile(source_file, output_path / file_name)

    convert_audio_tokenizer(checkpoint_path, output_path, bfloat16=bfloat16)

    if push_to_hub:
        model.push_to_hub(push_to_hub, max_shard_size=max_shard_size)


__all__ = ["convert", "convert_audio_tokenizer"]
