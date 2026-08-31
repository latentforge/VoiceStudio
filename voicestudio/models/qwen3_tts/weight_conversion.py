"""Checkpoint conversion for Qwen3-TTS."""

import re
from pathlib import Path

import torch
from transformers.conversion_mapping import Concatenate
from transformers.models.qwen3_tts.convert_qwen3_tts_to_hf import convert_checkpoint, load_original_checkpoint

from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


# `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.lm_head` is a single `nn.Linear`
# covering every codebook at once, but the checkpoint stores it as one `nn.Linear` per codebook
# (`talker.code_predictor.lm_head.<i>.weight`). `transformers`'s own `convert_qwen3_tts_to_hf`
# doesn't concatenate these, so they load as MISSING/UNEXPECTED and the code predictor's output
# head ends up at its random init.
_CODE_PREDICTOR_LM_HEAD_PATTERN = re.compile(r"^talker\.code_predictor\.lm_head\.(\d+)\.weight$")


def convert(checkpoint_path, output_dir, push_to_hub=None, bfloat16=True, max_shard_size="5GB"):
    """
    Converts a raw Qwen3-TTS checkpoint to the HF format via `transformers`'s own
    `convert_qwen3_tts_to_hf.convert_checkpoint`, then patches in the code predictor's real
    per-codebook `lm_head` weights, which that conversion drops.
    """
    convert_checkpoint(checkpoint_path, output_dir, push_to_hub=None, bfloat16=bfloat16, max_shard_size=max_shard_size)

    original_state_dict = load_original_checkpoint(Path(checkpoint_path))
    lm_head_keys = sorted(
        (int(match.group(1)), key)
        for key in original_state_dict
        if (match := _CODE_PREDICTOR_LM_HEAD_PATTERN.match(key)) is not None
    )
    if lm_head_keys:
        source_patterns = [key for _, key in lm_head_keys]
        merged = Concatenate(dim=0).convert(
            {key: original_state_dict[key] for key in source_patterns},
            source_patterns=source_patterns,
            target_patterns=["code_predictor.lm_head.weight"],
        )

        dtype = torch.bfloat16 if bfloat16 else torch.float32
        model = Qwen3TTSForConditionalGeneration.from_pretrained(output_dir, dtype=dtype)
        model.code_predictor.lm_head.weight.data.copy_(
            merged["code_predictor.lm_head.weight"].to(model.code_predictor.lm_head.weight.dtype)
        )
        model.save_pretrained(output_dir, max_shard_size=max_shard_size)
        if push_to_hub:
            model.push_to_hub(push_to_hub, max_shard_size=max_shard_size)


__all__ = ["convert"]
