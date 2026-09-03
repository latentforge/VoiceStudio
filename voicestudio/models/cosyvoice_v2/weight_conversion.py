"""Checkpoint conversion for CosyVoice v2."""

from pathlib import Path

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .configuration_cosyvoice_v2 import CosyVoiceV2Config


PUBLISHED_CHECKPOINTS = ("FunAudioLLM/CosyVoice2-0.5B",)

# The Qwen2 directory the language model is built from, which the released directory ships beside
# the three network files.
TEXT_MODEL_SUBDIR = "CosyVoice-BlankEN"

SPEECH_TOKENIZER_FILE = "speech_tokenizer_v2.onnx"


def build_config(directory: "str | Path", **overrides) -> CosyVoiceV2Config:
    r"""
    Builds the [`CosyVoiceV2Config`] of a released CosyVoice v2 directory.

    The geometry of the three CosyVoice networks is the class defaults; the Qwen2 sub configuration
    is read from the [`TEXT_MODEL_SUBDIR`] directory the checkpoint ships.

    Args:
        directory (`str` or `os.PathLike`):
            Local directory of the released checkpoint.
        overrides (`dict`, *optional*):
            Configuration fields overriding the released geometry.

    Returns:
        [`CosyVoiceV2Config`]: The configuration.
    """
    text_config = Qwen2Config.from_pretrained(str(Path(directory) / TEXT_MODEL_SUBDIR))
    return CosyVoiceV2Config(text_config=text_config, **overrides)


__all__ = [
    "PUBLISHED_CHECKPOINTS",
    "SPEECH_TOKENIZER_FILE",
    "TEXT_MODEL_SUBDIR",
    "build_config",
]
