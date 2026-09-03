"""Checkpoint conversion for CosyVoice v1."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError

from .configuration_cosyvoice_v1 import CosyVoiceV1Config


# The v1 repositories the CosyVoice authors published. `CosyVoice-300M` is the base model;
# `-SFT` and `-Instruct` add a `spk2info.pt` holding the built in speakers.
PUBLISHED_CHECKPOINTS = (
    "FunAudioLLM/CosyVoice-300M",
    "FunAudioLLM/CosyVoice-300M-SFT",
    "FunAudioLLM/CosyVoice-300M-Instruct",
)

# The three files of a released directory, one per network, keyed by the submodule that holds them.
# Their key namespaces overlap, `spk_embed_affine_layer` sits in `llm.pt` and in `flow.pt` alike, so
# the merge has to prefix them apart before any rename rule sees a key.
CHECKPOINT_FILES = {"llm": "llm.pt", "flow": "flow.pt", "hift": "hift.pt"}

# The speech tokenizer, the speaker encoder and the speaker table the processor opens out of the
# same directory. Only the base model ships no `spk2info.pt`.
SPEECH_TOKENIZER_FILE = "speech_tokenizer_v1.onnx"
SPEAKER_ENCODER_FILE = "campplus.onnx"
SPEAKER_INFO_FILE = "spk2info.pt"

# Upstream tokenizes text with `whisper.tokenizer.get_tokenizer(multilingual=True,
# num_languages=100, language='en', task='transcribe')`, whose 51866 entry vocabulary is the one
# `openai/whisper-large-v3` ships.
TEXT_TOKENIZER_ID = "openai/whisper-large-v3"

# Fields of `from_pretrained` that select a revision and a cache rather than shape the model.
DOWNLOAD_KWARGS = ("cache_dir", "force_download", "local_files_only", "proxies", "revision", "token")


def resolve_checkpoint(
    source: "str | Path | None", filenames: tuple[str, ...], patterns: tuple[str, ...] = (), **kwargs
) -> "Path | None":
    r"""
    Fetches a released CosyVoice directory and returns where it landed.

    Args:
        source (`str` or `os.PathLike`, *optional*):
            Repository id or local directory.
        filenames (`tuple[str, ...]`):
            Files the caller reads. Their presence is what tells a released directory apart from a
            directory holding a saved model.
        patterns (`tuple[str, ...]`, *optional*):
            Further `allow_patterns` fetched alongside `filenames`, which need not be present.
        kwargs (`dict`, *optional*):
            Fields of [`DOWNLOAD_KWARGS`] are forwarded to `snapshot_download`; the rest are ignored.

    Returns:
        `Path` or `None`: The local directory, or `None` when `source` holds no released checkpoint.
    """
    if source is None:
        return None
    directory = Path(source)
    if not directory.is_dir():
        download_kwargs = {name: kwargs[name] for name in DOWNLOAD_KWARGS if name in kwargs}
        try:
            directory = Path(
                snapshot_download(str(source), allow_patterns=list(filenames) + list(patterns), **download_kwargs)
            )
        except (HFValidationError, OSError):
            return None
    if all((directory / name).is_file() for name in filenames):
        return directory
    return None


def load_checkpoint(directory: "str | Path") -> dict[str, torch.Tensor]:
    r"""
    Merges the three files of a released CosyVoice directory into one state dict.

    Args:
        directory (`str` or `os.PathLike`):
            Local directory holding [`CHECKPOINT_FILES`].

    Returns:
        `dict[str, torch.Tensor]`: The tensors of the three files, each key prefixed by the submodule
        the file belongs to.
    """
    state_dict = {}
    for prefix, name in CHECKPOINT_FILES.items():
        tensors = torch.load(Path(directory) / name, map_location="cpu", weights_only=True)
        for key, value in tensors.items():
            state_dict[f"{prefix}.{key}"] = value.contiguous()
    return state_dict


def build_config(directory: "str | Path", **overrides) -> CosyVoiceV1Config:
    r"""
    Builds the [`CosyVoiceV1Config`] of a released CosyVoice v1 directory.

    Every released v1 directory ships the same `cosyvoice.yaml`, so the geometry is the class
    defaults and only the overrides a caller passes change it.

    Args:
        directory (`str` or `os.PathLike`):
            Local directory of the released checkpoint.
        overrides (`dict`, *optional*):
            Configuration fields overriding the released geometry.

    Returns:
        [`CosyVoiceV1Config`]: The configuration.
    """
    return CosyVoiceV1Config(**overrides)


__all__ = [
    "CHECKPOINT_FILES",
    "DOWNLOAD_KWARGS",
    "PUBLISHED_CHECKPOINTS",
    "SPEAKER_ENCODER_FILE",
    "SPEAKER_INFO_FILE",
    "SPEECH_TOKENIZER_FILE",
    "TEXT_TOKENIZER_ID",
    "build_config",
    "load_checkpoint",
    "resolve_checkpoint",
]
