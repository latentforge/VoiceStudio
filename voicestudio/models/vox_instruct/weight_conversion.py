"""Checkpoint conversion for VoxInstruct."""

import importlib.abc
import importlib.util
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import joblib
import torch
from huggingface_hub import snapshot_download
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.encodec.configuration_encodec import EncodecConfig
from transformers.models.hubert.configuration_hubert import HubertConfig
from transformers.models.mt5.configuration_mt5 import MT5Config

from ..vocos.weight_conversion import build_model_files as build_vocoder_files
from .configuration_vox_instruct import VoxInstructARConfig, VoxInstructConfig, VoxInstructNARConfig
from .feature_extraction_vox_instruct import VoxInstructFeatureExtractor
from .modeling_vox_instruct import VoxInstructForConditionalGeneration
from .processing_vox_instruct import VoxInstructProcessor


# Layout of the `niobures/VoxInstruct` mirror of the released Google Drive folder.
CHECKPOINT_ROOT = "models/VoxInstruct/pretrained"
AR_CHECKPOINT = f"{CHECKPOINT_ROOT}/voxinstruct-sft-checkpoint/ar_1800k.pyt"
NAR_CHECKPOINT = f"{CHECKPOINT_ROOT}/voxinstruct-sft-checkpoint/nar_1800k.pyt"
SEMANTIC_CHECKPOINT = f"{CHECKPOINT_ROOT}/hubert-base-checkpoint/hubert_base_ls960.pt"
SEMANTIC_KMEANS = f"{CHECKPOINT_ROOT}/hubert-base-checkpoint/hubert_base_ls960_L9_km500.bin"
AUDIO_CHECKPOINT = f"{CHECKPOINT_ROOT}/encodec-checkpoint/encodec_24khz-d7cc33bc.th"
AUDIO_CONFIG = f"{CHECKPOINT_ROOT}/encodec-checkpoint/encodec_processor/config.json"
TEXT_ENCODER_CONFIG = f"{CHECKPOINT_ROOT}/google-mt5-base-checkpoint/config.json"
VOCODER_DIR = f"{CHECKPOINT_ROOT}/vocos-encodec-24khz"

# The released `hubert_base_ls960.pt` pickles the `fairseq` training task and its `omegaconf` configuration next to
# the tensors. Only the tensors are read, so those classes are resolved to inert placeholders.
_PLACEHOLDER_PACKAGES = ("fairseq", "omegaconf")


class _PickledPlaceholder:
    """Stands in for a class of an uninstalled package while a legacy checkpoint is unpickled."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _PlaceholderModule(types.ModuleType):
    """Module whose every attribute is a fresh [`_PickledPlaceholder`] subclass."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (_PickledPlaceholder,), {"__module__": self.__name__})
        setattr(self, name, placeholder)
        return placeholder


class _PlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import hook serving [`_PlaceholderModule`] for anything under the packages it is built with."""

    def __init__(self, roots: tuple[str, ...]):
        self.roots = roots

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.roots:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _PlaceholderModule(spec.name)

    def exec_module(self, module):
        pass


@contextmanager
def placeholder_packages(*names: str):
    r"""
    Resolves the named packages, and anything under them, to inert placeholder classes for the duration of the block,
    unless they are really installed. The hook sits last on the import path, so an installed package still wins.

    Args:
        *names (`str`):
            Top level package names to stand in for.
    """
    finder = _PlaceholderFinder(names)
    sys.meta_path.append(finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for loaded in [key for key in sys.modules if isinstance(sys.modules[key], _PlaceholderModule)]:
            del sys.modules[loaded]


def _resolve(source: str) -> Path:
    """Returns a local directory holding the released layout, downloading the repository if needed."""
    path = Path(source)
    if path.is_dir():
        return path
    return Path(
        snapshot_download(
            source,
            allow_patterns=[f"{CHECKPOINT_ROOT}/**"],
        )
    )


def convert_text_encoder_key(key: str) -> str:
    r"""
    Maps one key of the released instruction encoder onto its [`VoxInstructTextEncoder`] name.

    Args:
        key (`str`):
            Key below the `t5_model.` prefix of a released checkpoint, in the layout `peft` gives a wrapped
            [`MT5EncoderModel`].

    Returns:
        `str`: The corresponding key of [`VoxInstructTextEncoder`].
    """
    key = key.removeprefix("base_model.model.")
    return re.sub(r"\.(lora_[AB])\.default\.weight$", r".\1.weight", key)


def convert_semantic_encoder_key(key: str) -> str:
    r"""
    Maps one key of the released `fairseq` HuBERT checkpoint onto its [`HubertModel`] name.

    Args:
        key (`str`):
            Key of the `model` entry of `hubert_base_ls960.pt`.

    Returns:
        `str`: The corresponding key of [`HubertModel`].
    """
    if key == "mask_emb":
        return "masked_spec_embed"
    if key.startswith("layer_norm."):
        return f"feature_projection.{key}"
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "feature_projection.projection.")
    key = re.sub(r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv.", key)
    key = re.sub(r"^feature_extractor\.conv_layers\.(\d+)\.2\.", r"feature_extractor.conv_layers.\1.layer_norm.", key)
    key = key.replace("encoder.pos_conv.0.bias", "encoder.pos_conv_embed.conv.bias")
    key = key.replace("encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.parametrizations.weight.original0")
    key = key.replace("encoder.pos_conv.0.weight_v", "encoder.pos_conv_embed.conv.parametrizations.weight.original1")
    key = re.sub(r"^encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention.", key)
    key = re.sub(r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"encoder.layers.\1.layer_norm.", key)
    key = re.sub(r"^encoder\.layers\.(\d+)\.fc1\.", r"encoder.layers.\1.feed_forward.intermediate_dense.", key)
    key = re.sub(r"^encoder\.layers\.(\d+)\.fc2\.", r"encoder.layers.\1.feed_forward.output_dense.", key)
    return key


def convert_audio_codec_key(key: str) -> str:
    r"""
    Maps one key of the released standalone EnCodec checkpoint onto its [`EncodecModel`] name.

    Args:
        key (`str`):
            Key of `encodec_24khz-d7cc33bc.th`.

    Returns:
        `str`: The corresponding key of [`EncodecModel`].
    """
    key = key.replace("quantizer.vq.layers.", "quantizer.layers.").replace("._codebook.", ".codebook.")
    key = re.sub(r"^(encoder|decoder)\.model\.", r"\1.layers.", key)
    for module in ("conv", "convtr"):
        key = key.replace(f".{module}.{module}.weight_g", ".conv.parametrizations.weight.original0")
        key = key.replace(f".{module}.{module}.weight_v", ".conv.parametrizations.weight.original1")
        key = key.replace(f".{module}.{module}.bias", ".conv.bias")
    return key


def load_legacy_checkpoint(path: Path) -> dict:
    r"""
    Reads a checkpoint that pickles classes of packages this project does not depend on.

    Args:
        path (`Path`):
            File to read.

    Returns:
        `dict`: The unpickled object.
    """
    with placeholder_packages(*_PLACEHOLDER_PACKAGES):
        return torch.load(path, map_location="cpu", weights_only=False)


def build_config(directory: Path) -> VoxInstructConfig:
    r"""
    Builds a [`VoxInstructConfig`] from the configuration files the released checkpoint ships.

    The two stage architectures come from `configs/train_ar.yaml` and `configs/train_nar.yaml` of the
    [thuhcsi/VoxInstruct](https://github.com/thuhcsi/VoxInstruct) release, which the defaults of
    [`VoxInstructARConfig`] and [`VoxInstructNARConfig`] already carry.

    Args:
        directory (`Path`):
            Local directory holding the released layout.

    Returns:
        [`VoxInstructConfig`]: The equivalent VoiceStudio configuration.
    """
    text_encoder_config = MT5Config.from_json_file(directory / TEXT_ENCODER_CONFIG)
    audio_encoder_config = EncodecConfig.from_json_file(directory / AUDIO_CONFIG)
    vocoder_config, _ = build_vocoder_files(str(directory / VOCODER_DIR))
    return VoxInstructConfig(
        ar_config=VoxInstructARConfig(text_encoder_config=text_encoder_config),
        nar_config=VoxInstructNARConfig(text_encoder_config=text_encoder_config),
        audio_encoder_config=audio_encoder_config,
        semantic_encoder_config=HubertConfig(feat_proj_dropout=0.1),
        vocoder_config=vocoder_config,
    )


def convert_state_dict(directory: Path, config: VoxInstructConfig) -> dict[str, torch.Tensor]:
    r"""
    Reads the six released weight files and maps them onto [`VoxInstructForConditionalGeneration`] keys.

    Args:
        directory (`Path`):
            Local directory holding the released layout.
        config ([`VoxInstructConfig`]):
            Configuration the weights are checked against.

    Returns:
        `dict[str, torch.Tensor]`: The converted state dictionary.

    Raises:
        ValueError: If the k-means codebook does not match `config.semantic_num_clusters`.
    """
    state_dict = {}

    for prefix, checkpoint in (("ar", AR_CHECKPOINT), ("nar", NAR_CHECKPOINT)):
        stage = torch.load(directory / checkpoint, map_location="cpu", weights_only=False)["model"]
        for key, value in stage.items():
            if key.startswith("t5_model."):
                key = f"text_encoder.{convert_text_encoder_key(key.removeprefix('t5_model.'))}"
            state_dict[f"{prefix}.{key}"] = value

    semantic = load_legacy_checkpoint(directory / SEMANTIC_CHECKPOINT)["model"]
    for key, value in semantic.items():
        # The pretraining projection and its label embeddings sit outside `HubertModel`.
        if key.startswith("final_proj.") or key == "label_embs_concat":
            continue
        state_dict[f"semantic_encoder.encoder.{convert_semantic_encoder_key(key)}"] = value

    kmeans = joblib.load(directory / SEMANTIC_KMEANS)
    centers = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32)
    if centers.shape[0] != config.semantic_num_clusters:
        raise ValueError(
            f"The k-means codebook holds {centers.shape[0]} centroids, but the configuration declares "
            f"{config.semantic_num_clusters}."
        )
    state_dict["semantic_encoder.cluster_centers"] = centers

    audio = torch.load(directory / AUDIO_CHECKPOINT, map_location="cpu", weights_only=False)
    for key, value in audio.items():
        state_dict[f"audio_encoder.{convert_audio_codec_key(key)}"] = value

    _, vocoder = build_vocoder_files(str(directory / VOCODER_DIR))
    for key, value in vocoder.items():
        state_dict[f"vocoder.{key}"] = value

    return state_dict


def build_processor(directory: Path, config: VoxInstructConfig, tokenizer_id: str) -> VoxInstructProcessor:
    r"""
    Builds the [`VoxInstructProcessor`] matching a released checkpoint.

    Args:
        directory (`Path`):
            Local directory holding the released layout.
        config ([`VoxInstructConfig`]):
            Configuration the processor is built against.
        tokenizer_id (`str`):
            Repository holding a serialized fast tokenizer for the mT5 sentencepiece vocabulary. The `spiece.model`
            the checkpoint ships is byte for byte the one of `google/mt5-base`, whose repository carries no
            serialized fast tokenizer of its own.

    Returns:
        [`VoxInstructProcessor`]: The processor.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, subfolder="onnx")
    feature_extractor = VoxInstructFeatureExtractor(
        sampling_rate=config.sampling_rate,
        semantic_sampling_rate=config.semantic_sampling_rate,
        semantic_frame_multiple=config.semantic_frame_multiple,
    )
    return VoxInstructProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_text_len=config.ar_config.max_text_len,
        max_len=config.ar_config.max_position_embeddings - config.ar_config.max_text_len,
        num_language_ids=config.ar_config.num_language_ids,
        semantic_vocab_size=config.ar_config.semantic_vocab_size,
        acoustic_vocab_size=config.ar_config.acoustic_vocab_size,
        num_codebooks=config.num_codebooks,
    )


def convert(
    source: str = "niobures/VoxInstruct",
    output_dir: str = "voxinstruct-converted",
    tokenizer_id: str = "google/mt5-small",
) -> None:
    r"""
    Converts a released VoxInstruct checkpoint into a directory [`VoxInstructForConditionalGeneration`] and
    [`VoxInstructProcessor`] load from.

    Args:
        source (`str`, *optional*, defaults to `"niobures/VoxInstruct"`):
            Local directory or Hugging Face repository holding the released `models/VoxInstruct/pretrained` layout.
        output_dir (`str`, *optional*, defaults to `"voxinstruct-converted"`):
            Directory the converted checkpoint is written to.
        tokenizer_id (`str`, *optional*, defaults to `"google/mt5-small"`):
            Repository holding a serialized fast tokenizer for the mT5 sentencepiece vocabulary.

    Raises:
        RuntimeError: If the converted state dictionary does not cover the model exactly.
    """
    directory = _resolve(source)
    config = build_config(directory)
    state_dict = convert_state_dict(directory, config)

    model = VoxInstructForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_dir)
    build_processor(directory, config, tokenizer_id).save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="niobures/VoxInstruct")
    parser.add_argument("--output_dir", type=str, default="voxinstruct-converted")
    parser.add_argument("--tokenizer_id", type=str, default="google/mt5-small")
    args = parser.parse_args()

    convert(source=args.source, output_dir=args.output_dir, tokenizer_id=args.tokenizer_id)
