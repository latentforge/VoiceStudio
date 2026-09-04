"""Checkpoint conversion for Spark-TTS."""

import json
import os
import re
import shutil
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2Config, Wav2Vec2Model

from ...utils.checkpoint_cache import CheckpointWriter, cached_conversion, file_identity
from ..spark_tts_bicodec.configuration_spark_tts_bicodec import SparkTTSBiCodecConfig
from .configuration_spark_tts import SparkTTSConfig
from .feature_extraction_spark_tts import SparkTTSFeatureExtractor


# The published `SparkAudio/Spark-TTS-0.5B` repo is three independently saved models in three subfolders plus two
# YAML files, none of which `from_pretrained` can read. BiCodec is stored as the plain `nn.Module` tree of the
# original `sparktts` package, with weight-normalized convolutions kept as separate `weight_g`/`weight_v` tensors,
# an `nn.Sequential` wave generator addressed by index, and the two ConvNeXt resampling stages nested under
# `downsample` for both the down-sampling encoder and the up-sampling prenet/postnet.
_BICODEC_RENAMES = (
    (r"^encoder\.encoder\.", "semantic_encoder.backbone."),
    (r"^encoder\.downsample\.(\d+)\.0\.", r"semantic_encoder.resample_layers.\1.sampler."),
    (r"^encoder\.downsample\.(\d+)\.1\.", r"semantic_encoder.resample_layers.\1.backbone."),
    (r"^encoder\.project\.", "semantic_encoder.project."),
    (r"^quantizer\.in_project\.", "quantizer.in_proj."),
    (r"^quantizer\.out_project\.", "quantizer.out_proj."),
    (r"^(prenet|postnet)\.downsample\.(\d+)\.0\.", r"\1.resample_layers.\2.sampler."),
    (r"^(prenet|postnet)\.downsample\.(\d+)\.1\.", r"\1.resample_layers.\2.backbone."),
    (r"^(prenet|postnet)\.vocos_backbone\.", r"\1.backbone."),
    (r"^speaker_encoder\.speaker_encoder\.layer1\.", "speaker_encoder.encoder.layer1."),
    (r"^speaker_encoder\.speaker_encoder\.conv\.", "speaker_encoder.encoder.mfa_conv."),
    (r"^speaker_encoder\.speaker_encoder\.(pool|bn|linear)\.", r"speaker_encoder.encoder.\1."),
    (
        r"^speaker_encoder\.perceiver_sampler\.layers\.(\d+)\.0\.",
        r"speaker_encoder.perceiver_resampler.layers.\1.attn.",
    ),
    (
        r"^speaker_encoder\.perceiver_sampler\.layers\.(\d+)\.1\.0\.",
        r"speaker_encoder.perceiver_resampler.layers.\1.ff.fc1.",
    ),
    (
        r"^speaker_encoder\.perceiver_sampler\.layers\.(\d+)\.1\.2\.",
        r"speaker_encoder.perceiver_resampler.layers.\1.ff.fc2.",
    ),
    (r"^speaker_encoder\.perceiver_sampler\.", "speaker_encoder.perceiver_resampler."),
    (r"\.convnext\.(\d+)\.", r".layers.\1."),
    (r"\.de_conv_upsampler\.", ".upsampler."),
    (r"\.conv_downsampler\.", ".downsampler."),
)

_SE_RES2BLOCK_PARTS = {"0": "conv1", "1": "res2conv", "2": "conv2", "3": "se"}

_WAVE_GENERATOR_RES_UNIT_PARTS = {"0": "snake1", "1": "conv1", "2": "snake2", "3": "conv2"}

_TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

# The files `convert` reads. The published repo also carries figures and logos under `src/`, which nothing here
# reads.
_PUBLISHED_PATTERNS = ("config.yaml", "BiCodec/*", "LLM/*", "wav2vec2-large-xlsr-53/*")

# Directory of `convert`'s output that BiCodec is written to, which is what `SparkTTSProcessor` reads it back from.
AUDIO_TOKENIZER_SUBFOLDER = "audio_tokenizer"

# The configuration files of the published layout, whose presence together is what tells it apart from a directory
# `convert` wrote and whose snapshot names the revision the conversion is keyed on. They are the whole of what a
# load has to resolve before the cache answers it, so none of them is large.
_PUBLISHED_FILES = (
    "config.yaml",
    "BiCodec/config.yaml",
    "LLM/config.json",
    "wav2vec2-large-xlsr-53/config.json",
)


def _rename_bicodec_key(key: str) -> str:
    """
    Rewrite one original BiCodec parameter path into its [`SparkTTSBiCodecModel`] equivalent.

    Args:
        key (`str`):
            Parameter path as stored in `BiCodec/model.safetensors`.

    Returns:
        `str`: The corresponding path on [`SparkTTSBiCodecModel`].
    """
    match = re.match(r"^speaker_encoder\.speaker_encoder\.layer(\d+)\.se_res2block\.(\d+)\.(.*)$", key)
    if match is not None:
        layer_index, part_index, suffix = match.groups()
        part = _SE_RES2BLOCK_PARTS[part_index]
        return f"speaker_encoder.encoder.layers.{int(layer_index) - 2}.{part}.{suffix}"

    match = re.match(r"^decoder\.model\.(\d+)\.block\.(\d+)(?:\.block\.(\d+))?\.(.*)$", key)
    if match is not None:
        block_index, inner_index, unit_index, suffix = match.groups()
        block = f"wave_generator.blocks.{int(block_index) - 1}"
        if unit_index is None:
            return f"{block}.{'snake' if inner_index == '0' else 'conv_t'}.{suffix}"
        part = _WAVE_GENERATOR_RES_UNIT_PARTS[unit_index]
        return f"{block}.res_unit{int(inner_index) - 1}.{part}.{suffix}"

    match = re.match(r"^decoder\.model\.(\d+)\.(.*)$", key)
    if match is not None:
        index, suffix = match.groups()
        return {"0": "wave_generator.conv_in", "5": "wave_generator.snake_out", "6": "wave_generator.conv_out"}[
            index
        ] + f".{suffix}"

    for pattern, replacement in _BICODEC_RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


def _fold_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Merge every `weight_g`/`weight_v` pair into the single `weight` tensor the module tree declares.

    The original BiCodec applies weight normalization at construction time and strips it again right after loading a
    checkpoint, so the folded tensors are exactly the ones it runs with.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            Weights as stored in `BiCodec/model.safetensors`.

    Returns:
        `dict[str, torch.Tensor]`: The same weights with weight normalization folded in.
    """
    folded = {}
    for key, value in state_dict.items():
        if key.endswith(".weight_v"):
            prefix = key[: -len(".weight_v")]
            magnitude = state_dict[f"{prefix}.weight_g"]
            norm = value.norm(dim=tuple(range(1, value.ndim)), keepdim=True)
            folded[f"{prefix}.weight"] = value * magnitude / norm
        elif not key.endswith(".weight_g"):
            folded[key] = value
    return folded


def _build_bicodec_config(bicodec_config: dict, semantic_model_config, sampling_rate: int) -> SparkTTSBiCodecConfig:
    """
    Build a [`SparkTTSBiCodecConfig`] from the `audio_tokenizer` section of the original `BiCodec/config.yaml`.

    Args:
        bicodec_config (`dict`):
            The `audio_tokenizer` mapping of `BiCodec/config.yaml`.
        semantic_model_config ([`PreTrainedConfig`]):
            Configuration of the self-supervised model bundled in the checkpoint.
        sampling_rate (`int`):
            Sample rate declared by the repo-level `config.yaml`.

    Returns:
        [`SparkTTSBiCodecConfig`]: The equivalent `transformers` configuration.
    """
    encoder = bicodec_config["encoder"]
    decoder = bicodec_config["decoder"]
    quantizer = bicodec_config["quantizer"]
    speaker_encoder = bicodec_config["speaker_encoder"]
    prenet = bicodec_config["prenet"]
    postnet = bicodec_config["postnet"]

    return SparkTTSBiCodecConfig(
        semantic_model_config=semantic_model_config,
        sampling_rate=sampling_rate,
        hop_length=bicodec_config["mel_params"]["hop_length"],
        hidden_size=quantizer["input_dim"],
        vocos_dim=encoder["vocos_dim"],
        vocos_intermediate_dim=encoder["vocos_intermediate_dim"],
        encoder_num_layers=encoder["vocos_num_layers"],
        encoder_sample_ratios=encoder["sample_ratios"],
        prenet_num_layers=prenet["vocos_num_layers"],
        prenet_sample_ratios=prenet.get("sample_ratios", [1, 1]),
        prenet_use_tanh_at_final=prenet.get("use_tanh_at_final", False),
        postnet_num_layers=postnet["vocos_num_layers"],
        postnet_sample_ratios=postnet.get("sample_ratios", [1, 1]),
        postnet_use_tanh_at_final=postnet.get("use_tanh_at_final", False),
        codebook_size=quantizer["codebook_size"],
        codebook_dim=quantizer["codebook_dim"],
        commitment_weight=quantizer["commitment"],
        codebook_loss_weight=quantizer["codebook_loss_weight"],
        threshold_ema_dead_code=quantizer["threshold_ema_dead_code"],
        num_mel_bins=bicodec_config["mel_params"]["num_mels"],
        speaker_latent_dim=speaker_encoder["latent_dim"],
        num_speaker_tokens=speaker_encoder["token_num"],
        fsq_levels=speaker_encoder["fsq_levels"],
        fsq_num_quantizers=speaker_encoder["fsq_num_quantizers"],
        wave_generator_hidden_size=decoder["channels"],
        upsample_rates=decoder["rates"],
        upsample_kernel_sizes=decoder["kernel_sizes"],
    )


def convert(checkpoint_path, output_dir):
    """
    Convert a published Spark-TTS checkpoint into one that loads directly with
    [`SparkTTSForConditionalGeneration`] and [`SparkTTSProcessor`].

    The language model, its tokenizer and the feature extractor are written to `output_dir`; BiCodec and the
    self-supervised model it reads features from are written to `output_dir/audio_tokenizer`, which is where
    [`SparkTTSProcessor`] loads its `audio_tokenizer` from.

    Args:
        checkpoint_path (`str`):
            A Hugging Face repo id or a local directory holding the published checkpoint.
        output_dir (`str`):
            Directory the converted checkpoint is written to.

    Returns:
        `str`: The `output_dir` that was written.

    Raises:
        ValueError: If the checkpoint rounds the reference excerpt to a hop the mel spectrogram does not use, or if
            the resulting BiCodec weights do not match [`SparkTTSBiCodecModel`]'s parameter tree.
    """
    source = Path(checkpoint_path)
    if not source.is_dir():
        source = Path(snapshot_download(checkpoint_path, allow_patterns=list(_PUBLISHED_PATTERNS)))
    target = Path(output_dir)
    audio_tokenizer_target = target / AUDIO_TOKENIZER_SUBFOLDER
    audio_tokenizer_target.mkdir(parents=True, exist_ok=True)

    repo_config = yaml.safe_load((source / "config.yaml").read_text())
    bicodec_config = yaml.safe_load((source / "BiCodec" / "config.yaml").read_text())["audio_tokenizer"]

    # The repo-level `latent_hop_length`, which rounds the reference excerpt, and the codec's own mel hop size are
    # two separate knobs upstream that happen to agree. `SparkTTSFeatureExtractor` exposes a single `hop_length`.
    hop_length = bicodec_config["mel_params"]["hop_length"]
    if repo_config["latent_hop_length"] != hop_length:
        raise ValueError(
            f"This checkpoint sets latent_hop_length={repo_config['latent_hop_length']} but a mel hop size of "
            f"{hop_length}. SparkTTSFeatureExtractor uses one `hop_length` for both."
        )

    semantic_model = Wav2Vec2Model.from_pretrained(source / "wav2vec2-large-xlsr-53")
    audio_tokenizer_config = _build_bicodec_config(bicodec_config, semantic_model.config, repo_config["sample_rate"])

    bicodec_state_dict = _fold_weight_norm(load_file(source / "BiCodec" / "model.safetensors"))
    converted = {_rename_bicodec_key(key): value for key, value in bicodec_state_dict.items()}
    converted.update({f"semantic_model.{key}": value for key, value in semantic_model.state_dict().items()})

    from ..spark_tts_bicodec.modeling_spark_tts_bicodec import SparkTTSBiCodecModel

    with torch.device("meta"):
        expected = dict(SparkTTSBiCodecModel(audio_tokenizer_config).state_dict())
    if set(converted) != set(expected):
        raise ValueError(
            f"Converted BiCodec weights do not match SparkTTSBiCodecModel: "
            f"missing {sorted(set(expected) - set(converted))}, unexpected {sorted(set(converted) - set(expected))}."
        )
    mismatched = {key: (tuple(value.shape), tuple(expected[key].shape)) for key, value in converted.items() if value.shape != expected[key].shape}
    if mismatched:
        raise ValueError(f"Converted BiCodec weights have the wrong shape: {mismatched}.")

    audio_tokenizer_config.save_pretrained(audio_tokenizer_target)
    with CheckpointWriter(audio_tokenizer_target) as writer:
        for key in list(converted):
            writer.add(key, converted.pop(key))

    text_config = Qwen2Config.from_pretrained(source / "LLM").to_dict()
    for key in ("model_type", "architectures", "transformers_version"):
        text_config.pop(key, None)
    config = SparkTTSConfig(
        **text_config,
        sampling_rate=repo_config["sample_rate"],
        ref_segment_duration=repo_config["ref_segment_duration"],
        volume_normalize=repo_config["volume_normalize"],
        architectures=["SparkTTSForConditionalGeneration"],
    )
    config.save_pretrained(target)
    shutil.copy(source / "LLM" / "model.safetensors", target / "model.safetensors")
    for name in _TOKENIZER_FILES:
        if (source / "LLM" / name).is_file():
            shutil.copy(source / "LLM" / name, target / name)

    from .processing_spark_tts import SparkTTSProcessor

    feature_extractor = SparkTTSFeatureExtractor(
        sampling_rate=repo_config["sample_rate"],
        volume_normalize=repo_config["volume_normalize"],
        ref_segment_duration=repo_config["ref_segment_duration"],
        hop_length=hop_length,
        n_fft=bicodec_config["mel_params"]["n_fft"],
        win_length=bicodec_config["mel_params"]["win_length"],
        num_mel_bins=bicodec_config["mel_params"]["num_mels"],
        mel_fmin=bicodec_config["mel_params"]["mel_fmin"],
        mel_fmax=bicodec_config["mel_params"]["mel_fmax"],
    )
    SparkTTSProcessor(
        feature_extractor=feature_extractor,
        tokenizer=AutoTokenizer.from_pretrained(source / "LLM"),
        audio_tokenizer=SparkTTSBiCodecModel.from_pretrained(audio_tokenizer_target),
    ).save_pretrained(target)

    return str(target)


def retarget_audio_tokenizer(directory) -> None:
    """
    Point a converted directory's processor at the copy of BiCodec that sits beside it.

    `ProcessorMixin.save_pretrained` records the audio tokenizer as the path it was loaded from rather than as a
    location inside the directory being written, so a converted directory that ends up somewhere other than where
    it was written records a path that is not there.

    Args:
        directory (`str` or `os.PathLike`):
            A directory [`convert`] wrote.
    """
    config_file = Path(directory) / "processor_config.json"
    processor_config = json.loads(config_file.read_text())
    audio_tokenizer = processor_config.get("audio_tokenizer", {})
    target = str(Path(directory) / AUDIO_TOKENIZER_SUBFOLDER)
    if audio_tokenizer.get("audio_tokenizer_name_or_path") == target:
        return

    audio_tokenizer["audio_tokenizer_name_or_path"] = target
    written = config_file.with_name(f"{config_file.name}.{os.getpid()}")
    written.write_text(json.dumps(processor_config, indent=2, sort_keys=True) + "\n")
    os.replace(written, config_file)


def convert_published_checkpoint(pretrained_model_name_or_path, **kwargs):
    """
    Convert `pretrained_model_name_or_path` if it is a published Spark-TTS checkpoint, reusing an earlier conversion
    of the same checkpoint when one is already there.

    The conversion is cached on the checkpoint's resolved revision and written into place with a single rename, so
    a second process reading the cache never sees a half written directory. Only the configuration files of
    [`_PUBLISHED_FILES`] are fetched to name that revision, and the three models are fetched inside the conversion,
    so a checkpoint the cache already holds is never downloaded again.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            A Hugging Face repo id or a local directory.
        kwargs (`dict[str, Any]`, *optional*):
            Forwarded to [`~huggingface_hub.snapshot_download`], of which `revision`, `token` and `cache_dir` are
            used.

    Returns:
        `str` or `None`: The converted checkpoint's directory, or `None` if the given checkpoint is neither a
        repository nor in the published Spark-TTS layout, in which case its caller's own load error is the one
        worth reporting.
    """
    download_kwargs = {key: value for key, value in kwargs.items() if key in ("revision", "token", "cache_dir")}

    def resolve(patterns):
        if os.path.isdir(pretrained_model_name_or_path):
            return Path(pretrained_model_name_or_path)
        return Path(
            snapshot_download(pretrained_model_name_or_path, allow_patterns=list(patterns), **download_kwargs)
        )

    try:
        source = resolve(_PUBLISHED_FILES)
    except (HFValidationError, RepositoryNotFoundError):
        return None

    if not all((source / name).is_file() for name in _PUBLISHED_FILES):
        return None

    target = cached_conversion(
        "spark_tts", [file_identity(source)], lambda staging: convert(resolve(_PUBLISHED_PATTERNS), staging)
    )
    retarget_audio_tokenizer(target)
    return str(target)


__all__ = ["AUDIO_TOKENIZER_SUBFOLDER", "convert", "convert_published_checkpoint", "retarget_audio_tokenizer"]
