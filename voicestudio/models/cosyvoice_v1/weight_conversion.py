"""Checkpoint conversion for CosyVoice v1."""

import mmap
from pathlib import Path

import numpy as np
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

# The speech tokenizer and the speaker table the processor opens out of the same directory. Only the
# base model ships no `spk2info.pt`. The speech tokenizer is published as an ONNX graph and as
# nothing else, so [`convert_speech_tokenizer`] reads its weights out of that graph.
SPEECH_TOKENIZER_FILE = "speech_tokenizer_v1.onnx"
SPEAKER_INFO_FILE = "spk2info.pt"

# Element types of the `TensorProto` the speech tokenizer graphs hold, by their ONNX enumerator.
ONNX_TENSOR_DTYPES = {1: "<f4", 6: "<i4", 7: "<i8", 10: "<f2", 11: "<f8"}

# Module names upstream exported the speech tokenizer under, rewritten to the ones
# [`CosyVoiceV1SpeechTokenizer`] uses, applied in order as plain substring replacements. The encoder
# is a Whisper encoder, so its blocks take the names the `transformers` Whisper encoder gives them.
SPEECH_TOKENIZER_CONVERSION = (
    ("positional_embedding", "embed_positions.weight"),
    ("blocks.", "layers."),
    (".attn_ln.", ".self_attn_layer_norm."),
    (".mlp_ln.", ".final_layer_norm."),
    (".attn.query.", ".self_attn.q_proj."),
    (".attn.key.", ".self_attn.k_proj."),
    (".attn.value.", ".self_attn.v_proj."),
    (".attn.out.", ".self_attn.out_proj."),
    (".attn.fsmn_block.", ".self_attn.fsmn_block."),
    (".mlp.mlp.0.", ".fc1."),
    (".mlp.mlp.2.", ".fc2."),
    ("quantizer.rq.model.layers.0._codebook.weight", "quantizer.embedding.weight"),
)

# The speaker encoder the CosyVoice directories carry is an export of the CAM++ checkpoint its
# authors published separately, whose PyTorch release [`CosyVoiceV1SpeakerEncoder`] reads instead.
SPEAKER_ENCODER_REPO = "funasr/campplus"
SPEAKER_ENCODER_WEIGHTS = "campplus_cn_common.bin"

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


def _read_varint(buffer: memoryview, position: int) -> tuple[int, int]:
    r"""
    Reads one base 128 variable width integer.

    Args:
        buffer (`memoryview`):
            Encoded protocol buffer.
        position (`int`):
            Offset the integer starts at.

    Returns:
        `tuple[int, int]`: The integer and the offset just past it.
    """
    value, shift = 0, 0
    while True:
        byte = buffer[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7


def _read_protobuf(buffer: memoryview):
    r"""
    Walks the fields of one encoded protocol buffer message.

    Args:
        buffer (`memoryview`):
            Encoded message.

    Yields:
        `tuple[int, int, Any]`: The field number, its wire type, and its value, which is an `int` for
        a variable width field and a `memoryview` for every other one.

    Raises:
        ValueError: If a field carries a wire type protocol buffers no longer defines.
    """
    position, end = 0, len(buffer)
    while position < end:
        key, position = _read_varint(buffer, position)
        wire_type = key & 7
        if wire_type == 0:
            value, position = _read_varint(buffer, position)
        elif wire_type == 2:
            length, position = _read_varint(buffer, position)
            value, position = buffer[position : position + length], position + length
        elif wire_type in (1, 5):
            width = 8 if wire_type == 1 else 4
            value, position = buffer[position : position + width], position + width
        else:
            raise ValueError(f"unsupported protocol buffer wire type {wire_type}")
        yield key >> 3, wire_type, value


def _read_onnx_tensor(buffer: memoryview) -> tuple[str, np.ndarray]:
    r"""
    Reads one ONNX `TensorProto`.

    Args:
        buffer (`memoryview`):
            Encoded `TensorProto`.

    Returns:
        `tuple[str, np.ndarray]`: The tensor's name and its value.

    Raises:
        ValueError: If the tensor holds its values somewhere other than `raw_data`, which is where
            every initializer of the released graphs holds them.
    """
    dims, data_type, name, raw = [], 1, "", None
    for field, wire_type, value in _read_protobuf(buffer):
        if field == 1:
            dims.extend([value] if wire_type == 0 else _read_packed_varints(value))
        elif field == 2:
            data_type = value
        elif field == 8:
            name = bytes(value).decode()
        elif field == 9:
            raw = value
    if raw is None:
        raise ValueError(f"the initializer {name} carries no raw data")
    return name, np.frombuffer(raw, dtype=ONNX_TENSOR_DTYPES[data_type]).reshape(dims).copy()


def _read_packed_varints(buffer: memoryview) -> list[int]:
    r"""
    Reads a packed repeated field of variable width integers.

    Args:
        buffer (`memoryview`):
            Encoded field.

    Returns:
        `list[int]`: The integers.
    """
    values, position = [], 0
    while position < len(buffer):
        value, position = _read_varint(buffer, position)
        values.append(value)
    return values


def _read_onnx_node(buffer: memoryview) -> tuple[str, list[str], list[str]]:
    r"""
    Reads one ONNX `NodeProto`.

    Args:
        buffer (`memoryview`):
            Encoded `NodeProto`.

    Returns:
        `tuple[str, list[str], list[str]]`: The node's operator, the names of its inputs and the
        names of its outputs.
    """
    op_type, inputs, outputs = "", [], []
    for field, _, value in _read_protobuf(buffer):
        if field == 1:
            inputs.append(bytes(value).decode())
        elif field == 2:
            outputs.append(bytes(value).decode())
        elif field == 4:
            op_type = bytes(value).decode()
    return op_type, inputs, outputs


def read_onnx_graph(path: "str | Path") -> tuple[list[tuple[str, list[str], list[str]]], dict[str, np.ndarray]]:
    r"""
    Reads the nodes and the initializers of an ONNX graph.

    Only the three fields the speech tokenizer conversion needs are decoded, so this reads a graph
    rather than validating one.

    Args:
        path (`str` or `os.PathLike`):
            Path of the `.onnx` file.

    Returns:
        `tuple[list[tuple[str, list[str], list[str]]], dict[str, np.ndarray]]`: The nodes, each as
        its operator and the names of its inputs and outputs, in the order the graph lists them, and
        the initializers by name.
    """
    with open(path, "rb") as handle:
        # The view owns the mapping, which outlives the descriptor and is unmapped once every slice
        # taken from it here has been dropped.
        buffer = memoryview(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
    graph = next(value for field, _, value in _read_protobuf(buffer) if field == 7)
    nodes, initializers = [], {}
    for field, _, value in _read_protobuf(graph):
        if field == 1:
            nodes.append(_read_onnx_node(value))
        elif field == 5:
            name, array = _read_onnx_tensor(value)
            initializers[name] = array
    return nodes, initializers


def _onnx_scope(name: str) -> str:
    r"""
    Reads the module a node belonged to off the name of its output.

    Args:
        name (`str`):
            Name of a node output, which the exporter builds out of the module path the node was
            traced inside, as in `/blocks.0/attn/query/MatMul_output_0`.

    Returns:
        `str`: The module path, as in `blocks.0.attn.query`.
    """
    return ".".join(name.split("/")[1:-1])


def convert_speech_tokenizer(path: "str | Path") -> dict[str, torch.Tensor]:
    r"""
    Reads the weights of a `speech_tokenizer_*.onnx` graph into a state dict.

    The exporter kept a readable name for the initializers it could not fold, and gave the rest a
    generated one such as `onnx::MatMul_1532`. Those are recovered from the graph itself: every node
    that consumes an initializer names its output after the module it was traced inside, so the
    module path and the operator together say which parameter the initializer is. A `MatMul` weight
    is transposed on the way, since ONNX holds it the way the multiplication reads it.

    Args:
        path (`str` or `os.PathLike`):
            Path of the `.onnx` file.

    Returns:
        `dict[str, torch.Tensor]`: The weights, keyed the way [`CosyVoiceV1SpeechTokenizer`] names
        its parameters.
    """
    nodes, initializers = read_onnx_graph(path)
    named = {name: array for name, array in initializers.items() if not name.startswith("onnx::")}
    state_dict = {name.removeprefix("encoders."): array for name, array in named.items()}
    for op_type, inputs, outputs in nodes:
        consumed = [name for name in inputs if name in initializers]
        if not consumed:
            continue
        scope = _onnx_scope(outputs[0])
        if op_type == "Conv":
            state_dict[f"{scope}.weight"] = initializers[inputs[1]]
            if len(inputs) > 2:
                state_dict[f"{scope}.bias"] = initializers[inputs[2]]
        elif op_type == "MatMul":
            state_dict[f"{scope}.weight"] = initializers[inputs[1]].T
        elif op_type == "LayerNormalization":
            state_dict[f"{scope}.weight"] = initializers[inputs[1]]
            state_dict[f"{scope}.bias"] = initializers[inputs[2]]
        elif op_type in ("Add", "Mul"):
            state_dict[f"{scope}.{'bias' if op_type == 'Add' else 'weight'}"] = initializers[consumed[0]]
    converted = {}
    for name, array in state_dict.items():
        for source, target in SPEECH_TOKENIZER_CONVERSION:
            name = name.replace(source, target)
        converted[name] = torch.from_numpy(np.ascontiguousarray(array))
    return converted


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
    "ONNX_TENSOR_DTYPES",
    "PUBLISHED_CHECKPOINTS",
    "SPEAKER_ENCODER_REPO",
    "SPEAKER_ENCODER_WEIGHTS",
    "SPEAKER_INFO_FILE",
    "SPEECH_TOKENIZER_FILE",
    "SPEECH_TOKENIZER_CONVERSION",
    "TEXT_TOKENIZER_ID",
    "build_config",
    "convert_speech_tokenizer",
    "load_checkpoint",
    "read_onnx_graph",
    "resolve_checkpoint",
]
