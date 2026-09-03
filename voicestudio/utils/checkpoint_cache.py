"""Cached streaming conversion of published checkpoints."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import torch
from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HOME
from huggingface_hub.errors import CacheNotFound
from huggingface_hub.utils import DeleteCacheStrategy
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


# Directory under `HF_HOME` the converted checkpoints are written to, beside the `hub` cache the sources
# they were converted from live in.
CACHE_DIR_NAME = "converted"

# Bytes a shard holds before the writer starts another one, which is what bounds how much of a converted
# checkpoint is resident at a time.
MAX_SHARD_SIZE = 2 * 1024**3

# Bytes at which a source file counts as weights and is reclaimed once it has been converted. A configuration,
# a tokenizer or the discriminator a loader probes for stays, since it costs nothing and saves a round trip.
RECLAIMED_FILE_SIZE = 16 * 1024**2

# Suffixes a source file carries when it holds text rather than weights, which is never reclaimed whatever its
# size, since a serialized tokenizer can run to tens of megabytes.
RECLAIMED_FILE_EXCEPTIONS = frozenset({".json", ".md", ".model", ".py", ".tsv", ".txt", ".vocab", ".yaml", ".yml"})

# Seconds a staging directory goes untouched before a sweep takes it for an interrupted conversion. This has to
# stay above the longest a running conversion leaves its staging directory alone, which is the time it spends
# downloading and reading its source before the first shard is written.
STAGING_MAX_AGE = 6 * 60 * 60

# Layout `huggingface_hub` gives a downloaded file, whose two components pin the repository and the commit
# the file was read from.
_SNAPSHOT = re.compile(r"(?:^|/)(?P<repo>(?:models|datasets|spaces)--[^/]+)/snapshots/(?P<revision>[^/]+)(?:/|$)")

# The same two components as [`file_identity`] writes them, which is how a cache key names a downloaded source.
_IDENTITY = re.compile(r"(?P<repo>(?:models|datasets|spaces)--[^/@]+)@(?P<revision>[^/@]+)")

# Name `tempfile.mkdtemp` builds for a staging directory, a cache key followed by a random suffix. A finished
# entry is the bare key, so the dot is what tells the two apart.
_STAGING = re.compile(r"^[0-9a-f]{32}\.")


def cache_root() -> Path:
    r"""
    Returns:
        `Path`: Directory the converted checkpoints of every model are cached under.
    """
    return Path(HF_HOME) / CACHE_DIR_NAME


def file_identity(path) -> str:
    r"""
    Names the exact revision of a file or directory a conversion reads, so that a cache key built from it
    changes when what it names does.

    A download is named by the repository and the commit its snapshot directory carries, which is what a moved
    tag or branch changes and a repository id on its own does not. A local path is named by itself, by the size
    and modification time of the file, or of every file under the directory.

    Args:
        path (`str` or `os.PathLike`):
            Local path, as `huggingface_hub` resolved it or as the caller holds it.

    Returns:
        `str`: The path's identity.
    """
    match = _SNAPSHOT.search(Path(path).as_posix())
    if match is not None:
        return f"{match['repo']}@{match['revision']}"

    real = Path(os.path.realpath(path))
    if real.is_file():
        status = real.stat()
        return f"{real.as_posix()}:{status.st_size}:{status.st_mtime_ns}"

    digest = hashlib.sha256()
    for entry in sorted(real.rglob("*")):
        if entry.is_file():
            status = entry.stat()
            digest.update(f"{entry.relative_to(real).as_posix()}:{status.st_size}:{status.st_mtime_ns}\0".encode())
    return f"{real.as_posix()}:{digest.hexdigest()[:32]}"


def source_identity(source, resolved) -> str:
    r"""
    Names the revision of the checkpoint a loader was handed.

    Args:
        source (`str` or `os.PathLike`):
            Repository id or local directory the checkpoint was asked for by.
        resolved (`str` or `os.PathLike`):
            Local path of a file already read out of `source`, whose snapshot names the commit the whole
            repository resolved to and so covers every other file of it.

    Returns:
        `str`: The checkpoint's identity.
    """
    return file_identity(source if Path(source).is_dir() else resolved)


def cache_key(parts: Iterable) -> str:
    r"""
    The key covers the source a conversion reads and the options it takes, and nothing about the code that
    writes it, so a converter that starts writing something else keeps being served the entry it wrote before.
    Changing what a conversion produces means deleting its directories under [`cache_root`] by hand.

    Args:
        parts (`Iterable`):
            Everything the conversion's output depends on, as [`file_identity`] strings for the files it reads
            and as plain values for the options it takes.

    Returns:
        `str`: Hexadecimal digest naming the directory the conversion is cached in.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def sweep_staging(directory) -> None:
    r"""
    Removes the staging directories interrupted conversions left in one model's cache folder.

    A conversion that runs to completion renames its staging directory into place, so one still carrying a
    staging name either belongs to a conversion running right now or to one that died. The two are told apart by
    when the directory and its contents were last written: a running conversion keeps extending the shard it is
    on, so nothing older than `STAGING_MAX_AGE` can belong to one.

    Args:
        directory (`str` or `os.PathLike`):
            Cache folder of one model, holding its converted checkpoints and any staging directory beside them.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return

    now = time.time()
    for entry in directory.iterdir():
        if _STAGING.match(entry.name) is None or not entry.is_dir():
            continue
        try:
            touched = max(path.stat().st_mtime for path in (entry, *entry.rglob("*")))
        except OSError:
            continue
        if now - touched > STAGING_MAX_AGE:
            shutil.rmtree(entry, ignore_errors=True)


def reclaim_sources(parts: Iterable) -> int:
    r"""
    Removes from the `huggingface_hub` cache the weight files of every downloaded revision a conversion read,
    which the converted checkpoint replaces as what later loads are served from.

    The revisions come from the [`file_identity`] strings in `parts`, so a local directory the caller handed the
    conversion is never touched. Only files of at least `RECLAIMED_FILE_SIZE` go, and only where the revision
    being reclaimed is the one revision of its repository linking the blob behind them. Deletion runs through
    `huggingface_hub`'s own [`~huggingface_hub.utils.DeleteCacheStrategy`], which drops each snapshot link before
    the blob it points at, so no later scan of the cache meets a broken link.

    Args:
        parts (`Iterable`):
            Cache key of the conversion, see [`cache_key`].

    Returns:
        `int`: Bytes freed.
    """
    revisions = {
        (match["repo"], match["revision"])
        for match in (_IDENTITY.fullmatch(str(part)) for part in parts)
        if match is not None
    }
    if not revisions:
        return 0

    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return 0

    links: set[Path] = set()
    blobs: set[Path] = set()
    freed = 0
    for repo in cache.repos:
        for revision in repo.revisions:
            if (repo.repo_path.name, revision.commit_hash) not in revisions:
                continue
            elsewhere = {
                file.blob_path for other in repo.revisions if other is not revision for file in other.files
            }
            for file in revision.files:
                if file.size_on_disk < RECLAIMED_FILE_SIZE or file.file_path.suffix in RECLAIMED_FILE_EXCEPTIONS:
                    continue
                if file.blob_path in elsewhere:
                    continue
                links.add(file.file_path)
                # A filesystem without symlinks holds the bytes in the snapshot itself, so there is no blob
                # under it to drop separately.
                if file.blob_path != file.file_path:
                    blobs.add(file.blob_path)
                freed += file.size_on_disk

    if not links:
        return 0
    DeleteCacheStrategy(
        expected_freed_size=freed,
        blobs=frozenset(blobs),
        refs=frozenset(),
        repos=frozenset(),
        snapshots=frozenset(links),
    ).execute()
    return freed


def cached_conversion(name: str, parts: Iterable, write: Callable[[Path], None]) -> Path:
    r"""
    Returns a directory holding the converted checkpoint `parts` names, running `write` to produce it the first
    time it is asked for.

    The conversion is written to a temporary directory beside the cached one and moved into place with a single
    rename, so a second process reading the cache sees either nothing or a directory whose conversion ran to
    completion, and two processes converting the same checkpoint at once leave one result rather than a mixture.
    The process whose rename lands is also the one that reclaims the downloaded source, see [`reclaim_sources`],
    so nothing is removed under a conversion that is still reading it.

    Args:
        name (`str`):
            Model the checkpoint belongs to, which is the directory the cache groups it under.
        parts (`Iterable`):
            Cache key of the conversion, see [`cache_key`].
        write (`Callable[[Path], None]`):
            Writes the converted checkpoint into the directory it is handed.

    Returns:
        `Path`: The directory holding the converted checkpoint.
    """
    parts = tuple(parts)
    directory = cache_root() / name / cache_key(parts)
    sweep_staging(directory.parent)
    if directory.is_dir():
        return directory

    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{directory.name}.", dir=directory.parent))
    try:
        write(staging)
        try:
            os.replace(staging, directory)
        except OSError:
            if not directory.is_dir():
                raise
        else:
            reclaim_sources(parts)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return directory


class CheckpointWriter:
    r"""
    Writes a converted checkpoint to disk a shard at a time, for a conversion that reads its source the same
    way and so never holds the whole model.

    Shards are named once the last one is written, since the name `transformers` reads carries the total.

    Args:
        directory (`str` or `os.PathLike`):
            Directory the shards and, when there is more than one, their index are written to.
        max_shard_size (`int`, *optional*, defaults to `MAX_SHARD_SIZE`):
            Bytes a shard holds before the next tensor starts another one.
    """

    def __init__(self, directory, max_shard_size: int = MAX_SHARD_SIZE):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_shard_size = max_shard_size
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffered_storages: set[int] = set()
        self._buffered_size = 0
        self._shards: list[list[str]] = []
        self._total_size = 0

    def __enter__(self) -> "CheckpointWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()

    def add(self, key: str, tensor: torch.Tensor) -> None:
        r"""
        Buffers one tensor, writing the buffer out first if this one would take it past a shard.

        Args:
            key (`str`):
                Name the tensor is stored under.
            tensor (`torch.Tensor`):
                The tensor. It is copied where safetensors cannot store it as it stands, which is a view of a
                larger storage, or a second name for a tensor already in the shard.
        """
        tensor = tensor.detach().contiguous()
        if tensor.numel() * tensor.element_size() != tensor.untyped_storage().nbytes():
            tensor = tensor.clone()

        size = tensor.numel() * tensor.element_size()
        if self._buffer and self._buffered_size + size > self.max_shard_size:
            self.flush()
        if tensor.untyped_storage().data_ptr() in self._buffered_storages:
            tensor = tensor.clone()

        self._buffer[key] = tensor
        self._buffered_storages.add(tensor.untyped_storage().data_ptr())
        self._buffered_size += size

    def update(self, tensors: Mapping[str, torch.Tensor]) -> None:
        r"""
        Args:
            tensors (`Mapping[str, torch.Tensor]`):
                Tensors to buffer, in the order they are to be written.
        """
        for key, tensor in tensors.items():
            self.add(key, tensor)

    def flush(self) -> None:
        """Writes the buffered tensors out as a shard of their own, and drops the buffer holding them."""
        if not self._buffer:
            return
        save_file(self._buffer, str(self.directory / self._staged_name(len(self._shards))), metadata={"format": "pt"})
        self._shards.append(sorted(self._buffer))
        self._total_size += self._buffered_size
        self._buffer = {}
        self._buffered_storages = set()
        self._buffered_size = 0

    def close(self) -> None:
        r"""
        Writes what is left buffered, gives the shards the names `transformers` reads and, for more than one,
        writes their index.

        Raises:
            ValueError: If the conversion wrote no tensor at all.
        """
        self.flush()
        if not self._shards:
            raise ValueError(f"The conversion wrote no tensors to {self.directory}.")

        if len(self._shards) == 1:
            (self.directory / self._staged_name(0)).rename(self.directory / SAFE_WEIGHTS_NAME)
            return

        weight_map = {}
        for index, keys in enumerate(self._shards):
            name = f"model-{index + 1:05d}-of-{len(self._shards):05d}.safetensors"
            (self.directory / self._staged_name(index)).rename(self.directory / name)
            weight_map.update(dict.fromkeys(keys, name))
        (self.directory / SAFE_WEIGHTS_INDEX_NAME).write_text(
            json.dumps({"metadata": {"total_size": self._total_size}, "weight_map": weight_map}, indent=2)
        )

    @staticmethod
    def _staged_name(index: int) -> str:
        return f"shard-{index:05d}.safetensors"


__all__ = [
    "CACHE_DIR_NAME",
    "MAX_SHARD_SIZE",
    "RECLAIMED_FILE_EXCEPTIONS",
    "RECLAIMED_FILE_SIZE",
    "STAGING_MAX_AGE",
    "CheckpointWriter",
    "cache_key",
    "cache_root",
    "cached_conversion",
    "file_identity",
    "reclaim_sources",
    "source_identity",
    "sweep_staging",
]
