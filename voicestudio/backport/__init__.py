"""Loader for the vendored transformers models that have no released version yet."""

import builtins
import importlib
import sys
from pathlib import Path


# Holds the vendored packages under the names `transformers` gives them, so that the import system
# finds them as `transformers.models.<model>` and their `from ...` imports resolve into `transformers`.
BACKPORT_DIR = Path(__file__).parent / "models"

# Package name to config class name for each vendored model. `auto_docstring` resolves a model's
# config through `CONFIG_MAPPING_NAMES` while the modeling file is being imported, and emits a
# placeholder docstring when the lookup misses. The Auto preprocessing classes read the same mapping
# to turn a checkpoint's `model_type` into a class.
BACKPORTED_MODELS = {
    "qwen3_tts": "Qwen3TTSConfig",
    "qwen3_tts_tokenizer_multi_codebook": "Qwen3TTSTokenizerMultiCodebookConfig",
    "qwen3_tts_tokenizer_single_codebook": "Qwen3TTSTokenizerSingleCodebookConfig",
}


def install() -> None:
    r"""
    Make the vendored packages importable the way a released `transformers` would serve them.

    The directory is appended rather than prepended, so a `transformers` that ships one of these
    models resolves to its own copy and the vendored one is never reached. Call this before importing
    anything that imports `transformers.models.<model>`.
    """
    import transformers.models
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
    from transformers.utils.import_utils import define_import_structure

    # `transformers.processing_utils` runs `transformers/__init__.py` a second time through
    # `direct_transformers_import`, which replaces the root module object. Reaching it first means the
    # names registered below land on the module that survives.
    import transformers.processing_utils  # noqa: F401

    if str(BACKPORT_DIR) not in transformers.models.__path__:
        transformers.models.__path__.append(str(BACKPORT_DIR))

    _mute_vendored_docstring_reports()

    root = sys.modules["transformers"]
    for package, config_class in BACKPORTED_MODELS.items():
        CONFIG_MAPPING_NAMES.setdefault(package, config_class)
        # `transformers.<Name>` is what the vendored conversion scripts import, and the structure is
        # read off the `__all__` of each file rather than by importing it.
        structure = define_import_structure(str(BACKPORT_DIR / package), prefix=f"models.{package}")
        for modules in structure.values():
            for module, names in modules.items():
                for name in names:
                    if name not in root._class_to_module:
                        root._class_to_module[name] = module
                        root.__all__.append(name)

        for name in _feature_extractor_names(structure, package):
            FEATURE_EXTRACTOR_MAPPING_NAMES.setdefault(package, name)

    _refresh_auto_mapping_keys()


def _feature_extractor_names(structure: dict, package: str) -> list[str]:
    r"""
    Read the feature extractor class names a vendored package exports.

    Args:
        structure (`dict`):
            The import structure `define_import_structure` built for the package.
        package (`str`):
            The package name, which its feature extraction file is named after.

    Returns:
        `list[str]`: The exported names, empty for a package that ships no feature extraction file.
    """
    return sorted(
        name
        for modules in structure.values()
        for module, names in modules.items()
        if module.endswith(f".feature_extraction_{package}")
        for name in names
    )


def _refresh_auto_mapping_keys() -> None:
    r"""
    Rebuild the config-name index that each `_LazyAutoMapping` derives in its constructor.

    A `_LazyAutoMapping` resolves a config class to a preprocessor or model class by looking the
    config class name up in a dict it derives from `CONFIG_MAPPING_NAMES` once, at construction. A
    mapping built before `install` registers the vendored model types reports each of their configs
    as absent, so `AutoFeatureExtractor.from_pretrained` reaches its `Unrecognized feature extractor`
    error even though `FEATURE_EXTRACTOR_MAPPING_NAMES` carries the entry. Rebuilding from the live
    `CONFIG_MAPPING_NAMES` leaves the entries a released `transformers` supplies as they are.
    """
    from transformers.models.auto.auto_factory import _LazyAutoMapping

    for name, module in list(sys.modules.items()):
        if not name.startswith("transformers.models.auto.") or module is None:
            continue
        for value in getattr(module, "__dict__", {}).values():
            if isinstance(value, _LazyAutoMapping):
                value._reverse_config_mapping = {v: k for k, v in value._config_mapping.items()}


def _mute_vendored_docstring_reports() -> None:
    r"""
    Drop the undocumented-argument report `auto_docstring` prints for the vendored files.

    `auto_docstring` writes one line per undocumented argument to stdout while a modeling file is
    imported, and it writes them through `print` rather than a logger. Only the lines naming a
    vendored file are dropped, so the report still reaches stdout for every other model.
    """
    # `transformers.utils.auto_docstring` is the name of the decorator as well as of the module that
    # defines it, and the attribute lookup resolves to the decorator.
    auto_docstring = importlib.import_module("transformers.utils.auto_docstring")

    if "print" in vars(auto_docstring):
        return

    def filtered_print(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            kept = [line for line in args[0].split("\n") if str(BACKPORT_DIR) not in line]
            if not kept:
                return
            args = ("\n".join(kept),)
        builtins.print(*args, **kwargs)

    auto_docstring.print = filtered_print


__all__ = ["BACKPORT_DIR", "BACKPORTED_MODELS", "install"]
