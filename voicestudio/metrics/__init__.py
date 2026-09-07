from transformers.utils.import_utils import _is_package_available


# `evaluate` and the `datasets` backing its accumulation buffers ship in the optional `eval` extra, so
# importing this package must not require them. `_is_package_available` returns a
# `(available, version)` pair even when a version was not asked for, and the pair is truthy either way.
_evaluate_available, _ = _is_package_available("evaluate")

if _evaluate_available:
    from .cer import Cer
    from .ffe import Ffe
    from .mcd import Mcd
    from .utmos import Utmos
    from .wer import Wer
