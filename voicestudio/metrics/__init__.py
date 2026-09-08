from transformers.utils.import_utils import _is_package_available


# `jiwer` ships in the optional `eval` extra and the audio stack in `audio`, so importing this
# package must not require either. What is missing is left out rather than raising.
_jiwer_available, _ = _is_package_available("jiwer")
_soundfile_available, _ = _is_package_available("soundfile")

from .base import LauncherType, Metric, MetricConfig
from .judge import ChatCompletionClient, Judge, Qwen3OmniJudge, Transcriber, VllmServer


if _jiwer_available:
    from .cer import Cer
    from .error_rate import ErrorRate
    from .wer import Wer

if _soundfile_available:
    from .ffe import Ffe
    from .mcd import Mcd
    from .ssim import Ssim
    from .utmos import Utmos
