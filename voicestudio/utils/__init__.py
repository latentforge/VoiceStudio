from transformers.utils.import_utils import is_torch_available, is_torchao_available


# `torch` arrives through the optional `cloud` extra and `torchao` through `train`, so importing
# this package must not require either. What is missing is left out rather than raising, the same
# way `voicestudio.metrics` handles the `eval` extra. Which version satisfies the extra is the
# resolver's answer, taken from the floor `pyproject.toml` declares, so nothing here repeats it.

from .log_utils import BaseLogger, WandBLogger


if is_torch_available():
    from .evaluate import EvaluatorMode, EvaluationRecord, Evaluator
    from .stage import TrainingStage
    from .train import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingConfig

# Importing the module is what registers `nvfp4` as a quantization method, so a checkpoint saved
# under it reloads without the caller naming the class first.
if is_torchao_available():
    from .quantization_utils import (
        HadamardNVFP4Linear,
        NVFP4Config,
        NVFP4HfQuantizer,
        convert_nvfp4_qat,
    )
