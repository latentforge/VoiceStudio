# coding=utf-8
# Copyright 2026 LatentForge and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The trainer a VoiceStudio run drives its loop with."""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from transformers import Trainer as HfTrainer
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments as TrainingConfig

from .log_utils import BaseLogger


logger = logging.getLogger(__name__)


# TODO: Graceful shutdown


class Trainer(HfTrainer):
    r"""[`~transformers.Trainer`] for a run whose linear projections are quantized on every forward.

    Adds two things to the loop and changes nothing else: an objective the caller supplies, since
    that is the one hook `transformers` has no constructor argument for, and a check that anything
    is training at all. A freeze policy that reaches further than intended leaves the optimizer
    with an empty parameter list, and the run reports a finite loss for as long as it is allowed
    to continue without a single weight moving.

    Args:
        compute_loss_fn (`Callable`, *optional*):
            Scores one micro-batch, returning the scalar to call `backward` on, the terms behind
            it, and the model's output. Left out, the model's own objective is used.
        evaluate_fn (`Callable`, *optional*):
            Takes the readings this loop's evaluations report, replacing the loop over
            `eval_dataset` rather than running beside it. What is measured produces its own inputs
            from its own rows, so there is nothing for the trainer to iterate and nothing to
            accumulate. Left out, `transformers` evaluates the way it normally does.
        metric_logger (`BaseLogger`, *optional*):
            Where the terms `compute_loss_fn` reported are written, taking the optimizer step
            alongside them. Left out, they are computed and dropped.
    """

    def __init__(
        self,
        *args: Any,
        compute_loss_fn: Callable[..., tuple[torch.Tensor, dict[str, float], Any]] | None = None,
        evaluate_fn: Callable[..., dict[str, float]] | None = None,
        metric_logger: BaseLogger | None = None,
        **kwargs: Any,
    ):
        self.compute_loss_fn = compute_loss_fn
        self.evaluate_fn = evaluate_fn
        self.metric_logger = metric_logger
        super().__init__(*args, **kwargs)
        self._report_trainable_parameters()

    def compute_loss(
        self, model: nn.Module, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any
    ):
        """Scores one micro-batch through `compute_loss_fn`, and writes the terms behind the total.

        Args:
            model (`nn.Module`):
                The model as the step runs it.
            inputs (`dict[str, Any]`):
                The micro-batch, already on the model's device.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether the model's output is returned beside the loss.
            **kwargs:
                Passed through to `compute_loss_fn`, such as `num_items_in_batch`.

        Returns:
            `torch.Tensor` or `tuple`: The scalar to call `backward` on, with the model's
            output beside it when `return_outputs` is set.
        """
        if self.compute_loss_fn is None:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)
        loss, terms, outputs = self.compute_loss_fn(model, inputs, **kwargs)
        # Through the caller's metric_logger rather than the trainer's own reporting integrations, so a
        # curriculum's terms land on one axis instead of one per segment.
        if terms and self.metric_logger is not None:
            self.metric_logger.log(terms, int(self.state.global_step))
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Takes this loop's readings, through `evaluate_fn` where one was given.

        Args:
            eval_dataset (`Any`, *optional*):
                Ignored when `evaluate_fn` is set, since what is measured carries its own rows.
            ignore_keys (`list[str]`, *optional*):
                Ignored when `evaluate_fn` is set.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                Prefixed to every reading, so `metric_for_best_model` can name one.

        Returns:
            `dict[str, float]`: The readings, prefixed.
        """
        if self.evaluate_fn is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        readings = self.evaluate_fn(self.model, int(self.state.global_step))
        metrics = {f"{metric_key_prefix}_{key}": value for key, value in readings.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def _report_trainable_parameters(self) -> None:
        """Reports what the optimizer will see, and refuses a run where that is nothing.

        Raises:
            ValueError: If no parameter carries `requires_grad`.
        """
        trainable = [name for name, param in self.model.named_parameters() if param.requires_grad]
        if not trainable:
            raise ValueError("No parameter carries a gradient, so this run would optimize nothing.")
        counted = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        quantizer = getattr(self.model, "hf_quantizer", None)
        method = getattr(getattr(quantizer, "quantization_config", None), "quant_method", None)
        logger.info(
            "%d trainable tensors, %d parameters, quantized as %s",
            len(trainable),
            counted,
            method or "none",
        )


__all__ = [
    "Trainer",
    "TrainerCallback",
    "TrainerControl",
    "TrainerState",
    "TrainingConfig",
]
