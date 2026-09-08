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
"""One phase of a curriculum, and the adapters that hand its Evaluator to a Trainer."""

import gc
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .evaluate import EvaluatorMode, Evaluator
from .log_utils import BaseLogger
from .train import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingConfig


logger = logging.getLogger(__name__)


@dataclass
class TrainingStageConfig:
    stage_name: str
    output_dir: Path | str


class TrainingStage(ABC):
    r"""One phase of a curriculum: what it trains on, what it measures, what it holds still.

    A curriculum advances by handing the next stage the same logger:

    ```python
    for stage in stages:
        stage.run(model_init(), args_for(stage))
    ```

    Args:
        stage_config (`TrainingStageConfig`):
            Configuration for the training stage.
        metric_logger (`BaseLogger`, *optional*):
            Where this stage's metrics are written. Built outside the stage and handed in, so one
            logger spans a whole curriculum: a stage prefixes its keys with its own name rather
            than opening a run of its own, and the axis survives the transition to the next stage.
            The stage never closes it.
    """

    def __init__(
        self,
        stage_config: TrainingStageConfig,
        metric_logger: BaseLogger | None = None,
    ):
        self.stage_name = stage_config.stage_name
        self.stage_config = stage_config
        self.metric_logger = metric_logger

        self._state: TrainerState | None = None
        self._trainer: Trainer | None = None
        self._evaluators: list[Evaluator] | None = None

    @property
    def global_step(self) -> int:
        """Returns the optimizer step a reading is filed under.

        Returns:
            `int`: The trainer's global step, or `0` before training has begun.
        """
        return int(self._state.global_step) if self._state is not None else 0

    @property
    def trainer(self) -> Trainer:
        """The trainer that runs this stage, built once and kept.

        Building one loads a model onto the card, so this is not something to reach for twice.
        [`TrainingStage.release_trainer`] is how it is let go.
        """
        if self._trainer is None:
            model = self.prepare_model(self.initialize_model())
            trainer_kwargs: dict[str, Any] = {
                "compute_loss_fn": self.compute_loss,
                "metric_logger": self.metric_logger,
                "evaluate_fn": self.evaluate,
                "callbacks": self.callbacks(),
            }
            self._trainer = self.create_trainer(model, **trainer_kwargs)
        return self._trainer

    def release_trainer(self) -> None:
        """Drops the trainer and frees the card it held.

        A benchmark reads a written checkpoint precisely so that the model, its gradients and its
        optimizer are off the card before a vocoder, a recognizer or a judge is built.
        """
        self._trainer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def evaluators(self) -> list[Evaluator]:
        if self._evaluators is None:
            self._evaluators = list(self.create_evaluators())
            for evaluator in self._evaluators:
                if evaluator.metric_logger is None:
                    evaluator.metric_logger = self.metric_logger
        return self._evaluators

    @property
    def validators(self) -> list[Evaluator]:
        return [evaluator for evaluator in self.evaluators if evaluator.mode is EvaluatorMode.EVALUATE]

    @property
    def testers(self) -> list[Evaluator]:
        return [evaluator for evaluator in self.evaluators if evaluator.mode is EvaluatorMode.BENCHMARK]

    @abstractmethod
    def initialize_model(self, checkpoint: str | None = None) -> nn.Module:
        """Builds the model this stage trains and evaluates on.

        Returns:
            `nn.Module`: The model, already on its device.
        """

    @abstractmethod
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Applies this stage's policy to the model.

        The one thing a subclass has to write, because what a phase holds still is the phase, and
        a default that trains everything would be a policy nobody wrote down. Freezing is the
        usual content, and it is written over named subtrees rather than module by module, since
        holding the first twenty of twenty four blocks is a slice and not a predicate. A policy
        that really is per module can still reach for `model.apply` from in here.

        [`TrainingStage.create_trainer`] calls this before it builds anything. The order matters:
        the optimizer's parameter groups are taken from the parameters that carry a gradient at
        that moment, so a parameter frozen afterwards is already in a decaying group and is eroded
        by weight decay whatever its gradient says.

        Args:
            model (`nn.Module`):
                The assembled model, already on its device.

        Returns:
            `nn.Module`: The model after the stage's policy has been applied to it.
        """

    @abstractmethod
    def create_trainer(self, model: nn.Module, **kwargs: Any) -> Trainer:
        """Prepares `model` and builds the trainer that runs this stage on it.

        Args:
            model (`nn.Module`):
                The assembled model, already on its device. This stage's policy is applied to
                it here.
            **kwargs:
                Passed to the trainer, overriding anything the stage would have supplied. Name
                `trainer_cls` to build something other than [`~voicestudio.utils.train.Trainer`].

        Returns:
            `Trainer`: The trainer, with this stage attached.
        """

    def create_evaluators(self) -> list[Evaluator]:
        """Builds what this stage reads with.

        Mirrors [`TrainingStage.create_trainer`]: a stage makes what it owns rather than being
        handed it. Called once, the first time an evaluator is needed, so a benchmark's dataset is
        not opened while the phase is still training.

        Each evaluator carries its own name, its own rows and the mode that says when it runs, so
        nothing here restates any of those. A stage that measures nothing returns an empty list,
        which is a real answer for a phase that only warms weights up.

        Returns:
            `list[Evaluator]`: The evaluators, in the order they should be reported.
        """
        return []

    @abstractmethod
    def compute_loss(
        self, model: nn.Module, inputs: dict[str, Any], **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, float], Any]:
        """Scores one micro-batch and prices it.

        Returns the model's own objective, which every model here computes from `labels` and
        returns on its output. An objective that sums terms reports each one, since a total that
        stays finite says nothing about which term stopped moving.

        Args:
            model (`nn.Module`):
                The model as the step will run it.
            inputs (`dict[str, Any]`):
                The micro-batch, already on the model's device.
            **kwargs:
                What the trainer passes through, such as `num_items_in_batch`.

        Returns:
            `tuple[torch.Tensor, dict[str, float], Any]`: The scalar to call `backward` on, the
            terms behind it, which are empty for a single term objective, and the model's output,
            which the trainer's prediction step reads its logits from.

        Raises:
            ValueError: If the model returned no loss, which means it was called without `labels`.
        """

    def train(self, resume_from_checkpoint: Any = None) -> dict[str, dict[str, float]]:
        """Runs this stage end to end: train, release the card, then benchmark what was written.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                Passed to the trainer, to continue a segment that stopped.

        Returns:
            `dict[str, dict[str, float]]`: The metrics, keyed by the checkpoint they came from.
        """
        _ = self.evaluators

        # TODO: For loop required
        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.log(dict(result.metrics), self.global_step)
        self.release_trainer()

        checkpoints = _written_checkpoints(self.stage_config.output_dir)
        return {
            str(checkpoint): metrics
            for checkpoint in checkpoints
            if (metrics := self.benchmark(checkpoint))
        }

    def evaluate(self, model: nn.Module, step: int, **kwargs: Any) -> dict[str, float]:
        """Runs every `evaluate` mode evaluator over the model as it stands.

        This is what the trainer's evaluation is, rather than something taken beside it. Each
        evaluator produces its own artifacts from its own rows, so there is nothing for the loop
        to iterate over and no predictions to accumulate.

        Args:
            model (`nn.Module`):
                The model under evaluation.
            step (`int`):
                The optimizer step the metrics are filed under.
            **kwargs:
                Passed through to every evaluator.

        Returns:
            `dict[str, float]`: Every reading, prefixed by the evaluator that produced it.
        """
        evaluators = self.validators
        merged_metrics: dict[str, float] = {}

        for evaluator in evaluators:
            try:
                metrics = evaluator.evaluate(model, step=step, **kwargs)
            except Exception:
                logger.exception("evaluator %r failed at step %d", evaluator.name, step)
                continue
            merged_metrics.update(
                {f"{evaluator.name}/{key}": value for key, value in metrics.items()}
            )

        self.log(merged_metrics, step)
        return merged_metrics

    def benchmark(self, checkpoint: Any, *, step: int | None = None, **kwargs: Any) -> dict[str, float]:
        """Scores one written checkpoint in `benchmark` mode.

        This runs after a trainer has exited, in a process that trains nothing. Generation needs a
        vocoder, a recognizer or a judge, and none of those belong on a card that still holds a
        sharded model, an optimizer and a live CUDA context.

        Args:
            checkpoint (`str` or `os.PathLike` or `nn.Module`):
                The checkpoint to score, or an already loaded model.
            step (`int`, *optional*):
                The optimizer step to file the metrics under. Read off a `checkpoint-<step>`
                directory name when not given, and `0` when that fails.
            **kwargs:
                Passed through to the evaluator.

        Returns:
            `dict[str, float]`: Every reading, prefixed by the evaluator that produced it, empty
            when the stage takes no reading in that mode.

        """
        evaluators = self.testers
        if not evaluators:
            return {}
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        else:
            model = self.initialize_model(checkpoint)
        if step is None:
            step = _step_of(checkpoint)

        model.eval()
        merged_metrics: dict[str, float] = {}

        for evaluator in evaluators:
            try:
                metrics = evaluator.evaluate(model, step=step, **kwargs)
            except Exception:
                logger.exception("evaluator %r failed at step %d", evaluator.name, step)
                continue
            merged_metrics.update(
                {f"{evaluator.name}/{key}": value for key, value in metrics.items()}
            )

        self.log(merged_metrics, step)
        return merged_metrics

    def callbacks(self) -> list[TrainerCallback]:
        """Builds the callbacks the trainer needs to serve this stage.

        Returns:
            `list[TrainerCallback]`: The step record the stage reads the step off.
        """
        return [_StageStateCallback(self)]

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Writes one set of metrics under this stage's name.

        Args:
            metrics (`dict[str, Any]`):
                The metrics, keyed by name.
            step (`int`):
                The optimizer step they were taken at.
        """
        if metrics and self.metric_logger is not None:
            self.metric_logger.log(
                {f"{self.stage_name}/{key}": value for key, value in metrics.items()}, step
            )


class _StageStateCallback(TrainerCallback):
    """Keeps the trainer state where a stage's adapters can read the step off it.

    Args:
        stage (`TrainingStage`):
            The stage to hand the state to.
    """

    def __init__(self, stage: TrainingStage):
        self.stage = stage

    def on_train_begin(
        self, args: TrainingConfig, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Records the state at the start of training."""
        self.stage._state = state

    def on_evaluate(
        self, args: TrainingConfig, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Records the state before an evaluation that was not preceded by training."""
        self.stage._state = state


def _step_of(checkpoint: Any) -> int:
    """Reads the optimizer step out of a checkpoint directory's name.

    Args:
        checkpoint (`str` or `os.PathLike`):
            A path the trainer wrote, named `checkpoint-<step>`.

    Returns:
        `int`: The step, or `0` when the name does not carry one.
    """
    tail = os.path.basename(str(checkpoint).rstrip("/"))
    prefix = f"{PREFIX_CHECKPOINT_DIR}-"
    if tail.startswith(prefix) and tail[len(prefix) :].isdigit():
        return int(tail[len(prefix) :])
    return 0


def _written_checkpoints(output_dir: Any) -> list[str]:
    """Lists the checkpoints a run wrote, oldest first.

    Args:
        output_dir (`str` or `os.PathLike`):
            The directory the trainer saved into.

    Returns:
        `list[str]`: Paths to the `checkpoint-<step>` directories, ordered by step. Empty when the
        directory holds none, which is what a run that saved nothing leaves behind.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return []
    found = [
        os.path.join(output_dir, entry)
        for entry in os.listdir(output_dir)
        if entry.startswith(f"{PREFIX_CHECKPOINT_DIR}-")
        and os.path.isdir(os.path.join(output_dir, entry))
    ]
    return sorted(found, key=_step_of)


__all__ = ["TrainingStage"]
