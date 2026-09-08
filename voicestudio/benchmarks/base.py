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
"""What a benchmark fixes, so that two runs of it are comparable."""

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..metrics.base import Metric, MetricConfig
from ..utils.evaluate import EvaluatorMode, Evaluator, EvaluatorConfig
from ..utils.log_utils import BaseLogger


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """The settings a benchmark fixes, which two runs must share to be comparable.

    A benchmark is not a metric and not a dataset. It is the protocol binding them: which rows are
    read, what the model is asked to produce from them, and what scores the result. This holds
    that protocol as data. It builds nothing and runs nothing, so a module can declare one at
    import and a caller can read it, print it, or record it beside a number without paying for a
    dataset or a judge.

    What makes a reading reproducible is here rather than in whoever runs it. Pinning
    `dataset_revision` is part of that: a moved tag serves different rows under the same name, and
    nothing downstream would notice.

    A benchmark ships as a subclass that fixes its own defaults, so its identity is a type a
    caller can name rather than a constant it has to remember to import, and so a run can move
    what belongs to the run without restating what belongs to the protocol:

    ```python
    @dataclass
    class InstructTTSEval(BenchmarkConfig):
        name: str = "instruct_tts_eval"
        version: str = "open_v1"
        dataset_id: str | None = "CaasiHUANG/InstructTTSEval"
        dataset_revision: str | None = "0f0b8c1"
        dataset_split: str | None = "test"
        metrics: Sequence[MetricConfig] = field(
            default_factory=lambda: [MetricConfig(name="aps"), MetricConfig(name="dsd")]
        )

    benchmark = InstructTTSEval()
    smoke = InstructTTSEval(max_batches=4)
    ```

    Args:
        name (`str`):
            What the benchmark is called.
        version (`str`, *optional*, defaults to `"v1"`):
            Which version of it this is. Together with `name` it is what a reading is filed under,
            and two runs filed under one pair are claimed to be comparable. A benchmark that
            revises its rows gets a new version, and so does one run under a substituted judge or
            a substituted baseline: the readings are real, they are simply not the published
            benchmark's. `open_v1` is the convention for the latter.
        metrics (`Sequence[MetricConfig]`, *optional*):
            How each metric is run, including where its model lives. Held as configuration rather
            than as metrics, since which class reads them is the benchmark module's business.
        mode (`EvaluatorMode`, *optional*, defaults to `EvaluatorMode.BENCHMARK`):
            When the reading is taken. A benchmark reads a written checkpoint, after a trainer has
            exited.
        dataset_id (`str`, *optional*):
            Repository id of the rows read.
        dataset_revision (`str`, *optional*):
            Which commit of `dataset_id` was read.
        dataset_split (`str`, *optional*):
            Which split of it.
        max_batches (`int`, *optional*):
            Stop after this many batches. Set on the benchmark this is part of the protocol; passed
            at build time it is a smoke run, and the two should not be confused.
        score_batch_size (`int`, *optional*, defaults to 16):
            Artifacts handed to a metric per call.
        options (`dict[str, Any]`, *optional*):
            Anything else the benchmark fixes, such as a judge's prompt template or the baseline a
            win rate is measured against. Recorded alongside the readings.
    """

    name: str
    version: str = "v1"
    metrics: Sequence[MetricConfig] = ()
    mode: EvaluatorMode = EvaluatorMode.BENCHMARK
    dataset_id: str | None = None
    dataset_revision: str | None = None
    dataset_split: str | None = None
    max_batches: int | None = None
    score_batch_size: int = 16
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """The name a reading is filed under, which is the benchmark and the version together."""
        return f"{self.name}/{self.version}"

    def evaluator_config(self, **overrides: Any) -> EvaluatorConfig:
        """Builds the configuration an evaluator of this benchmark runs under.

        Args:
            **overrides:
                Fields of [`~voicestudio.utils.evaluate.EvaluatorConfig`] to replace, such as
                `output_dir` and `model_name`, which belong to the run rather than the protocol.

        Returns:
            `EvaluatorConfig`: The configuration, named after this benchmark.
        """
        fields: dict[str, Any] = {
            "name": self.full_name,
            "mode": self.mode,
            "max_batches": self.max_batches,
            "score_batch_size": self.score_batch_size,
        }
        fields.update(overrides)
        return EvaluatorConfig(**fields)

    def with_launcher(self, launcher: Any, **overrides: Any) -> "BenchmarkConfig":
        """Returns the same protocol with every metric's model placed somewhere else.

        Where a metric's model runs is the run's decision rather than the benchmark's, so moving
        one behind a service does not make the readings incomparable.

        Args:
            launcher (`LauncherType`):
                Where the metrics' models run.
            **overrides:
                Fields of [`MetricConfig`] to replace on every metric alongside it, such as
                `endpoint`.

        Returns:
            `BenchmarkConfig`: A copy, with the same name and the same rows.
        """
        moved = [replace(metric, launcher=launcher, **overrides) for metric in self.metrics]
        return replace(self, metrics=moved)


def build_evaluator(
    benchmark: BenchmarkConfig,
    metrics: Sequence[Metric],
    produce_artifacts: Callable[..., list[dict[str, Any]]],
    dataset: Iterable[Any],
    metric_logger: BaseLogger | None = None,
    **overrides: Any,
) -> Evaluator:
    """Assembles one benchmark into the evaluator a stage takes.

    Args:
        benchmark (`BenchmarkConfig`):
            What the benchmark fixes.
        metrics (`Sequence[Metric]`):
            What scores the artifacts, built from the benchmark's metric configurations.
        produce_artifacts (`Callable`):
            The loop this benchmark runs the model through, returning one artifact per sample.
            Written here rather than shared, because what a model is asked to produce is part of
            what the benchmark fixes: rolling two streams against each other and synthesizing a
            script are different questions, and a reading is only comparable to another taken the
            same way.
        dataset (`Iterable[Any]`):
            The rows the benchmark names, already loaded.
        metric_logger (`BaseLogger`, *optional*):
            Where the pooled readings are streamed. Built outside and handed in, so a
            curriculum's benchmarks share one axis.
        **overrides:
            Fields of [`~voicestudio.utils.evaluate.EvaluatorConfig`] belonging to the run rather
            than the protocol, such as `output_dir` and `model_name`.

    Returns:
        `Evaluator`: The evaluator, ready to be handed a checkpoint.
    """
    logger.info(
        "benchmark %r reads %s (%s) and scores it with %s",
        benchmark.full_name,
        benchmark.dataset_id or "rows given at build time",
        benchmark.dataset_revision or "an unpinned revision",
        ", ".join(metric.name for metric in metrics) or "nothing",
    )
    return Evaluator(
        config=benchmark.evaluator_config(**overrides),
        metrics=metrics,
        produce_artifacts=produce_artifacts,
        dataset=dataset,
        metric_logger=metric_logger,
    )


__all__ = ["BenchmarkConfig", "build_evaluator"]
