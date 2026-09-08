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
"""What a checkpoint is measured with."""

import collections
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import nn
from transformers.utils import is_in_notebook

from .log_utils import BaseLogger


if TYPE_CHECKING:
    # `voicestudio.metrics.base` reaches for the `audio` extra at import, and nothing here
    # needs the protocol at runtime.
    from ..metrics.base import Metric


logger = logging.getLogger(__name__)


_REMOTE_LAUNCHERS = frozenset({"vllm", "endpoint"})
"""Launchers whose model is reached over a service, and so is called from one rank."""


class EvaluatorMode(StrEnum):
    """Which set a reading is taken on, and what it is allowed to cost.

    Attributes:
        EVALUATE:
            The validation set, read from a forward pass inside the training loop. Cheap enough to
            run between checkpoints.
        BENCHMARK:
            The test set, read from generated output at a milestone. Needs a vocoder, a recognizer
            or a judge, none of which belong on a card that already holds the model being trained.
    """

    EVALUATE = "evaluate"
    BENCHMARK = "benchmark"


@dataclass
class MetricsRecord:
    """Every metric's pooled reading, which is the section a curve is read off.

    Attributes:
        aggregated (`dict[str, dict[str, float]]`):
            Each metric's corpus readings, keyed by metric name.
    """

    aggregated: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, metric_name: str, metrics: Mapping[str, float]) -> None:
        """Records one metric's corpus readings.

        Args:
            metric_name (`str`):
                Which metric produced them.
            metrics (`Mapping[str, float]`):
                The readings, keyed by name.
        """
        self.aggregated.setdefault(metric_name, {}).update(metrics)


@dataclass
class DetailsRecord:
    """Every per sample reading, which is what a recorded number is reproducible from.

    A rate without the transcripts behind it cannot be told apart from a later one that diverged,
    so this section is kept beside the pooled one rather than folded into it.

    Attributes:
        rows (`dict[str, list[dict[str, Any]]]`):
            Each metric's per sample readings, in the order they were measured.
    """

    rows: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: collections.defaultdict(list)
    )

    def add(self, metric_name: str, statistics: Iterable[Mapping[str, Any]]) -> None:
        """Records one batch of per sample readings.

        Args:
            metric_name (`str`):
                Which metric produced them.
            statistics (`Iterable[Mapping[str, Any]]`):
                One mapping per sample.
        """
        self.rows[metric_name].extend(dict(statistic) for statistic in statistics)


@dataclass
class ConfigRecord:
    """What the run was, so a reading can be told from a later one taken differently.

    Attributes:
        model_name (`str`):
            What was measured.
        mode (`str`):
            Which set it was measured on.
        step (`int`):
            The optimizer step the checkpoint was taken at.
        checkpoint (`str` or `None`):
            Where the checkpoint was read from, when it was read from one.
        started_at (`str`):
            When the evaluation began, in ISO 8601.
        ended_at (`str` or `None`):
            When it finished.
        metrics (`list[str]`):
            The metrics that ran, by name.
    """

    model_name: str = ""
    mode: str = ""
    step: int = 0
    checkpoint: str | None = None
    started_at: str = ""
    ended_at: str | None = None
    metrics: list[str] = field(default_factory=list)

    def update(self, **fields: Any) -> None:
        """Sets what is known about the run.

        Args:
            **fields:
                Attributes of this record to set. A name it does not carry is ignored, since a
                caller recording extra provenance should not break the evaluation it describes.
        """
        for name, value in fields.items():
            if hasattr(self, name):
                setattr(self, name, value)


class EvaluationRecords:
    """One evaluation's document, in the three sections it is written as.

    This is what an evaluation produced, not the thing that produced it: [`Evaluator`] builds one
    per call, fills it as the run goes, and asks it to write itself. Holding the sections together
    is what keeps a number and the samples behind it in one place.

    - [`MetricsRecord`] holds the pooled reading of every metric.
    - [`DetailsRecord`] holds the per sample readings behind them.
    - [`ConfigRecord`] holds what was measured, on which set, at which step.

    A [`~voicestudio.utils.log_utils.BaseLogger`] is a destination, and it takes a curve rather
    than a report, so [`Evaluator`] streams it the pooled section while this writes all three to
    disk. A transcript is not something a metrics dashboard has a place for.

    Details are saved beside the results rather than folded into them, because a recorded number
    is not reproducible without the samples behind it: a transcript, the clip it came from, and
    which recognizer produced it are what tell a reproduction from a divergence.

    Args:
        output_dir (`str`, *optional*):
            Directory the results and details are written under. Nothing is written without one.
        save_details (`bool`, *optional*, defaults to `True`):
            Whether the per sample metrics are written alongside the pooled ones.

    Example:
        ```python
        record = EvaluationRecords(output_dir="out/eval")
        record.metrics.add("wer", {"wer": 0.041})
        record.details.add("wer", [{"reference": "...", "hypothesis": "..."}])
        record.save()
        ```
    """

    def __init__(
        self,
        output_dir: str | None = None,
        save_details: bool = True,
    ):
        self.metrics = MetricsRecord()
        self.details = DetailsRecord()
        self.config = ConfigRecord()

        self.output_dir = output_dir
        self.should_save_details = save_details

    @property
    def results(self) -> dict[str, Any]:
        """Everything worth keeping about the evaluation except the samples.

        Returns:
            `dict[str, Any]`: The run's configuration and every metric's pooled metrics.
        """
        return {
            "config": asdict(self.config),
            "results": self.metrics.aggregated,
        }

    @property
    def flat_results(self) -> dict[str, float]:
        """The pooled metrics, flattened onto one namespace for a logger to take.

        Returns:
            `dict[str, float]`: Every reading, keyed `<metric>/<reading>`, except where a metric
            names a reading after itself, which is left unprefixed so `wer` does not read `wer/wer`.
        """
        flattened: dict[str, float] = {}
        for metric_name, metrics in self.metrics.aggregated.items():
            for key, value in metrics.items():
                flattened[key if key == metric_name else f"{metric_name}/{key}"] = value
        return flattened

    def save(self) -> str | None:
        """Writes the results, and the details when they were asked for.

        Returns:
            `str` or `None`: Path to the results file, or `None` when no `output_dir` was given.
        """
        if self.output_dir is None:
            return None

        date_id = datetime.now().isoformat().replace(":", "-")
        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"results_{date_id}.json")
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2, ensure_ascii=False, default=str)
        logger.info("saved results to %s", results_path)

        if self.should_save_details:
            details_dir = os.path.join(self.output_dir, "details")
            os.makedirs(details_dir, exist_ok=True)
            for metric_name, rows in self.details.rows.items():
                details_path = os.path.join(details_dir, f"{metric_name}_{date_id}.jsonl")
                with open(details_path, "w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            logger.info("saved details for %d metrics to %s", len(self.details.rows), details_dir)

        return results_path


@dataclass
class EvaluatorConfig:
    """How one evaluation is run.

    Args:
        name (`str`, *optional*, defaults to `"default_evaluator"`):
            What this evaluation is called. Prefixes every reading it produces, so two evaluations
            on one axis stay apart. A benchmark that ships as a pipeline names itself here rather
            than leaving it to whoever assembles it, since the same benchmark renamed twice is two
            unrelated curves.
        mode (`EvaluatorMode`, *optional*, defaults to `EvaluatorMode.EVALUATE`):
            Which set is read, and when the caller is meant to run it. Nothing branches on this;
            it groups evaluators and it is recorded beside the metrics.
        output_dir (`str`, *optional*):
            Where [`EvaluationRecords`] writes. Nothing is written without one.
        save_details (`bool`, *optional*, defaults to `True`):
            Whether the per sample metrics are written alongside the pooled ones.
        max_batches (`int`, *optional*):
            Stop producing after this many batches. A validation reading taken between checkpoints
            is worth more often than it is worth complete.
        score_batch_size (`int`, *optional*, defaults to 16):
            Artifacts handed to a metric per call.
        model_name (`str`, *optional*, defaults to `""`):
            What is being measured, recorded beside the metrics.
    """

    name: str = "default_evaluator"
    mode: EvaluatorMode = EvaluatorMode.EVALUATE
    output_dir: str | None = None
    save_details: bool = True
    max_batches: int | None = None
    score_batch_size: int = 16
    model_name: str = ""


class Evaluator:
    """One checkpoint's measurements, taken without owning the loop that produced it.

    An evaluator takes the live model and returns numbers. It never loads a model, opens a
    tracking run, or reads a step out of a path: the caller owns all three and passes what it has.

    It runs in two passes. The benchmark's own loop turns the dataset into artifacts, which is
    the only time the model runs. Then every metric scores those same artifacts, so a generation
    is paid for once however many metrics read it, and a run can be scored again under a different
    judge without generating again.

    Under a distributed group each rank produces its own shard. Statistics are gathered before
    they are pooled rather than pooling per rank and reducing the results, because the pooling
    rule is the metric's own and is not the same rule twice: an error rate divides summed edits by
    summed reference length, and a mean opinion score averages per clip after dropping what could
    not be scored. Gathering first leaves `pool` unaware that a group exists.

    Args:
        config (`EvaluatorConfig`):
            How the evaluation is run and where it is written.
        metrics (`Sequence[Metric]`):
            What is measured. An empty sequence measures nothing and returns nothing.
        produce_artifacts (`Callable`):
            Runs the model over one batch and returns one artifact per sample. This is the only
            place the model under evaluation runs, and what it should do differs per benchmark:
            one rolls two streams against each other, another synthesizes a script, and a
            validation reading teacher forces instead of generating. The benchmark writes that
            loop; this class only calls it. Its artifacts carry whatever the metrics read, and an
            `id` that survives sharding lets a distributed run drop the rows a sampler duplicated
            to square its last batch.
        dataset (`Iterable`, *optional*):
            Batches to produce over. Left out, the batches have to arrive as `batches` in the
            call's context, which is how a trainer hands over the loader it already built.
        metric_logger (`BaseLogger`, *optional*):
            Where the pooled metrics are streamed, alongside being saved.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        metrics: "Sequence[Metric]",
        produce_artifacts: Callable[..., list[dict[str, Any]]],
        dataset: Iterable[Any] | None = None,
        metric_logger: BaseLogger | None = None,
    ):
        self.config = config
        self.metrics = list(metrics)
        self.produce_artifacts = produce_artifacts
        self.dataset = dataset
        self.metric_logger = metric_logger
        self.record: EvaluationRecords | None = None

    @property
    def name(self) -> str:
        """What this evaluation is called, which prefixes every reading it produces."""
        return self.config.name

    @property
    def mode(self) -> EvaluatorMode:
        """Which set this evaluation reads, and when the caller is meant to run it."""
        return self.config.mode

    def create_record(self) -> EvaluationRecords:
        """Builds the record one call records into.

        A record holds one evaluation, so a fresh one is built per call rather than accumulating
        a second checkpoint's metrics behind the first.

        Returns:
            `EvaluationRecords`: The record, with nothing recorded in it yet.
        """
        return EvaluationRecords(
            output_dir=self.config.output_dir,
            save_details=self.config.save_details,
        )

    def __call__(self, model: nn.Module, *, step: int, **kwargs: Any) -> dict[str, float]:
        """Measures `model`, so an evaluator can stand in wherever a callable is expected.

        Args:
            model (`nn.Module`):
                The model under evaluation, on its device.
            step (`int`):
                The optimizer step the reading is filed under.
            **kwargs:
                Passed to [`Evaluator.evaluate`].

        Returns:
            `dict[str, float]`: What [`Evaluator.evaluate`] returned.
        """
        return self.evaluate(model, step=step, **kwargs)

    def evaluate(self, model: nn.Module, *, step: int, **kwargs: Any) -> dict[str, float]:
        """Measures `model` and returns its metrics.

        Args:
            model (`nn.Module`):
                The model under evaluation, on its device.
            step (`int`):
                The optimizer step the reading is filed under.
            **kwargs:
                Whatever the producer and the metrics need beyond the model. `batches` replaces
                this evaluator's own dataset, and `checkpoint` is recorded beside the metrics.

        Returns:
            `dict[str, float]`: The pooled metrics. Empty when nothing was measurable, which is
            not the same as zero.

        Raises:
            ValueError: If no batches were given, here or at construction.
        """
        if not self.metrics:
            return {}
        batches = kwargs.pop("batches", None) or self.dataset
        if batches is None:
            raise ValueError(
                "the evaluator was given no batches, either at construction or in the call."
            )

        record = self.create_record()
        record.config.update(
            model_name=self.config.model_name,
            mode=str(self.config.mode),
            step=step,
            checkpoint=kwargs.get("checkpoint"),
            started_at=datetime.now().isoformat(),
            metrics=[metric.name for metric in self.metrics],
        )

        artifacts = self._run_producer(model, batches, kwargs)
        self._run_metrics(artifacts, record, kwargs)

        for metric in self.metrics:
            statistics = record.details.rows.get(metric.name, [])
            if statistics:
                record.metrics.add(metric.name, metric.pool(statistics))

        record.config.update(ended_at=datetime.now().isoformat())
        if _is_main_process():
            record.save()
        self.record = record

        # The record is the document and the logger takes a curve, so the pooled section is what
        # goes out. Only one rank writes it, or a group of eight reports the same reading eight
        # times.
        metrics = record.flat_results
        if metrics and self.metric_logger is not None and _is_main_process():
            self.metric_logger.log(metrics, step)
        return metrics

    def _run_producer(
        self, model: nn.Module, batches: Iterable[Any], kwargs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Runs the model over the batches this rank was given.

        Args:
            model (`nn.Module`):
                The model under evaluation.
            batches (`Iterable[Any]`):
                The batches to produce over.
            kwargs (`dict[str, Any]`):
                Passed through to the producer.

        Returns:
            `list[dict[str, Any]]`: Every artifact this rank produced, in order.
        """
        produced: list[dict[str, Any]] = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for index, batch in enumerate(batches):
                    if self.config.max_batches is not None and index >= self.config.max_batches:
                        break
                    produced.extend(self.produce_artifacts(model, batch, **kwargs))
        finally:
            model.train(was_training)
        return produced

    def _run_metrics(
        self, artifacts: list[dict[str, Any]], record: EvaluationRecords, kwargs: dict[str, Any]
    ) -> None:
        """Scores the artifacts with every metric and records each reading.

        A metric whose model runs in this process scores the shard this rank produced, and its
        statistics are gathered afterwards. A metric that reaches a service scores the gathered
        artifacts on one rank instead, because eight ranks calling one endpoint at once is a way
        to lose a benchmark rather than to finish it sooner.

        A metric that raises costs its own reading. It does not cost the others, which have
        already been measured.

        Args:
            artifacts (`list[dict[str, Any]]`):
                What this rank produced.
            record (`EvaluationRecords`):
                Where the per sample metrics are recorded.
            kwargs (`dict[str, Any]`):
                Passed through to every metric.
        """
        gathered: list[dict[str, Any]] | None = None
        for metric in self.metrics:
            remote = getattr(metric.config, "launcher", None) in _REMOTE_LAUNCHERS
            if remote:
                if gathered is None:
                    gathered = _deduplicate(_gather(artifacts))
                mine = gathered if _is_main_process() else []
            else:
                mine = artifacts
            try:
                with metric:
                    statistics = self._score_metric(metric, mine, kwargs)
            except Exception:
                logger.exception("metric %r failed", metric.name)
                continue
            if not remote:
                statistics = _deduplicate(_gather(statistics))
            record.details.add(metric.name, statistics)

    def _score_metric(
        self, metric: "Metric", artifacts: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Scores one metric over its artifacts, in the batches it asked for.

        Args:
            metric (`Metric`):
                The metric to run.
            artifacts (`list[dict[str, Any]]`):
                What it scores.
            kwargs (`dict[str, Any]`):
                Passed through to the metric.

        Returns:
            `list[dict[str, Any]]`: One statistic per artifact, in order.
        """
        size = max(1, int(self.config.score_batch_size))
        statistics: list[dict[str, Any]] = []
        for start in range(0, len(artifacts), size):
            statistics.extend(metric.score(artifacts[start : start + size], **kwargs))
        return statistics

    def get_results(self) -> dict[str, Any] | None:
        """Returns everything the last evaluation recorded except the samples.

        Returns:
            `dict[str, Any]` or `None`: The run's configuration and every metric's pooled
            metrics, or `None` when nothing has been evaluated yet.
        """
        return self.record.results if self.record is not None else None

    def show_results(self) -> None:
        """Renders the last evaluation's metrics, as a table the surrounding session can show.

        Both branches report the way the library's own trainer does, so an evaluation reads the
        way a training run does. Under a notebook kernel the table is drawn by
        `transformers.utils.notebook`, which is what [`~transformers.Trainer`] reports its metrics
        through there; elsewhere the metrics are printed the way
        [`~transformers.PrinterCallback`] prints them. Detection is
        [`~transformers.utils.is_in_notebook`].
        """
        results = self.get_results()
        if results is None:
            print("nothing has been evaluated yet")
            return

        config = results["config"]
        rows = [
            (metric_name, key, value)
            for metric_name, metrics in results["results"].items()
            for key, value in metrics.items()
        ]
        caption = f"{config['model_name'] or 'model'} | {config['mode']} | step {config['step']}"
        if not rows:
            print(f"{caption}\nno reading was measurable")
            return

        if is_in_notebook():
            # `transformers.utils.notebook` imports IPython at module scope, so it is only
            # reachable on the branch where a kernel is answering.
            from IPython.display import HTML, display
            from transformers.utils.notebook import text_to_html_table

            table = [["Metric", "Reading", "Value"]] + [list(row) for row in rows]
            display(HTML(f"<p>{caption}</p>" + text_to_html_table(table)))
        else:
            metrics = {
                key: (f"{value:.4g}" if isinstance(value, float) else value)
                for key, value in self.record.flat_results.items()
            }
            print(caption)
            print(metrics)


def _process_group_ready() -> bool:
    """Whether a distributed group is up and worth reaching for.

    Returns:
        `bool`: `True` when several ranks are running, `False` for a single process.
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _is_main_process() -> bool:
    """Whether this is the rank that writes.

    Returns:
        `bool`: `True` outside a group, and on rank 0 inside one.
    """
    return not _process_group_ready() or dist.get_rank() == 0


def _gather(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collects every rank's rows onto every rank, in rank order.

    Args:
        rows (`list[dict[str, Any]]`):
            What this rank holds.

    Returns:
        `list[dict[str, Any]]`: Every rank's rows concatenated, or the input unchanged outside a
        group.
    """
    if not _process_group_ready():
        return rows
    collected: list[list[dict[str, Any]]] = [None] * dist.get_world_size()  # type: ignore[list-item]
    dist.all_gather_object(collected, rows)
    return [row for shard in collected for row in shard]


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drops the rows a distributed sampler repeated to square its last batch.

    A sampler pads by handing an already seen sample to another rank, so a gathered corpus counts
    those twice unless the rows carry something to tell them apart. Rows without an `id` are kept
    as they are, since guessing which of two identical metrics was the padding would be worse
    than counting it.

    Args:
        rows (`list[dict[str, Any]]`):
            The gathered rows.

    Returns:
        `list[dict[str, Any]]`: The rows, with a repeated `id` kept only the first time.
    """
    if not rows or "id" not in rows[0]:
        if _process_group_ready() and rows:
            logger.warning(
                "gathered %d rows carrying no `id`, so a sampler's padding cannot be told from a "
                "real sample and is counted with the rest",
                len(rows),
            )
        return rows
    seen: set[Any] = set()
    unique = []
    for row in rows:
        key = row.get("id")
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


__all__ = [
    "EvaluatorMode",
    "ConfigRecord",
    "DetailsRecord",
    "EvaluationRecords",
    "Evaluator",
    "EvaluatorConfig",
    "MetricsRecord",
]
