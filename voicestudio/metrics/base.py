"""What a metric is, and the recognizer the ones that read text share."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .judge import Judge


class LauncherType(StrEnum):
    """Where a metric's model runs.

    Which one a metric uses is the benchmark's decision rather than the model's. The same
    recognizer is worth holding in process for a reading taken between checkpoints and worth
    putting behind a service for a run that scores thousands of clips, and a judge is worth
    pinning to an environment of its own whatever it is scoring.

    Attributes:
        INPROCESS:
            Loaded onto this process's device. Costs the memory for as long as it is held, and
            costs nothing to reach.
        VLLM:
            Served by a vLLM process this metric brings up, and reached over its OpenAI compatible
            endpoint.
        ENDPOINT:
            Reached over an OpenAI compatible endpoint somebody else brought up.
    """

    INPROCESS = "inprocess"
    VLLM = "vllm"
    ENDPOINT = "endpoint"


@dataclass
class MetricConfig:
    """How one metric is run.

    Args:
        name (`str`):
            What the metric is called. Prefixes every reading it produces.
        launcher (`LauncherType`, *optional*, defaults to `LauncherType.INPROCESS`):
            Where the metric's model runs, when it has one. A metric that is arithmetic over the
            artifacts alone ignores this.
        model_id (`str`, *optional*):
            Repository id of the model the metric measures with. A recorded number is not
            reproducible without it: two recognizers disagreed on the same audio three times in
            one day here, once making a healthy generation read as a failure.
        revision (`str`, *optional*):
            Which commit of `model_id` was used, for the same reason.
        endpoint (`str`, *optional*):
            Base URL the metric reaches its model at, under `VLLM` and `ENDPOINT`.
        device (`str`, *optional*):
            Device an in process model is loaded onto. Defaults to CUDA where there is one.
        dtype (`str`, *optional*):
            Precision an in process model is loaded in.
        batch_size (`int`, *optional*, defaults to 8):
            Artifacts scored per call to the model.
        options (`dict[str, Any]`, *optional*):
            Anything the launcher or the metric needs beyond these, such as a quantization choice
            or a judge's decoding settings. Recorded alongside the readings.
    """

    name: str
    launcher: LauncherType = LauncherType.INPROCESS
    model_id: str | None = None
    revision: str | None = None
    endpoint: str | None = None
    device: str | None = None
    dtype: str | None = None
    batch_size: int = 8
    options: dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    r"""One measurement, taken over artifacts an evaluator already produced.

    A metric never runs the model under evaluation. It reads what that model produced, which is
    what lets several metrics score one generation instead of each synthesizing its own, and what
    lets a run be scored again under a different judge without generating again.

    The measurement is split at the point where a sample stops being a sample. A metric reports
    per sample statistics and pools them separately, because the two readings a run needs are
    taken at different places: a training loop wants one number, and a benchmark wants the sample
    that produced it. Splitting also keeps the pooling rule explicit, which matters because it is
    not the same rule twice. An error rate divides summed edits by summed reference length, and a
    mean opinion score averages per clip after dropping what could not be scored; averaging the
    first the way the second is averaged silently answers a different question.

    Args:
        config (`MetricConfig`):
            What the metric is called, and where its model runs.
        judge (`Judge`, *optional*):
            The model this metric asks about an artifact. Shared rather than built per metric, so
            a word rate and a character rate over one run load one recognizer instead of two.
    """

    def __init__(self, config: MetricConfig, judge: "Judge | None" = None):
        self.config = config
        self.judge = judge

    @property
    def name(self) -> str:
        """What this metric is called, which prefixes every reading it produces."""
        return self.config.name

    def load(self) -> None:
        """Brings up whatever this metric measures with, once.

        Delegates to the judge a metric holds, which is what knows whether its model runs in this
        process, behind a vLLM this run brought up, or at a service somebody else did. A metric
        that is arithmetic over the artifacts alone holds none and loads nothing.
        """
        if self.judge is not None:
            self.judge.load()

    def release(self) -> None:
        """Drops whatever [`Metric.load`] brought up, and frees what it held."""
        if self.judge is not None:
            self.judge.release()

    @abstractmethod
    def score(self, artifacts: Sequence[Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Measures a batch of artifacts, one reading per artifact.

        Args:
            artifacts (`Sequence[Any]`):
                What the evaluator produced, in the order it produced them.
            **kwargs:
                Whatever the metric needs beyond the artifacts.

        Returns:
            `list[dict[str, Any]]`: One mapping per artifact, holding whatever [`Metric.pool`]
            needs to reach the corpus number, plus anything worth keeping beside it. What was
            measured with, and what it answered, belong here: a rate without the transcripts
            behind it cannot be reproduced.
        """

    @abstractmethod
    def pool(self, statistics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Reduces every artifact's statistics to the corpus reading.

        Args:
            statistics (`Sequence[Mapping[str, Any]]`):
                Everything [`Metric.score`] returned, in the order it was measured.

        Returns:
            `dict[str, float]`: The corpus readings, keyed by name.
        """


__all__ = [
    "LauncherType",
    "Metric",
    "MetricConfig",
]
