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
"""EmergentTTS-Eval: win rate against a baseline on prompts built to be hard to say."""

import os
from collections.abc import Callable, Sequence
from typing import Any
from dataclasses import dataclass, field

from ..metrics.base import LauncherType, MetricConfig
from ..utils.evaluate import Evaluator
from .audio_basics import SynthesisLoop, build_metrics, intelligibility_metrics
from .base import BenchmarkConfig, build_evaluator
from .instruct_tts_eval import JUDGE_MODEL_ID


BASELINE = "gpt-4o-mini-tts/alloy"
"""What the win rate is measured against.

The dataset ships that system's rendering of every prompt in its `audio` column, so the comparison
is against a fixed recording rather than against a service answering today. A reading is a win rate
over this baseline and means nothing without it named.
"""

CATEGORIES = (
    "emotions",
    "paralinguistics",
    "foreign_words",
    "syntactic_complexity",
    "complex_pronunciation",
    "questions",
)
"""The six ways a prompt is made hard, reported separately as well as pooled."""


def win_rate_metrics(
    model_id: str = JUDGE_MODEL_ID,
    launcher: LauncherType = LauncherType.ENDPOINT,
    endpoint: str | None = None,
) -> list[MetricConfig]:
    """The judged comparison against the baseline recording.

    Args:
        model_id (`str`, *optional*, defaults to [`~benchmarks.instruct_tts_eval.JUDGE_MODEL_ID`]):
            Repository id of the judge. Substituting one is what a version is for, so a reading
            taken under a different judge belongs under a different version.
        launcher (`LauncherType`, *optional*, defaults to `LauncherType.ENDPOINT`):
            Where the judge runs.
        endpoint (`str`, *optional*):
            Base URL it is reached at.

    Returns:
        `list[MetricConfig]`: One configuration, which reports the pooled win rate and one per
        category beside it.
    """
    return [
        MetricConfig(
            name="win_rate",
            launcher=launcher,
            model_id=model_id,
            endpoint=endpoint,
            options={
                "kind": "judge",
                "baseline": BASELINE,
                "baseline_column": "audio",
                "categories": CATEGORIES,
            },
        )
    ]


@dataclass
class EmergentTTSEval(BenchmarkConfig):
    """Prompts evolved to be progressively harder to say, scored against a fixed baseline.

    Each row carries a `text_to_synthesize`, the `category` of difficulty it exercises, an
    `evolution_depth` saying how far that difficulty was pushed, and the baseline system's own
    rendering. A judge hears both and says which followed the prompt better.

    Depth is worth reading as a curve rather than pooled: a model that holds up to depth 1 and
    collapses at 3 and one that is mediocre throughout can report the same average.

    `open_v1` is the published rows and the published baseline judged by
    [`~benchmarks.instruct_tts_eval.JUDGE_MODEL_ID`], since the model the paper judged with has
    been retired.
    """

    name: str = "emergent_tts_eval"
    version: str = "open_v1"
    dataset_id: str | None = "bosonai/EmergentTTS-Eval"
    dataset_split: str | None = "train"
    metrics: Sequence[MetricConfig] = field(
        default_factory=lambda: win_rate_metrics() + intelligibility_metrics()
    )


def build(
    benchmark: EmergentTTSEval,
    dataset: Any,
    output_dir: str,
    generate: Callable[..., Any] | None = None,
    metric_logger: Any = None,
    **overrides: Any,
) -> Evaluator:
    """Wires this benchmark into the evaluator a stage takes.

    Args:
        benchmark (`EmergentTTSEval`):
            What the benchmark fixes.
        dataset (`Any`):
            The rows to read, already loaded.
        output_dir (`str`):
            Where the clips and the readings are written.
        generate (`Callable`, *optional*):
            How the model is asked to speak under an instruction. Left out, `model.generate` is
            called with the script alone, which measures nothing this benchmark asks about.
        metric_logger (`BaseLogger`, *optional*):
            Where the pooled readings are streamed.
        **overrides:
            Fields of [`~voicestudio.utils.evaluate.EvaluatorConfig`] belonging to the run.

    Returns:
        `Evaluator`: The evaluator, ready to be handed a checkpoint.
    """
    return build_evaluator(
        benchmark,
        metrics=build_metrics(benchmark.metrics),
        produce_artifacts=SynthesisLoop(
            output_dir=os.path.join(output_dir, "audio"),
            text_column='text_to_synthesize',
            instruction_column=None,
            reference_column='audio',
            generate=generate,
        ),
        dataset=dataset,
        metric_logger=metric_logger,
        output_dir=output_dir,
        **overrides,
    )


__all__ = [
    "build","BASELINE", "CATEGORIES", "EmergentTTSEval", "win_rate_metrics"]
