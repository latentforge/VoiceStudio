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
"""InstructTTSEval: whether a model follows an instruction about how to speak."""

import os
from collections.abc import Callable, Sequence
from typing import Any
from dataclasses import dataclass, field

from ..metrics.base import LauncherType, MetricConfig
from ..utils.evaluate import Evaluator
from .audio_basics import SynthesisLoop, build_metrics, intelligibility_metrics
from .base import BenchmarkConfig, build_evaluator


JUDGE_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
"""The judge the `open_v1` version is scored by, in place of the retired Gemini model."""

TASKS = ("APS", "DSD", "RP")
"""The three instruction columns, in the order the benchmark escalates them.

`APS` specifies acoustic parameters outright, `DSD` describes a style, and `RP` gives a character
or a scenario and leaves the style to be inferred. They share one `text`, so a model can be read
across all three on the same script.
"""


def judge_metrics(
    model_id: str = JUDGE_MODEL_ID,
    launcher: LauncherType = LauncherType.ENDPOINT,
    endpoint: str | None = None,
) -> list[MetricConfig]:
    """One judged reading per instruction task.

    Args:
        model_id (`str`, *optional*, defaults to [`JUDGE_MODEL_ID`]):
            Repository id of the judge. Substituting one is what a version is for, so a reading
            taken under a different judge belongs under a different version.
        launcher (`LauncherType`, *optional*, defaults to `LauncherType.ENDPOINT`):
            Where it runs. A judge this size is worth a process of its own, and putting it behind
            a service is also what keeps its answers from moving when the training environment does.
        endpoint (`str`, *optional*):
            Base URL it is reached at.

    Returns:
        `list[MetricConfig]`: One configuration per entry of [`TASKS`].
    """
    return [
        MetricConfig(
            name=task.lower(),
            launcher=launcher,
            model_id=model_id,
            endpoint=endpoint,
            options={"kind": "judge", "instruction_column": task},
        )
        for task in TASKS
    ]


@dataclass
class InstructTTSEval(BenchmarkConfig):
    """Complex natural language instruction following in text to speech.

    Each row carries one `text` and three instructions of escalating indirection, plus a
    `reference_audio` showing what the instruction was meant to produce. A model is asked to speak
    the script under one instruction at a time and a judge reads whether it did.

    The intelligibility readings ride along because an instruction followed at the cost of the
    script is not a pass, and a judge scoring style will not always say so.

    `open_v1` is the published rows judged by [`JUDGE_MODEL_ID`], since the model the paper judged
    with has been retired.
    """

    name: str = "instruct_tts_eval"
    version: str = "open_v1"
    dataset_id: str | None = "CaasiHUANG/InstructTTSEval"
    dataset_split: str | None = "test"
    metrics: Sequence[MetricConfig] = field(
        default_factory=lambda: judge_metrics() + intelligibility_metrics()
    )


def build(
    benchmark: InstructTTSEval,
    dataset: Any,
    output_dir: str,
    generate: Callable[..., Any] | None = None,
    metric_logger: Any = None,
    **overrides: Any,
) -> Evaluator:
    """Wires this benchmark into the evaluator a stage takes.

    Args:
        benchmark (`InstructTTSEval`):
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
            text_column='text',
            instruction_column=None,
            reference_column='reference_audio',
            generate=generate,
        ),
        dataset=dataset,
        metric_logger=metric_logger,
        output_dir=output_dir,
        **overrides,
    )


__all__ = [
    "build","JUDGE_MODEL_ID", "TASKS", "InstructTTSEval", "judge_metrics"]
