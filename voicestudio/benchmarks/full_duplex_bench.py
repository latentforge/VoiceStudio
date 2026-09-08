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
"""Full-Duplex-Bench: what a model does while the other speaker is still talking."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..metrics.base import MetricConfig
from .base import BenchmarkConfig


class FullDuplexBench(BenchmarkConfig):
    """Turn taking behaviour, which only shows up when both streams run at once.

    This benchmark does not reuse [`~benchmarks.audio_basics.SynthesisLoop`]. A duplex model is not
    handed a script and asked to read it: it is played a user stream and its own stream is rolled
    against it, so what an artifact holds is a frame grid rather than a clip, and when it spoke
    matters as much as what it said. The loop that produces those artifacts belongs to this module.

    Upstream ships four versions and they measure different things, so each is its own class rather
    than a flag: `v1` and `v1_5` score a fixed corpus offline, `v2` drives a live conversation
    against an examiner, and `v3` scores tool use under disfluent speech.
    """

    name: str = "full_duplex_bench"


@dataclass
class FullDuplexBenchV1(FullDuplexBench):
    """Pause handling, backchanneling, smooth turn taking and user interruption, read offline.

    The readings this fixes are still open. Which of the four dimensions are carried over, and how
    each is scored, has not been decided, so `metrics` is empty rather than guessed at.
    """

    version: str = "v1"
    metrics: Sequence[MetricConfig] = field(default_factory=list)


@dataclass
class FullDuplexBenchV15(FullDuplexBench):
    """`v1` extended with overlap: listener backchannel, side conversation and ambient speech.

    Still an offline corpus, so it shares `v1`'s producer. Which readings are carried over is open.
    """

    version: str = "v1_5"
    metrics: Sequence[MetricConfig] = field(default_factory=list)


@dataclass
class FullDuplexBenchV2(FullDuplexBench):
    """A live conversation against an automated examiner, scored by a judge.

    Not an offline corpus at all: the rows are a session driven over WebRTC or WebSocket, so the
    producer holds a conversation rather than reading one. Whether that is in scope here is open.
    """

    version: str = "v2"
    metrics: Sequence[MetricConfig] = field(default_factory=list)


@dataclass
class FullDuplexBenchV3(FullDuplexBench):
    """Tool use under real world disfluency, over human recordings annotated for five kinds.

    Fillers, pauses, hesitations, false starts and self corrections, against multi step API calls.
    The rows are distributed outside the Hugging Face hub, so `dataset_id` names nothing and the
    path has to be handed in. Which readings are carried over is open.
    """

    version: str = "v3"
    metrics: Sequence[MetricConfig] = field(default_factory=list)


__all__ = [
    "FullDuplexBench",
    "FullDuplexBenchV1",
    "FullDuplexBenchV15",
    "FullDuplexBenchV2",
    "FullDuplexBenchV3",
]
