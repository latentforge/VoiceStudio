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
"""Reference and synthesis pairs, which is what a speech model is asked first."""

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import shutil

import soundfile
import torch
from torch import nn

from ..metrics.base import Metric, MetricConfig
from ..metrics.judge import Transcriber
from ..metrics.cer import Cer
from ..metrics.ffe import Ffe
from ..metrics.mcd import Mcd
from ..metrics.ssim import Ssim
from ..metrics.utmos import Utmos
from ..metrics.wer import Wer
from ..utils.evaluate import Evaluator
from .base import BenchmarkConfig, build_evaluator


logger = logging.getLogger(__name__)


class SynthesisLoop:
    """Speaks each row's transcript back in the voice of its own recording, and keeps the pair.

    One row makes one pair: the recording is copied out as the reference and the model is asked to
    say the same words in that voice. Everything downstream reads the pair, so the copy is not
    redundant. A reference left in the dataset is a path that means nothing once the rows are
    reshuffled, and the metrics that compare against it would be reading a different corpus than
    the one that was scored.

    A benchmark that asks for something else, such as rolling two streams against each other,
    writes its own loop instead of reusing this.

    The clips are written rather than held, because scoring reads them several times over, a judge
    reaching a service needs a file to send, and a run scored again under a different judge should
    not have to synthesize again.

    Args:
        output_dir (`str`):
            Directory the pair is written under, as `ref/` and `syn/`. Created if it does not
            exist.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Rate the syntheses are written at. Recorded on every artifact, since a metric that
            resamples and one that does not answer differently.
        text_column (`str`, *optional*, defaults to `"text"`):
            Which field of a row carries the transcript to speak.
        reference_column (`str`, *optional*, defaults to `"audio"`):
            Which field carries the recording the voice is taken from and the synthesis is
            compared against.
        id_column (`str`, *optional*, defaults to `"id"`):
            Which field identifies a row. A distributed run drops the rows a sampler duplicated by
            this, so a benchmark whose rows carry no id gets one from its position.
        style_column (`str`, *optional*):
            Which field carries a style prompt, for a dataset that has one.
        speaker_column (`str`, *optional*):
            Which field names the speaker, for a model that takes one.
        generate (`Callable`, *optional*):
            How the model is asked to speak. Called with the model and the row, and returns the
            waveform. Left out, `model.generate` is called with the transcript and the reference,
            which is what a model following the `transformers` conventions supports.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Decoding settings. Recorded on every artifact: two runs decoded differently are not
            comparable, and nothing downstream would notice.
        seed (`int`, *optional*):
            Seed set before each row. Given one, a run repeats; recorded either way.
    """

    def __init__(
        self,
        output_dir: str,
        sampling_rate: int = 24000,
        text_column: str = "text",
        reference_column: str = "audio",
        id_column: str = "id",
        style_column: str | None = None,
        speaker_column: str | None = None,
        generate: Callable[..., Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ):
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        self.text_column = text_column
        self.reference_column = reference_column
        self.id_column = id_column
        self.style_column = style_column
        self.speaker_column = speaker_column
        self.generate = generate
        self.generation_kwargs = generation_kwargs or {}
        self.seed = seed
        self._written = 0

    def __call__(self, model: nn.Module, batch: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Synthesizes one batch of rows and returns one artifact per pair.

        Args:
            model (`nn.Module`):
                The model under evaluation, in eval mode and under `no_grad`.
            batch (`Any`):
                A mapping of columns to lists, or a sequence of row mappings.
            **kwargs:
                Unused.

        Returns:
            `list[dict[str, Any]]`: One artifact per row, carrying `id`, `audio`,
            `reference_audio`, the transcript both are meant to say, and what produced the
            synthesis.
        """
        reference_dir = os.path.join(self.output_dir, "ref")
        synthesis_dir = os.path.join(self.output_dir, "syn")
        os.makedirs(reference_dir, exist_ok=True)
        os.makedirs(synthesis_dir, exist_ok=True)

        artifacts = []
        for row in _rows(batch):
            if self.seed is not None:
                torch.manual_seed(self.seed)
            identifier = row.get(self.id_column, self._written)
            reference_path = os.path.join(reference_dir, f"ref_{identifier}.wav")
            synthesis_path = os.path.join(synthesis_dir, f"syn_{identifier}.wav")

            source = row.get(self.reference_column)
            if source is not None:
                shutil.copyfile(source, reference_path)
            waveform = self._synthesize(model, row)
            soundfile.write(synthesis_path, waveform, self.sampling_rate)
            self._written += 1

            artifacts.append(
                {
                    "id": identifier,
                    "audio": synthesis_path,
                    "reference_audio": reference_path if source is not None else None,
                    "reference": row.get(self.text_column),
                    "speaker": row.get(self.speaker_column) if self.speaker_column else None,
                    "sampling_rate": self.sampling_rate,
                    "seed": self.seed,
                    "generation_kwargs": dict(self.generation_kwargs),
                }
            )
        return artifacts

    def _synthesize(self, model: nn.Module, row: Mapping[str, Any]):
        """Runs the model over one row and returns the waveform to write.

        Args:
            model (`nn.Module`):
                The model under evaluation.
            row (`Mapping[str, Any]`):
                One row of the benchmark's dataset.

        Returns:
            `numpy.ndarray`: The clip, as `soundfile` takes it.
        """
        if self.generate is not None:
            output = self.generate(model, row, **self.generation_kwargs)
        else:
            arguments = dict(self.generation_kwargs)
            if self.reference_column in row:
                arguments["reference_audio"] = row[self.reference_column]
            if self.style_column:
                arguments["style_prompt"] = row.get(self.style_column)
            output = model.generate(row[self.text_column], **arguments)
        waveform = getattr(output, "audio", output)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().to(torch.float32).cpu().squeeze().numpy()
        return waveform


def _rows(batch: Any) -> list[Mapping[str, Any]]:
    """Reads a batch as a sequence of rows, however the loader hands it over.

    Args:
        batch (`Any`):
            A mapping of columns to lists, or a sequence of row mappings.

    Returns:
        `list[Mapping[str, Any]]`: The rows.
    """
    if isinstance(batch, Mapping):
        keys = list(batch)
        return [{key: batch[key][index] for key in keys} for index in range(len(batch[keys[0]]))]
    return list(batch)


def build_transcriber(config: MetricConfig) -> Transcriber:
    """Builds the recognizer an error rate reads its transcripts from.

    Args:
        config (`MetricConfig`):
            Which recognizer, at what precision, on which device.

    Returns:
        `Transcriber`: The recognizer, not yet loaded.

    Raises:
        ValueError: If the configuration names no `model_id`.
    """
    if not config.model_id:
        raise ValueError(f"metric {config.name!r} needs a `model_id` to transcribe with")
    return Transcriber(config)


def build_metrics(configs: Sequence[MetricConfig]) -> list[Metric]:
    """Builds the reference free and reference paired readings from their configurations.

    One recognizer is built for every error rate that names the same one, so scoring a word rate
    and a character rate over a run loads it once.

    Args:
        configs (`Sequence[MetricConfig]`):
            What to build, keyed by the `kind` each carries in its options.

    Returns:
        `list[Metric]`: The metrics, in the order they were configured.

    Raises:
        ValueError: If a configuration names a kind this module does not build.
    """
    transcribers: dict[str, Transcriber] = {}
    built: list[Metric] = []
    for config in configs:
        kind = config.options.get("kind", config.name)
        if kind in ("wer", "cer"):
            if config.model_id not in transcribers:
                transcribers[config.model_id] = build_transcriber(config)
            built.append((Wer if kind == "wer" else Cer)(config, transcribers[config.model_id]))
        elif kind == "utmos":
            built.append(Utmos(config, model_id=config.model_id or "sarulab-speech/UTMOSv2"))
        elif kind == "ssim":
            built.append(Ssim(config, model_id=config.model_id or "speechbrain/spkrec-ecapa-voxceleb"))
        elif kind == "mcd":
            built.append(Mcd(config))
        elif kind == "ffe":
            built.append(Ffe(config))
        else:
            raise ValueError(f"metric {config.name!r} names kind {kind!r}, which is not built here")
    return built


def intelligibility_metrics(transcriber: str = "openai/whisper-large-v3") -> list[MetricConfig]:
    """The readings that answer whether a generation says what it was asked to.

    Args:
        transcriber (`str`, *optional*, defaults to `"openai/whisper-large-v3"`):
            Repository id of the recognizer. Recorded beside every rate, because two recognizers
            disagreed on the same audio three times in one day here.

    Returns:
        `list[MetricConfig]`: A word rate and a character rate over the same recognizer.
    """
    return [
        MetricConfig(name="wer", model_id=transcriber, options={"kind": "wer"}),
        MetricConfig(name="cer", model_id=transcriber, options={"kind": "cer"}),
    ]


def quality_metrics() -> list[MetricConfig]:
    """The reading that answers how a generation sounds, with nothing to compare it against.

    Returns:
        `list[MetricConfig]`: A predicted opinion score.
    """
    return [MetricConfig(name="utmos", model_id="sarulab-speech/UTMOSv2", options={"kind": "utmos"})]


def similarity_metrics() -> list[MetricConfig]:
    """The readings that answer how close a generation is to a reference recording.

    Returns:
        `list[MetricConfig]`: Speaker similarity, mel cepstral distortion and pitch error.
    """
    return [
        MetricConfig(
            name="ssim", model_id="speechbrain/spkrec-ecapa-voxceleb", options={"kind": "ssim"}
        ),
        MetricConfig(name="mcd", options={"kind": "mcd"}),
        MetricConfig(name="ffe", options={"kind": "ffe"}),
    ]


@dataclass
class AudioBasics(BenchmarkConfig):
    """One synthesis per reference recording, scored against the recording it came from.

    The first question asked of a speech model: given a recording and the words it says, can the
    model say those words in that voice. Every reading here compares a synthesis against its own
    reference, so the corpus is pairs rather than clips.

    The three families answer different failures and none of them substitutes for another. An
    error rate says whether the words came out; a predicted opinion score says whether the result
    is listenable; speaker similarity, cepstral distortion and pitch error say whether it is the
    same voice saying them. A model can pass any two and fail the third.

    Args:
        pairs (`int`, *optional*, defaults to 100):
            How many reference and synthesis pairs the reading is taken over. Part of the protocol:
            a rate over 20 pairs and one over 100 are not the same measurement.
    """

    name: str = "audio_basics"
    pairs: int = 100
    metrics: Sequence[MetricConfig] = field(
        default_factory=lambda: (
            intelligibility_metrics() + quality_metrics() + similarity_metrics()
        )
    )


def build(
    benchmark: AudioBasics,
    dataset: Any,
    output_dir: str,
    text_column: str = "text",
    reference_column: str = "audio",
    style_column: str | None = None,
    speaker_column: str | None = None,
    generate: Callable[..., Any] | None = None,
    metric_logger: Any = None,
    **overrides: Any,
) -> Evaluator:
    """Wires this benchmark into the evaluator a stage takes.

    Args:
        benchmark (`AudioBasics`):
            What the benchmark fixes.
        dataset (`Any`):
            The rows to read, already loaded.
        output_dir (`str`):
            Where the pairs and the readings are written. The pairs go under `pairs/ref` and
            `pairs/syn`, so a rescoring can find them without the readings being in the way.
        text_column (`str`, *optional*, defaults to `"text"`):
            Which field of a row carries the transcript to speak.
        reference_column (`str`, *optional*, defaults to `"audio"`):
            Which field carries the recording the voice is taken from.
        style_column (`str`, *optional*):
            Which field carries a style prompt, for a dataset that has one.
        speaker_column (`str`, *optional*):
            Which field names the speaker, for a model that takes one.
        generate (`Callable`, *optional*):
            How the model is asked to speak. Left out, `model.generate` is called with the script.
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
            output_dir=os.path.join(output_dir, "pairs"),
            text_column=text_column,
            reference_column=reference_column,
            style_column=style_column,
            speaker_column=speaker_column,
            generate=generate,
        ),
        dataset=dataset,
        metric_logger=metric_logger,
        output_dir=output_dir,
        **overrides,
    )


__all__ = [
    "AudioBasics",
    "build",
    "SynthesisLoop",
    "build_metrics",
    "build_transcriber",
    "intelligibility_metrics",
    "quality_metrics",
    "similarity_metrics",
]
