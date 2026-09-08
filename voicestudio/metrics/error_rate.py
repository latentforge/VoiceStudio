"""Edit distance between what a model was asked to say and what came out."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .base import Metric, MetricConfig
from .judge import Transcriber


if TYPE_CHECKING:
    from jiwer import AbstractTransform, CharacterOutput, Compose, WordOutput


def normalizing_transform(reduction: "AbstractTransform") -> "Compose":
    """Builds a jiwer transform that normalizes before reducing to tokens.

    Args:
        reduction (`jiwer.AbstractTransform`):
            The reduction the caller's metric aligns on, such as
            [`jiwer.ReduceToListOfListOfWords`] or [`jiwer.ReduceToListOfListOfChars`].

    Returns:
        `jiwer.Compose`: The transform to pass as both `reference_transform` and `hypothesis_transform`.
    """
    import jiwer

    # jiwer's own defaults only collapse whitespace. Transcribers disagree on casing and punctuation far
    # more often than they disagree on words, so comparing without stripping those reports differences
    # that are not errors.
    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            reduction,
        ]
    )


def per_utterance_rates(output: "WordOutput | CharacterOutput") -> list[float]:
    """Recovers a per-utterance error rate from an already computed alignment.

    Args:
        output (`jiwer.WordOutput` or `jiwer.CharacterOutput`):
            The result of `jiwer.process_words` or `jiwer.process_characters`.

    Returns:
        `list[float]`: One rate per utterance, in input order, `inf` where the reference is empty.
    """
    rates = []
    for reference, alignment in zip(output.references, output.alignments):
        # An insertion spans hypothesis tokens and has no width on the reference side.
        errors = sum(
            chunk.hyp_end_idx - chunk.hyp_start_idx
            if chunk.type == "insert"
            else chunk.ref_end_idx - chunk.ref_start_idx
            for chunk in alignment
            if chunk.type != "equal"
        )
        rates.append(errors / len(reference) if reference else float("inf"))
    return rates


def per_utterance_counts(output: "WordOutput | CharacterOutput") -> list[dict[str, int]]:
    """Recovers each utterance's edit counts from an already computed alignment.

    Summing these over a corpus reproduces the totals jiwer reports for it, which is what lets an
    error rate be pooled from per sample statistics rather than recomputed over the whole set.

    Args:
        output (`jiwer.WordOutput` or `jiwer.CharacterOutput`):
            The result of `jiwer.process_words` or `jiwer.process_characters`.

    Returns:
        `list[dict[str, int]]`: One mapping per utterance, in input order, holding `substitutions`,
        `deletions`, `insertions` and `reference_length`.
    """
    counts = []
    for alignment in output.alignments:
        # An insertion spans hypothesis tokens and has no width on the reference side.
        tallied = {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
        for chunk in alignment:
            if chunk.type == "insert":
                tallied["insertions"] += chunk.hyp_end_idx - chunk.hyp_start_idx
            elif chunk.type == "substitute":
                tallied["substitutions"] += chunk.ref_end_idx - chunk.ref_start_idx
            elif chunk.type == "delete":
                tallied["deletions"] += chunk.ref_end_idx - chunk.ref_start_idx
            else:
                tallied["hits"] += chunk.ref_end_idx - chunk.ref_start_idx
        counts.append(
            {
                "substitutions": tallied["substitutions"],
                "deletions": tallied["deletions"],
                "insertions": tallied["insertions"],
                "reference_length": tallied["hits"] + tallied["substitutions"] + tallied["deletions"],
            }
        )
    return counts


class ErrorRate(Metric):
    r"""Edit distance between what a model was asked to say and what came out, pooled by length.

    Reads `audio` off each artifact and transcribes it when a transcriber is configured, or reads
    `hypothesis` when the artifact already carries text, and compares it against `reference`. The
    transcript it scored is kept on every statistic, because a rate recorded without the
    transcripts behind it cannot be told apart from a later one that diverged.

    Pooling sums the edits and sums the reference tokens before dividing, so an utterance weighs by
    its length rather than by being one utterance. Averaging the per utterance rates instead
    answers a different question and reads as the same number.

    Args:
        config (`MetricConfig`):
            What the metric is called, and where its transcriber runs.
        transcriber (`Transcriber`, *optional*):
            What turns generated audio into text. Shared rather than built per metric, so scoring
            word and character rates over one run loads one recognizer instead of two.
    """

    reduction: str = "words"
    """Which jiwer reduction the alignment is taken over, `"words"` or `"characters"`."""

    def __init__(self, config: MetricConfig, transcriber: Transcriber | None = None):
        super().__init__(config, judge=transcriber)

    @property
    def transcriber(self) -> Transcriber | None:
        """The recognizer this metric reads its transcripts from, or `None` for scored text."""
        return self.judge

    def score(self, artifacts: Sequence[Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Aligns each artifact's transcript against its reference.

        Args:
            artifacts (`Sequence[Any]`):
                Artifacts carrying `reference`, and either `audio` to transcribe or `hypothesis`
                already as text.
            **kwargs:
                Unused.

        Returns:
            `list[dict[str, Any]]`: One mapping per artifact, holding the edit counts, the rate,
            the transcript scored and the recognizer that produced it.
        """
        import jiwer

        references = [str(artifact["reference"]) for artifact in artifacts]
        if self.transcriber is not None:
            hypotheses = self.transcriber.transcribe([artifact["audio"] for artifact in artifacts])
        else:
            hypotheses = [str(artifact["hypothesis"]) for artifact in artifacts]

        if self.reduction == "characters":
            transform = normalizing_transform(jiwer.ReduceToListOfListOfChars())
            output = jiwer.process_characters(references, hypotheses, transform, transform)
        else:
            transform = normalizing_transform(jiwer.ReduceToListOfListOfWords())
            output = jiwer.process_words(references, hypotheses, transform, transform)

        statistics = []
        for artifact, hypothesis, counts in zip(artifacts, hypotheses, per_utterance_counts(output)):
            errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
            length = counts["reference_length"]
            statistics.append(
                {
                    "id": artifact.get("id"),
                    **counts,
                    "rate": errors / length if length else float("inf"),
                    "reference": artifact["reference"],
                    "hypothesis": hypothesis,
                    "transcriber": self.transcriber.model_id if self.transcriber else None,
                }
            )
        return statistics

    def pool(self, statistics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Divides the summed edits by the summed reference length.

        Args:
            statistics (`Sequence[Mapping[str, Any]]`):
                Everything [`ErrorRate.score`] returned.

        Returns:
            `dict[str, float]`: The rate under this metric's name, the edit operations behind it,
            and the reference tokens it was divided by. Reading the operations beside the rate is
            what tells a truncated generation, which scores as deletions, from a mispronounced one,
            which scores as substitutions.
        """
        totals = {key: 0 for key in ("substitutions", "deletions", "insertions", "reference_length")}
        for statistic in statistics:
            for key in totals:
                totals[key] += int(statistic.get(key, 0))
        length = totals["reference_length"]
        errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
        return {
            self.name: errors / length if length else float("inf"),
            **{key: float(value) for key, value in totals.items()},
            "utterances": float(len(statistics)),
        }


__all__ = ["ErrorRate", "normalizing_transform", "per_utterance_counts", "per_utterance_rates"]
