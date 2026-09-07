"""Word error rate between reference prompts and transcribed generations."""

import datasets
import evaluate

from .base import TranscriberMixin, normalizing_transform, per_utterance_rates


_DESCRIPTION = """
Word error rate between a script a speech model was asked to speak and the transcript of the audio it
generated. Reports the edit operations behind the rate alongside it, so that a truncated generation,
which scores as deletions, is distinguishable from a mispronounced one, which scores as substitutions.
Given a `transcriber`, takes the generated audio itself and transcribes it before scoring.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list[str]`): Transcripts of the generated audio, or paths to it when a `transcriber`
        was configured.
    references (`list[str]`): Scripts the model was asked to speak.

Returns:
    wer (`float`): Errors over reference words, pooled across every utterance.
    substitutions (`int`): Words aligned to a differing word.
    deletions (`int`): Reference words absent from the transcript.
    insertions (`int`): Transcript words absent from the reference.
    reference_length (`int`): Reference words, the denominator of `wer`.
    utterances (`list[float]`): Per-utterance word error rate, in input order.
    hypotheses (`list[str]`): The text that was scored, which is the transcript when a `transcriber`
        produced it. A recorded rate is not reproducible without the transcripts behind it.
    transcriber (`str` or `None`): Repository id of the model that produced them.

Examples:
    >>> wer = Wer()
    >>> wer.add_batch(predictions=["she sells sea shells"], references=["She sells sea shells by the sea shore."])
    >>> wer.compute()["deletions"]
    4
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Wer(TranscriberMixin, evaluate.Metric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            codebase_urls=["https://github.com/jitsi/jiwer"],
            reference_urls=["https://jitsi.github.io/jiwer/"],
        )

    def _compute(self, predictions: list[str], references: list[str]) -> dict:
        import jiwer

        hypotheses = self.transcribe(predictions) if self.transcriber else predictions
        transform = normalizing_transform(jiwer.ReduceToListOfListOfWords())
        output = jiwer.process_words(references, hypotheses, transform, transform)
        return {
            "wer": output.wer,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "reference_length": output.hits + output.substitutions + output.deletions,
            "utterances": per_utterance_rates(output),
            "hypotheses": list(hypotheses),
            "transcriber": self.transcriber,
        }


__all__ = ["Wer"]
