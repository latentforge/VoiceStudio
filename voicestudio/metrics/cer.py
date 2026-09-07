"""Character error rate between reference prompts and transcribed generations."""

import datasets
import evaluate

from .base import TranscriberMixin, normalizing_transform, per_utterance_rates


_DESCRIPTION = """
Character error rate between a script a speech model was asked to speak and the transcript of the audio
it generated. Scores a language written without spaces, which word error rate reduces to a single token
per utterance, and separates a near miss from an unrelated word, which word error rate scores alike.
Given a `transcriber`, takes the generated audio itself and transcribes it before scoring.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list[str]`): Transcripts of the generated audio, or paths to it when a `transcriber`
        was configured.
    references (`list[str]`): Scripts the model was asked to speak.

Returns:
    cer (`float`): Errors over reference characters, pooled across every utterance.
    substitutions (`int`): Characters aligned to a differing character.
    deletions (`int`): Reference characters absent from the transcript.
    insertions (`int`): Transcript characters absent from the reference.
    reference_length (`int`): Reference characters, the denominator of `cer`.
    utterances (`list[float]`): Per-utterance character error rate, in input order.
    hypotheses (`list[str]`): The text that was scored, which is the transcript when a `transcriber`
        produced it. A recorded rate is not reproducible without the transcripts behind it.
    transcriber (`str` or `None`): Repository id of the model that produced them.

Examples:
    >>> cer = Cer()
    >>> cer.add_batch(predictions=["she sells sea shell"], references=["She sells sea shells."])
    >>> cer.compute()["deletions"]
    1
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Cer(TranscriberMixin, evaluate.Metric):
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
        transform = normalizing_transform(jiwer.ReduceToListOfListOfChars())
        output = jiwer.process_characters(references, hypotheses, transform, transform)
        return {
            "cer": output.cer,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
            "reference_length": output.hits + output.substitutions + output.deletions,
            "utterances": per_utterance_rates(output),
            "hypotheses": list(hypotheses),
            "transcriber": self.transcriber,
        }


__all__ = ["Cer"]
