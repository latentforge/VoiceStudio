"""Word error rate between reference prompts and transcribed generations."""

from .error_rate import ErrorRate


class Wer(ErrorRate):
    r"""Word error rate between a script a model was asked to speak and what it said.

    Edit distance over whitespace split words, over the reference word count, summed across the
    corpus before dividing. Reports the edit operations behind the rate alongside it, so that a
    truncated generation, which scores as deletions, is distinguishable from a mispronounced one,
    which scores as substitutions.

    Args:
        config (`MetricConfig`):
            What the metric is called, and where its transcriber runs.
        transcriber (`Transcriber`, *optional*):
            What turns generated audio into text. Left out, each artifact has to carry a
            `hypothesis` already.
    """

    reduction = "words"


__all__ = ["Wer"]
