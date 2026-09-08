"""Character error rate between reference prompts and transcribed generations."""

from .error_rate import ErrorRate


class Cer(ErrorRate):
    r"""Character error rate between a script a model was asked to speak and what it said.

    Edit distance over despaced characters, so a spacing disagreement is not an error, over the
    reference character count, summed across the corpus before dividing. Reads beside a word rate
    on a language whose words a recognizer segments inconsistently.

    Args:
        config (`MetricConfig`):
            What the metric is called, and where its transcriber runs.
        transcriber (`Transcriber`, *optional*):
            What turns generated audio into text. Left out, each artifact has to carry a
            `hypothesis` already.
    """

    reduction = "characters"


__all__ = ["Cer"]
