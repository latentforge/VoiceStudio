"""Processor class for CosyVoice v3."""

from ..cosyvoice_v2.processing_cosyvoice_v2 import CosyVoiceV2Processor


class CosyVoiceV3Processor(CosyVoiceV2Processor):
    r"""
    Constructs a CosyVoice v3 processor which wraps the Qwen tokenizer shipped with the `CosyVoice-BlankEN`
    checkpoint. See [`CosyVoiceV1Processor`] for behavior; CosyVoice v3 additionally expects the tokenized text
    to contain an `<|endofprompt|>` marker separating the instruction/prompt span from the text to synthesize.
    """


__all__ = ["CosyVoiceV3Processor"]
