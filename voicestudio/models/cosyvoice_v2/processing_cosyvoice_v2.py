"""Processor class for CosyVoice v2."""

from ..cosyvoice_v1.processing_cosyvoice_v1 import CosyVoiceV1Processor


class CosyVoiceV2Processor(CosyVoiceV1Processor):
    r"""
    Constructs a CosyVoice v2 processor which wraps the Qwen tokenizer shipped with the `CosyVoice-BlankEN`
    checkpoint. See [`CosyVoiceV1Processor`] for behavior; CosyVoice v2 outputs 24 kHz audio.
    """

    def decode(self, waveform, sample_rate: int | None = None):
        """
        Args:
            waveform (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
                Waveform produced by [`~CosyVoiceV2ForConditionalGeneration.generate`].
            sample_rate (`int`, *optional*):
                Overrides the model's configured output sample rate.

        Returns:
            `tuple(torch.FloatTensor, int)`: The waveform and its sample rate.
        """
        return waveform, sample_rate or 24000


__all__ = ["CosyVoiceV2Processor"]
