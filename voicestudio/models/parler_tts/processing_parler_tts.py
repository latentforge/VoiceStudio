"""Processor class for Parler-TTS."""

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.utils.import_utils import requires


logger = logging.get_logger(__name__)

# The published `parler-tts/parler-tts-*` checkpoints keep the DAC codec fused into the composite
# model's own `model.safetensors`, under an `audio_encoder.` prefix, rather than in a repository of its
# own. `weight_conversion.convert` additionally saves it standalone under this subfolder so it can be
# loaded here the normal `DacModel.from_pretrained` way.
AUDIO_TOKENIZER_SUBFOLDER = "audio_encoder"


class ParlerTTSProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "return_tensors": "pt",
        },
    }


@requires(backends=("torch",))
class ParlerTTSProcessor(ProcessorMixin):
    r"""
    Constructs a Parler-TTS processor which wraps an [`AutoTokenizer`] and a [`DacModel`] into a single
    processor. See [`~ParlerTTSProcessor.__call__`] and [`~ParlerTTSProcessor.decode`] for more information.

    Args:
        tokenizer ([`AutoTokenizer`]):
            The T5 text tokenizer, used both for the voice description and for the transcript to speak. The
            tokenizer is a required input.
        audio_tokenizer ([`DacModel`], *optional*):
            The DAC audio codec. [`ParlerTTSForConditionalGeneration.generate`] already returns a decoded
            waveform using the codec it owns internally; this one decodes codes obtained some other way.
        chat_template (`str`, *optional*):
            A template string for chat formatting when combining text and audio interactions.
    """

    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "DacModel"

    def __init__(self, tokenizer=None, audio_tokenizer=None, chat_template=None):
        super().__init__(tokenizer, audio_tokenizer=audio_tokenizer, chat_template=chat_template)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Loads the processor of a Parler-TTS checkpoint, together with the DAC codec saved standalone in its
        `audio_encoder` subfolder by [`weight_conversion.convert`].

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Repository id or local path of the checkpoint.
            kwargs (`dict[str, Any]`, *optional*):
                Forwarded to the tokenizer and audio tokenizer loaders.

        Returns:
            [`ParlerTTSProcessor`]: The loaded processor. Its `audio_tokenizer` is `None` when the checkpoint
            bundles none, in which case [`~ParlerTTSProcessor.decode`] is unavailable.
        """
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if getattr(processor, "audio_tokenizer", None) is None:
            from transformers import DacModel

            try:
                processor.audio_tokenizer = DacModel.from_pretrained(
                    pretrained_model_name_or_path, subfolder=AUDIO_TOKENIZER_SUBFOLDER, **kwargs
                )
            except OSError:
                logger.warning_once(
                    f"'{pretrained_model_name_or_path}' bundles no '{AUDIO_TOKENIZER_SUBFOLDER}' audio "
                    "tokenizer. `decode` will be unavailable until `audio_tokenizer` is set."
                )
        return processor

    def __call__(
        self,
        description: TextInput | PreTokenizedInput | list[TextInput],
        transcript: TextInput | PreTokenizedInput | list[TextInput],
        **kwargs: Unpack[ParlerTTSProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Args:
            description (`str` or `list[str]`):
                Free-form natural language description of the voice to generate, for example its pitch, pace
                and recording quality.
            transcript (`str` or `list[str]`):
                The text to speak.

        Returns:
            [`~feature_extraction_utils.BatchFeature`]: A dictionary holding `input_ids` and `attention_mask`
            for `description`, and `prompt_input_ids` and `prompt_attention_mask` for `transcript`, ready to
            be splatted into [`ParlerTTSForConditionalGeneration.generate`].
        """
        output_kwargs = self._merge_kwargs(
            ParlerTTSProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        text_kwargs = output_kwargs["text_kwargs"]

        description_inputs = self.tokenizer(description, **text_kwargs)
        prompt_inputs = self.tokenizer(transcript, **text_kwargs)

        return BatchFeature(
            data={
                "input_ids": description_inputs["input_ids"],
                "attention_mask": description_inputs["attention_mask"],
                "prompt_input_ids": prompt_inputs["input_ids"],
                "prompt_attention_mask": prompt_inputs["attention_mask"],
            },
            tensor_type=text_kwargs.get("return_tensors"),
        )

    def decode(self, audio_codes: torch.LongTensor) -> torch.Tensor:
        r"""
        Decodes DAC codes into a waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
                DAC codes to decode.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 1, num_samples)`: The decoded waveform.

        Raises:
            ValueError: If this processor has no `audio_tokenizer`.
        """
        if self.audio_tokenizer is None:
            raise ValueError(
                "This ParlerTTSProcessor has no `audio_tokenizer` (the checkpoint it was loaded from does not "
                "ship one), so codes cannot be decoded to a waveform."
            )
        audio_codes = audio_codes.to(self.audio_tokenizer.device)
        with torch.no_grad():
            return self.audio_tokenizer.decode(audio_codes=audio_codes).audio_values

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


__all__ = ["ParlerTTSProcessor"]
