"""
Parler-TTS synthesizer implementation.
"""

from pathlib import Path
from typing import Optional, List
import torch
import soundfile as sf
from tqdm import tqdm

from .base import BaseSynthesizer


# Default description if none provided
DEFAULT_DESCRIPTION = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

class ParlerTTSSynthesizer(BaseSynthesizer):
    """Parler-TTS synthesizer."""

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None
        self.sampling_rate = None

    def load_model(self) -> None:
        """Load Parler-TTS model and tokenizer."""
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError:
            raise RuntimeError("Parler-TTS not installed. Install with: pip install parler-tts")

        try:
            model_name = "parler-tts/parler-tts-mini-v1"
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.config.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sampling_rate = self.model.config.sampling_rate
            self.is_loaded = True
            print(f"Loaded Parler-TTS model on {self.config.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Parler-TTS model: {e}")

    def synthesize(
            self,
            text: str,
            output_path: Path,
            reference_audio: Optional[Path] = None,
            style_prompt: Optional[str] = None,
            speaker_id: Optional[str] = None
    ) -> bool:
        """Synthesize speech using Parler-TTS.

        Args:
            text: Text to synthesize (prompt).
            output_path: Path to save synthesized audio.
            reference_audio: Optional Path to reference audio (unused in Parler-TTS).
            style_prompt: Optional style prompt (used as description).
            speaker_id: Optional speaker identifier (unused).

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_loaded:
            self.load_model()

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            prompt = text
            description = style_prompt if style_prompt else DEFAULT_DESCRIPTION

            input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.config.device)
            prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.config.device)

            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)

            generation = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
            )

            audio_arr = generation.cpu().numpy().squeeze()

            sf.write(str(output_path), audio_arr, self.sampling_rate)

            # Verify output file was created
            if not output_path.exists() or output_path.stat().st_size == 0:
                print(f"Warning: Output file {output_path} was not created or is empty")
                return False

            return True

        except Exception as e:
            print(f"Failed to synthesize audio: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up model resources."""
        super().cleanup()
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
