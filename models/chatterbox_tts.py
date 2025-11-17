"""
Chatterbox-TTS synthesizer implementation.
"""

from pathlib import Path
from typing import Optional
import torch
import torchaudio
from .base import BaseSynthesizer

class ChatterboxSynthesizer(BaseSynthesizer):
    """Chatterbox-TTS synthesizer."""

    def __init__(self, config):
        super().__init__(config)
        self.sampling_rate = None

    def load_model(self) -> None:
        """Load Chatterbox-TTS model."""
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise RuntimeError(
                "Chatterbox library not installed. "
                "Please install it by running: pip install chatterbox-tts"
            )

        try:
            self.model = ChatterboxTTS.from_pretrained(device=self.config.device)
            self.sampling_rate = self.model.sr
            self.is_loaded = True
            print(f"Loaded Chatterbox-TTS model on {self.config.device} with sample rate {self.sampling_rate}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Chatterbox-TTS model: {e}")

    def synthesize(
            self,
            text: str,
            output_path: Path,
            reference_audio: Optional[Path] = None,
            style_prompt: Optional[str] = None,
            speaker_id: Optional[str] = None,
    ) -> bool:
        """Synthesize speech using Chatterbox-TTS.

        Args:
            text: Text to synthesize.
            output_path: Path to save synthesized audio.
            reference_audio: Optional Path to reference audio for voice cloning.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_loaded:
            self.load_model()

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use reference_audio for voice cloning if provided, otherwise generate with default voice.
            audio_prompt_path = str(reference_audio) if reference_audio else None

            # Generate waveform
            waveform = self.model.generate(text, audio_prompt_path=audio_prompt_path)

            # Save the generated audio
            torchaudio.save(str(output_path), waveform, self.sampling_rate)

            if not output_path.exists() or output_path.stat().st_size == 0:
                print(f"Warning: Output file {output_path} was not created or is empty")
                return False
            return True

        except Exception as e:
            print(f"Failed to synthesize audio with Chatterbox-TTS: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up Chatterbox model resources."""
        super().cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
