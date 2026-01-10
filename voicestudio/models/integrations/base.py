"""
Base synthesizer for TTS models.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseSynthesizer(ABC):
    """Base class for TTS synthesizers."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the TTS model."""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        reference_audio: Path | None = None,
        style_prompt: str | None = None,
        speaker_id: str | None = None
    ) -> bool:
        """Synthesize speech from text using reference audio or style prompt.

        Args:
            text: Text to synthesize
            output_path: Path to save synthesized audio
            reference_audio: Optional Path to reference audio for voice cloning
            style_prompt: Optional style prompt for synthesis
            speaker_id: Optional speaker identifier

        Returns:
            True if successful, False otherwise
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        if not self.is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
