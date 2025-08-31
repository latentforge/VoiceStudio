"""
Base synthesizer for TTS models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


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
            reference_audio: Path,
            output_path: Path,
            speaker_id: Optional[str] = None
    ) -> bool:
        """Synthesize speech from text using reference audio.

        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio for voice cloning
            output_path: Path to save synthesized audio
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