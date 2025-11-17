"""
XTTS v2 synthesizer implementation.
"""

from pathlib import Path
from typing import Optional, List
import torch
from tqdm import tqdm

from .base import BaseSynthesizer


class XTTSSynthesizer(BaseSynthesizer):
    """XTTS v2 synthesizer using Coqui TTS."""

    def __init__(self, config):
        super().__init__(config)

    def load_model(self) -> None:
        """Load XTTS v2 model."""
        try:
            from TTS.api import TTS

            # Load XTTS v2 model
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            self.model = TTS(model_name).to(self.config.device)
            self.is_loaded = True

            print(f"Loaded XTTS v2 model on {self.config.device}")

        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Install with: pip install TTS")
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS model: {e}")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        reference_audio: Path = None,
        style_prompt: Optional[str] = None,
        speaker_id: Optional[str] = None
    ) -> bool:
        """Synthesize speech using XTTS voice cloning.

        text: Text to synthesize
            output_path: Path to save synthesized audio
            reference_audio: Path to reference audio for voice cloning
            style_prompt: Optional style prompt (unused in XTTS)
            speaker_id: Optional speaker identifier (unused in XTTS)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded:
            self.load_model()

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate speech with voice cloning
            self.model.tts_to_file(
                text=text,
                speaker_wav=str(reference_audio),
                language=self.config.language,
                file_path=str(output_path),
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p
            )

            # Verify output file was created
            if not output_path.exists() or output_path.stat().st_size == 0:
                print(f"Warning: Output file {output_path} was not created or is empty")
                return False
            return True

        except Exception as e:
            print(f"Failed to synthesize audio with XTTS: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up XTTS model resources."""
        super().cleanup()

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
