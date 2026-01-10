"""
Higgs-v2 synthesizer implementation.
"""

from pathlib import Path
from typing import Optional

import torch
import torchaudio

from voicestudio.boson_multimodal.data_types import ChatMLSample, Message
from voicestudio.boson_multimodal.serve.serve_engine import HiggsAudioResponse, HiggsAudioServeEngine

from .base import BaseSynthesizer

class HiggsV2Synthesizer(BaseSynthesizer):
    """Higgs-v2 synthesizer using bosonai."""

    def __init__(self, config):
        super().__init__(config)
        self.sampling_rate = None  # Higgs Audio V2 default sampling rate

    def load_model(self) -> None:
        """Load Higgs-v2 model and tokenizer."""
        try:
            model_path = "bosonai/higgs-audio-v2-generation-3B-base"
            audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"

            self.model = HiggsAudioServeEngine(
                model_path,
                audio_tokenizer_path,
                device=self.config.device
            )
            self.is_loaded = True
            print(f"Loaded Higgs-v2 model on {self.config.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Higgs-v2 model: {e}")

    def synthesize(
            self,
            text: str,
            output_path: Path,
            reference_audio: Optional[Path] = None,
            style_prompt: Optional[str] = None,
            speaker_id: Optional[str] = None
    ) -> bool:
        """Synthesize speech using Higgs-v2.

        Args:
            text: Text to synthesize (prompt).
            output_path: Path to save synthesized audio.
            reference_audio: Optional Path to reference audio for voice cloning.
            style_prompt: Optional style prompt.
            speaker_id: Optional speaker identifier (unused).

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_loaded:
            self.load_model()

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            system_prompt = (
                "Generate audio following instruction.\n\n<|scene_desc_start|>\n"
                f"{style_prompt if style_prompt else 'Audio is recorded from a quiet room.'}"
                "\n<|scene_desc_end|>"
            )

            messages = [
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=text,
                ),
            ]

            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)

            output: HiggsAudioResponse = self.model.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=1024,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )

            if output.audio is not None:
                torchaudio.save(str(output_path), torch.from_numpy(output.audio)[None, :], output.sampling_rate)

                # Verify output file was created
                if not output_path.exists() or output_path.stat().st_size == 0:
                    print(f"Warning: Output file {output_path} was not created or is empty")
                    return False
                return True
            else:
                print("Audio generation failed, no audio data produced.")
                return False

        except Exception as e:
            print(f"Failed to synthesize audio: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up model resources."""
        super().cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
