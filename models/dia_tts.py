"""
Dia-TTS synthesizer implementation.
"""

from pathlib import Path
from typing import Optional
import torch
import torchaudio
import numpy as np
from .base import BaseSynthesizer

class DiaSynthesizer(BaseSynthesizer):
    """Dia-TTS synthesizer using nari-labs/Dia-1.6B-0626."""

    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.sampling_rate = 44100  # Dia-TTS default sampling rate
        self.min_audio_duration = 6.0
        self.max_audio_duration = 20.0

    def load_model(self) -> None:
        """Load Dia-TTS model and processor."""
        try:
            from transformers import AutoProcessor, DiaForConditionalGeneration
        except ImportError:
            raise RuntimeError("transformers not installed. Install with: pip install transformers")

        try:
            model_name = "nari-labs/Dia-1.6B-0626"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = DiaForConditionalGeneration.from_pretrained(model_name).to(self.config.device)
            self.is_loaded = True
            print(f"Loaded Dia-TTS model on {self.config.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Dia-TTS model: {e}")

    def synthesize(
            self,
            text: str,
            output_path: Path,
            reference_audio: Optional[Path] = None,
            style_prompt: Optional[str] = None,
            speaker_id: Optional[str] = None
    ) -> bool:
        """Synthesize speech using Dia-TTS.

        Args:
            text: Text to synthesize.
            output_path: Path to save synthesized audio.
            reference_audio: Path to reference audio for voice cloning.
            style_prompt: Optional style prompt (unused).
            speaker_id: Optional speaker identifier (unused).

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_loaded:
            self.load_model()

        if reference_audio is None:
            print("Error: Dia-TTS requires a reference audio for voice cloning.")
            return False

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Load and process reference audio
            waveform, sample_rate = torchaudio.load(str(reference_audio))
            original_duration = waveform.shape[1] / sample_rate

            # Resample if needed
            if sample_rate != self.sampling_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            current_duration = waveform.shape[1] / self.sampling_rate
            repeat_count = 1

            # Pad short audios by repeating with short silence
            if current_duration < self.min_audio_duration:
                target_samples = int(self.min_audio_duration * self.sampling_rate)

                repeat_count = (target_samples - 1) // waveform.shape[1]

                silence_duration = 0.1
                silence_samples = int(silence_duration * self.sampling_rate)
                silence = torch.zeros(1, silence_samples)

                repeated_waveform = []
                for _ in range(repeat_count):
                    repeated_waveform.append(waveform)
                    repeated_waveform.append(silence)
                waveform = torch.cat(repeated_waveform, dim=1)[:, :target_samples]

            # Trim long audios
            elif current_duration > self.max_audio_duration:
                target_samples = int(self.max_audio_duration * self.sampling_rate)
                waveform = waveform[:, :target_samples]
                print(f"[Dia Debug] Trimmed input from {original_duration:.2f}s to {self.max_audio_duration:.2f}s")

            # Repeat S1 text according to repeat_count
            s1_text = " ".join([text] * repeat_count)
            print(repeat_count)
            input_text = [f"[S1] {s1_text} [S2] {text}"]

            inputs = self.processor(
                text=input_text,
                audio=waveform.squeeze().numpy(),
                sampling_rate=self.sampling_rate,
                padding=True,
                return_tensors="pt"
            ).to(self.config.device)

            estimated_tokens = len(text.split()) * 40
            max_tokens = min(estimated_tokens, 1536)
            prompt_len = self.processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

            # Generate audio
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                )

            # Decode and save audio
            decoded_outputs = self.processor.batch_decode(outputs, audio_prompt_len=prompt_len)
            self.processor.save_audio(decoded_outputs, str(output_path))

            if not output_path.exists() or output_path.stat().st_size == 0:
                print(f"Warning: Output file {output_path} was not created or is empty")
                return False
            return True

        except Exception as e:
            print(f"Failed to synthesize audio with Dia-TTS: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up Dia model resources."""
        super().cleanup()
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()