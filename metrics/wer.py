"""
WER (Word Error Rate) calculator using Whisper for transcription.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import jiwer

from .base import BaseMetricCalculator, ModelConfig, MetricCalculationError


class WERCalculator(BaseMetricCalculator):
    """Word Error Rate calculator using Whisper for transcription."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.whisper_model = None
        self.transform = self._setup_transform()

    @staticmethod
    def _setup_transform() -> jiwer.Compose:
        """Setup text transformation for WER calculation."""
        return jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])

    def _load_model_impl(self) -> None:
        """Load Whisper model."""
        try:
            import whisper

            model_name = self.config.additional_params.get('model_name', 'large-v3')

            self.whisper_model = whisper.load_model(
                model_name,
                device=self.get_device()
            )

            self.logger.info(f"Loaded Whisper model: {model_name}")

        except ImportError as e:
            raise MetricCalculationError(f"Whisper not installed: {e}")
        except Exception as e:
            raise MetricCalculationError(f"Failed to load Whisper model: {e}")

    def transcribe_audio(self, audio_path: Path) -> str:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            # Whisper transcription options
            options = {
                'language': self.config.additional_params.get('language', 'en'),
                'task': 'transcribe',
                'fp16': torch.cuda.is_available(),
            }

            result = self.whisper_model.transcribe(
                str(audio_path),
                **options
            )

            return result['text'].strip()

        except Exception as e:
            self.logger.error(f"Failed to transcribe {audio_path}: {e}")
            raise MetricCalculationError(f"Transcription failed: {e}")

    def calculate_wer(self, ref_text: str, syn_text: str) -> float:
        """
        Calculate WER between reference and synthesis transcriptions.

        Args:
            ref_text: Reference transcription
            syn_text: Synthesis transcription

        Returns:
            WER score (0.0 = perfect match, 1.0 = completely different)
        """
        try:
            if not ref_text.strip() or not syn_text.strip():
                self.logger.warning("Empty transcription detected")
                return 1.0  # Maximum error for empty transcriptions

            wer_score = jiwer.wer(
                ref_text,
                syn_text,
                reference_transform=self.transform,
                hypothesis_transform=self.transform
            )

            return float(wer_score)

        except Exception as e:
            self.logger.error(f"Failed to calculate WER: {e}")
            raise MetricCalculationError(f"WER calculation failed: {e}")

    def _calculate_pair_impl(self, ref_path: Path, syn_path: Path) -> float:
        """Calculate WER for a reference-synthesis pair."""
        try:
            # Transcribe both audio files
            ref_text = self.transcribe_audio(ref_path)
            syn_text = self.transcribe_audio(syn_path)

            # Calculate WER
            wer_score = self.calculate_wer(ref_text, syn_text)

            self.logger.debug(f"REF: {ref_text[:100]}...")
            self.logger.debug(f"SYN: {syn_text[:100]}...")
            self.logger.debug(f"WER: {wer_score:.4f}")

            return wer_score

        except Exception as e:
            raise MetricCalculationError(f"Failed to calculate WER for pair: {e}")

    def calculate_batch_optimized(self, pairs: List[Tuple[Path, Path]]) -> List[float]:
        """
        Optimized batch calculation for WER.
        Transcribes all audio files first, then calculates WER scores.
        """
        try:
            # Extract all unique audio files
            all_paths = set()
            for ref_path, syn_path in pairs:
                all_paths.add(ref_path)
                all_paths.add(syn_path)

            # Batch transcribe all audio files
            transcriptions = {}

            self.logger.info(f"Transcribing {len(all_paths)} unique audio files")

            for audio_path in all_paths:
                try:
                    transcriptions[audio_path] = self.transcribe_audio(audio_path)
                except Exception as e:
                    self.logger.warning(f"Failed to transcribe {audio_path}: {e}")
                    transcriptions[audio_path] = ""  # Empty string for failed transcriptions

            # Calculate WER for all pairs
            results = []
            for ref_path, syn_path in pairs:
                try:
                    ref_text = transcriptions.get(ref_path, "")
                    syn_text = transcriptions.get(syn_path, "")

                    if ref_text and syn_text:
                        wer_score = self.calculate_wer(ref_text, syn_text)
                        results.append(wer_score)
                    else:
                        results.append(np.nan)  # NaN for failed transcriptions

                except Exception as e:
                    self.logger.warning(f"Failed WER calculation for pair: {e}")
                    results.append(np.nan)

            return results

        except Exception as e:
            self.logger.warning(f"Batch processing failed, falling back to individual: {e}")
            return super().calculate_batch_optimized(pairs)

    def get_transcription_cache(self) -> Dict[Path, str]:
        """Get cached transcriptions (for debugging/analysis purposes)."""
        return getattr(self, '_transcription_cache', {})

    def get_name(self) -> str:
        return "WER"


if __name__ == '__main__':
    from pathlib import Path
    import torch

    ref_path = Path("data/test/ref.wav")
    syn_path = Path("data/test/syn.wav")

    config = ModelConfig(
        name="wer",
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        additional_params={'model_name': 'base', 'language': 'en'}
    )

    try:
        with WERCalculator(config) as calculator:
            print(f"Testing {calculator.get_name()} calculator...")
            score = calculator.calculate_pair(ref_path, syn_path)
            print(f"WER Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")