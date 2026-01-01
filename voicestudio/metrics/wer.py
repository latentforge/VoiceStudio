"""
WER (Word Error Rate) calculator using Whisper for transcription.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import jiwer
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .base import BaseMetricCalculator, MetricCalculationError, ModelConfig


class WERCalculator(BaseMetricCalculator):
    """Word Error Rate calculator using Whisper for transcription."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.whisper_model = None
        self.processor = None
        self.transform = self._setup_transform()

    @staticmethod
    def _setup_transform() -> jiwer.Compose:
        """Setup text transformation for WER calculation."""
        return jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.RemoveMultipleSpaces(),
                jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
            ]
        )

    def _load_model_impl(self) -> None:
        """Load Whisper model."""
        try:
            model_name = self.config.additional_params.get("model_name", "large-v3")

            # Convert short model names to HuggingFace model IDs
            model_id = (
                model_name if "/" in model_name else f"openai/whisper-{model_name}"
            )

            # Get dtype configuration
            dtype_str = self.config.additional_params.get("dtype", "float16")
            dtype = getattr(torch, dtype_str, torch.float16)

            # Load processor and model
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype
            ).to(self.get_device())
            self.whisper_model.eval()

            self.logger.info(f"Loaded Whisper model: {model_id} (dtype: {dtype})")

        except ImportError as e:
            raise MetricCalculationError(f"Transformers not installed: {e}")
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
            sampling_rate = self.processor.feature_extractor.sampling_rate

            audio, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)

            input_features = self.processor(
                audio, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features

            input_features = input_features.to(
                device=self.get_device(), dtype=self.whisper_model.dtype
            )

            language = self.config.additional_params.get("language", "en")

            generate_kwargs = {
                "language": language,
                "task": "transcribe",
            }

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    input_features, **generate_kwargs
                )

            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            return transcription.strip()

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
                hypothesis_transform=self.transform,
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

            for audio_path in tqdm(all_paths, desc="Transcribing audio files"):
                try:
                    transcriptions[audio_path] = self.transcribe_audio(audio_path)
                except Exception as e:
                    self.logger.warning(f"Failed to transcribe {audio_path}: {e}")
                    transcriptions[audio_path] = (
                        ""  # Empty string for failed transcriptions
                    )

            # Calculate WER for all pairs
            results = []
            for ref_path, syn_path in tqdm(pairs, desc="Calculating WER scores"):
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
            self.logger.warning(
                f"Batch processing failed, falling back to individual: {e}"
            )
            return super().calculate_batch_optimized(pairs)

    def get_transcription_cache(self) -> Dict[Path, str]:
        """Get cached transcriptions (for debugging/analysis purposes)."""
        return getattr(self, "_transcription_cache", {})

    def get_name(self) -> str:
        return "WER"

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ref_text = "Your daughter will be married to morrow, if not to day-in a week, if not to morrow; and I do not think you can regret the intended husband of your daughter."
    syn_audio_path = Path("audio.wav")

    config = ModelConfig(
        name="wer",
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        additional_params={"model_name": "base", "language": "en"},
    )

    try:
        with WERCalculator(config) as calculator:
            print(f"Testing {calculator.get_name()} calculator...")
            print(f"Reference text: {ref_text}")

            # Transcribe audio
            syn_text = calculator.transcribe_audio(syn_audio_path)
            print(f"Transcribed text: {syn_text}")

            # Calculate WER
            score = calculator.calculate_wer(ref_text, syn_text)
            print(f"WER Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
