"""
WER (Word Error Rate) calculator using Whisper for transcription.
"""

from pathlib import Path

import jiwer
import numpy as np
import torch
from tqdm.auto import tqdm
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
            waveform = self._prepare_audio_input(audio_path, sample_rate=sampling_rate)
            audio = waveform.numpy()

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

    def _forward_impl(
        self,
        synthesis: torch.Tensor | str,
        reference: torch.Tensor | str | None = None,
        **kwargs,
    ) -> float:
        """
        Calculate WER. 
        - If synthesis is Tensor, transcribes it first.
        - If target_text is provided in kwargs, uses it as reference.
        - Otherwise, if reference is Tensor, transcribes it; if str, uses it directly.
        """
        try:
            # 1. Prepare synthesis text
            if isinstance(synthesis, torch.Tensor):
                syn_text = self._transcribe_tensor(synthesis)
            else:
                syn_text = synthesis

            # 2. Prepare reference text
            # Check if target_text is provided (for Method 2/3 evaluation)
            target_text = kwargs.get('target_text')
            if target_text is not None:
                ref_text = target_text
            elif reference is None:
                raise MetricCalculationError("WER requires a reference (text or audio) or target_text")
            elif isinstance(reference, torch.Tensor):
                ref_text = self._transcribe_tensor(reference)
            else:
                ref_text = reference

            # 3. Calculate WER
            return self.calculate_wer(ref_text, syn_text)

        except Exception as e:
            raise MetricCalculationError(f"WER forward pass failed: {e}")

    def _transcribe_tensor(self, waveform: torch.Tensor) -> str:
        """Internal helper to transcribe a tensor."""
        try:
            sampling_rate = self.processor.feature_extractor.sampling_rate
            
            # Whisper expects 1D [samples]
            if waveform.dim() > 1:
                waveform = waveform.mean(0)  # Mono
            
            audio = waveform.cpu().numpy()
            
            input_features = self.processor(
                audio, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features.to(device=self.get_device(), dtype=self.whisper_model.dtype)

            language = self.config.additional_params.get("language", "en")
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    input_features, language=language, task="transcribe"
                )
            
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            return transcription.strip()
        except Exception as e:
            raise MetricCalculationError(f"Transcription from tensor failed: {e}")

    def calculate_batch_optimized(self, pairs: list[tuple[Path, Path]]) -> list[float]:
        """
        Optimized batch calculation for WER.
        Transcribes all audio files first, then calculates WER scores.
        """
        if not pairs:
            return []

        try:
            # 1. Collect all unique paths
            all_paths = list(set([p for pair in pairs for p in pair]))
            transcriptions = {}
            batch_size = self.config.batch_size
            sampling_rate = self.processor.feature_extractor.sampling_rate
            language = self.config.additional_params.get("language", "en")

            self.logger.info(f"Transcribing {len(all_paths)} unique files in batches of {batch_size}")

            # 2. Transcribe in batches
            for i in tqdm(range(0, len(all_paths), batch_size), desc="Transcribing"):
                batch_paths = all_paths[i : i + batch_size]
                
                batch_audios = []
                valid_mask = []
                
                for p in batch_paths:
                    try:
                        waveform = self._prepare_audio_input(p, sample_rate=sampling_rate)
                        batch_audios.append(waveform.numpy())
                        valid_mask.append(True)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {p}: {e}")
                        batch_audios.append(np.zeros(sampling_rate))  # Dummy audio
                        valid_mask.append(False)

                input_features = self.processor(
                    batch_audios, sampling_rate=sampling_rate, return_tensors="pt"
                ).input_features.to(device=self.get_device(), dtype=self.whisper_model.dtype)

                with torch.no_grad():
                    predicted_ids = self.whisper_model.generate(
                        input_features, language=language, task="transcribe"
                    )

                batch_txt = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

                # Store valid transcriptions
                for idx, (p, is_valid) in enumerate(zip(batch_paths, valid_mask)):
                    transcriptions[p] = batch_txt[idx].strip() if is_valid else ""

            # 3. Calculate WER for each pair
            self.logger.info(f"Calculating WER for {len(pairs)} pairs")
            results = []
            for ref_path, syn_path in tqdm(pairs, desc="Calculating WER scores"):
                try:
                    ref_text = transcriptions.get(ref_path, "")
                    syn_text = transcriptions.get(syn_path, "")
                    
                    if ref_text and syn_text:
                        results.append(self.calculate_wer(ref_text, syn_text))
                    else:
                        results.append(np.nan)
                except Exception as e:
                    self.logger.warning(f"Failed calculation: {e}")
                    results.append(np.nan)

            return results

        except Exception as e:
            self.logger.error(f"WER batch optimization failed: {e}. Falling back to individual.")
            return super().calculate_batch_optimized(pairs)

    def get_transcription_cache(self) -> dict[Path, str]:
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

            # Calculate WER using the new callable interface (handles Path and str automatically)
            score = calculator(reference=ref_text, synthesis=syn_audio_path)
            print(f"WER Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
