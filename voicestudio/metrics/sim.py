"""
SIM (Speaker Similarity) calculator using ECAPA-TDNN for speaker embeddings.
"""

from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

from ..utils.loader import AudioLoader
from .base import BaseMetricCalculator, MetricCalculationError, ModelConfig


class SIMCalculator(BaseMetricCalculator):
    """Speaker similarity calculator using ECAPA-TDNN."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.encoder = None
        self.classifier = None
        self.target_sr = 16000

    def _load_model_impl(self) -> None:
        """Load ECAPA-TDNN model from speechbrain."""
        try:
            model_name = self.config.additional_params.get(
                "model_name", "speechbrain/spkrec-ecapa-voxceleb"
            )

            self.classifier = EncoderClassifier.from_hparams(
                source=model_name, run_opts={"device": str(self.get_device())}
            )

            self.logger.info(f"Loaded ECAPA-TDNN model: {model_name}")

        except ImportError as e:
            raise MetricCalculationError(f"SpeechBrain not installed: {e}")
        except Exception as e:
            raise MetricCalculationError(f"Failed to load ECAPA-TDNN model: {e}")

    def _load_and_preprocess_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Load and preprocess audio for speaker embedding extraction.

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed audio tensor
        """
        try:
            loader = AudioLoader(sr=self.target_sr, mono=True, cache=False)
            waveform = loader.load(audio_path)

            # Ensure waveform is 2D [channels, samples]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Ensure minimum length (e.g., 1 second)
            min_length = self.target_sr * 1  # 1 second
            if waveform.shape[1] < min_length:
                # Repeat waveform to meet minimum length
                repeat_times = (min_length // waveform.shape[1]) + 1
                waveform = waveform.repeat(1, repeat_times)[:, :min_length]

            return waveform.squeeze(0)  # Remove channel dimension

        except Exception as e:
            raise MetricCalculationError(f"Failed to load audio {audio_path}: {e}")

    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        """
        Extract speaker embedding from audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Speaker embedding vector
        """
        try:
            # Load and preprocess audio
            waveform = self._load_and_preprocess_audio(audio_path)

            # Extract embedding using SpeechBrain
            with torch.no_grad():
                embedding = self.classifier.encode_batch(waveform.unsqueeze(0))
                embedding = embedding.squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            self.logger.error(f"Failed to extract embedding from {audio_path}: {e}")
            raise MetricCalculationError(f"Embedding extraction failed: {e}")

    def calculate_cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two speaker embeddings.

        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding

        Returns:
            Cosine similarity score [-1, 1]
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                self.logger.warning("Zero norm embedding detected")
                return 0.0

            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2

            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)

            return float(similarity)

        except Exception as e:
            raise MetricCalculationError(f"Cosine similarity calculation failed: {e}")

    def _calculate_pair_impl(self, ref_path: Path, syn_path: Path) -> float:
        """Calculate speaker similarity for a reference-synthesis pair."""
        try:
            # Extract embeddings
            ref_embedding = self.extract_embedding(ref_path)
            syn_embedding = self.extract_embedding(syn_path)

            # Calculate similarity
            similarity = self.calculate_cosine_similarity(ref_embedding, syn_embedding)

            self.logger.debug(f"Speaker similarity: {similarity:.4f}")

            return similarity

        except Exception as e:
            raise MetricCalculationError(f"Failed to calculate speaker similarity: {e}")

    def calculate_batch_optimized(self, pairs: list[tuple[Path, Path]]) -> list[float]:
        """
        Optimized batch calculation for speaker similarity.
        Extracts all embeddings first, then calculates similarities.
        """
        try:
            # Extract all unique audio files
            all_paths = set()
            for ref_path, syn_path in pairs:
                all_paths.add(ref_path)
                all_paths.add(syn_path)

            # Batch extract embeddings
            embeddings = {}

            self.logger.info(
                f"Extracting embeddings for {len(all_paths)} unique audio files"
            )

            # Process in batches to manage memory
            batch_size = self.config.batch_size
            all_paths_list = list(all_paths)

            for i in tqdm(
                range(0, len(all_paths_list), batch_size), desc="Extracting embeddings"
            ):
                batch_paths = all_paths_list[i : i + batch_size]

                for audio_path in batch_paths:
                    try:
                        embeddings[audio_path] = self.extract_embedding(audio_path)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to extract embedding for {audio_path}: {e}"
                        )
                        embeddings[audio_path] = None

            # Calculate similarities for all pairs
            results = []
            for ref_path, syn_path in tqdm(pairs, desc="Calculating similarities"):
                try:
                    ref_embedding = embeddings.get(ref_path)
                    syn_embedding = embeddings.get(syn_path)

                    if ref_embedding is not None and syn_embedding is not None:
                        similarity = self.calculate_cosine_similarity(
                            ref_embedding, syn_embedding
                        )
                        results.append(similarity)
                    else:
                        results.append(np.nan)

                except Exception as e:
                    self.logger.warning(f"Failed similarity calculation for pair: {e}")
                    results.append(np.nan)

            return results

        except Exception as e:
            self.logger.warning(
                f"Batch processing failed, falling back to individual: {e}"
            )
            return super().calculate_batch_optimized(pairs)

    def get_embedding_cache(self) -> dict[Path, np.ndarray]:
        """Get cached embeddings (for debugging/analysis purposes)."""
        return getattr(self, "_embedding_cache", {})

    def get_name(self) -> str:
        return "SIM"


if __name__ == "__main__":
    ref_path = Path("data/test/ref.wav")
    syn_path = Path("data/test/syn.wav")

    config = ModelConfig(
        name="sim",
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        additional_params={"model_name": "speechbrain/spkrec-ecapa-voxceleb"},
    )

    try:
        with SIMCalculator(config) as calculator:
            print(f"Testing {calculator.get_name()} calculator...")
            score = calculator.calculate_pair(ref_path, syn_path)
            print(f"SIM Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
