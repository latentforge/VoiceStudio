"""
SIM (Speaker Similarity) calculator using ECAPA-TDNN for speaker embeddings.
"""

from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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

    def _forward_impl(
        self,
        synthesis: torch.Tensor,
        reference: torch.Tensor | None = None,
        **kwargs,
    ) -> float:
        """Calculate speaker similarity between synthesis and reference tensors."""
        if reference is None:
            raise MetricCalculationError("SIM requires a reference audio tensor")

        try:
            with torch.no_grad():
                syn_embed = self.classifier.encode_batch(synthesis.unsqueeze(0)).squeeze()
                ref_embed = self.classifier.encode_batch(reference.unsqueeze(0)).squeeze()

            syn_norm = torch.nn.functional.normalize(syn_embed, dim=0)
            ref_norm = torch.nn.functional.normalize(ref_embed, dim=0)
            
            similarity = torch.dot(syn_norm, ref_norm).item()
            similarity = max(0.0, float(similarity))
            
            return similarity

        except Exception as e:
            raise MetricCalculationError(f"SIM forward pass failed: {e}")

    def calculate_batch_optimized(self, pairs: list[tuple[Path, Path]]) -> list[float]:
        """
        Optimized batch implementation for speaker similarity.
        Processes unique audio files in batches to extract embeddings.
        """
        if not pairs:
            return []

        results = [np.nan] * len(pairs)
        batch_size = self.config.batch_size
        
        try:
            all_paths = list(set([p for pair in pairs for p in pair]))
            embeddings = {}

            self.logger.info(f"Extracting embeddings for {len(all_paths)} files in batches of {batch_size}")

            for i in tqdm(range(0, len(all_paths), batch_size), desc="Extracting embeddings"):
                batch_paths = all_paths[i : i + batch_size]
                
                batch_audios = []
                for p in batch_paths:
                    try:
                        audio = self._prepare_audio_input(p)
                        batch_audios.append(audio)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {p}: {e}")
                        batch_audios.append(None)

                valid_batch_data = [(idx, a) for idx, a in enumerate(batch_audios) if a is not None]
                if not valid_batch_data:
                    continue

                valid_indices, valid_audios = zip(*valid_batch_data)
                
                padded_batch = pad_sequence(valid_audios, batch_first=True).to(self.get_device())

                with torch.no_grad():
                    batch_embeds = self.classifier.encode_batch(padded_batch).squeeze(1)
                    batch_embeds = torch.nn.functional.normalize(batch_embeds, dim=1)
                    
                    for idx, orig_batch_idx in enumerate(valid_indices):
                        embeddings[batch_paths[orig_batch_idx]] = batch_embeds[idx]

            self.logger.info(f"Calculating similarities for {len(pairs)} pairs")
            for idx, (ref_path, syn_path) in tqdm(enumerate(pairs), total=len(pairs), desc="Calculating similarities"):
                if ref_path in embeddings and syn_path in embeddings:
                    ref_embed = embeddings[ref_path]
                    syn_embed = embeddings[syn_path]
                    
                    if ref_embed is not None and syn_embed is not None:
                        similarity = torch.dot(ref_embed, syn_embed).item()
                        results[idx] = max(0.0, float(similarity))
                
            return results
        except Exception as e:
            self.logger.error(f"SIM batch optimization failed: {e}. Falling back to individual processing.")
            return super().calculate_batch_optimized(pairs)

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
            score = calculator(reference=ref_path, synthesis=syn_path)
            print(f"SIM Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
