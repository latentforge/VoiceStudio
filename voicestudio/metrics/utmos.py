"""
UTMOS calculator using UTMOSv2.
"""

import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from utmosv2 import create_model

from .base import BaseMetricCalculator, MetricCalculationError, ModelConfig


class UTMOSCalculator(BaseMetricCalculator):
    """UTMOS quality score calculator using UTMOSv2."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.utmos_model = None

    def _load_model_impl(self) -> None:
        """Load UTMOSv2 model."""
        try:
            seed = self.config.additional_params.get("seed", 42)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            model_name = self.config.additional_params.get(
                "model_name", "fusion_stage3"
            )
            fold = self.config.additional_params.get("fold", 0)

            self.utmos_model = create_model(
                pretrained=True,
                config=model_name,
                fold=fold,
                seed=seed,
                device=self.get_device(),
            )
            
            self.utmos_model.eval()

            self.logger.info(f"Loaded UTMOSv2 model: {model_name}")

        except ImportError as e:
            raise MetricCalculationError(f"UTMOSv2 not installed: {e}")
        except Exception as e:
            raise MetricCalculationError(f"Failed to load UTMOSv2 model: {e}")

    def _forward_impl(
        self,
        synthesis: torch.Tensor,
        reference: torch.Tensor | None = None,
        **kwargs,
    ) -> float:
        """Calculate UTMOS score for synthesis audio."""
        try:         
            orig_synthesis = kwargs.get("orig_synthesis")
            if orig_synthesis and isinstance(orig_synthesis, (Path, str)):
                score = self.utmos_model.predict(input_path=orig_synthesis)
            else:
                raise MetricCalculationError("UTMOSv2 requires a file path for prediction. Use forward(synthesis=Path).")

            return float(score)

        except Exception as e:
            raise MetricCalculationError(f"UTMOS forward pass failed: {e}")

    def calculate_batch_optimized(self, pairs: list[tuple[Path, Path]]) -> list[float]:
        """Optimized batch calculation for UTMOS."""
        try:
            # Extract synthesis paths only (UTMOS doesn't use reference)
            results = []
            self.logger.info(f"Calculating UTMOS scores for {len(pairs)} pairs")
            for _, syn_path in tqdm(pairs, desc="Calculating UTMOS scores"):
                # Path input is now automatically tracked as orig_synthesis in base.py
                results.append(self(synthesis=syn_path))
            
            return results

        except Exception as e:
            self.logger.warning(
                f"Batch processing failed, falling back to individual: {e}"
            )
            return super().calculate_batch_optimized(pairs)

    def get_name(self) -> str:
        return "UTMOS"


if __name__ == "__main__":
    import torch

    ref_path = Path("data/test/ref.wav")
    syn_path = Path("data/test/syn.wav")

    config = ModelConfig(
        name="utmos",
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    try:
        with UTMOSCalculator(config) as calculator:
            print(f"Testing {calculator.get_name()} calculator...")
            score = calculator(synthesis=syn_path, orig_synthesis=syn_path)
            print(f"UTMOS Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
