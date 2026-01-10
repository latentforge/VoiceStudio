"""
UTMOS calculator using UTMOSv2.
"""

import random
from pathlib import Path

import numpy as np
import torch
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

    def _calculate_pair_impl(self, ref_path: Path, syn_path: Path) -> float:
        """Calculate UTMOS score for synthesis audio (reference not used)."""
        try:
            # UTMOS evaluates synthesis audio only
            score = self.utmos_model.predict(input_path=syn_path)
            return float(score)

        except Exception as e:
            raise MetricCalculationError(f"Failed to calculate UTMOS: {e}")

    def calculate_batch_optimized(self, pairs: list[tuple[Path, Path]]) -> list[float]:
        """Optimized batch calculation for UTMOS."""
        try:
            # Extract synthesis paths only (UTMOS doesn't use reference)
            syn_paths = [syn_path for _, syn_path in pairs]

            # Use UTMOSv2's batch prediction
            results = self.utmos_model.predict(
                input_dir=None,  # Will be handled by individual paths
                batch_size=self.config.batch_size,
                num_workers=4,
            )

            # If batch prediction is not available, fall back to individual predictions
            if results is None:
                return super().calculate_batch_optimized(pairs)

            return [float(score) for score in results]

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
            score = calculator.calculate_pair(ref_path, syn_path)
            print(f"UTMOS Score: {score:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
