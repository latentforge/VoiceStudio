"""
Base generation strategy for synthesis.
"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path



class BaseGenerationStrategy(ABC):
    """Base class for generation strategies."""

    def __init__(self, config, dataset, synthesizer):
        self.config = config
        self.dataset = dataset
        self.synthesizer = synthesizer
        self.output_dir = config.generation.output_dir

    @abstractmethod
    def generate_all(self, dataset_name: str, model_name: str) -> bool:
        """Execute the generation strategy.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model

        Returns:
            True if successful, False otherwise
        """
        pass

    @staticmethod
    def copy_reference_audio(src_path: Path, dst_path: Path) -> bool:
        """Copy reference audio to target location."""
        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as e:
            print(f"Failed to copy {src_path} to {dst_path}: {e}")
            return False

    def create_output_paths(
        self, dataset_name: str, model_name: str, method_name: str
    ) -> tuple[Path, Path]:
        """Create output directory paths for reference and synthesis."""
        ref_dir = self.output_dir / "ref" / dataset_name / method_name
        syn_dir = self.output_dir / "syn" / dataset_name / model_name / method_name

        ref_dir.mkdir(parents=True, exist_ok=True)
        syn_dir.mkdir(parents=True, exist_ok=True)

        return ref_dir, syn_dir
