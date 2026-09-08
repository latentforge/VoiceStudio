"""
Base generation strategy for synthesis.
"""

import json
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

    def select_unique_speakers(self, num_refs: int) -> list[int]:
        """Select sample indices with unique speakers.
        
        Args:
            num_refs: Number of unique speakers to select
            
        Returns:
            List of sample indices with unique speakers
        """
        initial_samples_to_check = self.dataset.select_samples(num_refs * 5)
        initial_samples_to_check = self.dataset.filter_by_duration(initial_samples_to_check)

        sample_indices = []
        used_speakers = set()
        
        for sample_idx in initial_samples_to_check:
            if len(sample_indices) >= num_refs:
                break

            _, _, _, speaker_id = self.dataset.get_sample(sample_idx)

            if speaker_id not in used_speakers:
                used_speakers.add(speaker_id)
                sample_indices.append(sample_idx)

        if len(sample_indices) < num_refs:
            print(f"Warning: Only {len(sample_indices)} samples available, requested {num_refs}")

        return sample_indices

    @staticmethod
    def save_metadata(set_dir: Path, metadata: dict) -> None:
        """Save metadata to JSON file in set directory.
        
        Args:
            set_dir: Path to the set directory
            metadata: Dictionary containing metadata to save
        """
        metadata_path = set_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
