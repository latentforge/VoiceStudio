"""
Method 1: 100 reference-synthesis pairs generation.
"""

from pathlib import Path
from tqdm import tqdm

from .base import BaseGenerationStrategy


class Method1Strategy(BaseGenerationStrategy):
    """Generate 100 1:1 reference-synthesis pairs."""

    def generate_all(self, dataset_name: str, model_name: str) -> bool:
        """Generate 100 reference-synthesis pairs.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model

        Returns:
            True if successful, False otherwise
        """
        print(f"Starting Method 1 generation for {dataset_name} -> {model_name}")

        # Create output directories
        ref_dir, syn_dir = self.create_output_paths(dataset_name, model_name, "method1")

        # Select samples
        num_samples = self.config.generation.method1_samples
        sample_indices = self.dataset.select_samples(num_samples)
        sample_indices = self.dataset.filter_by_duration(sample_indices)

        if len(sample_indices) < num_samples:
            print(f"Warning: Only {len(sample_indices)} samples available, requested {num_samples}")

        success_count = 0

        # Process each sample
        for i, sample_idx in enumerate(tqdm(sample_indices, desc="Generating Method1 pairs")):
            try:
                # Get sample data
                transcript, audio_path, style_prompt, speaker_id = self.dataset.get_sample(sample_idx)

                # Create file names
                ref_filename = f"ref_{i:03d}.wav"
                syn_filename = f"syn_{i:03d}.wav"

                ref_output_path = ref_dir / ref_filename
                syn_output_path = syn_dir / syn_filename

                # Copy reference audio
                if not self.copy_reference_audio(audio_path, ref_output_path):
                    print(f"Failed to copy reference audio for sample {i}")
                    continue

                # Synthesize audio
                if self.synthesizer.synthesize(
                        text=transcript,
                        output_path=syn_output_path,
                        reference_audio=audio_path,
                        style_prompt=style_prompt,
                        speaker_id=speaker_id
                ):
                    success_count += 1
                else:
                    print(f"Failed to synthesize audio for sample {i}")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        print(f"Method 1 completed: {success_count}/{len(sample_indices)} pairs generated")
        return success_count > 0
