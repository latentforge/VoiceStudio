"""
Method 1: 100 reference-synthesis pairs generation.
"""
import traceback
from tqdm.auto import tqdm

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
        for i, sample_idx in enumerate(
            tqdm(sample_indices, desc="Generating Method1 pairs")
        ):
            try:
                # Get sample data
                transcript, audio_path, style_prompt, speaker_id = (
                    self.dataset.get_sample(sample_idx)
                )

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
                    speaker_id=speaker_id,
                ):
                    success_count += 1
                else:
                    print(f"Failed to synthesize audio for sample {i}")

            except Exception as e:
                print(f"Error processing sample {i}")
                traceback.print_exc()
                continue

        print(f"Method 1 completed: {success_count}/{len(sample_indices)} pairs generated")
        return success_count > 0

    def generate_batch_all(self, dataset_name: str, model_name: str, verbose: bool = False) -> bool:
        if verbose: print(f"Starting Method 1 batch generation for {dataset_name} -> {model_name}")

        # Create output directories
        ref_dir, syn_dir = self.create_output_paths(dataset_name, model_name, "method1")

        # Select samples
        num_samples = self.config.generation.method1_samples
        sample_indices = self.dataset.select_samples(num_samples)
        sample_indices = self.dataset.filter_by_duration(sample_indices)

        if len(sample_indices) < num_samples:
            print(f"Warning: Only {len(sample_indices)} samples available, requested {num_samples}")

        # Collect batch data
        batch_text, batch_prompt = [], []
        batch_ref, batch_spk_id = [], []
        batch_output_path = []

        for i, sample_idx in enumerate(
            tqdm(sample_indices, desc="Preparing Method1 batch", leave=verbose)
        ):
            try:
                # Get sample data
                transcript, audio_path, style_prompt, speaker_id = (
                    self.dataset.get_sample(sample_idx)
                )

                # Copy reference audio
                ref_filename = f"ref_{i:03d}.wav"
                ref_output_path = ref_dir / ref_filename

                if not self.copy_reference_audio(audio_path, ref_output_path):
                    print(f"Failed to copy reference audio for sample {i}")
                    continue

                syn_filename = f"syn_{i:03d}.wav"
                syn_output_path = syn_dir / syn_filename

                batch_text.append(transcript)
                batch_prompt.append(style_prompt)
                batch_ref.append(audio_path)
                batch_spk_id.append(speaker_id)
                batch_output_path.append(syn_output_path)

            except Exception as e:
                print(f"Error preparing sample {i}")
                traceback.print_exc()
                continue

        # Batch synthesize
        if batch_text:
            success = self.synthesizer.synthesize(
                text=batch_text,
                output_path=batch_output_path,
                reference_audio=batch_ref,
                style_prompt=batch_prompt,
                speaker_id=batch_spk_id,
            )
            success_count = len(batch_text) if success else 0
        else:
            success_count = 0

        if verbose: print(f"Method 1 batch completed: {success_count}/{len(sample_indices)} pairs generated")
        return success_count > 0
