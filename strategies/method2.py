"""
Method 2: 10 references × 10 synthesis each for speaker consistency.
"""

from pathlib import Path
from tqdm import tqdm

from .base import BaseGenerationStrategy


class Method2Strategy(BaseGenerationStrategy):
    """Generate 10 reference audios with 10 synthesis each."""

    def generate_all(self, dataset_name: str, model_name: str) -> bool:
        """Generate 10 refs × 10 synthesis for speaker consistency evaluation.

        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model

        Returns:
            True if successful, False otherwise
        """
        print(f"Starting Method 2 generation for {dataset_name} -> {model_name}")

        # Create output directories
        ref_dir, syn_dir = self.create_output_paths(dataset_name, model_name, "method2")

        # Select reference samples
        num_refs = self.config.generation.method2_ref_samples
        syn_per_ref = self.config.generation.method2_syn_per_ref

        sample_indices = self.dataset.select_samples(num_refs)
        sample_indices = self.dataset.filter_by_duration(sample_indices)

        if len(sample_indices) < num_refs:
            print(f"Warning: Only {len(sample_indices)} samples available, requested {num_refs}")

        total_success = 0

        # Process each reference
        for ref_idx, sample_idx in enumerate(tqdm(sample_indices, desc="Processing references")):
            try:
                # Get reference sample
                transcript, audio_path, style_prompt, speaker_id = self.dataset.get_sample(sample_idx)

                # Copy reference audio
                ref_filename = f"ref_{ref_idx:03d}.wav"
                ref_output_path = ref_dir / ref_filename

                if not self.copy_reference_audio(audio_path, ref_output_path):
                    print(f"Failed to copy reference audio {ref_idx}")
                    continue

                # Create set directory for this reference
                set_dir = syn_dir / f"set_{ref_idx:03d}"
                set_dir.mkdir(exist_ok=True)

                set_success = 0

                # Generate multiple synthesis for this reference
                for syn_idx in tqdm(range(syn_per_ref), desc=f"Set {ref_idx}", leave=False):
                    syn_filename = f"syn_{ref_idx:03d}_{syn_idx:02d}.wav"
                    syn_output_path = set_dir / syn_filename

                    # Synthesize with same reference but potentially different randomness
                    if self.synthesizer.synthesize(
                        text=transcript,
                        output_path=syn_output_path,
                        reference_audio=audio_path,
                        style_prompt=style_prompt,
                        speaker_id=speaker_id
                    ):
                        set_success += 1
                    else:
                        print(f"Failed synthesis: set {ref_idx}, syn {syn_idx}")

                total_success += set_success
                print(f"Set {ref_idx}: {set_success}/{syn_per_ref} synthesis generated")

            except Exception as e:
                print(f"Error processing reference {ref_idx}: {e}")
                continue

        expected_total = len(sample_indices) * syn_per_ref
        print(f"Method 2 completed: {total_success}/{expected_total} synthesis generated")
        return total_success > 0