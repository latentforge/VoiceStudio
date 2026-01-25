"""
Method 2: 10 references × 10 synthesis each for speaker consistency.
"""
import traceback
from tqdm.auto import tqdm

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

        sample_indices = self.select_unique_speakers(num_refs)

        comparison_texts = [
            "At least, no friend came forwards immediately, and mrs Thornton is not one, I fancy, to wait till tardy kindness comes to find her out.",
            "And the poor men around him-they were poor because they were vicious-out of the pale of his sympathies because they had not his iron nature, and the capabilities that it gives him for being rich.'",
            "The modes of treatment may be ranged under three heads: (one) To eliminate the poison; (two) to antagonize its action; (three) to avert the tendency to death.",
            "Visiting register offices, seeing all manner of unlikely people, and very few in the least likely, absorbed Margaret's time and thoughts for several days.",
            "But though she received caresses and fond words back again, in such profusion as would have gladdened her formerly, yet she felt that there was a secret withheld from her, and she believed it bore serious reference to her mother's health.",
            "And the poor men around him-they were poor because they were vicious-out of the pale of his sympathies because they had not his iron nature, and the capabilities that it gives him for being rich.",
            "mr Bell said they absolutely lived upon water porridge for years-how, he did not know; but long after the creditors had given up hope of any payment of old mr Thornton's debts (if, indeed, they ever had hoped at all about it, after his suicide,) this young man returned to Milton, and went quietly round to each creditor, paying him the first instalment of the money owing to him.",
            "In using the elastic stomach tube, some fluid should be introduced into the stomach before attempting to empty it, or a portion of the mucous membrane may be sucked into the aperture.",
            "'Margaret!' said mr Hale, as he returned from showing his guest downstairs; 'I could not help watching your face with some anxiety, when mr Thornton made his confession of having been a shop boy.",
        ]

        total_success = 0

        # Process each reference
        for ref_idx, sample_idx in enumerate(
            tqdm(sample_indices, desc="Processing references")
        ):
            try:
                # Get reference sample
                transcript, audio_path, style_prompt, speaker_id = (
                    self.dataset.get_sample(sample_idx)
                )

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
                set_metadata = {}

                # Generate multiple synthesis for this reference
                for syn_idx in tqdm(
                    range(syn_per_ref), desc=f"Set {ref_idx}", leave=False
                ):
                    syn_filename = f"syn_{ref_idx:03d}_{syn_idx:02d}.wav"
                    syn_output_path = set_dir / syn_filename

                    if syn_idx == 0:
                        text_to_synthesize = transcript
                    else:
                        text_to_synthesize = comparison_texts[syn_idx - 1]

                    # Store metadata for WER evaluation
                    set_metadata[syn_filename] = {
                        "target_text": text_to_synthesize,
                        "speaker_id": speaker_id,
                        "reference_audio": str(audio_path)
                    }

                    if self.synthesizer.synthesize(
                        text=text_to_synthesize,
                        output_path=syn_output_path,
                        reference_audio=audio_path,
                        style_prompt=style_prompt,
                        speaker_id=speaker_id,
                    ):
                        set_success += 1
                    else:
                        print(f"Failed synthesis: set {ref_idx}, syn {syn_idx}")

                # Save metadata for this set
                self.save_metadata(set_dir, set_metadata)

                total_success += set_success
                print(f"Set {ref_idx}: {set_success}/{syn_per_ref} synthesis generated")

            except Exception as e:
                print(f"Error processing reference {ref_idx}")
                traceback.print_exc()
                continue

        expected_total = len(sample_indices) * syn_per_ref
        print(f"Method 2 completed: {total_success}/{expected_total} synthesis generated")
        return total_success > 0
