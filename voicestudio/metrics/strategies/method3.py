"""
Method 3: 10 references × 3 synthesis each for speaker consistency.
"""

import re
import traceback
from tqdm.auto import tqdm

from .base import BaseGenerationStrategy


class Method3Strategy(BaseGenerationStrategy):
    """Generate 10 reference audios with 3 synthesis each."""

    def _generate_text_variations(self, original_text: str) -> list[tuple[str, str]]:
        texts = []
        # 1. Original Text (T1)
        texts.append((original_text, "original"))
        # 2. No Period Text (T2)
        text_no_period = re.sub(r"\.$", "", original_text.strip())
        texts.append((text_no_period, "no_period"))
        # 3. Word Changed Text (T3)
        words = re.findall(r"\b[A-Za-z]+\b", original_text)
        text_changed = original_text

        if words:
            word_to_replace = words[0]
            replacement_word = "hello" if word_to_replace.lower() != "hello" else "test"
            text_changed = original_text.replace(word_to_replace, replacement_word, 1)
        else:
            text_changed = original_text
            self.logger.warning(f"Could not find a word to replace in: {original_text}")

        texts.append((text_changed, "word_changed"))

        return texts

    def generate_all(self, dataset_name: str, model_name: str) -> bool:
        print(f"Starting Method 3 generation for {dataset_name} -> {model_name}")

        ref_dir, syn_dir = self.create_output_paths(dataset_name, model_name, "method3")

        num_refs = self.config.generation.method2_ref_samples
        sample_indices = self.select_unique_speakers(num_refs)

        total_success = 0
        num_syn_per_ref = 3

        # Process each reference
        for ref_idx, sample_idx in enumerate(
            tqdm(sample_indices, desc="Processing Method 3 references")
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

                texts_to_synthesize = self._generate_text_variations(transcript)

                # Generate multiple synthesis for this reference
                for syn_idx, (text_to_synthesize, text_type_suffix) in enumerate(
                    texts_to_synthesize
                ):
                    syn_filename = (
                        f"syn_{ref_idx:03d}_{syn_idx:02d}_{text_type_suffix}.wav"
                    )
                    syn_output_path = set_dir / syn_filename

                    # Store metadata for WER evaluation
                    set_metadata[syn_filename] = {
                        "target_text": text_to_synthesize,
                        "text_type": text_type_suffix,
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
                        print(f"Failed synthesis: set {ref_idx}, syn {syn_idx} ({text_type_suffix})")

                # Save metadata for this set
                self.save_metadata(set_dir, set_metadata)

                total_success += set_success
                print(f"Set {ref_idx}: {set_success}/{num_syn_per_ref} synthesis generated")

            except Exception as e:
                print(f"Error processing reference {ref_idx}")
                traceback.print_exc()
                continue

        expected_total = len(sample_indices) * num_syn_per_ref
        print(f"Method 3 completed: {total_success}/{expected_total} synthesis generated")
        return total_success > 0
