"""
VCTK dataset loader for synthesis.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import torchaudio

from .base import BaseSynthesisDataset


class VCTKSynthesisDataset(BaseSynthesisDataset):
    """VCTK dataset loader using torchaudio."""

    def __init__(self, config, root_dir: str = "./data", download: bool = True):
        super().__init__(config, download)
        self.root_dir = Path(root_dir)
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load VCTK dataset using torchaudio."""
        try:
            from torchaudio.datasets import VCTK_092
            self.dataset = VCTK_092(
                root=str(self.root_dir),
                download=self.download
            )
            print(f"Loaded VCTK dataset with {len(self.dataset)} samples")
        except Exception as e:
            raise RuntimeError(f"Failed to load VCTK dataset: {e}")

    def get_sample(self, index: int) -> Tuple[str, Path, Optional[str], str]:
        """Get a sample from VCTK dataset.

        Returns:
            Tuple of (transcript, audio_path, style_prompt, speaker_id)
        """
        try:
            # VCTK_092 returns (waveform, sample_rate, transcript, speaker_id, utterance_id)
            _, _, transcript, speaker_id, utterance_id = self.dataset[index]

            # Construct audio path (this might need adjustment based on actual VCTK structure)
            audio_path = self.root_dir / "VCTK-Corpus-0.92" / "wav22_trimmed" / speaker_id / f"{utterance_id}.wav"

            return transcript, audio_path, None, speaker_id

        except Exception as e:
            raise RuntimeError(f"Failed to get sample {index}: {e}")

    def get_total_samples(self) -> int:
        """Get total number of samples in VCTK dataset."""
        return len(self.dataset) if self.dataset else 0

    def get_speakers(self) -> List[str]:
        """Get list of unique speakers in VCTK."""
        # This is a simplified approach - in practice you might want to cache this
        speakers = set()
        for i in range(min(100, self.get_total_samples())):  # Sample first 100 to get speakers
            try:
                _, _, _, speaker_id, _ = self.dataset[i]
                speakers.add(speaker_id)
            except:
                continue
        return list(speakers)

    def select_samples_by_speaker(self, num_samples: int, speaker_id: Optional[str] = None, seed: int = 42) -> List[int]:
        """Select samples from specific speaker or random speakers."""
        if speaker_id:
            # Find all samples from specific speaker
            speaker_indices = []
            for i in range(self.get_total_samples()):
                try:
                    _, _, _, spk_id, _ = self.dataset[i]
                    if spk_id == speaker_id:
                        speaker_indices.append(i)
                except:
                    continue

            if len(speaker_indices) < num_samples:
                print(f"Warning: Speaker {speaker_id} has only {len(speaker_indices)} samples, requested {num_samples}")
                return speaker_indices

            import random
            random.seed(seed)
            return random.sample(speaker_indices, num_samples)
        else:
            # Random selection from all speakers
            return self.select_samples(num_samples, seed)