"""
VCTK dataset loader for synthesis.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import torchaudio
import ssl

from .base import BaseSynthesisDataset


class VctkSynthesisDataset(BaseSynthesisDataset):
    """VCTK dataset loader using torchaudio."""

    def __init__(self, config, root_dir: str = "./data", download: bool = True, **kwargs):
        super().__init__(config, download)
        self.root_dir = Path(root_dir)
        self.dataset = None
        self.speakers = []
        self._dataset_path = self.root_dir / "VCTK-Corpus-0.92"
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load VCTK dataset using torchaudio."""
        try:
            ssl._create_default_https_context = ssl._create_unverified_context

            self.dataset = torchaudio.datasets.VCTK_092(
                root=str(self.root_dir),
                download=self.download
            )

            all_speaker_ids = [self.dataset[i][3] for i in range(len(self.dataset))]
            self.speakers = sorted(list(set(all_speaker_ids)))

            print(f"Loaded VCTK dataset with {len(self.dataset)} samples and {len(self.speakers)} speakers")

        except Exception as e:
            raise RuntimeError(f"Failed to load VCTK dataset: {e}")

    def get_sample(self, index: int) -> Tuple[str, Path, Optional[str], str]:
        """Get a sample from VCTK dataset.

        Returns:
            Tuple of (transcript, audio_path, style_prompt, speaker_id)
        """
        try:
            _, _, transcript, speaker_id, utterance_id = self.dataset[index]

            filename = f"{speaker_id}_{utterance_id}_mic1.flac"

            audio_path = self._dataset_path / "wav48_silence_trimmed" / speaker_id / filename

            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found at expected path: {audio_path}")

            return transcript, audio_path, None, speaker_id

        except Exception as e:
            raise RuntimeError(f"Failed to get sample {index}: {e}")

    def get_total_samples(self) -> int:
        """Get total number of samples in VCTK dataset."""
        return len(self.dataset) if self.dataset else 0

    def get_speakers(self) -> List[str]:
        """Get list of unique speakers in VCTK."""
        return self.speakers