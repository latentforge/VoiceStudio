"""
Custom LJSpeech dataset loader for synthesis.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from .base import BaseSynthesisDataset


class LJSpeechSynthesisDataset(BaseSynthesisDataset):
    """LJSpeech dataset loader that reads from local files directly."""

    def __init__(self, config, root_dir: str = "./data", download: bool = False):
        """
        Initializes the dataset loader. The `download` parameter is ignored.

        Args:
            config: Dataset configuration object.
            root_dir (str): The root directory where the dataset is stored.
        """
        super().__init__(config, download)
        self.root_dir = Path(root_dir)
        self.dataset_path = self.root_dir / "LJSpeech-1.1"
        self.wavs_path = self.dataset_path / "wavs"
        self._speaker_id = "ljspeech_speaker"
        self.metadata = None
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load LJSpeech dataset from the metadata.csv file."""
        metadata_file = self.dataset_path / "metadata.csv"

        try:
            self.metadata = pd.read_csv(
                metadata_file,
                sep='|',
                header=None,
                names=['fileid', 'transcript', 'normalized_transcript']
            )
            print(f"Loaded LJSpeech dataset with {len(self.metadata)} samples")
        except Exception as e:
            raise RuntimeError(f"Failed to load LJSpeech metadata.csv: {e}")

    def get_sample(self, index: int) -> Tuple[str, Path, Optional[str], str]:
        """
        Get a sample from the LJSpeech dataset by index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple of (transcript, audio_path, style_prompt, speaker_id)
        """
        if not 0 <= index < len(self.metadata):
            raise IndexError("Index out of range")

        try:
            sample_info = self.metadata.iloc[index]
            fileid = sample_info['fileid']
            transcript = sample_info['normalized_transcript']

            audio_path = self.wavs_path / f"{fileid}.wav"

            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            return transcript, audio_path, None, self._speaker_id

        except Exception as e:
            raise RuntimeError(f"Failed to get sample {index}: {e}")

    def get_total_samples(self) -> int:
        """Get the total number of samples in the LJSpeech dataset."""
        return len(self.metadata) if self.metadata is not None else 0

    def get_speakers(self) -> List[str]:
        """Get the list of unique speakers. For LJSpeech, there is only one."""
        return [self._speaker_id]