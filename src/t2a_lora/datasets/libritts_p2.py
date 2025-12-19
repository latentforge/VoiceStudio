"""
Custom LibriTTS dataset loader for synthesis.
"""

from pathlib import Path
from typing import Any

import torch
import torchaudio
from torchaudio.datasets import LIBRITTS

from datasets import load_dataset


class LIBRITTSP(LIBRITTS):
    """LibriTTS-P dataset loader that reads from Hugging Face."""

    def __init__(self, config, **kwargs):
        self.split = kwargs.get("split", "test.other")
        self.root_dir = Path(kwargs.get("root_dir", "./data"))

        try:
            print(f"INFO: Loading '{self.split}' split of LibriTTS dataset...")
            self.dataset = load_dataset(
                "tictap11/libritts_p_dataset_20250821_095157",
                split=self.split,
                cache_dir=str(self.root_dir / "cache")
            )
            print(f"Loaded LibriTTS '{self.split}' split with {len(self.dataset)} samples")
        except Exception as e:
            raise RuntimeError(f"Failed to load LibriTTS-P dataset: {e}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.dataset is None or not 0 <= index < len(self.dataset):
            raise IndexError("Index out of range or dataset not loaded")

        try:
            sample = self.dataset[index]
            if not isinstance(sample['style_prompt'], list):
                sample['style_prompt'] = sample['style_prompt'].split(";")
            if not isinstance(sample['audio']['array'], torch.Tensor):
                sample['sampling_rate'] = sample["audio"]["sampling_rate"]
                sample['audio'] = torch.from_numpy(sample['audio']['array']).unsqueeze(0)

            return sample

        except KeyError as e:
            raise RuntimeError(f"Failed to get sample {index}: Key {e} not found. Features: {self.dataset.features}")
        except Exception as e:
            raise RuntimeError(f"Failed to get sample {index}: {e}")

    def __len__(self) -> int:
        return len(self.dataset) if self.dataset is not None else 0
