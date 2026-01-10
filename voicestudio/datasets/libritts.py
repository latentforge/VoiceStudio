"""
Custom LibriTTS dataset loader for synthesis.
"""

from pathlib import Path

import torch
import torchaudio

from datasets import load_dataset

from .base import BaseSynthesisDataset


class LibriTTSSynthesisDataset(BaseSynthesisDataset):
    """LibriTTS dataset loader that reads from Hugging Face."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.split = kwargs.get("split", "test.other")
        self.root_dir = Path(kwargs.get("root_dir", "./data"))
        self.dataset = None
        self.temp_audio_dir = self.root_dir / "temp_libritts_audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load LibriTTS dataset from Hugging Face."""
        try:
            print(f"INFO: Loading '{self.split}' split of LibriTTS dataset...")
            self.dataset = load_dataset(
                "tictap11/libritts_p_dataset_20250821_095157",
                split=self.split,
                cache_dir=str(self.root_dir / "cache")
            )
            print(f"Loaded LibriTTS '{self.split}' split with {len(self.dataset)} samples")
        except Exception as e:
            raise RuntimeError(f"Failed to load LibriTTS dataset: {e}")

    def get_sample(self, index: int) -> tuple[str, Path, str | None, str | None]:
        if self.dataset is None or not 0 <= index < len(self.dataset):
            raise IndexError("Index out of range or dataset not loaded")

        try:
            sample = self.dataset[index]
            style_prompt = sample["style_prompt"].split(";")[0] if sample["style_prompt"] else None
            transcript = sample["content_prompt"]
            speaker_id = str(sample["spk_id"])
            audio_data = torch.from_numpy(sample["audio"]["array"]).unsqueeze(0)
            sampling_rate = sample["audio"]["sampling_rate"]
            audio_path = self.temp_audio_dir / f"temp_audio_{index}.wav"
            torchaudio.save(str(audio_path), audio_data, sampling_rate)

            return transcript, audio_path, style_prompt, speaker_id

        except KeyError as e:
            raise RuntimeError(f"Failed to get sample {index}: Key {e} not found. Features: {self.dataset.features}")
        except Exception as e:
            raise RuntimeError(f"Failed to get sample {index}: {e}")

    def get_total_samples(self) -> int:
        return len(self.dataset) if self.dataset is not None else 0
