"""LibriTTS-P dataset implementation.

LibriTTS-P extends LibriTTS with style and speaker prompts for controllable TTS.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from torch import Tensor
from torchaudio._internal import download_url_to_file
from torchaudio.datasets import LIBRITTS
from torchaudio.datasets.libritts import load_libritts_item


_METADATA_FILENAME = "LibriTTS/metadata_w_style_prompt_tags_v230922.csv"
_STYLE_FILENAME = "LibriTTS/style_prompt_candidates_v230922.csv"

_BASE_URL = "https://raw.githubusercontent.com/line/LibriTTS-P/main/data/"
_URLS = {
    "metadata": _BASE_URL + "metadata_w_style_prompt_tags_v230922.csv",
    "style": _BASE_URL + "style_prompt_candidates_v230922.csv",
    "df1": _BASE_URL + "df1_en.csv",
    "df2": _BASE_URL + "df2_en.csv",
    "df3": _BASE_URL + "df3_en.csv",
}


class LIBRITTS_P(LIBRITTS):
    """LibriTTS-P dataset with style and speaker prompts.
    
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to download.
            Allowed values: ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``, 
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"``, 
            ``"train-other-500"``
            (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriTTS"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        annotator (str or None, optional):
            Speaker prompt annotator. One of ``["df1", "df2", "df3"]`` or ``None``.
            If ``None``, only style prompts are available. (default: ``None``)
    
    Example:
        >>> dataset = LIBRITTS_P(root="./data", url="train-clean-100", annotator="df1")
        >>> waveform, sr, orig_text, norm_text, spk_id, ch_id, utt_id, styles, speakers = dataset[0]
        >>> print(f"Waveform: {waveform.shape}, Sample rate: {sr}")
        >>> print(f"Style variants: {len(styles)}, Speaker variants: {len(speakers)}")
    """

    def __init__(
        self,
        root: Union[str, Path],
        url: str = "train-clean-100",
        folder_in_archive: str = "LibriTTS",
        download: bool = False,
        annotator: Optional[str] = None,
    ) -> None:
        super().__init__(root, url, folder_in_archive, download)

        root = os.fspath(root)
        self._annotator = annotator

        self._metadata_path = os.path.join(root, _METADATA_FILENAME)
        self._style_path = os.path.join(root, _STYLE_FILENAME)

        if annotator is not None and annotator not in ["df1", "df2", "df3"]:
            raise ValueError(
                f"``annotator`` must be one of ['df1', 'df2', 'df3'] or None, "
                f"but got '{annotator}'"
            )

        if download:
            if not os.path.isfile(self._metadata_path):
                download_url_to_file(_URLS["metadata"], self._metadata_path)
            if not os.path.isfile(self._style_path):
                download_url_to_file(_URLS["style"], self._style_path)

            if annotator is not None:
                speaker_path = os.path.join(root, f"{annotator}_en.csv")
                if not os.path.isfile(speaker_path):
                    download_url_to_file(_URLS[annotator], speaker_path)

        if not os.path.isfile(self._metadata_path):
            raise RuntimeError(
                f"Metadata file not found at ``{self._metadata_path}``. "
                "Please set ``download=True`` to download it"
            )
        if not os.path.isfile(self._style_path):
            raise RuntimeError(
                f"Style file not found at ``{self._style_path}``. "
                "Please set ``download=True`` to download it"
            )

        self._load_prompts(root, annotator)
        self._filter_valid_samples()

    def _load_prompts(self, root: str, annotator: Optional[str]) -> None:
        """Load style and speaker prompts from CSV files."""
        # Load style prompts
        style_df = pd.read_csv(
            self._style_path,
            sep='|',
            header=None,
            names=['style_prompt_key', 'style_prompt'],
        )
        style_prompts_by_key: Dict[str, List[str]] = {
            row.style_prompt_key: row.style_prompt.split(";")
            for row in style_df.itertuples(index=False)
        }

        # Load utterance metadata
        metadata_df = pd.read_csv(self._metadata_path)
        self._prompts: Dict[str, Tuple[List[str], int]] = {}

        for row in metadata_df.itertuples(index=False):
            utterance_id = row.item_name
            style_prompt_key = row.style_prompt_key
            speaker_id = int(row.spk_id)

            if style_prompt_key in style_prompts_by_key:
                style_prompts = style_prompts_by_key[style_prompt_key]
                self._prompts[utterance_id] = (style_prompts, speaker_id)

        # Load speaker prompts if annotator specified
        self._speaker_prompts: Optional[Dict[int, List[str]]] = None
        if annotator is not None:
            speaker_path = os.path.join(root, f"{annotator}_en.csv")
            if not os.path.isfile(speaker_path):
                raise RuntimeError(
                    f"Speaker prompt file not found at ``{speaker_path}``. "
                    "Please set ``download=True`` to download it"
                )

            speaker_df = pd.read_csv(
                speaker_path,
                sep='|',
                header=None,
                names=['spk_id', 'speaker_prompt'],
            )
            self._speaker_prompts = {
                int(row.spk_id): row.speaker_prompt.split(",")
                for row in speaker_df.itertuples(index=False)
            }
    
    def _filter_valid_samples(self) -> None:
        """Filter valid samples during initialization."""
        self._valid_indices = []

        for i in range(len(self._walker)):
            fileid = self._walker[i]

            if fileid not in self._prompts:
                continue

            _, metadata_speaker_id = self._prompts[fileid]

            try:
                parts = fileid.split("_")
                if len(parts) < 4:
                    continue
                file_speaker_id = int(parts[0])
            except (ValueError, IndexError):
                continue

            if metadata_speaker_id != file_speaker_id:
                continue

            if self._speaker_prompts is not None:
                if file_speaker_id not in self._speaker_prompts:
                    continue

            self._valid_indices.append(i)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, int, int, str, List[str], Optional[List[str]]]:
        """Load the n-th sample from the dataset.
        
        Args:
            n (int): The index of the sample to be loaded
        
        Returns:
            Tuple of the following items;
            
            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Original text
            str:
                Normalized text
            int:
                Speaker ID
            int:
                Chapter ID
            str:
                Utterance ID
            List[str]:
                Style prompts (all variants)
            List[str] or None:
                Speaker prompts (all variants) if annotator was specified
        """
        actual_idx = self._valid_indices[n]
        fileid = self._walker[actual_idx]

        waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id = \
            load_libritts_item(
                fileid,
                self._path,
                self._ext_audio,
                self._ext_original_txt,
                self._ext_normalized_txt,
            )

        style_prompts, _ = self._prompts[utterance_id]

        speaker_prompts = None
        if self._speaker_prompts is not None:
            speaker_prompts = self._speaker_prompts[speaker_id]

        return waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id, \
            style_prompts, speaker_prompts
