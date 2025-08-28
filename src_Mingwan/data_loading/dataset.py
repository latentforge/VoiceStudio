import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class LibriTTSDataset(Dataset):
    """
    Dataset for LibriTTS with speaker characteristics
    """
    
    def __init__(
        self,
        dataset_name: str = "tictap11/libritts_p_dataset_20250821_095157",
        split: str = "train",
        sample_rate: int = 48000,
        max_audio_length: int = 480000,  # 10 seconds at 480000 Hz
        cache_dir: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir
        )

        # Audio sampling rate resampling with Huggingface datasets library (Audio)
        self.dataset.cast_column("audio", Audio(
            sampling_rate=self.sample_rate,
            ))
        
        print(f"Loaded {len(self.dataset)} samples from {dataset_name}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset
        
        Returns:
            Dictionary containing:
            - audio: Audio waveform tensor [audio_length]
            - text: Transcription text
            - speaker_id: Speaker identifier
            - speaker_description: Speaker characteristics description
            - metadata: Additional metadata including gender, pitch, speaking_speed, etc.
        """
        sample = self.dataset[idx]
        
        # Load audio
        if 'audio' in sample:
            # Handle audio data from the dataset
            audio_data = sample['audio']
            # print("audio_data: ", audio_data)
            # print("type(audio_data): ", type(audio_data))

            # Handle AudioDecoder objects (new torchcodec format)
            if hasattr(audio_data, 'get_all_samples'):
                audio_samples = audio_data.get_all_samples()
                # print("audio_samples: ", audio_samples)
                # audio = audio_data["array"]  # you can get numpy array with this approach.
                # audio = audio_samples.data.squeeze().numpy().astype(np.float32)  # another way to get numpy array.
                audio = audio_samples.data # you can get tensor with this approach.
                # print("audio: ", audio)
                # print("type(audio): ", type(audio))
                # print("audio.shape: ", audio.shape)
                # print(" audio_samples.data: ",  audio_samples.data)
                original_sr = audio_samples.sample_rate
                # print("original_sr: ", original_sr)
        
        # # Ensure audio is a proper numpy array with float32 dtype
        # if not isinstance(audio, np.ndarray):
        #     audio = np.array(audio, dtype=np.float32)
        # else:
        #     audio = audio.astype(np.float32)
        
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            else:
                error_msg = (
                    f"Invalid audio shape: {audio.shape}. "
                    f"Expected mono audio with shape (1, samples), "
                    f"but received multi-channel audio. "
                    f"Dataset index: {idx}, "
                    f"Sample info: {sample.get('item_name', 'unknown')}"
                )
                raise ValueError(error_msg)
        
        # # Convert to tensor
        # audio = torch.tensor(audio, dtype=torch.float32)
        
        # Trim or pad audio to fixed length
        audio = self._process_audio_length(audio)
        
        # Extract text (use content_prompt as the main text)
        text = sample.get('content_prompt', sample.get('text', sample.get('transcription', '')))
        
        # Extract speaker information
        speaker_id = sample.get('spk_id', sample.get('speaker_id', sample.get('speaker', 'unknown')))
        
        # Additional metadata
        metadata = {
            'speaker_id': speaker_id,
            'text': text,
            'item_name': sample.get('item_name', ''),
            'gender': sample.get('gender', ''),
            'pitch': sample.get('pitch', ''),
            'speaking_speed': sample.get('speaking_speed', ''),
            'energy': sample.get('energy', ''),
            'style_prompt_key': sample.get('style_prompt_key', ''),
        }
        
        # Speaker description (use existing speaker_prompt or style_prompt)
        # 우선은 speaker_prompt 만 사용하는 중.
        speaker_description = (
            sample.get('speaker_prompt', '') or 
            sample.get('style_prompt', '') or 
            self._get_speaker_description(sample)
        )
        
        return {
            'audio': audio,
            'text': text,
            'speaker_id': speaker_id,
            'speaker_description': speaker_description,
            'metadata': metadata
        }
    
    def _process_audio_length(self, audio: torch.Tensor) -> torch.Tensor:
        """Trim or pad audio to max_audio_length"""
        if len(audio) > self.max_audio_length:
            # Trim to max length
            audio = audio[:self.max_audio_length]
        elif len(audio) < self.max_audio_length:
            # Pad with zeros
            padding = self.max_audio_length - len(audio)
            audio = torch.cat([audio, torch.zeros(padding)])
        
        return audio
    
    def _get_speaker_description(self, sample: Dict[str, Any]) -> str:
        """Extract or generate speaker description"""
        # Check if speaker description is available in the dataset
        if 'speaker_prompt' in sample and sample['speaker_prompt']:
            return sample['speaker_prompt']
        if 'style_prompt' in sample and sample['style_prompt']:
            return sample['style_prompt']
        
        # 민관: 이 함수에서, 여기 아래 부분들을 필요 없을 수도 있음.
        # Generate basic description from available metadata
        speaker_id = sample.get('spk_id', sample.get('speaker_id', 'unknown'))
        gender = sample.get('gender', 'unknown')
        pitch = sample.get('pitch', 'unknown')
        speed = sample.get('speaking_speed', 'unknown')
        energy = sample.get('energy', 'unknown')
        
        # Create a description
        description_parts = []
        if gender != 'unknown':
            description_parts.append(f"gender: {gender}")
        if pitch != 'unknown':
            description_parts.append(f"pitch: {pitch}")
        if speed != 'unknown':
            description_parts.append(f"speaking speed: {speed}")
        if energy != 'unknown':
            description_parts.append(f"energy: {energy}")
        if speaker_id != 'unknown':
            description_parts.append(f"speaker ID: {speaker_id}")
        
        if description_parts:
            return f"Speaker with {', '.join(description_parts)}"
        else:
            return f"Speaker {speaker_id}"


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader
    """
    # Stack audio tensors
    audios = torch.stack([item['audio'] for item in batch])
    
    # Collect texts
    texts = [item['text'] for item in batch]
    
    # Collect speaker information
    speaker_ids = [item['speaker_id'] for item in batch]
    speaker_descriptions = [item['speaker_description'] for item in batch]
    
    # Collect metadata
    metadata = [item['metadata'] for item in batch]
    
    return {
        'audio': audios,
        'text': texts,
        'speaker_ids': speaker_ids,
        'speaker_descriptions': speaker_descriptions,
        'metadata': metadata
    }


def create_dataloader(
    dataset_name: str = "tictap11/libritts_p_dataset_20250821_095157",
    split: str = "train",
    batch_size: int = 4,
    sample_rate: int = 48000,
    max_audio_length: int = 480000,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    seed: Optional[int] = None  # seed 매개변수 추가
) -> DataLoader:
    """
    Create DataLoader for LibriTTS dataset
    """
    dataset = LibriTTSDataset(
        dataset_name=dataset_name,
        split=split,
        sample_rate=sample_rate,
        max_audio_length=max_audio_length,
        cache_dir=cache_dir
    )

    # Seed 설정
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        generator=generator # generator 추가.
    )
    
    return dataloader


class AudioProcessor:
    """
    Audio processing utilities
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 80,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to mel spectrogram"""
        # Convert to numpy for librosa
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.tensor(log_mel, dtype=torch.float32)
    
    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range"""
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def add_noise(self, audio: torch.Tensor, noise_factor: float = 0.01) -> torch.Tensor:
        """Add random noise to audio for data augmentation"""
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise
