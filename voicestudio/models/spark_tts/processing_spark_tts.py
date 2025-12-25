# Copyright (c) 2025 SparkAudio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SparkTTS audio processing."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import soxr
import torch

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging

from .modeling_spark_tts import BiCodecModel


logger = logging.get_logger(__name__)


class SparkTTSProcessor(ProcessorMixin):
    """
    Constructs a SparkTTS processor which wraps a Wav2Vec2 feature extractor and BiCodec audio tokenizer.
    
    This processor can be used to prepare audio for the model and tokenize/detokenize audio.
    
    Args:
        feature_extractor (`Wav2Vec2FeatureExtractor`): The feature extractor for Wav2Vec2.
        bicodec (`BiCodecModel`): The BiCodec model for audio tokenization.
        wav2vec2 (`Wav2Vec2Model`, *optional*): The Wav2Vec2 model for feature extraction.
    
    Example:
    
    ```python
    >>> from voicestudio.models.spark_tts import SparkTTSProcessor, BiCodecModel
    >>> from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    
    >>> # Load components
    >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    >>> bicodec = BiCodecModel.from_pretrained("SparkAudio/Spark-TTS-0.5B", subfolder="BiCodec")
    >>> wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    
    >>> # Create processor
    >>> processor = SparkTTSProcessor(feature_extractor=feature_extractor, bicodec=bicodec, wav2vec2=wav2vec2)
    ```
    """
    
    attributes = ["feature_extractor"]
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    
    def __init__(
        self,
        feature_extractor: Wav2Vec2FeatureExtractor,
        bicodec: BiCodecModel,
        wav2vec2: Optional[Wav2Vec2Model] = None,
        sample_rate: int = 16000,
        ref_segment_duration: float = 6.0,
        latent_hop_length: int = 320,
        volume_normalize: bool = True,
    ):
        super().__init__(feature_extractor)
        self.bicodec = bicodec
        self.wav2vec2 = wav2vec2
        self.sample_rate = sample_rate
        self.ref_segment_duration = ref_segment_duration
        self.latent_hop_length = latent_hop_length
        self.volume_normalize = volume_normalize
        
        if wav2vec2 is not None:
            wav2vec2.config.output_hidden_states = True
    
    def load_audio(
        self, 
        audio_path: Union[str, Path], 
        sampling_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        Load audio file and resample if necessary.
        
        Args:
            audio_path (`str` or `Path`): Path to audio file.
            sampling_rate (`int`, *optional*): Target sampling rate. Defaults to self.sample_rate.
            
        Returns:
            `np.ndarray`: Audio waveform.
        """
        sampling_rate = sampling_rate or self.sample_rate
        
        # Load audio
        wav, sr = sf.read(str(audio_path))
        
        # Convert to mono if stereo
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        # Resample if necessary
        if sr != sampling_rate:
            wav = soxr.resample(wav, sr, sampling_rate)
        
        # Volume normalization
        if self.volume_normalize:
            wav = wav / (np.abs(wav).max() + 1e-6) * 0.95
        
        return wav.astype(np.float32)
    
    def get_reference_clip(self, wav: np.ndarray) -> np.ndarray:
        """
        Extract reference audio clip for speaker embedding.
        
        Args:
            wav (`np.ndarray`): Input waveform.
            
        Returns:
            `np.ndarray`: Reference audio clip.
        """
        ref_segment_length = (
            int(self.sample_rate * self.ref_segment_duration)
            // self.latent_hop_length
            * self.latent_hop_length
        )
        wav_length = len(wav)
        
        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = np.tile(wav, ref_segment_length // wav_length + 1)
        
        return wav[:ref_segment_length]
    
    def extract_wav2vec2_features(self, wavs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Extract Wav2Vec2 features from audio.
        
        Args:
            wavs (`np.ndarray` or `torch.Tensor`): Input waveforms.
            
        Returns:
            `torch.Tensor`: Extracted features.
        """
        if self.wav2vec2 is None:
            raise ValueError("Wav2Vec2 model is required for feature extraction")
        
        # Convert to list if single waveform
        if isinstance(wavs, (np.ndarray, torch.Tensor)) and wavs.ndim == 1:
            wavs = [wavs]
        
        # Process with feature extractor
        inputs = self.feature_extractor(
            wavs,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        # Extract features
        with torch.no_grad():
            outputs = self.wav2vec2(inputs.input_values.to(self.wav2vec2.device))
            # Mix hidden states from layers 11, 14, 16
            feats_mix = (
                outputs.hidden_states[11] + 
                outputs.hidden_states[14] + 
                outputs.hidden_states[16]
            ) / 3
        
        return feats_mix
    
    def tokenize(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        reference_audio: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        return_tensors: str = "pt",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize audio into semantic and global tokens.
        
        Args:
            audio (`str`, `Path`, `np.ndarray`, or `torch.Tensor`): Input audio.
            reference_audio (`str`, `Path`, `np.ndarray`, or `torch.Tensor`, *optional*): 
                Reference audio for speaker embedding. If None, uses input audio.
            return_tensors (`str`, *optional*, defaults to "pt"): Return tensor type.
            
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: Semantic tokens and global speaker tokens.
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio = self.load_audio(audio)
        
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Get reference audio
        if reference_audio is None:
            reference_audio = audio
        elif isinstance(reference_audio, (str, Path)):
            reference_audio = self.load_audio(reference_audio)
        
        if isinstance(reference_audio, np.ndarray):
            reference_audio = torch.from_numpy(reference_audio).float()
        
        # Get reference clip
        ref_clip = self.get_reference_clip(reference_audio.cpu().numpy())
        ref_clip = torch.from_numpy(ref_clip).unsqueeze(0).float()
        
        # Extract features
        features = self.extract_wav2vec2_features(audio.cpu().numpy())
        
        # Tokenize
        with torch.no_grad():
            semantic_tokens, global_tokens = self.bicodec.tokenize(
                features.to(self.bicodec.device),
                ref_clip.to(self.bicodec.device)
            )
        
        return semantic_tokens, global_tokens
    
    def detokenize(
        self,
        semantic_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """
        Detokenize semantic and global tokens into audio waveform.
        
        Args:
            semantic_tokens (`torch.Tensor`): Semantic tokens.
            global_tokens (`torch.Tensor`): Global speaker tokens.
            return_tensors (`str`, *optional*, defaults to "pt"): Return tensor type.
            
        Returns:
            `torch.Tensor`: Reconstructed waveform.
        """
        with torch.no_grad():
            waveform = self.bicodec.detokenize(
                semantic_tokens.to(self.bicodec.device),
                global_tokens.to(self.bicodec.device)
            )
        
        return waveform
    
    def __call__(
        self,
        audio: Union[str, Path, np.ndarray, torch.Tensor],
        reference_audio: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        """
        Main method to process audio.
        
        Args:
            audio (`str`, `Path`, `np.ndarray`, or `torch.Tensor`): Input audio.
            reference_audio (`str`, `Path`, `np.ndarray`, or `torch.Tensor`, *optional*): 
                Reference audio for speaker embedding.
            sampling_rate (`int`, *optional*): Sampling rate.
            return_tensors (`str`, *optional*, defaults to "pt"): Return tensor type.
            
        Returns:
            Dictionary with processed audio features.
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio = self.load_audio(audio, sampling_rate)
        
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Get reference audio
        if reference_audio is None:
            reference_audio = audio
        elif isinstance(reference_audio, (str, Path)):
            reference_audio = self.load_audio(reference_audio, sampling_rate)
        
        if isinstance(reference_audio, np.ndarray):
            reference_audio = torch.from_numpy(reference_audio).float()
        
        # Get reference clip
        ref_clip = self.get_reference_clip(reference_audio.cpu().numpy())
        ref_clip = torch.from_numpy(ref_clip).float()
        
        # Extract features
        features = self.extract_wav2vec2_features(audio.cpu().numpy())
        
        return {
            "input_features": features,
            "reference_waveform": ref_clip,
            "waveform": audio,
        }
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from pretrained model.
        
        Args:
            pretrained_model_name_or_path (`str` or `Path`): Path to pretrained model.
            
        Returns:
            `SparkTTSProcessor`: Loaded processor.
        """
        # Load feature extractor
        wav2vec2_path = Path(pretrained_model_name_or_path) / "wav2vec2-large-xlsr-53"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_path, **kwargs)
        
        # Load Wav2Vec2 model
        wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_path, **kwargs)
        
        # Load BiCodec
        from .configuration_spark_tts import SparkTTSConfig
        config = SparkTTSConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        bicodec = BiCodecModel(config)
        
        # Load BiCodec weights
        from safetensors.torch import load_file
        bicodec_path = Path(pretrained_model_name_or_path) / config.bicodec_path
        bicodec_weights = load_file(bicodec_path / "model.safetensors")
        bicodec.load_state_dict(bicodec_weights, strict=False)
        
        return cls(
            feature_extractor=feature_extractor,
            bicodec=bicodec,
            wav2vec2=wav2vec2,
            sample_rate=config.sample_rate,
            ref_segment_duration=config.ref_segment_duration,
            latent_hop_length=config.latent_hop_length,
            volume_normalize=config.volume_normalize,
        )
