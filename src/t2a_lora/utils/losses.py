import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import librosa
import numpy as np
import torchaudio.transforms as T


class MultiScaleSpectralLoss(nn.Module):
    """
    Computes multi-scale spectral loss, including L1 and L2 losses on mel spectrograms.
    """
    def __init__(self, sample_rate, n_ffts, n_mels, hop_lengths, win_lengths):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_ffts = n_ffts
        self.n_mels = n_mels
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def _compute_mel_spectrogram(self, audio, hop_length, win_length, n_mels: Optional[int] = None, n_fft: Optional[int] = None):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # The module has no parameters, so we infer the device from the input tensor.
        device = audio.device
        audio = audio.to(device)

        # Resolve per-scale parameters
        scale_idx = self.hop_lengths.index(hop_length)
        use_n_fft = n_fft if n_fft is not None else self.n_ffts[scale_idx]
        use_n_mels = n_mels if n_mels is not None else self.n_mels[scale_idx]

        transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=use_n_fft,
            n_mels=use_n_mels,
            hop_length=hop_length,
            win_length=win_length,
            power=1.0,
            normalized=True,
            center=True,
            pad_mode="reflect",
        ).to(device)
        mel_spec = transform(audio)
        return torch.log(torch.clamp(mel_spec, min=1e-5))

    def forward(self, pred, target_audio):
        """
        Args:
            pred (Tensor): Predicted mel spectrogram (B, n_mels, T) or waveform (B, T).
            target_audio (Tensor): Target audio waveform, shape (B, T_audio).
        """
        losses = {}
        total_spectral_loss = 0.0

        for i, (hop_length, win_length) in enumerate(zip(self.hop_lengths, self.win_lengths)):
            # Target audio is waveform, compute its mel spectrogram
            target_mel = self._compute_mel_spectrogram(target_audio, hop_length, win_length)
            
            # Determine predicted mel for this scale
            # Case 1: pred is waveform [B, T]
            if pred.dim() == 1:
                pred_wave = pred.unsqueeze(0)
                pred_mel_i = self._compute_mel_spectrogram(pred_wave, hop_length, win_length)
            elif pred.dim() == 2:
                # [B, T]
                pred_mel_i = self._compute_mel_spectrogram(pred, hop_length, win_length)
            elif pred.dim() == 3:
                # [B, C, T] or [B, T, C]; align channels to n_mels for this scale
                pred_mel_i = pred
                # If time is in dim 1 and channels in dim 2 -> transpose to [B, C, T]
                if pred_mel_i.shape[2] < pred_mel_i.shape[1]:
                    # Heuristic: treat last dim as smaller time? Keep as-is unless exact match below fails
                    pass
                # Accept [B, n_mels, T]
                if pred_mel_i.shape[1] == self.n_mels[i]:
                    pass
                # Accept [B, T, n_mels]
                elif pred_mel_i.shape[2] == self.n_mels[i]:
                    pred_mel_i = pred_mel_i.transpose(1, 2)
                else:
                    # Resize channel dimension to match this scale's n_mels using interpolation
                    import torch.nn.functional as F
                    B, C, Tt = pred_mel_i.shape if pred_mel_i.shape[1] <= pred_mel_i.shape[2] else pred_mel_i.transpose(1,2).shape
                    src = pred_mel_i if pred_mel_i.shape[1] <= pred_mel_i.shape[2] else pred_mel_i.transpose(1,2)
                    src4d = src.unsqueeze(1)  # [B, 1, C, T]
                    resized = F.interpolate(src4d, size=(self.n_mels[i], src.shape[2]), mode='bilinear', align_corners=False)
                    pred_mel_i = resized.squeeze(1)
            else:
                # Unsupported shape; fallback to computing mel from flattened waveform if possible
                if pred.numel() > target_audio.numel():
                    pred_wave = pred.view(target_audio.shape)
                    pred_mel_i = self._compute_mel_spectrogram(pred_wave, hop_length, win_length)
                else:
                    # As a last resort, broadcast zeros to expected shape to avoid crash (will increase loss)
                    pred_mel_i = torch.zeros_like(target_mel)
            
            # Align the time dimension of predicted and target spectrograms
            min_len = min(pred_mel_i.size(-1), target_mel.size(-1))
            pred_mel_aligned = pred_mel_i[..., :min_len]
            target_mel_aligned = target_mel[..., :min_len]

            # L1 loss on mel spectrograms
            mel_l1 = self.l1_loss(pred_mel_aligned, target_mel_aligned)
            losses[f'mel_l1_scale_{i}'] = mel_l1

            # L2 loss on mel spectrograms
            mel_l2 = self.l2_loss(pred_mel_aligned, target_mel_aligned)
            losses[f'mel_l2_scale_{i}'] = mel_l2

            total_spectral_loss += mel_l1 + mel_l2

        losses['total_spectral'] = total_spectral_loss
        return losses


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained feature extractors
    """
    
    def __init__(self, model_name: str = "laion/larger_clap_general"):
        super().__init__()
        from transformers import ClapModel, ClapProcessor
        
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using CLAP features
        
        Args:
            pred_audio: Predicted audio [batch_size, audio_length]
            target_audio: Target audio [batch_size, audio_length]
            
        Returns:
            Perceptual loss
        """
        # Extract features from both audios
        pred_features = self._extract_features(pred_audio)
        target_features = self._extract_features(target_audio)
        
        # Compute MSE loss between features
        perceptual_loss = self.mse_loss(pred_features, target_features)
        
        return perceptual_loss
    
    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract CLAP features from audio"""
        # Convert to numpy for processing
        audio_np = audio.detach().cpu().numpy()
        
        # Process with CLAP
        inputs = self.processor(
            audios=audio_np,
            sampling_rate=22050,
            return_tensors="pt"
        )
        
        # Move to device
        device = audio.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            features = self.model.get_audio_features(**inputs)
        
        return features


class SpeakerConsistencyLoss(nn.Module):
    """
    Loss to ensure speaker consistency between reference and generated audio
    """
    
    def __init__(self, clap_model_name: str = "laion/larger_clap_general"):
        super().__init__()
        from transformers import ClapModel, ClapProcessor
        
        self.processor = ClapProcessor.from_pretrained(clap_model_name)
        self.model = ClapModel.from_pretrained(clap_model_name)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(
        self, 
        generated_audio: torch.Tensor, 
        reference_audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute speaker consistency loss
        
        Args:
            generated_audio: Generated audio [batch_size, audio_length]
            reference_audio: Reference audio [batch_size, audio_length]
            
        Returns:
            Speaker consistency loss
        """
        # Extract embeddings from both audios
        gen_embeddings = self._extract_embeddings(generated_audio)
        ref_embeddings = self._extract_embeddings(reference_audio)
        
        # Cosine similarity loss (we want high similarity)
        target = torch.ones(gen_embeddings.shape[0]).to(gen_embeddings.device)
        consistency_loss = self.cosine_loss(gen_embeddings, ref_embeddings, target)
        
        return consistency_loss
    
    def _extract_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings from audio"""
        # Convert to numpy for processing
        audio_np = audio.detach().cpu().numpy()
        
        # Process with CLAP
        inputs = self.processor(
            audios=audio_np,
            sampling_rate=22050,
            return_tensors="pt"
        )
        
        # Move to device
        device = audio.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)
        
        return embeddings


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions for TTS training.
    """
    def __init__(self, sample_rate=48000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Multi-scale spectral reconstruction loss settings
        # Note: The number of mel bins (n_mels) must be consistent across all scales
        # and match the output of the TTS model (which is 16).
        self.n_ffts = [1024, 2048, 512]
        self.n_mels = [16, 16, 16] # Use 16 for all scales
        self.hop_lengths = [120, 240, 60]
        self.win_lengths = [600, 1200, 300]

        self.spectral_loss = MultiScaleSpectralLoss(
            sample_rate=self.sample_rate,
            n_ffts=self.n_ffts,
            n_mels=self.n_mels,
            hop_lengths=self.hop_lengths,
            win_lengths=self.win_lengths
        )
        
        # CLAP-based style loss (optional, can be added later)
        # self.clap_loss = CLAPLoss(...)

    def forward(self, generated_audio, target_audio, reference_audio):
        """
        Args:
            generated_audio (Tensor): Model output (waveform [B, T] or mel [B, C, T]).
            target_audio (Tensor): Ground-truth waveform [B, T].
            reference_audio (Tensor): Reference waveform [B, T] (unused placeholder).
        """
        losses: Dict[str, torch.Tensor] = {}

        # Spectral reconstruction loss between prediction and target waveform
        spectral_losses = self.spectral_loss(pred=generated_audio, target_audio=target_audio)
        losses.update(spectral_losses)

        # Total loss (extend with other terms later)
        total_loss = spectral_losses['total_spectral']
        losses['total'] = total_loss

        return losses
