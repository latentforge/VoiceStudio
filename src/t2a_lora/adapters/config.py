"""Generation configuration for T2A-LoRA."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GenerationConfig:
    """Configuration for T2A-LoRA generation."""
    
    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # Generation parameters
    batch_size: int = 1
    max_length: Optional[int] = None
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Text processing
    max_text_length: int = 512
    truncation: bool = True
    padding: bool = True
    
    # Audio processing
    audio_sample_rate: int = 22050
    audio_hop_length: int = 256
    audio_win_length: int = 1024
    n_mels: int = 80
    
    # Output settings
    output_format: str = "wav"
    output_sample_rate: int = 22050
    normalize_audio: bool = True
    
    # Device settings
    device: str = "auto"
    dtype: str = "float32"
    
    # Cache settings
    use_cache: bool = True
    cache_size: int = 100
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerationConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
