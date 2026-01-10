"""Parler-TTS synthesizer with SelectiveTuner support."""

from __future__ import annotations

import random
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

from voicestudio.models.parler_tts import ParlerTTSForConditionalGeneration
from voicestudio.models.selective_tuner import (
    SelectiveTunerForConditionalGeneration,
    SelectiveTunerConfig,
)
from voicestudio.components.style_anchor import (
    DirectStyleAnchorEmbedding,
    EncoderStyleAnchorEmbedding,
)

from .base import BaseSynthesizer

DEFAULT_VOICE_DESCRIPTION = (
    "A female speaker delivers a slightly expressive and animated speech "
    "with a moderate speed and pitch. The recording is of very high quality, "
    "with the speaker's voice sounding clear and very close up."
)


class AnchorMode(Enum):
    """Anchor embedding mode for SelectiveTuner."""
    ENCODER = "encoder"
    DIRECT = "direct"
    BASE = "base"


class ParlerTTSSynthesizer(BaseSynthesizer):
    """Parler-TTS synthesizer with SelectiveTuner support.
    
    Supports automatic detection of anchor embedding modes from checkpoints.
    """
    
    MODEL_NAME = "parler-tts/parler-tts-mini-v1"
    ANCHOR_TOKEN = "</s>"
    ANCHOR_TOKEN_ID = 1

    def __init__(
        self,
        config,
        pretrained_model_name_or_path: str | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        """Initialize Parler-TTS synthesizer.
        
        Args:
            config: Synthesizer configuration object
            pretrained_model_name_or_path: HuggingFace model ID or local path
            checkpoint_path: Optional path to fine-tuned checkpoint
        """
        super().__init__(config)
        
        self.pretrained_model_name_or_path = pretrained_model_name_or_path or self.MODEL_NAME
        self.checkpoint_path = checkpoint_path
        self.tokenizer: AutoTokenizer | None = None
        self.sample_rate: int | None = None
        self.anchor_mode: AnchorMode | None = None

    def _detect_anchor_mode(self, state_dict: dict) -> AnchorMode:
        """Detect anchor mode from checkpoint keys.
        
        Args:
            state_dict: Model checkpoint state dictionary
            
        Returns:
            Detected anchor mode
        """
        # Single pass through keys
        for key in state_dict:
            if "anchor_bases" in key or "anchor_encoders" in key:
                return AnchorMode.ENCODER
            if "anchor_deltas" in key:
                return AnchorMode.DIRECT
        return AnchorMode.BASE

    def _freeze_base_parameters(self) -> None:
        """Freeze all parameters except anchor embeddings."""
        self.model.requires_grad_(False)
        
        for module in self.model.modules():
            if isinstance(module, (DirectStyleAnchorEmbedding, EncoderStyleAnchorEmbedding)):
                module.requires_grad_(True)
    
    def _load_encoder_weights(self, state_dict: dict) -> None:
        """Load encoder anchor weights from checkpoint.
        
        Args:
            state_dict: Checkpoint state dictionary
        """
        components = [
            (self.model.text_encoder.shared, "text_encoder.shared"),
            (self.model.text_encoder.encoder.embed_tokens, "text_encoder.encoder.embed_tokens"),
            (self.model.embed_prompts, "embed_prompts"),
        ]
        
        weight_keys = ["0.0.weight", "0.0.bias", "0.2.weight", "0.2.bias"]
        
        for module, prefix in components:
            module.anchor_bases.load_state_dict({"0": state_dict[f"{prefix}.anchor_bases.0"]})
            module.anchor_encoders.load_state_dict({
                key: state_dict[f"{prefix}.anchor_encoders.{key}"]
                for key in weight_keys
            })

    def _init_tuner_model(self, use_direct: bool) -> None:
        """Initialize model with SelectiveTuner.
        
        Args:
            use_direct: Whether to use direct anchor mode
        """
        base_config = ParlerTTSForConditionalGeneration.config_class.from_pretrained(
            self.pretrained_model_name_or_path
        )
        
        tuner_config = SelectiveTunerConfig.from_pretrained(
            base_config,
            anchor_token=self.ANCHOR_TOKEN,
            anchor_token_id=self.ANCHOR_TOKEN_ID,
            use_direct_anchor=use_direct,
        )
        tuner_config.hidden_size = tuner_config.decoder.hidden_size
        
        self.model = SelectiveTunerForConditionalGeneration.from_pretrained(
            self.pretrained_model_name_or_path,
            config=tuner_config,
        )
        self._freeze_base_parameters()

    def load_model(self) -> None:
        """Load model with automatic checkpoint detection."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        
        # Detect anchor mode from checkpoint
        if self.checkpoint_path:
            print(f"\nLoading checkpoint: {self.checkpoint_path}")
            state_dict = torch.load(
                self.checkpoint_path,
                map_location=self.config.device,
                weights_only=True,
            )
            self.anchor_mode = self._detect_anchor_mode(state_dict)
            print(f"Detected {self.anchor_mode.value} mode")
        else:
            self.anchor_mode = AnchorMode.BASE
            state_dict = None
        
        # Initialize model
        if self.anchor_mode is AnchorMode.BASE:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.pretrained_model_name_or_path
            )
        else:
            self._init_tuner_model(use_direct=(self.anchor_mode is AnchorMode.DIRECT))
            
            # Load checkpoint weights
            if self.anchor_mode is AnchorMode.ENCODER:
                self._load_encoder_weights(state_dict)
            else:  # DIRECT
                self.model.load_state_dict(state_dict, strict=False)
        
        # Finalize
        self.model.to(self.config.device)
        self.sample_rate = self.model.config.sampling_rate
        self.is_loaded = True
        print(f"Loaded Parler-TTS ({self.anchor_mode.value}) on {self.config.device}\n")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        reference_audio: Path | None = None,
        style_prompt: str | None = None,
        speaker_id: str | None = None,
    ) -> bool:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            reference_audio: Unused in Parler-TTS
            style_prompt: Voice style description
            speaker_id: Unused in Parler-TTS
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded:
            self.load_model()

        try:
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Tokenize and move to device
            description = style_prompt or DEFAULT_VOICE_DESCRIPTION
            description_ids = self.tokenizer(description, return_tensors="pt").input_ids
            text_ids = self.tokenizer(text, return_tensors="pt").input_ids
            
            # Generate audio
            with torch.inference_mode():
                audio = self.model.generate(
                    input_ids=description_ids.to(self.config.device),
                    prompt_input_ids=text_ids.to(self.config.device),
                )

            # Save and verify
            sf.write(output_path, audio.cpu().numpy().squeeze(), self.sample_rate)
            
            # Single stat call for verification
            try:
                return output_path.stat().st_size > 0
            except FileNotFoundError:
                return False

        except Exception as e:
            print(f"Synthesis failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self.tokenizer = None
