import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch
import torch.nn as nn
from transformers import AutoModelForTextToWaveform, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from .hypernetwork import Hypernetwork, CLAPEncoder, QwenTextEncoder

class TTSWithLoRA(nn.Module):
    """
    TTS model with LoRA adaptation controlled by Hypernetwork
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,  # For backward compatibility with TTSLoRA
        tts_model_name: str = "nari-labs/Dia-1.6B-0626",
        clap_model_name: str = "laion/larger_clap_general",
        qwen_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        lora_config: Optional[LoraConfig] = None,
        hypernetwork_config: Optional[Dict[str, Any]] = None,
        use_qwen: bool = False,
        manual_target_modules: Optional[List[str]] = None,
        # For backward compatibility with TTSLoRA
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
    ):
        super().__init__()
        
        # --- Backward compatibility ---
        if model_id:
            tts_model_name = model_id
        
        # Load base TTS model
        try:
            self.tts_model = AutoModel.from_pretrained(tts_model_name)
        except Exception:
            self.tts_model = AutoModelForTextToWaveform.from_pretrained(tts_model_name)

        self.tts_processor = AutoProcessor.from_pretrained(tts_model_name)
        self.debug_once = True

        # Initialize LoRA configuration
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r or 16,
                lora_alpha=lora_alpha or 32,
                lora_dropout=lora_dropout or 0.1,
                target_modules=None
            )
        
        if manual_target_modules is None:
            raise ValueError(
                "LoRA target modules must be specified manually via `manual_target_modules`. "
                "Automatic detection has been removed."
            )
        lora_config.target_modules = manual_target_modules
        
        self.lora_config = lora_config
        
        self.tts_model_with_lora = get_peft_model(self.tts_model, lora_config)
        
        if use_qwen:
            self.text_encoder = QwenTextEncoder(qwen_model_name)
            text_embedding_dim = 1536
        else:
            self.clap_encoder = CLAPEncoder(clap_model_name)
            text_embedding_dim = 512
        
        self.audio_encoder = CLAPEncoder(clap_model_name)
        
        hypernetwork_config = hypernetwork_config or {}
        clap_audio_dim = 512
        
        if use_qwen:
            hypernetwork_input_dim = clap_audio_dim + text_embedding_dim
        else:
            hypernetwork_input_dim = clap_audio_dim + text_embedding_dim
        
        self.hypernetwork = Hypernetwork(
            tts_model=self.tts_model,
            input_dim=hypernetwork_input_dim,
            hidden_dim=hypernetwork_config.get('hidden_dim', 512),
            num_layers=hypernetwork_config.get('num_layers', 3),
            lora_config=lora_config,
            target_modules=lora_config.target_modules
        )
        
        self.use_qwen = use_qwen

    def _apply_lora_weights(self, lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """Apply generated LoRA weights to the model's LoRA layers."""
        if self.debug_once:
            print("[DEBUG] --- Applying LoRA Weights ---")
            print(f"[DEBUG] LoRA weight keys: {list(lora_weights.keys())}")

        applied_count = 0
        # named_modules 를 사용하여 정확한 모듈 경로 이름을 그대로 사용
        for name, peft_module in self.tts_model_with_lora.named_modules():
            if hasattr(peft_module, 'lora_A') and ('default' in getattr(peft_module, 'lora_A', {})):
                # Try exact match first
                key = name
                weights = lora_weights.get(key)
                if weights is None:
                    # Try matching by decoder-suffix to ignore peft wrapper prefixes
                    if 'decoder.' in name:
                        suffix = name[name.find('decoder.'):]
                        weights = lora_weights.get(suffix)
                if weights is None:
                    # Also try without leading 'model.' in case base model had it
                    if '.model.' in name:
                        alt = name.split('.model.', 1)[1]
                        weights = lora_weights.get(alt)
                        if weights is None and 'decoder.' in alt:
                            alt_suffix = alt[alt.find('decoder.'):]
                            weights = lora_weights.get(alt_suffix)
                if weights is not None:
                    lora_A_w, lora_B_w = weights
                    peft_module.lora_A['default'].weight.data = lora_A_w
                    peft_module.lora_B['default'].weight.data = lora_B_w
                    applied_count += 1
                    if self.debug_once:
                        print(f"  -> SUCCESS: Applied weights to {name}")
                elif self.debug_once:
                    print(f"  -> WARNING: No weights found for {name} in generated weights dict.")

        if self.debug_once:
            print(f"[DEBUG] Total LoRA layers updated: {applied_count}")
            print("[DEBUG] --- Finished Applying LoRA Weights ---")


    def _get_target_module(self, module_name: str):
        """Get target module by name"""
        parts = module_name.split('.')
        module = self.tts_model_with_lora
        
        try:
            for part in parts:
                if '*' in part:
                    continue
                module = getattr(module, part)
            return module
        except AttributeError:
            return None
    
    def encode_speaker_characteristics(
        self,
        audio: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
        sample_rate: int = 22050
    ) -> torch.Tensor:
        """Encode speaker characteristics from audio and/or text"""
        embeddings = []
        if audio is not None:
            audio_embeddings = self.audio_encoder.encode_audio(audio, sample_rate)
            embeddings.append(audio_embeddings)
        if text is not None:
            if self.use_qwen:
                text_embeddings = self.text_encoder.encode_text(text)
            else:
                text_embeddings = self.clap_encoder.encode_text(text)
            embeddings.append(text_embeddings)
        
        if not embeddings:
            raise ValueError("At least one of audio or text must be provided")
        
        return embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=-1)
    
    def forward(
        self,
        content_text: List[str],
        speaker_audio: Optional[torch.Tensor] = None,
        speaker_text: Optional[List[str]] = None,
        sample_rate: int = 48000,
        # For backward compatibility
        style_prompts: Optional[Any] = None,
        content_prompts: Optional[Any] = None,
        labels: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        **tts_kwargs
    ) -> torch.Tensor:
        """Forward pass through the TTS model with LoRA adaptation."""
        # --- Backward compatibility ---
        if style_prompts is not None:
            # This part is tricky as the old hypernetwork is different.
            # Assuming style_prompts are text for clap encoder
            speaker_text = style_prompts
        if content_prompts is not None:
            # Assuming content_prompts are tokenized input_ids
            text_inputs = {'input_ids': content_prompts, 'attention_mask': attention_mask}
            device = next(self.tts_model_with_lora.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            batch_size = content_prompts.shape[0]
        else:
            batch_size = len(content_text)
            device = next(self.tts_model_with_lora.parameters()).device
            text_inputs = self.tts_processor(
                text=content_text, return_tensors="pt", padding=True, truncation=True
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        all_outputs = []
        
        for i in range(batch_size):
            if self.debug_once:
                print(f"[DEBUG] --- Processing batch item {i} ---")

            single_speaker_audio = speaker_audio[i:i+1] if speaker_audio is not None else None
            single_speaker_text = [speaker_text[i]] if speaker_text is not None else None
            
            speaker_embeddings = self.encode_speaker_characteristics(
                audio=single_speaker_audio, text=single_speaker_text, sample_rate=sample_rate
            )
            if self.debug_once:
                print(f"[DEBUG] Speaker embedding shape: {speaker_embeddings.shape}")

            lora_weights = self.hypernetwork(speaker_embeddings)
            if self.debug_once:
                if lora_weights:
                    first_key = list(lora_weights.keys())[0]
                    lora_a_shape = lora_weights[first_key][0].shape
                    lora_b_shape = lora_weights[first_key][1].shape
                    print(f"[DEBUG] Generated LoRA weights shape (A, B for first module): {lora_a_shape}, {lora_b_shape}")
                else:
                    print("[DEBUG] Hypernetwork returned empty weights dict.")

            if speaker_embeddings.dim() > 1 and lora_weights and list(lora_weights.values())[0][0].dim() > 2:
                 lora_weights = {k: (v[0][0], v[1][0]) for k, v in lora_weights.items()}
                 if self.debug_once:
                    first_key = list(lora_weights.keys())[0]
                    lora_a_shape = lora_weights[first_key][0].shape
                    lora_b_shape = lora_weights[first_key][1].shape
                    print(f"[DEBUG] LoRA weights shape after un-batching: {lora_a_shape}, {lora_b_shape}")

            self._apply_lora_weights(lora_weights)

            single_text_inputs = {k: v[i:i+1] for k, v in text_inputs.items()}

            # Add labels for training if provided (for backward compatibility)
            if labels is not None:
                tts_kwargs['speech_inputs'] = labels[i:i+1] if labels.shape[0] > 1 else labels

            with torch.amp.autocast(device_type='cuda'):
                outputs = self.tts_model_with_lora(**single_text_inputs, **tts_kwargs)
            
            if hasattr(outputs, 'spectrogram'): output_audio = outputs.spectrogram
            elif hasattr(outputs, 'audio'): output_audio = outputs.audio
            elif hasattr(outputs, 'waveform'): output_audio = outputs.waveform
            elif hasattr(outputs, 'speech_values'): output_audio = outputs.speech_values # For backward compatibility
            else: output_audio = outputs.last_hidden_state
            
            if self.debug_once:
                print(f"[DEBUG] Raw model output shape: {output_audio.shape}")

            if output_audio.dim() == 4 and output_audio.shape[1] == 1:
                output_audio = output_audio.squeeze(1)

            all_outputs.append(output_audio)
        
        self.debug_once = False

        max_len = max(out.shape[-1] for out in all_outputs)
        padded_outputs = [torch.nn.functional.pad(out, (0, max_len - out.shape[-1])) for out in all_outputs]
        
        final_output = torch.cat(padded_outputs, dim=0)

        if self.training and labels is not None:
            loss = nn.L1Loss()(final_output, labels)
            return {"loss": loss, "logits": final_output}
        
        return final_output


    
    def generate_speech(
        self,
        content_text: str,
        speaker_audio: Optional[torch.Tensor] = None,
        speaker_text: Optional[str] = None,
        sample_rate: int = 48000,
        **generation_kwargs
    ) -> torch.Tensor:
        """Generate speech for a single input"""
        return self.forward(
            content_text=[content_text],
            speaker_audio=speaker_audio.unsqueeze(0) if speaker_audio is not None else None,
            speaker_text=[speaker_text] if speaker_text is not None else None,
            sample_rate=sample_rate,
            **generation_kwargs
        )

