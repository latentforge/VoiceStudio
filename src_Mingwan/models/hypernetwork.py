import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from peft import LoraConfig


class Hypernetwork(nn.Module):
    """
    Hypernetwork that generates LoRA weights from CLAP embeddings
    """
    
    def __init__(
        self,
        tts_model: nn.Module,
        input_dim: int,  # CLAP embedding dimension
        hidden_dim: int = 512,
        num_layers: int = 3,
        lora_config: LoraConfig = None,
        target_modules: List[str] = None
    ):
        super().__init__()
        
        self.tts_model = tts_model
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lora_config = lora_config
        self.target_modules = target_modules or []
        
        # Feature extractor from CLAP embeddings
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # LoRA weight generators
        self.lora_generators = nn.ModuleDict()
        self.expanded_target_modules = []
        
        if lora_config:
            self._init_lora_generators()
    
    def _init_lora_generators(self):
        """Initialize LoRA weight generators for each target module"""
        # Validate provided target module names against model's named_modules for exact match
        model_module_names = {name for name, _ in self.tts_model.named_modules()}
        incoming = list(self.target_modules or [])
        validated_ordered = []
        for n in incoming:
            if n in model_module_names:
                validated_ordered.append(n)
        # Preserve order and uniqueness
        self.expanded_target_modules = list(dict.fromkeys(validated_ordered))

        # One-time compact debug on validation outcome
        if not hasattr(self, "_printed_validation_once"):
            self._printed_validation_once = True
            missing = [n for n in incoming if n not in model_module_names]
            print(f"[DEBUG][Hypernetwork] validated targets: {len(self.expanded_target_modules)} kept, {len(missing)} missing")
            if len(self.expanded_target_modules) > 0:
                print("   e.g.", self.expanded_target_modules[:4])

        for module_name in self.expanded_target_modules:
            # 모듈 이름에서 점을 언더스코어로 대체 (PyTorch ModuleDict 제한 때문)
            safe_module_name = module_name.replace(".", "_")
            
            # 각 모듈에 대해 A와 B 행렬을 생성하는 네트워크
            self.lora_generators[f"{safe_module_name}_A"] = nn.Linear(
                self.hidden_dim, 
                self.lora_config.r * self._get_module_input_dim(module_name)
            )
            self.lora_generators[f"{safe_module_name}_B"] = nn.Linear(
                self.hidden_dim,
                self._get_module_output_dim(module_name) * self.lora_config.r
            )
    
    def _expand_target_modules(self):
        """Expand wildcard patterns in target_modules to actual module names"""
        expanded = []
        # Create a fast lookup set of all module names present in the model
        model_module_names = {name for name, _ in self.tts_model.named_modules()}
        for pattern in self.target_modules:
            if '*' in pattern:
                # Find all modules that match the pattern
                for name in model_module_names:
                    if self._match_pattern(pattern, name):
                        expanded.append(name)
            else:
                # Direct name: include only if it exactly exists in model's named_modules
                if pattern in model_module_names:
                    expanded.append(pattern)
                else:
                    print(f"Warning: Module {pattern} not found in TTS model, skipping...")
                    continue
        return list(set(expanded))  # Remove duplicates
    
    def _match_pattern(self, pattern: str, name: str) -> bool:
        """Check if a module name matches a wildcard pattern"""
        pattern_parts = pattern.split('.')
        name_parts = name.split('.')
        
        if len(pattern_parts) != len(name_parts):
            return False
        
        for pattern_part, name_part in zip(pattern_parts, name_parts):
            if pattern_part == '*':
                continue
            elif pattern_part != name_part:
                return False
        return True
    
    def _get_module_input_dim(self, module_name: str) -> int:
        """Get input dimension for a specific module"""
        try:
            module = self.tts_model.get_submodule(module_name)
            if hasattr(module, 'in_features'):
                return module.in_features
            elif hasattr(module, 'weight'):
                return module.weight.shape[1]
            else:
                raise AttributeError(f"Module {module_name} has no attribute 'in_features' or 'weight'")
        except AttributeError:
            raise ValueError(f"Module {module_name} not found in the TTS model.")

    def _get_module_output_dim(self, module_name: str) -> int:
        """Get output dimension for a specific module"""
        try:
            module = self.tts_model.get_submodule(module_name)
            if hasattr(module, 'out_features'):
                return module.out_features
            elif hasattr(module, 'weight'):
                return module.weight.shape[0]
            else:
                raise AttributeError(f"Module {module_name} has no attribute 'out_features' or 'weight'")
        except AttributeError:
            raise ValueError(f"Module {module_name} not found in the TTS model.")
    
    def forward(self, clap_embeddings: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate LoRA weights from CLAP embeddings
        
        Args:
            clap_embeddings: [batch_size, embedding_dim]
            
        Returns:
            Dictionary mapping module names to (lora_A, lora_B) weight tuples
        """
        # Extract features from CLAP embeddings
        features = self.feature_extractor(clap_embeddings)  # [batch_size, hidden_dim]
        
        lora_weights = {}
        # Optional: one-time debug (guarded by attribute)
        if not hasattr(self, "_printed_targets_once"):
            self._printed_targets_once = True
            sa = [m for m in self.expanded_target_modules if ".self_att" in m]
            ca = [m for m in self.expanded_target_modules if ".cross_att" in m]
            print(f"[DEBUG][Hypernetwork] target modules count -> self_attn: {len(sa)}, cross_attn: {len(ca)}")
            if len(ca) == 0:
                print("[DEBUG][Hypernetwork] No cross_attention modules in target list; cross-attn LoRA will not be generated.")
        
        for module_name in self.expanded_target_modules:
            # 모듈 이름에서 점을 언더스코어로 대체 (ModuleDict 키와 일치시키기 위해)
            safe_module_name = module_name.replace(".", "_")
            
            # Generate LoRA A matrix
            lora_A_flat = self.lora_generators[f"{safe_module_name}_A"](features)
            input_dim = self._get_module_input_dim(module_name)
            lora_A = lora_A_flat.view(-1, self.lora_config.r, input_dim)
            
            # Generate LoRA B matrix  
            lora_B_flat = self.lora_generators[f"{safe_module_name}_B"](features)
            output_dim = self._get_module_output_dim(module_name)
            lora_B = lora_B_flat.view(-1, output_dim, self.lora_config.r)
            
            # 원래 모듈 이름을 키로 사용 (점 포함)
            lora_weights[module_name] = (lora_A, lora_B)
        
        return lora_weights


class CLAPEncoder(nn.Module):
    """
    Wrapper for CLAP model to extract audio/text embeddings
    """
    
    def __init__(self, model_name: str = "laion/larger_clap_general"):
        super().__init__()
        from transformers import ClapModel, ClapProcessor
        
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        
        # Freeze CLAP parameters during training
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode_audio(self, audio: torch.Tensor, sample_rate: int = 48000) -> torch.Tensor:
        """Encode audio to embeddings"""
        # audio: [batch_size, audio_length] 
        audio_inputs = self.processor(
            audios=audio.cpu().numpy(), 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        
        audio_inputs = {k: v.to(audio.device) for k, v in audio_inputs.items()}
        
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_features(**audio_inputs)
        
        return audio_embeddings
    
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode text to embeddings"""
        text_inputs = self.processor(
            text=text,
            return_tensors="pt"
        )
        
        # Move to appropriate device
        device = next(self.model.parameters()).device
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**text_inputs)
        
        return text_embeddings
    
    def encode_multimodal(
        self, 
        audio: torch.Tensor = None, 
        text: List[str] = None,
        sample_rate: int = 48000
    ) -> torch.Tensor:
        """Encode audio and/or text to combined embeddings"""
        embeddings = []
        
        if audio is not None:
            audio_emb = self.encode_audio(audio, sample_rate)
            embeddings.append(audio_emb)
        
        if text is not None:
            text_emb = self.encode_text(text)
            embeddings.append(text_emb)
        
        if len(embeddings) == 1:
            return embeddings[0]
        elif len(embeddings) == 2:
            # Combine audio and text embeddings
            return torch.cat(embeddings, dim=-1)
        else:
            raise ValueError("At least one of audio or text must be provided")


class QwenTextEncoder(nn.Module):
    """
    Qwen3 embedding model for Korean text support
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze parameters during training
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """Encode Korean text to embeddings"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Move to appropriate device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
