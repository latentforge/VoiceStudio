import re
import torch

class ConversionOps:
    def convert(self, tensors: dict) -> torch.Tensor:
        raise NotImplementedError

class WeightRenaming:
    def __init__(self, source_pattern: str, target_pattern: str):
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern

class WeightConverter:
    def __init__(self, source_patterns: list, target_patterns: str, operations: list):
        self.source_patterns = source_patterns
        self.target_patterns = target_patterns
        self.operations = operations

def convert_and_load_state_dict_in_model(state_dict: dict, mapping: list) -> dict:
    new_state_dict = {}
    used_keys = set()
    
    for rule in mapping:
        if isinstance(rule, WeightRenaming):
            for k in list(state_dict.keys()):
                match = re.search(rule.source_pattern, k)
                if match:
                    new_k = re.sub(rule.source_pattern, rule.target_pattern, k)
                    new_state_dict[new_k] = state_dict[k]
                    used_keys.add(k)
        elif isinstance(rule, WeightConverter):
            if '*' in rule.target_patterns:
                idx_pattern = rule.source_patterns[0].replace('*', r'(\d+)')
                for k in list(state_dict.keys()):
                    m = re.match(idx_pattern, k)
                    if m:
                        idx = m.group(1)
                        tensors = {}
                        for sp in rule.source_patterns:
                            sp_k = sp.replace('*', idx)
                            tensors[sp_k] = state_dict.get(sp_k)
                            used_keys.add(sp_k)
                        
                        tg_k = rule.target_patterns.replace('*', idx)
                        new_state_dict[tg_k] = rule.operations[0].convert(tensors)
            else:
                tensors = {}
                for sp in rule.source_patterns:
                    if sp in state_dict:
                        tensors[sp] = state_dict[sp]
                        used_keys.add(sp)
                if len(tensors) == len(rule.source_patterns):
                    new_state_dict[rule.target_patterns] = rule.operations[0].convert(tensors)

    for k, v in state_dict.items():
        if k not in used_keys:
            new_state_dict[k] = v

    return new_state_dict

class WeightNormToWeight(ConversionOps):
    """
    PyTorch weight_norm parametrize된 두 텐서로부터 실제 weight를 복원합니다.
    weight = weight_g * (weight_v / ‖weight_v‖)
    """

    def convert(self, tensors: dict) -> torch.Tensor:
        g = next(v for k, v in tensors.items() if k.endswith("weight_g"))
        v = next(v for k, v in tensors.items() if k.endswith("weight_v"))
        norm = v.norm(p=2, dim=list(range(1, v.dim())), keepdim=True)
        return g.view(-1, *([1] * (v.dim() - 1))) * (v / norm.clamp(min=1e-12))


_WN_OP = [WeightNormToWeight()]


def _wn(src_g: str, src_v: str, tgt: str) -> WeightConverter:
    """weight_norm → weight 변환 헬퍼."""
    return WeightConverter(
        source_patterns=[src_g, src_v],
        target_patterns=tgt,
        operations=_WN_OP,
    )


def build_dac_conversion_mapping() -> list:
    rules = []

    # 1. 헬퍼 (Encoder / Decoder 공통 블록 추출)
    def _add_residual_units(old_pfx, new_pfx, offset=0):
        for j in range(3):
            old_ru = f"{old_pfx}.{j + offset}.block"
            new_ru = f"{new_pfx}.res_unit{j + 1}"
            
            # Conv1d
            rules.append(_wn(f"{old_ru}.1.parametrizations.weight.0.weight_g",
                             f"{old_ru}.1.parametrizations.weight.0.weight_v",
                             f"{new_ru}.conv1.weight"))
            rules.append(_wn(f"{old_ru}.3.parametrizations.weight.0.weight_g",
                             f"{old_ru}.3.parametrizations.weight.0.weight_v",
                             f"{new_ru}.conv2.weight"))

            # Snake1d alpha
            for k, sn in [(0, 1), (2, 2)]:
                rules.append(WeightRenaming(
                    re.escape(f"{old_ru}.{k}.alpha"),
                    f"{new_ru}.snake{sn}.alpha"
                ))

    # Encoder
    rules.append(_wn("model.encoder.block.0.parametrizations.weight.0.weight_g",
                     "model.encoder.block.0.parametrizations.weight.0.weight_v",
                     "encoder.conv1.weight"))

    rules.append(WeightRenaming(re.escape("model.encoder.block.5.alpha"), "encoder.snake1.alpha"))
    
    rules.append(_wn("model.encoder.block.6.parametrizations.weight.0.weight_g",
                     "model.encoder.block.6.parametrizations.weight.0.weight_v",
                     "encoder.conv2.weight"))

    for i in range(4):
        old_eb = f"model.encoder.block.{i+1}.block"
        new_eb = f"encoder.block.{i}"
        
        _add_residual_units(old_eb, new_eb, offset=0)
        
        rules.append(WeightRenaming(re.escape(f"{old_eb}.3.alpha"), f"{new_eb}.snake1.alpha"))
        rules.append(_wn(f"{old_eb}.4.parametrizations.weight.0.weight_g",
                         f"{old_eb}.4.parametrizations.weight.0.weight_v",
                         f"{new_eb}.conv1.weight"))

    # Decoder
    rules.append(_wn("model.decoder.model.0.parametrizations.weight.0.weight_g",
                     "model.decoder.model.0.parametrizations.weight.0.weight_v",
                     "decoder.conv1.weight"))

    rules.append(WeightRenaming(re.escape("model.decoder.model.5.alpha"), "decoder.snake1.alpha"))
    
    rules.append(_wn("model.decoder.model.6.parametrizations.weight.0.weight_g",
                     "model.decoder.model.6.parametrizations.weight.0.weight_v",
                     "decoder.conv2.weight"))

    for i in range(4):
        old_db = f"model.decoder.model.{i+1}.block"
        new_db = f"decoder.block.{i}"
        
        rules.append(WeightRenaming(re.escape(f"{old_db}.0.alpha"), f"{new_db}.snake1.alpha"))
        rules.append(_wn(f"{old_db}.1.parametrizations.weight.0.weight_g",
                         f"{old_db}.1.parametrizations.weight.0.weight_v",
                         f"{new_db}.conv_t1.weight"))
        
        _add_residual_units(old_db, new_db, offset=2)

    # Quantizer
    for proj in ("in_proj", "out_proj"):
        rules.append(WeightConverter(
            source_patterns=[
                f"model.quantizer.quantizers.*.{proj}.parametrizations.weight.0.weight_g",
                f"model.quantizer.quantizers.*.{proj}.parametrizations.weight.0.weight_v",
            ],
            target_patterns=f"quantizer.quantizers.*.{proj}.weight",
            operations=_WN_OP,
        ))

    # Codebook
    rules.append(WeightRenaming(
        r"model\.quantizer\.quantizers\.(\d+)\.codebook\.weight",
        r"quantizer.quantizers.\1.codebook.weight",
    ))

    return rules

def apply_dac_weight_conversion_if_needed(state_dict):
    """
    Check if the state_dict contains old-style DAC weights and convert them to the new style
    expected by the transformers DAC model natively.
    """
    # Custom `convert_and_load_state_dict_in_model` is defined at the module level
    
    # Check if the state dict has the old weights format (with 'weight_g' and 'model.encoder.block')
    has_old_weights = any('weight_g' in k or k.startswith('model.encoder.block.0') for k in state_dict.keys())
    
    # If the prefix 'audio_encoder.' is heavily used in state_dict (as it's a submodel of ParlerTTS)
    # we need to ensure the conversion rules apply. `build_dac_conversion_mapping` expects keys without prefix,
    # or if they are prefixed, we need to adapt it. Wait, the mapping explicitly maps 'model.encoder...' to 'encoder...'.
    # In a typical state_dict from ParlerTTS, the audio encoder's keys are prefixed with 'audio_encoder.'
    
    # Check if 'audio_encoder.model.encoder.block.0...' exists
    is_prefixed = any(k.startswith('audio_encoder.model.encoder') for k in state_dict.keys())
    
    if has_old_weights:
        # We temporarily strip the 'audio_encoder.' prefix for DAC weights, convert, and re-add.
        dac_sd = {}
        rest_sd = {}
        prefix = 'audio_encoder.' if is_prefixed else ''
        
        for k, v in state_dict.items():
            if k.startswith(prefix + 'model.encoder') or k.startswith(prefix + 'model.decoder') or k.startswith(prefix + 'model.quantizer'):
                dac_sd[k[len(prefix):]] = v
            elif k.startswith(prefix + 'encoder') and 'conv1' in k: 
                # Already converted
                dac_sd[k[len(prefix):]] = v
            else:
                rest_sd[k] = v

        if dac_sd:
            mapping = build_dac_conversion_mapping()
            # `convert_and_load_state_dict_in_model` actually returns a converted dict
            new_dac_sd = convert_and_load_state_dict_in_model(dac_sd, mapping)
            
            # Put back with prefix
            for k, v in new_dac_sd.items():
                rest_sd[prefix + k] = v
                
            return rest_sd

    return state_dict
