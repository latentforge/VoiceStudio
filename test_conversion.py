import torch
import sys
import os
import argparse
import importlib.util

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'parler_tts', 'weight_conversion.py'))
spec = importlib.util.spec_from_file_location("weight_conversion", module_path)
weight_conversion = importlib.util.module_from_spec(spec)
sys.modules["weight_conversion"] = weight_conversion
spec.loader.exec_module(weight_conversion)
apply_dac_weight_conversion_if_needed = weight_conversion.apply_dac_weight_conversion_if_needed

def compare_structures(checkpoint_path=None):
    # 1. New Structure (Transformers Native DAC)
    print("Loading Native transformers DACModel to get target keywords...")
    try:
        from transformers import DACConfig, DACModel
        new_model = DACModel(DACConfig())
        new_keys = set(new_model.state_dict().keys())
    except ImportError:
        print("Could not import DACModel from transformers. Cannot compare with new native structure exactly.")
        new_keys = set()
    
    # 2. Old Structure (Dummy or Real Checkpoint)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading actual checkpoint state_dict from {checkpoint_path}...")
        old_sd = torch.load(checkpoint_path, map_location='cpu')
        # Handle cases where state_dict is nested inside another key like "model" or "state_dict"
        if "state_dict" in old_sd:
            old_sd = old_sd["state_dict"]
        elif "model" in old_sd:
            old_sd = old_sd["model"]
    else:
        print("Using dummy weights simulating old DAC model for comparison...")
        old_sd = {
            "audio_encoder.model.encoder.block.1.block.0.block.1.parametrizations.weight.0.weight_g": torch.ones(512, 1, 1),
            "audio_encoder.model.encoder.block.1.block.0.block.1.parametrizations.weight.0.weight_v": torch.ones(512, 1, 3),
            "audio_encoder.model.encoder.block.5.alpha": torch.randn(512),
            "audio_encoder.model.quantizer.quantizers.0.out_proj.parametrizations.weight.0.weight_g": torch.ones(1024, 1, 1),
            "audio_encoder.model.quantizer.quantizers.0.out_proj.parametrizations.weight.0.weight_v": torch.ones(1024, 1, 1),
            "some_other_key": torch.tensor(1.0)
        }

    old_keys = set(old_sd.keys())
    
    print("\nApplying conversion logic...")
    try:
        new_sd = apply_dac_weight_conversion_if_needed(old_sd)
        converted_keys = set(new_sd.keys())
    except Exception as e:
        print("Error during conversion:")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    print("           OLD STRUCTURE (기존 구조) -> CONVERTED STRUCTURE (변환된 구조)")
    print("="*80)
    
    # Display keys that disappeared (like weight_g, weight_v)
    removed_keys = old_keys - converted_keys
    # Display keys that appeared (like conv.weight)
    added_keys = converted_keys - old_keys
    # Display keys that stayed the same
    untouched_keys = old_keys.intersection(converted_keys)
    
    print(f"\n[새롭게 생성된 신규 DAC Key (NEW STRUCTURE)] - {len(added_keys)}개")
    for k in sorted(added_keys):
        # Check if the generated key (stripped of prefix) exists in Native DACModel
        k_no_prefix = k.replace("audio_encoder.", "")
        target_exists_mark = "✔" if k_no_prefix in new_keys else "❌"
        print(f"  + {k:<70} [Native 매칭: {target_exists_mark}]")
        
    print(f"\n[원본에서 지워진/매핑된 구형 DAC Key (OLD STRUCTURE)] - {len(removed_keys)}개")
    for k in sorted(removed_keys):
        print(f"  - {k}")
        
    print(f"\n[변환 없이 유지된 Key] - {len(untouched_keys)}개")
    for k in sorted(list(untouched_keys))[:10]: # Print up to 10 for brevity if many
        print(f"  = {k}")
    if len(untouched_keys) > 10:
        print(f"    ... and {len(untouched_keys) - 10} more.")

    print("\n*참고*: 'Native 매칭: ✔' 표시는 생성된 키가 transformers 내부 DAC 모델에서 실제로 사용하는 변수명과 일치함을 의미합니다.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and compare old vs new DAC key structures")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to original .pt/.bin checkpoint file")
    args = parser.parse_args()
    
    compare_structures(args.checkpoint)
