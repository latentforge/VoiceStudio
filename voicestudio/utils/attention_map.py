import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil  # Used for cleaning up the directory
from tqdm.auto import tqdm
import warnings
from typing import List, Optional
import argparse
from PIL import Image  # PIL-only for GIF creation

try:
    import soundfile as sf  # For saving generated audio
except Exception:  # Graceful fallback if soundfile isn't available
    sf = None


def visualize_attention_for_prompts(
    save_audio: bool = False,
    audio_dir: Optional[str] = None,
    gif_duration: float = 0.5,
):
    """
    Extracts the Cross-Attention weights from the Parler-TTS model's decoder
    in a greedy manner, generates a GIF of the layer-wise attention maps,
    annotated with numerical scores and a fixed color bar range.

    Args:
        save_audio: When True, also generate and save audio as ex1.wav, ex2.wav, ex3.wav.
        audio_dir: Optional override for the output directory to save audio/GIFs (defaults to attention_maps_output).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = "parler-tts/parler-tts-mini-v1"

    # Set seed for deterministic (greedy) outputs
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        from transformers import AutoTokenizer  # type: ignore
        from parler_tts import ParlerTTSForConditionalGeneration

        model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id)
        model.to(device)  # type: ignore[misc]
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Ensure 'parler-tts' and 'transformers' are installed and compatible.")
        return

    # --- Prompt Settings ---
    style_prompt = "A man speaks slowly, with a low volume and a high-pitched voice"
    content_prompts = [
        "I am a little disappointed.",
        "I am a little disappointed",
        "I am a little disheartened.",
    ]
    
    output_base_dir = audio_dir or "results/attention_maps_output"

    # --- Clean up output directory before generation ---
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Pre-computation step to find the global max attention value ---
    # Pre-compute attention maps and global max for consistent color bar range
    all_attention_maps: List[List[torch.Tensor]] = []
    global_max_attention = 0.0

    # Tokenize style prompt once
    description_inputs = tokenizer(style_prompt, return_tensors="pt").to(device)
    description_tokens = tokenizer.convert_ids_to_tokens(description_inputs.input_ids[0])

    for content_prompt in tqdm(content_prompts, desc="Pre-compute attentions"):
        prompt_inputs = tokenizer(content_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=10,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # Validate attentions are present (be len-safe for static analyzers)
        cross_attns = getattr(outputs, "cross_attentions", None)
        if not cross_attns:
            print("Warning: No cross_attentions found in generation output. Skipping.")
            all_attention_maps.append([])
            continue

        num_layers = len(cross_attns[0])
        prompt_attention_maps = [[] for _ in range(num_layers)]
        for step_attentions in cross_attns:
            for layer_idx, layer_attention in enumerate(step_attentions):
                prompt_attention_maps[layer_idx].append(layer_attention)
        
        final_maps_for_prompt = [torch.cat(layer_attentions, dim=2) for layer_attentions in prompt_attention_maps]
        all_attention_maps.append(final_maps_for_prompt)

        # Update global max
        for layer_map in final_maps_for_prompt:
            current_max = layer_map.max().item()
            if current_max > global_max_attention:
                global_max_attention = current_max

    # --- Main loop for Visualization ---
    for i, content_prompt in enumerate(content_prompts):
        prompt_image_dir = os.path.join(output_base_dir, f"prompt_{i+1}")
        os.makedirs(prompt_image_dir, exist_ok=True)

        final_attention_maps = all_attention_maps[i]
        if not final_attention_maps:
            continue
        num_layers = len(final_attention_maps)
        image_files = []
        
        for layer_idx in tqdm(range(num_layers), desc=f"Render layers (prompt {i+1})"):
            attention_map = final_attention_maps[layer_idx].squeeze(0).cpu().numpy()
            avg_attention_map = np.mean(attention_map, axis=0)
            
            # Plotting
            fig, ax = plt.subplots(figsize=(12, 8))
            # Use fixed vmin and vmax for consistent color scale
            im = ax.imshow(avg_attention_map, cmap='viridis', aspect='auto', vmin=0, vmax=global_max_attention)
            
            # --- Add numerical annotations to each cell ---
            for row in range(avg_attention_map.shape[0]):
                for col in range(avg_attention_map.shape[1]):
                    ax.text(col, row, f"{avg_attention_map[row, col]:.2f}",
                            ha="center", va="center", color="w", fontsize=6)

            ax.set_title(f"Content: '{content_prompt}'\nLayer {layer_idx+1} Cross-Attention", fontsize=12)
            ax.set_ylabel("Generated Audio Timesteps", fontsize=10)
            ax.set_xlabel("Style Prompt Tokens", fontsize=10)
            # Align token labels to heatmap width safely
            key_len = avg_attention_map.shape[1]
            ax.set_xticks(np.arange(key_len))
            if len(description_tokens) != key_len:
                # Truncate or pad labels to match key_len
                labels = (description_tokens[:key_len] if len(description_tokens) >= key_len
                          else description_tokens + ["<pad>"] * (key_len - len(description_tokens)))
            else:
                labels = description_tokens
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            plt.tight_layout()
            
            # Save image file
            filepath = os.path.join(prompt_image_dir, f"layer_{layer_idx+1}.png")
            plt.savefig(filepath)
            plt.close(fig)
            image_files.append(filepath)

        # --- GIF Generation (PIL-only) ---
        gif_path = os.path.join(output_base_dir, f'attention_prompt_{i+1}.gif')
        frames = []
        for fp in image_files:
            im = Image.open(fp)
            frames.append(im.convert("RGB"))
            im.close()
        if frames:
            duration_ms = max(10, int(gif_duration * 1000))  # GIF expects ms per frame
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )

        # --- Audio Generation & Save as ex{i+1}.wav (optional) ---
        if save_audio:
            try:
                with torch.no_grad():
                    gen_audio = model.generate(
                        input_ids=description_inputs.input_ids,
                        attention_mask=description_inputs.attention_mask,
                        prompt_input_ids=tokenizer(content_prompt, return_tensors="pt").to(device).input_ids,
                        do_sample=False,
                        num_beams=1,
                        num_return_sequences=1,
                    )

                if isinstance(gen_audio, torch.Tensor):
                    audio_arr = gen_audio.detach().cpu().numpy().squeeze()
                    save_path = os.path.join(output_base_dir, f"ex{i+1}.wav")
                    if sf is not None:
                        sr = getattr(model.config, "sampling_rate", 24000)
                        sf.write(save_path, audio_arr, sr)
                    else:
                        # If soundfile isn't available, attempt a minimal WAV save via scipy if installed
                        try:
                            from scipy.io.wavfile import write as wav_write  # type: ignore

                            sr = getattr(model.config, "sampling_rate", 24000)
                            # Ensure data is in int16 for scipy writer
                            max_val = np.max(np.abs(audio_arr)) + 1e-12
                            norm = (audio_arr / max_val * 32767.0).astype(np.int16)
                            wav_write(save_path, sr, norm)
                        except Exception:
                            print("Audio save skipped: install 'soundfile' (pip install soundfile) for WAV output.")
                # If returned object isn't a Tensor, skip saving to avoid ambiguity
            except Exception as e:
                print(f"Audio generation failed for prompt {i+1}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize cross-attention maps (and optionally save audio)")
    parser.add_argument("--save-audio", action="store_true", help="Also generate audio files ex1.wav, ex2.wav, ex3.wav")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: attention_maps_output)")
    parser.add_argument("--gif-duration", type=float, default=0.5, help="Seconds per frame in GIF (bigger = slower). Default: 0.5")
    args = parser.parse_args()

    # guard against non-positive values
    gif_dur = args.gif_duration if args.gif_duration and args.gif_duration > 0 else 0.5
    visualize_attention_for_prompts(save_audio=args.save_audio, audio_dir=args.outdir, gif_duration=gif_dur)