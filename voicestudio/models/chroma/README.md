# Chroma

Chroma 1.0 is a real-time end-to-end spoken dialogue model for personalized voice cloning: a frozen Qwen2.5-Omni-3B thinker reasons over the user's speech, and its token embeddings and last hidden states are interleaved at a 1:2 ratio with audio frame embeddings into a Llama3 backbone (16 layers, 2048 hidden size) that predicts Mimi codebook 0. A second Llama3 stack (4 layers, 1024 hidden size) runs once per frame and predicts the remaining 7 Mimi codebooks, which a frozen 24 kHz Mimi codec turns back into a waveform.

Every component is reused from `transformers`: `Qwen2_5OmniThinkerForConditionalGeneration` for the reasoner, `LlamaModel` for both stacks, `MimiModel` for the codec, `CsmCodebooksHead` for the decoder's position-specific head, and `Qwen2_5OmniProcessor` for the reasoner inputs. Only the pieces Chroma does not share with any of them are written here: the flat multi-codebook embedding table, the voice-cloning prompt layout, and the interleaved generation loop.

Original model and code: [FlashLabs-AI-Corp/FlashLabs-Chroma](https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma)


## Usage

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "FlashLabs/Chroma-4B"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id)
model.to("cuda")
```

The model answers a spoken turn in the voice of a reference clip, so the processor takes the conversation, the reference waveform and its transcript together:

```python
import soundfile as sf

conversation = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are Chroma, an advanced virtual human created by FlashLabs."}],
        },
        {"role": "user", "content": [{"type": "audio", "audio": "question.wav"}]},
    ]
]

inputs = processor(
    conversation,
    prompt_audio=["reference.wav"],
    prompt_text=["Transcript of reference.wav."],
    add_generation_prompt=True,
    tokenize=False,
).to(model.device)

audio = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, output_audio=True)
sf.write("output.wav", audio[0].float().cpu().numpy(), model.config.codec_config.sampling_rate)
```

`generate` returns the `(batch_size, num_frames, audio_num_codebooks)` codebook ids unless `output_audio=True`, in which case the codec decodes each sequence up to its first all-codebook end-of-stream frame. Decoder sampling is parametrized separately, through `decoder_`-prefixed keyword arguments.


## Training

`ChromaForConditionalGeneration.forward` accepts `labels` of shape `(batch_size, sequence_length, audio_num_codebooks)` holding the codebook ids of the frame at each backbone position, with `-100` on the positions kept out of the loss. `labels[..., 0]` supervises the backbone, and the frames whose residual codebooks are not uniformly `-100` supervise the decoder. It returns `backbone_loss`, `decoder_loss` and their weighted sum

```
loss = (1 - decoder_loss_weight) * backbone_loss + decoder_loss_weight * decoder_loss
```

The reasoner and the codec are frozen on construction and after `from_pretrained`, so the released `decoder_loss_weight` of 0.5 reproduces the first training stage. The second stage is `model.freeze_backbone()` with `config.decoder_loss_weight = 1.0`. A 2D `labels` of shape `(batch_size, sequence_length)` supervises the backbone alone.
