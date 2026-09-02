# Dia2

Dia2 is a streaming dialogue text-to-speech model. Every frame it consumes carries two text stream channels plus one channel per Mimi codebook, and the decoder-only backbone predicts two things from it: a binary word-advance action, which drives a state machine that decides which script word feeds the text streams on the next frame, and the first codebook of the next frame. A depth decoder then predicts that frame's remaining 31 codebooks one position at a time, conditioned on the backbone hidden state and on the codebook it just produced. Because the text streams advance one word at a time under the model's own control, generation can start before the whole script is known, and conditioning audio can be pushed through the backbone first to clone a voice.

Original model and code: [nari-labs/dia2](https://github.com/nari-labs/dia2)


## Usage

The published `nari-labs/Dia2-*` checkpoints ship a bespoke config and weight layout, so they need a one-time conversion before they load:

```python
from voicestudio.models.dia2.weight_conversion import convert

convert("nari-labs/Dia2-2B", "dia2-2b-converted")
```

```python
import torch
from voicestudio.models.dia2 import Dia2ForConditionalGeneration, Dia2Processor

model_id = "dia2-2b-converted"

processor = Dia2Processor.from_pretrained(model_id)
model = Dia2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).to("cuda")
processor.audio_tokenizer.to(model.device)
```

```python
import soundfile as sf

inputs = processor(text="[S1] Hello Dia2! [S2] How are you doing today?").to(model.device)

audio_codes = model.generate(**inputs)
waveform = processor.decode(audio_codes)
sf.write("output.wav", waveform.squeeze().cpu().numpy(), processor.sampling_rate)
```

To condition on previous conversational context, pass one waveform per speaker together with its word-level alignment. Dia2 needs the alignment to place the conditioning words on the frame grid; the upstream project obtains it by running Whisper over each file, which is left to the caller here:

```python
inputs = processor(
    text="[S1] I think so too. [S2] Then let's do it.",
    audio=[speaker_1_waveform, speaker_2_waveform],
    transcript=[
        [{"text": "What", "start": 0.0, "end": 0.3}, {"text": "do", "start": 0.3, "end": 0.45}],
        [{"text": "you", "start": 0.0, "end": 0.2}, {"text": "think", "start": 0.2, "end": 0.6}],
    ],
).to(model.device)

audio_codes = model.generate(**inputs)
```

`generate` decodes one script at a time. Classifier-free guidance runs the conditional and unconditional branches as a batch of two, so `guidance_scale=1.0` halves the compute at the cost of guidance.

Training uses the standard `forward`: pass `labels` of shape `(batch_size, sequence_length, num_codebooks)` for the codebook grid and `action_labels` of shape `(batch_size, sequence_length)` for the word-advance stream. Both are shifted internally, and the returned `loss` sums the backbone, action and depth decoder cross-entropies.
