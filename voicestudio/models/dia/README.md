# Dia

This folder only re-exports `DiaConfig`, `DiaDecoderConfig`, `DiaEncoderConfig`, `DiaFeatureExtractor`, `DiaForConditionalGeneration`, `DiaModel`, `DiaPreTrainedModel`, `DiaProcessor`, and `DiaTokenizer`; it does not vendor or reimplement the model.

Original model and code: [nari-labs/dia](https://github.com/nari-labs/dia)


## Usage

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "nari-labs/Dia-1.6B-0626"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
model.to("cuda")
processor.audio_tokenizer.to(model.device)
```

```python
import soundfile as sf

inputs = processor(
    text=(
        "[S1] The sun rises in the east and sets in the west, and the moon follows it "
        "across the night sky, quietly, hour after hour, until the morning comes again."
    ),
    return_tensors="pt",
).to(model.device)

output_sequences = model.generate(**inputs)
waveform = processor.decode(output_sequences)
sf.write("output.wav", waveform.numpy(), processor.audio_tokenizer.config.sampling_rate)
```

Dia writes dialogue, so the script has to open with `[S1]` and alternate between `[S1]` and `[S2]`. Text with no speaker tag is spoken as something else entirely.

Script length matters as much as the tags. Upstream asks for text corresponding to between five and twenty seconds of speech. Below that floor the model frequently ignores the script and runs to the token limit on unrelated speech or near silence, and above the ceiling it speaks unnaturally fast.
