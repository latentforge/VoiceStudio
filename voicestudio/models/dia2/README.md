# Dia2

Dia2 is a streaming dialogue text-to-speech model. Every frame it consumes carries two text stream channels plus one channel per Mimi codebook, and the decoder-only backbone predicts two things from it: a binary word-advance action, which drives a state machine that decides which script word feeds the text streams on the next frame, and the first codebook of the next frame. A depth decoder then predicts that frame's remaining 31 codebooks one position at a time, conditioned on the backbone hidden state and on the codebook it just produced. Because the text streams advance one word at a time under the model's own control, generation can start before the whole script is known, and conditioning audio can be pushed through the backbone first to clone a voice.

Original model and code: [nari-labs/dia2](https://github.com/nari-labs/dia2)


## Usage

`from_pretrained` takes a published `nari-labs/Dia2-*` repository id as it stands:

```python
import torch

from voicestudio.models.dia2 import Dia2ForConditionalGeneration, Dia2Processor

model_id = "nari-labs/Dia2-1B"

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

The published repositories ship the training runtime's own `config.json`, a `data`/`model`/`runtime` tree that
declares no `model_type`, so `Dia2ForConditionalGeneration.from_pretrained` builds the configuration from it
itself. The weights are converted by a checkpoint conversion mapping registered with
`transformers.conversion_mapping`, which renames the backbone and depth decoder keys, concatenates the
per-codebook embedding tables, stacks the per-codebook output heads and splits the fused `wi` and `in_proj`
projections as the checkpoint loads. `Dia2Processor.from_pretrained` reads the published tokenizer files
directly, which is also why it keeps the added `[S1]`/`[S2]` and sound-effect tokens that a tokenizer
save/load round trip drops.

To condition on previous conversational context, pass one waveform per speaker. Dia2 needs a word-level alignment of that audio to place the conditioning words on the frame grid; pass one directly through `transcript`:

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

or leave `transcript` out and the processor derives it itself, running a Whisper model over each `audio` entry. Whisper decodes the clip with its own segment timestamps, then every segment is aligned again against the cross-attentions inside a window around those timestamps, the two-pass scheme `whisper-timestamped` uses upstream, inlined rather than added as a dependency:

```python
inputs = processor(
    text="[S1] I think so too. [S2] Then let's do it.",
    audio=[speaker_1_waveform, speaker_2_waveform],
).to(model.device)

audio_codes = model.generate(**inputs)
```

The Whisper checkpoint defaults to `openai/whisper-large-v3`, matching what upstream used, and is only loaded the first time this path runs. Set `processor.whisper_checkpoint` to a different hub id or local path before calling `processor(...)` to use a different one.

`processor.refine_whisper_precision` is the margin in seconds each segment's window is widened by on both sides before it is aligned again. It defaults to `0.5`, the same default `whisper-timestamped` ships, and must be a non-negative multiple of `0.02`, the 20 ms a Whisper cross-attention frame covers. Setting it to `None` skips the second pass and keeps the single whole-window alignment; boundaries then differ by up to about the margin.

`generate` decodes one script at a time. Classifier-free guidance runs the conditional and unconditional branches as a batch of two, so `guidance_scale=1.0` halves the compute at the cost of guidance.

`weight_conversion.convert` still writes a converted directory, for a checkpoint that has to be materialized once
and loaded many times or shipped elsewhere, and both `from_pretrained` calls above read it as readily as the
published repository:

```python
from voicestudio.models.dia2.weight_conversion import convert

convert("nari-labs/Dia2-2B", "dia2-2b-converted")
```


## Training

Training uses the standard `forward`: pass `labels` of shape `(batch_size, sequence_length, num_codebooks)` for the delayed codebook grid and `action_labels` of shape `(batch_size, sequence_length)` for the word-advance stream. Both are shifted internally, and the returned `loss` sums the codebook-0, action and depth decoder cross-entropies, each also reported on its own in `Dia2OutputWithPast`.

Upstream ships no training, loss, collator or evaluation code, so the objective above is derived from the three heads the released checkpoints carry and from the targets the upstream decode loop feeds each of them: `transformer.action_head` over `action_vocab_size` actions, `transformer.cb0_head` over the first codebook, and `depformer.logits.{i}` over codebook `i + 1` conditioned on the backbone hidden state of the preceding frame and on codebooks `0 .. i` of the current one. Nothing upstream freezes any module, and no weighting between the three terms is recorded anywhere in the released code or model cards, so they are summed unweighted. See the section below.


## Verification

`Dia2ForConditionalGeneration.from_pretrained("nari-labs/Dia2-1B")`, with no conversion call before it, reports no missing, unexpected or mismatched keys. All 440 tensors of the published `model.safetensors` are placed by the registered conversion mapping; the 51 parameter names the renaming rules never produce are exactly the fan-out targets of the three splitting converters, 33 `mlp.up_proj` and 9 each of `self_attn.k_proj` and `self_attn.v_proj`, which `Chunk` writes alongside the first target it renames to.

The weights that mapping produces are bit identical to what `weight_conversion.convert_state_dict` produces from the same file: over all 400 converted tensors the maximum absolute difference is 0.0. That is what pins the mapping to the traced conversion rather than to a load that merely reports no missing keys.

Generating `[S1] The quick brown fox jumps over the lazy dog. [S2] I could not agree with you more.` from that load gives 5.44 seconds of audio, which `openai/whisper-small.en` transcribes back as `The quick brown fox jumps over the lazy dog. I could not agree with you more.`, word for word.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **Term weights and masking of the training loss.** No upstream trainer, loss module, collator or evaluation script exists (the `nari-labs/dia2` GitHub tree is inference only), and no paper or technical report has been published. The three cross-entropy terms are therefore summed with weight 1.0 each, and no term is masked beyond the caller's own `-100` labels. If the real recipe weights the depth decoder term per codebook, or masks padding frames, this diverges from it.
- **Codebook logit width during training.** At generation time the upstream depth decoder slices its logits to `[..., :min(audio_pad_token_id, audio_bos_token_id)]`, which the migration reproduces by masking those two ids in `generate`. The training path leaves the full `vocab_size` head width in the cross-entropy, so the two beginning-of-stream and padding classes stay in the softmax denominator.
- **Word grouping and the timing of trailing punctuation.** `Dia2Processor._transcribe` groups tokens into words with `transformers.models.whisper.tokenization_whisper._combine_tokens_into_words`, the port of OpenAI Whisper's own `split_to_word_tokens` plus `merge_punctuations`. Upstream `whisper_timestamped` groups them with its own `split_tokens_on_spaces` / `split_tokens_on_unicode`, over a punctuation set of its own, and then ends a word one token early when its last token is punctuation (`num_punctuations_per_tokens` in `perform_word_alignment`). A word whose text ends in punctuation therefore keeps the punctuation's own span in its `end` here, where upstream would cut it before.
- **CUDA graph capture and `torch.compile` paths.** Upstream `dia2/runtime/generator.py` could capture the backbone step and each depth stage into CUDA graphs and compile them. `generate` runs eagerly.
- **Word timestamps in the generation result.** Upstream carried each consumed word's text and frame through the state machine and returned `(word, seconds)` pairs. `Dia2TextStreamState.word_start_frames` keeps the frames, but `generate` returns only the codes.
- **Audio file I/O, CLI, Gradio app and progress logger.** `dia2/runtime/audio_io.py`, `dia2/audio/grid.py:write_wav`, `dia2/cli.py`, `gradio_app.py` and `dia2/runtime/logger.py` are dropped along with `sphn`, `soundfile` and `gradio`.
- **Unused upstream config fields.** `data.first_word_min_start`, `data.max_pad`, `data.tokenizer_path`, `model.dropout` and the `assets` block are read by upstream `load_config` but never used by any upstream code path, and have no counterpart in `Dia2Config`. `model.rope_min_timescale` is likewise absent; `weight_conversion.build_config` raises when a checkpoint sets it to anything but `1`, which is the only value the two published checkpoints use.


## Repository integration

Two things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .dia2 import *` line.
- `PROJECT.md` needs a Dia2 status entry carrying the gaps listed above.

No new dependency is required. The migration removes `sphn`, `soundfile`, `whisper-timestamped` and `gradio` from what this model needs, leaving `torch`, `torchaudio`, `transformers`, `numpy`, `safetensors` and `huggingface_hub`; automatic word alignment resamples conditioning audio for Whisper with `torchaudio.functional.resample`, already a base dependency of the project.
