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

Upstream ships no training, loss, collator or evaluation code, so the objective above is derived from the three heads the released checkpoints carry and from the targets the upstream decode loop feeds each of them: `transformer.action_head` over `action_vocab_size` actions, `transformer.cb0_head` over the first codebook, and `depformer.logits.{i}` over codebook `i + 1` conditioned on the backbone hidden state of the preceding frame and on codebooks `0 .. i` of the current one. Nothing upstream freezes any module, and no weighting between the three terms is recorded anywhere in the released code or model cards, so they are summed unweighted. The depth decoder term is one pooled cross-entropy over the 31 remaining codebooks, so the acoustic stack enters that sum with the same weight as codebook 0 and as the action stream rather than scaling with its codebook count. See the section below.


## Verification

`Dia2ForConditionalGeneration.from_pretrained("nari-labs/Dia2-1B")`, with no conversion call before it, reports no missing, unexpected or mismatched keys. All 440 tensors of the published `model.safetensors` are placed by the registered conversion mapping; the 51 parameter names the renaming rules never produce are exactly the fan-out targets of the three splitting converters, 33 `mlp.up_proj` and 9 each of `self_attn.k_proj` and `self_attn.v_proj`, which `Chunk` writes alongside the first target it renames to.

The weights that mapping produces are bit identical to what `weight_conversion.convert_state_dict` produces from the same file: over all 400 converted tensors the maximum absolute difference is 0.0. That is what pins the mapping to the traced conversion rather than to a load that merely reports no missing keys.

Generating `[S1] The quick brown fox jumps over the lazy dog. [S2] I could not agree with you more.` from that load gives 5.44 seconds of audio, which `openai/whisper-small.en` transcribes back as `The quick brown fox jumps over the lazy dog. I could not agree with you more.`, word for word.

Automatic alignment is measured against `whisper_timestamped` 1.15.9, the release upstream's `pyproject.toml` and `uv.lock` pin. Four published clips carry it: `example_prefix1.wav` and `example_prefix2.wav` from the upstream GitHub tree, 5.40 s and 3.62 s, and `example_1.wav` and `example_2.wav` from `nari-labs/Dia2-2B`, 10.56 s and 9.12 s, 82 words in total. Both word groupings are applied to the same `openai/whisper-large-v3` token timestamps, which measures the grouping alone rather than the alignment underneath it. `_combine_tokens_into_words` and upstream's `split_tokens_on_spaces` partition the tokens identically on all four clips, word for word, so the two punctuation sets and the two splitters agree on this material. 16 of the 82 words end in a punctuation token, and that token's own span, which upstream's `num_punctuations_per_tokens` cuts off the word's `end`, runs 40 to 480 ms with a 200 ms median. `_transcribe` cuts the same span and reproduces upstream's `end` on every word of all four clips. The test that decides a word ends on a mark reads the decode of the word's trailing token group, the unit upstream's `num_punctuations_per_tokens` compares, so it fires on all eight of the marks `WORD_END_PUNCTUATION` appends beyond `string.punctuation`, including the five that `openai/whisper-large-v3` spells over more than one token: `，`, `！`, `？` and `：` over three each and `”` over two.

Of those, one boundary per clip reaches the model. `_align_words` reads `end` only on the last word of a speaker's alignment and takes every other word's hold time from the next word's `start`, so all 82 word start frames are identical either way and only the last word's trailing hold moves: 8 frames against 5, 4 against 2 and 12 against 8 on three of the clips, that is 240, 160 and 320 ms at the 12.5 Hz frame grid, and unchanged on `example_prefix2.wav`, where the `max(end_frame + 1, ...)` floor absorbs it. Cloning both upstream prefix speakers and generating the script above gives 6.24 seconds that `openai/whisper-small.en` transcribes back word for word.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **Term weights and masking of the training loss.** No upstream trainer, loss module, collator or evaluation script exists, and no paper or technical report has been published. Checked and empty: the `nari-labs/dia2` GitHub tree on `main`, 36 blobs with no train, loss, collator or eval file, and its full 19 commit history back to the initial one, which never held such a file for it to have been removed from; its branches, tags and releases, only `main`, none and none. The `nari-labs/Dia2-1B` and `nari-labs/Dia2-2B` repositories carry no `training_args.bin`, optimizer state, `trainer_state.json` or scheduler at any revision, their `model.safetensors` headers carry no `__metadata__`, and their `config.json` holds only the `data`, `model` and `runtime` trees, with no loss weight, term coefficient or masking field in any of them at any revision. The `nari-labs/Dia2-2B` Space bundles the same inference package and no training loop. arXiv for `Dia2` and for `Nari Labs`, Hugging Face papers, the `nari-labs` organisation overview whose `numPapers` is 0, Zenodo and the `nari-labs.com` blog return nothing. The README and both model cards acknowledge TPU Research Cloud compute and recommend fine-tuning without a recipe, and the two standing requests for one, GitHub issue 2 and the `Dia2-2B` discussion, have no maintainer reply.

  What that leaves open is the relative weight of the three terms, not their identity: the heads and the targets fed to them are fixed by the checkpoint and by the decode loop. The depth decoder term is the one carrying a real choice. `forward` computes it as one pooled cross-entropy over all 31 remaining codebooks, so those 31 together weigh the same as codebook 0 alone and the same as the binary action term. That is `CsmForConditionalGeneration`'s convention, `loss = backbone_loss + depth_decoder_loss` over a pooled depth term. `HiggsAudioV2ForConditionalGeneration` normalises the other way, summing a separate mean per codebook, which on this model's 31 depth codebooks would make the depth term about 31 times heavier against the other two. The two closest siblings therefore disagree about the normalisation, and weight 1.0 each names the CSM convention rather than a neutral fact.

  Masking follows CSM too. A frame whose codebooks `1 ..` are all `-100` is dropped from the depth decoder pass, which is the hook CSM uses to train its depth decoder on a subset of frames, and nothing else is masked beyond the caller's own labels, so a recipe that subsamples frames or masks padding is expressed through the labels the caller builds rather than through this `forward`.
- **Codebook logit width during training.** At generation time the upstream depth decoder slices its logits to `[..., :min(audio_pad_token_id, audio_bos_token_id)]`, which the migration reproduces by masking those two ids in `generate`. The training path leaves the full `vocab_size` head width in the cross-entropy, so the two beginning-of-stream and padding classes stay in the softmax denominator.
- **Word grouping.** `Dia2Processor._transcribe` groups tokens into words with `transformers.models.whisper.tokenization_whisper._combine_tokens_into_words`, the port of OpenAI Whisper's own `split_to_word_tokens` plus `merge_punctuations`. Upstream `whisper_timestamped` groups them with its own `split_tokens_on_spaces` / `split_tokens_on_unicode`. The two disagree over what opens a word. `_split_tokens_on_spaces` opens one on any subword whose stripped form is in `` !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ ``, a string that includes the hyphen and the apostrophe; upstream drops both from its `_punctuation` and opens a word only on a leading space. Common English separates the two: `state-of-the-art, one-of-a-kind` groups here as the eight words `state`, `-of`, `-the`, `-art,`, `one`, `-of`, `-a`, `-kind` and upstream as two, `said "stop"` as two words against one, `plan (a good one)` as four against three, and `At 3:45 p.m., call +1-555-0100.` as nine against four.

  The exposure is not symmetric with the punctuation trim measured above. A trim moves a word's `end`, and `_align_words` reads `end` only on the last word of a speaker's alignment, taking every other word's hold from the next word's `start`, so a shifted `end` is largely absorbed. A grouping difference moves `text` and `start`, and those become `input_ids`, `word_lengths` and `prefix_word_start_frames` directly. English conditioning audio containing a hyphenated compound therefore produces a different script partition here than upstream produces. The 82 words of the four measured clips carry none of these forms, which is why they partition identically there.

  The grouping stays `_combine_tokens_into_words`. Porting upstream's would buy 0.0 ms on the measured clips, and would mean carrying a hand ported grouping in place of the one `transformers` maintains. The word-final trim upstream applies on top of its grouping is implemented, and is measured in the section above.
- **CUDA graph capture and `torch.compile` paths.** Upstream `dia2/runtime/generator.py` could capture the backbone step and each depth stage into CUDA graphs and compile them. `generate` runs eagerly.
- **Word timestamps in the generation result.** Upstream carried each consumed word's text and frame through the state machine and returned `(word, seconds)` pairs. `Dia2TextStreamState.word_start_frames` keeps the frames, but `generate` returns only the codes.
- **Audio file I/O, CLI, Gradio app and progress logger.** `dia2/runtime/audio_io.py`, `dia2/audio/grid.py:write_wav`, `dia2/cli.py`, `gradio_app.py` and `dia2/runtime/logger.py` are dropped along with `sphn`, `soundfile` and `gradio`.
- **Unused upstream config fields.** `data.first_word_min_start`, `data.max_pad`, `data.tokenizer_path`, `model.dropout` and the `assets` block are read by upstream `load_config` but never used by any upstream code path, and have no counterpart in `Dia2Config`. `model.rope_min_timescale` is likewise absent; `weight_conversion.build_config` raises when a checkpoint sets it to anything but `1`, which is the only value the two published checkpoints use.


## Repository integration

Two things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .dia2 import *` line.
- `PROJECT.md` needs a Dia2 status entry carrying the gaps listed above.

No new dependency is required. The migration removes `sphn`, `soundfile`, `whisper-timestamped` and `gradio` from what this model needs, leaving `torch`, `torchaudio`, `transformers`, `numpy`, `safetensors` and `huggingface_hub`; automatic word alignment resamples conditioning audio for Whisper with `torchaudio.functional.resample`, already a base dependency of the project.
