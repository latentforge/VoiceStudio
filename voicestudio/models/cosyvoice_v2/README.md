# CosyVoice v2

CosyVoice v2 synthesizes speech in three stages, the same shape as v1 with every stage rebuilt. A
pretrained Qwen2 0.5B decoder turns its own text tokens and an optional speech token prompt into a
sequence of 6561 supervised semantic speech tokens at 25 Hz. A chunk aware conditional flow matching
model turns those tokens into an 80 bin mel spectrogram at 24 kHz, upsampling the token sequence by
two inside its own encoder rather than through a length regulator. A HiFTNet vocoder turns the mel
spectrogram into a 24 kHz waveform.

Three things change against v1 and they all reach the weights. The language model no longer has a
text encoder or a speaker embedding of its own: the text is read through the Qwen2 embedding table
and the utterance level speaker embedding conditions the flow matching model only. The flow encoder
gains a lookahead convolution, an upsampling layer and a second, shorter stack of encoder layers that
runs at the mel frame rate. The flow estimator is built from causal convolutions and its attention is
restricted to fixed chunks while streaming, so that a chunk can be decoded before the utterance is
finished.

The three networks are trained separately upstream and are released as three separate `.pt` files.

Original model and code: https://github.com/FunAudioLLM/CosyVoice

## Usage

```python
from voicestudio.models.cosyvoice_v2 import CosyVoiceV2ForConditionalGeneration, CosyVoiceV2Processor

model = CosyVoiceV2ForConditionalGeneration.from_pretrained("FunAudioLLM/CosyVoice2-0.5B")
processor = CosyVoiceV2Processor.from_pretrained("FunAudioLLM/CosyVoice2-0.5B")

inputs = processor(text="The quick brown fox jumps over the lazy dog.")

waveform = model.generate(
    input_ids=inputs.input_ids,
    speaker_embedding=speaker_embedding,
)
```

Passing `prompt_input_ids` and `prompt_speech_token_ids` together with
`flow_prompt_speech_token_ids` and `prompt_speech_feat` is upstream's zero shot mode; passing only
the flow ones is upstream's cross lingual mode; passing `source_speech_token_ids` bypasses the
language model and performs voice conversion. Passing a speaker embedding alone is upstream's sft
mode, which is the only mode that needs no reference waveform.

`stream=True` returns a generator. Each chunk re-encodes every token decoded so far and
`token_offset` selects the mel frames that were not rendered yet, so the flow matching model carries
no cache between chunks and only the vocoder does.

Passing a generator of `input_ids` tensors instead of one tensor selects upstream's
`inference_bistream`, the interleaved decode that reads text as it arrives and emits speech tokens
between the groups, at the same `mix_ratio` `[5, 15]` the interleaved training layout uses. It is
what upstream reaches for when the text comes from another language model rather than from a file,
and it is the input side of streaming, orthogonal to `stream=True` on the output side.

```python
def stream():
    for chunk in ["The quick ", "brown fox ", "jumps over ", "the lazy dog."]:
        yield processor(text=chunk).input_ids

waveform = model.generate(input_ids=stream(), speaker_embedding=speaker_embedding,
                          prompt_input_ids=prompt.prompt_input_ids,
                          prompt_speech_token_ids=prompt.prompt_speech_token_ids,
                          prompt_speech_feat=prompt.speech_feat)
```

Switching on `isinstance(input_ids, GeneratorType)` is what `transformers` itself does for a streamed
input: `nemotron_asr_streaming/generation_nemotron_asr_streaming.py` selects its streaming path with
`isinstance(input_features, GeneratorType)` in both `_prepare_model_inputs` and `generate`, and
`voxtral_realtime` does the same in four places. Those are the only two occurrences of
`GeneratorType` in `transformers`, and both are input side. The `BaseStreamer` family in
`generation/streamers.py` was rejected because it is an output side protocol, `put` and `end` called
by `generate` as it produces tokens, with no input side counterpart; a `TextIteratorStreamer` driving
this model is its consumer side, and reaches it as `(chunk for chunk in streamer)`.

The released directory holds one `.pt` file per network rather than a single checkpoint, beside the
`CosyVoice-BlankEN` directory the language model is built from. `from_pretrained` reads that layout
directly: it merges the three files under the name of the submodule each belongs to into a directory
under `HF_HOME`, keyed on the repository and the commit it resolved to, and the `WeightRenaming` rules
registered in `modeling_cosyvoice_v2.py` turn upstream's module names into this model's as that
directory loads. Later loads reuse it, and resolve nothing but the `cosyvoice2.yaml` and the
`CosyVoice-BlankEN/config.json` that name the revision. Once the merge is written, the three `.pt`
files are dropped from the `huggingface_hub` cache. The processor takes the same repository id and
picks up the text tokenizer, the speech tokenizer and the speaker encoder.

## Training

`CosyVoiceV2ForConditionalGeneration.forward(labels=...)` returns the language model objective only,
because upstream trains the three networks one at a time with three separate runs of
`cosyvoice/bin/train.py`. The other two objectives are on the submodules:

- **Language model.** `LabelSmoothingLoss` over `speech_vocab_size + 3` classes with `smoothing=0.0`
  and `normalize_length=True`, that is a cross entropy divided by the number of unmasked targets.
  Upstream's `Qwen2LM.prepare_lm_input_target` builds one of two layouts per sample. The unistream
  layout is the start of sequence embedding, the text, the task id embedding and the teacher forced
  speech tokens, with targets `-1` on everything before the speech tokens and the end of speech token
  `6561` appended. The bistream layout interleaves the two streams in groups of `mix_ratio[0]` text
  tokens and `mix_ratio[1]` speech tokens, each full group predicting its speech tokens and then the
  fill token `6564`, and the tail group closed with the task id embedding and the end of speech
  token. Upstream draws between them per sample with probability one half, and only when the speech
  to text ratio exceeds `mix_ratio[1] / mix_ratio[0]`; the migrated `forward` takes that draw as its
  `bistream` argument so it stays deterministic and a collator can reproduce the draw. Text is
  teacher forced through the Qwen2 embedding table and the speech tokens come from the speech
  tokenizer, not from the waveform. Upstream also reports a token accuracy, which
  `CosyVoiceV2Output.accuracy` carries and which is not part of the loss.
- **Flow matching.** `CosyVoiceV2FlowModel.forward` returns a single term,
  `CosyVoiceV2ConditionalCFM.compute_loss`, which is v1's optimal transport conditional flow matching
  loss unchanged: `sigma_min=1e-06`, one timestep per sample drawn from `U(0, 1)`,
  `y = (1 - (1 - sigma_min) t) z + t x1`, target `u = x1 - (1 - sigma_min) z`, scored with
  `mse_loss(pred * mask, u * mask, reduction="sum") / (mask.sum() * n_feats)`, and the same
  `training_cfg_rate` of 0.2 and the same prefix conditioning mel. What v2 adds is unified streaming
  training: upstream draws `streaming` once per batch with probability one half and threads it into
  both the encoder and the estimator, so half the batches are trained with chunked attention and half
  with full attention. `forward` takes it as an argument for the same reason `bistream` is one.
- **Vocoder.** `CosyVoiceV2HiFTGenerator.forward` returns the waveform and the predicted f0, which
  are the two generator side inputs of upstream's objective. `compute_loss`, inherited from v1,
  scores the two terms that need no discriminator, and takes the resolution of its mel term from
  the configuration: `cosyvoice2.yaml` sets `mel_spec_transform1` to `n_fft` 1920, `hop_size` 480
  and `win_size` 1920 at 24 kHz, against v1's 1024, 256 and 1024 at 22.05 kHz, so `CosyVoiceV2Config`
  overrides those three fields. The mel bin count, the frequency bounds and the weight of 45 are the
  same in both recipes. The rest of the objective is not implemented; see "Not carried over from
  upstream".

Upstream freezes nothing. The only `requires_grad` assignment in upstream's tree is
`Snake.alpha.requires_grad = alpha_trainable` in `cosyvoice/transformer/activation.py`, whose default
is `True` and which the vocoder never overrides. The only `.eval()` on a training path is
`self.ref_model.eval()` in `Executor`, and that reference model is a second, separately constructed
`Qwen2LM` belonging to the direct preference optimization recipe, not a frozen submodule of the model
being trained. There is nothing inside this model to freeze.

## Lineage

The base is `voicestudio/models/cosyvoice_v1/`, which upstream's own class hierarchy mirrors:
`CosyVoice2Model` subclasses `CosyVoiceModel`, `Qwen2LM` subclasses `TransformerLM`,
`CausalConditionalCFM` subclasses `ConditionalCFM` and `CausalConditionalDecoder` subclasses
`ConditionalDecoder`. What is inherited and what is not:

| Class | Base | Why |
|---|---|---|
| `CosyVoiceV2Config` | `CosyVoiceV1Config` | Same flat field set, different defaults, plus a `Qwen2Config` sub configuration named `text_config` so that `get_text_config()` finds it without an override. |
| `CosyVoiceV2ConditionalDecoder` | `CosyVoiceV1ConditionalDecoder` | Same module names and same forward shape; the convolutions become causal and the attention masks become chunked. |
| `CosyVoiceV2ConditionalCFM` | `CosyVoiceV1ConditionalCFM` | Same optimal transport objective and same guided Euler solver; the initial noise becomes a fixed tensor and `streaming` is threaded through. |
| `CosyVoiceV2CausalResnetBlock1D`, `CosyVoiceV2CausalBlock1D` | `CosyVoiceV1ResnetBlock1D`, `CosyVoiceV1Block1D` | Same residual algebra, with a causal convolution and a layer norm in place of the padded convolution and the group norm. |
| `CosyVoiceV2HiFTGenerator` | `CosyVoiceV1HiFTGenerator` | Identical class. v2 only changes the configuration: 24 kHz, upsample rates `[8, 5, 3]` and a third source residual block. |
| `CosyVoiceV2UpsampleEncoder` | composed of `CosyVoiceV1EncoderLayer`, `CosyVoiceV1InputProjection`, `CosyVoiceV1RelPositionalEmbedding` | The layers are the same WeNet blocks as v1; only the surrounding structure, a lookahead convolution and a second stack after an upsampling layer, is new. |
| `CosyVoiceV2FeatureExtractor`, `CosyVoiceV2Processor` | the v1 pair | Same code, 24 kHz mel geometry, and a truncation that keeps the prompt mel and the prompt speech tokens at exactly `token_mel_ratio` frames per token. |
| `CosyVoiceV2GenerationMixin` | reuses v1's `repetition_aware_sampling` and `fade_in_out` | The sampler is unchanged. The chunking is not: v2 re-encodes from the start of the utterance every chunk and carries no flow matching cache, so `token2wav` is written out rather than inherited. |
| `CosyVoiceV2SpeechTokenLM` | `transformers` `Qwen2Model` | Upstream wraps `Qwen2ForCausalLM` but only ever reads its last hidden state and its embedding table, so the head, which is tied to the embedding table and stored nowhere in `llm.pt`, is dropped. |

Against the sibling inheritance map, the v1 rejections carry over unchanged and for the same reasons:
`FeedForward` and `TimestepEmbedding` against f5_tts, because the v2 estimator is still the Matcha
one dimensional UNet whose checkpoint stores diffusers layout parameter paths; `Encoder` and
`EncoderLayer` against prompt_tts_pp, because these are WeNet blocks with a two `Linear` feed forward
and no convolution module or macaron branch; `SourceModule` against prompt_tts_pp, because v2 still
uses upstream's `sinegen_type='1'`.

`Snake` against bigvgan is **rejected**, which corrects the prediction v1's README makes.
`voicestudio/models/bigvgan/` has landed, and `BigVGANSnakeActivation.forward` unconditionally
upsamples its input by `config.anti_alias_ratio` through a Kaiser windowed sinc, applies the
nonlinearity and downsamples again. CosyVoice's `Snake(channels, alpha_logscale=False)` is the bare
`x + (1 / (alpha + 1e-9)) * sin(alpha * x) ** 2` with no resampling, and `anti_alias_ratio=1` is not
an identity because the transposed convolution with the filter taps is not one. The same argument
rejects `CosyVoiceV1ResBlock` inheriting `BigVGANAmpBlock`, which is built from the anti aliased
activation. Inheriting either would mean growing a no anti aliasing mode into the BigVGAN model.

## Dependencies

Everything v1 removed stays removed. What v2 adds on top, and what each turned into:

| Upstream dependency | What replaced it |
|---|---|
| `transformers.Qwen2ForCausalLM` inside `Qwen2Encoder` | Native `Qwen2Model`, already in `transformers-tts`. |
| `diffusers`, `matcha-tts`, `einops`, `omegaconf`, `hyperpyyaml`, `torchdiffeq` | Removed the same way v1 removed them. |
| `onnxruntime` | Removed. See below. |

`onnxruntime` was the last one, and it is gone. CosyVoice v2's speech tokenizer
(`speech_tokenizer_v2.onnx`, 496 MB) and speaker encoder (`campplus.onnx`, 28 MB) are both PyTorch
modules now. The speaker encoder is `CosyVoiceV1SpeakerEncoder` reading its authors' own PyTorch
release, `campplus_cn_common.bin` on `funasr/campplus`; `campplus.onnx` is byte identical at
28,303,423 bytes across v1, v2 and v3, so one port covers all three.

The speech tokenizer is `CosyVoiceV2SpeechTokenizer`, and it is not a port of
`xingchensong/S3Tokenizer`; that project is itself a reimplementation and the graph is the real
upstream source, so the graph is what was traced. Its weights are read out of the graph by
`cosyvoice_v1/weight_conversion.py`, which parses the six protocol buffer fields it needs rather than
depending on `onnx`. All 102 initializers map to exactly one parameter each and load with
`strict=True`.

The v2 tokenizer differs from v1's in five ways, each read off the graph rather than assumed. Its
first convolution strides by two as well as its second, so it runs at 25 Hz where v1 runs at 50 Hz.
It has no position table; each attention applies a rotary embedding of base 10000 built for 2048
positions, which `Qwen2RotaryEmbedding` reproduces bit for bit. Each attention adds a memory block
to its output, a depthwise convolution of kernel 31 over the value projection masked to the valid
frames, which is a scalar attention memory network. It leaves the attention scores unmasked and
zeroes the padded rows of the attention output instead, which is only correct at batch one, and is
why upstream ships a separate `speech_tokenizer_v2.batch.onnx`. And it closes with a finite scalar
quantizer over eight ternary dimensions, whose 3 to the 8th is the 6561 entry vocabulary, rather than
with v1's euclidean codebook.

## Upstream does not run correctly on transformers 5

Two defects in the vendored upstream code surfaced during verification. Both come from the same
place: upstream pins `transformers==4.51.3` and its code does not survive the move to 5. Neither is
an omission in this folder and neither is a choice this migration made, so they are recorded here
rather than under "Not carried over from upstream".

The naive reading of "the migrated model produces different tokens from upstream" is that the
migration is wrong. Here the measurement says the reverse, and the thing that makes the claim
checkable rather than asserted is that both cached decoders were compared against an **uncached full
sequence reference**, which is ground truth for either of them.

- **`Qwen2Encoder.forward_one_step` mishandles its attention mask.** It passes
  `masks[:, -1, :]`, a padding mask covering the current step alone, while the key value cache
  already holds the whole prefix. The first step is fine, because the last row of
  `tril(ones(1, L, L))` is a correct all ones mask of length `L` against an empty cache. From the
  second step the mask has length one against a key axis of length `L + 1`, and transformers 5 does
  not raise on that, it silently computes something else. Teacher forced against the uncached
  reference, upstream's cached logits are wrong by **9.762** where this implementation is at
  **9.5e-06**. Sampling from the two therefore parts company at index 1: on the same seed and the
  same text upstream runs to its **200** token length cap while this implementation stops on a stop
  token after **58**. With the one step mask dropped, which is what this implementation does and what
  the shapes imply was meant, the two sequences are **identical, 58 against 58**. This model
  deliberately does not reproduce upstream's sampled output on transformers 5, because reproducing it
  would mean reproducing the defect.
- **`Qwen2Encoder.__init__` leaves the backbone dtype to `from_pretrained`.** It calls
  `Qwen2ForCausalLM.from_pretrained(pretrain_path)` with no dtype. Under 4.51.3 that gives float32
  and the bfloat16 in `CosyVoice-BlankEN/config.json` is upcast, and `load_state_dict` then copies
  `llm.pt` into float32 parameters. Under transformers 5 `from_pretrained` follows the checkpoint and
  gives bfloat16, while `llm_embedding`, `speech_embedding` and `llm_decoder` are plain modules that
  stay float32, so upstream's own `Qwen2LM` is internally mixed and raises on the first matrix
  multiply across that seam. This model is uniformly float32, which is what upstream's own runs
  produced. [`CosyVoiceV2Config`] additionally makes the composite dtype authoritative over
  `text_config`, so a `from_pretrained` path that propagates a sub configuration dtype cannot
  reintroduce the split.

## Verification

Everything below ran on CPU in float32 in the project venv, against the real
`FunAudioLLM/CosyVoice2-0.5B` weights. Nothing ran on a GPU.

**Checkpoint coverage.** 1744 source tensors (295 from `llm.pt`, 1121 from `flow.pt`, 328 from
`hift.pt`) convert to 1743 tensors over 639,174,467 parameters, 505,803,812 under `llm`, 112,549,360
under `flow` and 20,821,295 under `hift`, with zero MISSING and zero UNEXPECTED keys and no buffer
skipped. Exactly one source tensor is dropped, `llm.model.lm_head.weight`, and the conversion check
demonstrates rather than assumes why: it searches the converted tensors for a bit identical match and
finds `llm.model.embed_tokens.weight`, which is the tie `Qwen2Config.tie_word_embeddings` declares.

Every one of those source tensors is bit identical to the model tensor it lands on, and no model
tensor is left unfilled, so the registered `WeightRenaming` rules account for the checkpoint exactly
rather than merely producing a clean report. `save_pretrained` followed by `from_pretrained` returns
the same tensors with zero missing, unexpected or mismatched keys, which is what keeps the reverse
of the mapping honest.

**Numeric parity against upstream's own classes.** Every migrated module was run side by side with
the class it replaces, loaded from the same tensors. Reported as `max|diff|`:

| Component | Upstream class | max abs difference |
|---|---|---|
| Flow encoder, full attention | `UpsampleConformerEncoder` | 0.0 |
| Flow encoder, chunked attention | `UpsampleConformerEncoder` | 0.0 |
| Flow encoder, lookahead context path | `UpsampleConformerEncoder` | 0.0 |
| Flow matching estimator, full attention | `CausalConditionalDecoder` | 0.0 |
| Flow matching estimator, chunked attention | `CausalConditionalDecoder` | 0.0 |
| Fixed initial noise | `CausalConditionalCFM.rand_noise` | 0.0 |
| Euler solver, 10 steps with guidance 0.7 | `CausalConditionalCFM.forward` | 0.0 |
| Euler solver, chunked attention | `CausalConditionalCFM.forward` | 0.0 |
| Flow matching loss, full and chunked | `CausalConditionalCFM.compute_loss` | 0.0 |
| Flow inference, both streaming and finalize settings | `CausalMaskedDiffWithXvec.inference` | 0.0 |
| Sine generator waves and voiced mask | `SineGen2` | 0.0 |
| Neural source filter | `SourceModuleHnNSF` | 0.0 |
| f0 predictor | `ConvRNNF0Predictor` | 0.0 |
| Vocoder, inference | `HiFTGenerator.inference` | 1.208e-05 |
| Vocoder, training entry point | `HiFTGenerator.forward` | 1.228e-05 |
| Language model text embedding | `Qwen2Encoder` | 0.0 |
| Language model logits, full sequence | `Qwen2Encoder` | 0.0 |
| Language model loss, unistream layout | `Qwen2LM.forward` | 0.0 |
| Language model loss, bistream layout | `Qwen2LM.forward` | 0.0 |

The vocoder is the only component that is not bit exact, and the cause is the one v1 already pinned:
upstream builds the analysis window with `scipy.signal.get_window("hann", 16, fftbins=True)` and this
model with `torch.hann_window(16, periodic=True)`, the same function rounded differently in float32.
The excitation, the f0 contour and the sine generator are all exactly 0.0, so the difference enters
at the short time Fourier transform of the excitation and nowhere earlier. That was established by
comparing the source module before the synthesis filter rather than only the waveform, which is what
separated this from the real fault the same check found first: the vocoder initially differed by
1.89 because upstream selects `SineGen2` for every sampling rate other than 22050 Hz and v1's
`SineGen` had been inherited unchanged.

The autoregressive loop was checked as a loop, not only as a forward pass. Against upstream with the
one step mask corrected, `Qwen2LM.inference` and `generate_speech_tokens` produce the **same 58
speech tokens** from the same seed, token for token, and both end on a stop token rather than on the
length cap. Against stock upstream they differ, 200 against 58; see the section above.

**Training objectives.** With the random number generators pinned, the language model loss matches
upstream at 0.0 on both sequence layouts, the unistream one and the interleaved bistream one, and so
does the flow matching loss with streaming on and off. Gradients from the language model objective
reach 294 parameter tensors, all of them under `llm`, and none under `flow` or `hift`.

**Round trip through the hub format.** `save_pretrained` followed by `from_pretrained` reproduces
every parameter with a largest difference of 0.0 and leaves no meta parameter or buffer behind. A
model reloaded that way computes bit for bit what the model it was saved from computes: 0.0 on the
flow encoder, 0.0 on the vocoder waveform, and the same speech tokens from the same seed. The fixed
initial noise also comes back at 0.0, which is the check that matters most here. It is not a
registered buffer and not in the checkpoint, so under the meta device initialisation transformers 5
uses it would otherwise be uninitialised memory, which is the failure that silently destroyed v1's
output before it was found. It is built on first use from a saved and restored generator state, and
outside inference mode so that the cached tensor stays usable by autograd.

**Generated speech, transcribed back.** Two utterances were synthesized and transcribed with
`facebook/wav2vec2-base-960h`. Model and processor were both built by
`from_pretrained("FunAudioLLM/CosyVoice2-0.5B")` with no conversion step before it, and the load
report carried no missing, unexpected or mismatched key over 639,174,467 parameters.

| Prompt text | Transcript |
|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DO` |
| `She sells sea shells by the sea shore.` | `SHE SELLS SEASHELLS BY THE SEASHORE` |

The second is word for word, with the connectionist temporal classification decoder merging both
compounds the way v1 does. The first drops the final `G` of `DOG`, a decoder edge on a word ending
the utterance rather than a missing word; the waveforms are 2.84 s at RMS 0.0964 and 2.60 s at RMS
0.0650, so nothing was truncated.

The conditioning was upstream's sft mode, `sft, speaker embedding only, no reference waveform`,
selected by the geometry test above rather than hardcoded: a 192 dimensional campplus speaker vector
taken from the v1 speaker table, no prompt speech tokens and no prompt mel spectrogram. That
exercises the language model, the flow matching model and the vocoder end to end, and it is a
meaningful test of all three precisely because the v2 language model does not read the speaker
embedding at all, so every word in these transcripts came out of the text path.

That run predates the ONNX port and used upstream's sft mode only. Zero shot voice cloning from a
reference clip, which is the path that derives speech tokens, a mel spectrogram and a speaker
embedding from a waveform, is exercised by the run recorded under "The ported ONNX components"
below.

**The ported ONNX components, against the graphs they replace.** Three LibriSpeech clips of 2.9, 2.5
and 6.5 seconds went through `onnxruntime` and through the PyTorch port, on the same features. The
speaker embedding, 192 dimensions reaching 2.83 in magnitude, differs by at most 8.106e-06. The
speech tokenizer produced 580 token ids over the three clips and every one is identical; its encoder
output, reaching 3.49 in magnitude, differs by at most 7.272e-06. That residual is float32
reassociation, not a difference in the computation.

**Zero shot voice cloning, end to end.** The port makes this path runnable, so it was run. A 5.86 s
LibriSpeech clip, which `facebook/wav2vec2-base-960h` transcribes as `MISTER QUILTER IS THE APOSTLE
OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL`, became 146 speech tokens and a 192
dimensional speaker embedding, and both were passed to the language model and to the flow matching
model together with the prompt mel spectrogram and the clip's transcript.

| Asked | Heard back | Waveform |
|---|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` | 4.56 s at 24000 Hz, RMS 0.0654, peak 0.7251 |
| `She sells seashells by the seashore.` | `SHE SELLS SEA SHELLS BY THE SEASHORE` | 4.40 s at 24000 Hz, RMS 0.0544, peak 0.4214 |

Every word is there; the second splits `SEASHELLS`, a decoder edge rather than a missing word.

**Streaming input text, generated and transcribed back.** The same 5.86 s LibriSpeech clip as the
zero shot run above, three seeds, transcribed with `facebook/wav2vec2-base-960h`, against the
sentence `The quick brown fox jumps over the lazy dog.` fed as the four chunks `The quick `,
`brown fox `, `jumps over `, `the lazy dog.`:

| Input | WER by seed | Heard back, seed 1 |
|---|---|---|
| the whole text at once | 0.000 / 0.000 / 0.000 | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` |
| a generator of the four chunks | 0.444 / 0.333 / 0.556 | `WELCOME HIS GOSPEL THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` |

Every seed of the streamed decode carries the whole sentence, preceded by the last three words of the
prompt utterance; seeds 0 and 2 add a connectionist temporal classification merge, `GASPEL` and
`DUG`, `QUITBROWN`. The prefix is the interleaved layout rather than a defect: the prompt transcript
is part of the interleaved stream by construction, so the decode continues out of it, and the same
run on v3 shows what happens when the prompt speech tokens are withheld and the transcript has
nothing to stand in for it.

The decode was also checked as a decode rather than only by its audio: from seed 0 it yields 135
speech tokens with a maximum id of 6558 against a `speech_vocab_size` of 6561, so every fill token
and the end of speech token were consumed inside the loop and none leaked into the output.

**The text frontend, generated and transcribed back.** The English digit reading v2 inherits from
`CosyVoiceV1Processor` was exercised on the real weights: zero shot from the same 5.86 s LibriSpeech
clip, `I paid 1234 dollars in 2025 for 7 books.`, three seeds each, on the local GPU, transcribed with
`facebook/wav2vec2-base-960h`, word error rate against the text handed to the model:

| Frontend | WER by seed | Heard back, seed 0 |
|---|---|---|
| on | 0.000 / 0.000 / 0.000 | `I PAID ONE THOUSAND TWO HUNDRED AND THIRTY FOUR DOLLARS IN TWO THOUSAND AND TWENTY FIVE FOR SEVEN BOOKS` |
| off | 1.111 / 1.000 / 0.889 | `I PAID TWONE HUNDRED AN D THIRTY FOUR DOLLARS INTWUNDRED INDE FY FOR SEVEN BOOKS` |

With the front end on, all three seeds are word for word. With it off, no seed reads either number
back and every one of them slurs the words around it.

**The interleaved decode against upstream's own `inference_bistream`.** Upstream's method was run
unmodified on the same weights, through an adapter exposing this model's language model under the
attribute names it reads, with both sides drawing from `repetition_aware_sampling` off the same
seed. It agrees **token for token**: 209 tokens against 209, no first difference, from a generator
yielding one text token at a time and again from one yielding five. The one library difference the
adapter absorbs is that upstream reads the cache length as `cache[0][0].size(2)`, the tuple layout
`transformers` carried before 5.0.

## Not carried over from upstream

Recorded per CLAUDE.md section 2.6.

- **The vocoder's adversarial objective.** `cosyvoice/hifigan/hifigan.py` is unchanged from v1: the
  generator loss is `generator_loss` plus `feat_match_loss_weight` (2.0) times the feature matching
  loss plus `multi_mel_spectral_recon_loss_weight` (45) times a mel reconstruction loss plus
  `tpr_loss_weight` (1.0) times `tpr_loss` at `tpr_loss_tau` 0.04 plus an L1 loss between the
  predicted f0 and the pitch feature, and the discriminator loss is `discriminator_loss` plus the same
  weighted `tpr_loss`, optimized in an alternating turn with its own optimizer. `MultipleDiscriminator`
  and the three loss functions upstream imports from `matcha.hifigan.models` are not implemented, so
  the vocoder cannot be trained the way upstream trained it. Leaving them out follows the
  `transformers` convention on GAN trained vocoders, measured over the 494 `modeling_*.py` files in
  the 510 model folders of `transformers` 5.16.1. `Discriminator` appears in two of those files,
  `electra/modeling_electra.py` and `funnel/modeling_funnel.py`, and both are pretraining heads over
  token logits, `ElectraDiscriminatorPredictions` and `FunnelDiscriminatorPredictions` under
  `FunnelForPreTraining`, not adversaries over a waveform. `adversarial` appears in none of them.
  Every vocoder shipped takes a spectrogram or codes and returns a bare tensor, and none takes
  `labels`:

  | Class | `forward` signature |
  |---|---|
  | `SpeechT5HifiGan` | `(self, spectrogram, **kwargs) -> torch.FloatTensor` |
  | `FastSpeech2ConformerHifiGan` | `(self, spectrogram, **kwargs) -> torch.FloatTensor` |
  | `VitsHifiGan` | `(self, spectrogram, global_conditioning=None) -> torch.FloatTensor` |
  | `SeamlessM4THifiGan`, `SeamlessM4Tv2HifiGan` | `(self, inputs_embeds) -> torch.FloatTensor` |
  | `SeamlessM4TCodeHifiGan` | `(self, input_ids, spkr_id, lang_id, **kwargs) -> tuple[torch.Tensor]` |
  | `SeamlessM4Tv2CodeHifiGan` | `(self, input_ids, speaker_id, lang_id, **kwargs) -> tuple[torch.Tensor]` |
  | `Qwen2_5OmniToken2WavBigVGANModel` | `(self, mel_spectrogram, **kwargs)`, returning a clamped waveform |

  VITS is the case that settles it, because upstream VITS is GAN trained end to end:
  `VitsModel.forward` declares `labels` and its body opens with
  `raise NotImplementedError("Training of VITS is not supported yet.")`. The one shipped speech
  synthesis model with a real objective, `FastSpeech2ConformerWithHifiGan`, sums L1 mel, duration,
  pitch and energy in `FastSpeech2ConformerLoss` and carries no adversarial term, and its vocoder
  half takes no labels. Against that convention, the two terms this folder does score, the mel
  reconstruction loss and the f0 loss, go beyond it rather than falling short of it. The consequence
  to know is that a vocoder trained through them alone would not reproduce a released checkpoint.
- **Direct preference optimization.** `Qwen2LM.forward_dpo` and `DPOLoss` in `cosyvoice/utils/losses.py`
  are a second training objective, reached only through `examples/libritts/cosyvoice2/run_dpo.sh`. It
  needs a second, separately constructed model instance held in `Executor.ref_model` and a batch
  carrying a rejected speech token sequence alongside the chosen one, so it does not fit inside a
  single `forward`. No released checkpoint is a preference tuned checkpoint. Leaving it out is a scope
  decision and it is **still open**. Its loss is
  `-logsigmoid(beta * ((chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)))`
  added to the ordinary cross entropy on the chosen half. Note also that upstream averages those log
  probabilities over the positions where the target **is** `IGNORE_ID` rather than where it is not,
  which reads like a sign error in upstream and is recorded here rather than corrected.
- **The text normalizer inside the text frontend.** The frontend itself is implemented:
  `CosyVoiceV2Processor` inherits `CosyVoiceV1Processor.normalize_text`, which is upstream's
  `CosyVoiceFrontEnd.text_normalize` with the sentence splitting of `split_paragraph` and the English
  digit reading of `number_to_words`, so digits and multi sentence input behave the way upstream does.
  What is missing is the normalizer upstream runs before all of that, `ttsfrd` if the resource pack is
  installed, otherwise `wetext`, otherwise nothing. So the branch this reproduces exactly is upstream's
  own "no frontend is avaliable" path, and abbreviations, dates and currency are not expanded the way
  an installed normalizer would expand them. `processor(text=..., normalize=True)` applies the
  rewriting; the processor tokenizes the text as given when it is not asked to.
- **A speaker table.** The released v2 directory ships no `spk2info.pt`, ModelScope's
  `iic/CosyVoice2-0.5B` ships none, and the one mirror that does, `lucyknada/CosyVoice2-0.5B`, turns
  out to hold a **v1** table. Its seven speakers top out at speech token id 4085 against v2's 6561
  vocabulary and carry about 1.72 mel frames per token against v2's exactly 2, which is the 50 Hz,
  22050 Hz, 256 hop geometry of v1. Only the 192 dimensional speaker embeddings transfer, since
  `campplus.onnx` is byte identical across v1, v2 and v3, and that is what upstream's sft mode needs
  and nothing more. Zero shot cloning from a reference clip needs no table.
- **The vendored upstream tree.** It is gone, removed in `86a9fa18` once v2 and v3 had landed and
  every file in it could be pointed at a counterpart or a category. Seven files stay in
  `voicestudio/models/cosyvoice_v1/`, each holding something no migration implemented, and the
  File map in that folder's README accounts for all 166. The two that matter to v2 are
  `cosyvoice/hifigan/hifigan.py` with `cosyvoice/hifigan/discriminator.py`, for the adversarial
  objective above, and `cosyvoice/utils/losses.py`, for `DPOLoss`.

## Repository integration

Three things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .cosyvoice_v2 import *` line in its alphabetical
  list, after `cosyvoice_v1`.
- `PROJECT.md`'s status table still records this model as not started.
- `pyproject.toml` still declares an `onnx` extra holding `onnxruntime`, `onnxruntime-gpu` and
  `onnx`. Nothing in these three folders imports any of them any more, and no other model folder does
  either, so that extra and its entry in `all` can go.
