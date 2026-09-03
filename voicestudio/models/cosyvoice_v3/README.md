# CosyVoice v3

CosyVoice v3 keeps v2's three network layout and replaces two of the three. A Qwen2 0.5B decoder
turns text into 6561 supervised semantic speech tokens at 25 Hz. A conditional flow matching model
turns those tokens into an 80 bin mel spectrogram at 24 kHz, and it has **no encoder at all**: a
lookahead convolution runs over the embedded tokens, a repeat interleave carries them to the mel
frame rate, and a diffusion transformer predicts the vector field. A causal HiFTNet vocoder turns the
mel spectrogram into a waveform, padding every convolution on one side only so a chunk can be
rendered before the frames that follow it exist.

Against v2 the language model moves its start of sequence, end of speech and task vectors out of a
separate two entry table and into the speech token table, which grows from `6561 + 3` to
`6561 + 200`. Because that is the only change to how a sequence is built, every packing routine and
the decode loop are inherited from v2 rather than copied.

The three networks are trained separately upstream and are released as three separate `.pt` files.

Original model and code: https://github.com/FunAudioLLM/CosyVoice

## Usage

```python
from voicestudio.models.cosyvoice_v3 import CosyVoiceV3ForConditionalGeneration, CosyVoiceV3Processor

model = CosyVoiceV3ForConditionalGeneration.from_pretrained("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
processor = CosyVoiceV3Processor.from_pretrained("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")

inputs = processor(text="<|endofprompt|>The quick brown fox jumps over the lazy dog.")

waveform = model.generate(input_ids=inputs.input_ids, speaker_embedding=speaker_embedding)
```

v3 requires the end of prompt token, id 151646, to appear in the text or the prompt text, and
`generate` raises if it does not. Upstream uses it to separate an instruction prefix from the text to
synthesize, as in `"用四川话说<|endofprompt|>扁担长..."`, so an empty instruction leaves the marker at
the front. The same conditioning modes as v2 are available, and passing a speaker embedding alone is
upstream's sft mode, the only one that needs no reference waveform.

The released directory holds one `.pt` file per network rather than a single checkpoint, beside the
`CosyVoice-BlankEN` directory the language model is built from. `from_pretrained` reads that layout
directly: it merges the three files under the name of the submodule each belongs to, and the
`WeightRenaming` rules registered in `modeling_cosyvoice_v3.py` turn upstream's module names into
this model's as the checkpoint loads. The processor takes the same repository id and picks up the
text tokenizer, the speech tokenizer and the speaker encoder.

## Training

`CosyVoiceV3ForConditionalGeneration.forward(labels=...)` returns the language model objective only,
because upstream trains the three networks one at a time. The other two objectives are on the
submodules.

- **Language model.** `LabelSmoothingLoss` over `speech_vocab_size + 200` classes with
  `smoothing=0.0` and `normalize_length=True`. The two sequence layouts are v2's unchanged, the
  unistream one and the interleaved bistream one at `mix_ratio` `[5, 15]`, and upstream draws between
  them per sample with probability one half when the speech to text ratio allows; `forward` takes
  that draw as its `bistream` argument so it stays deterministic. The only difference from v2 is that
  the control vectors come out of the speech token table.
- **Flow matching.** `CosyVoiceV3FlowModel.forward` returns
  `CosyVoiceV3ConditionalCFM.compute_loss`, which is v1's optimal transport conditional flow matching
  loss unchanged, with v2's unified streaming training on top: upstream draws `streaming` once per
  batch with probability one half and threads it into the estimator.
- **Vocoder.** `CosyVoiceV3HiFTGenerator.forward` returns the waveform and the predicted f0.
  `compute_loss`, inherited from v1, scores the two terms that need no discriminator, at the mel
  resolution `CosyVoiceV2Config` sets: `cosyvoice3.yaml` gives `mel_spec_transform1` the same
  `n_fft` 1920, `hop_size` 480 and `win_size` 1920 at 24 kHz that `cosyvoice2.yaml` does, so v3
  needs no override of its own. The rest of the objective is not implemented; see "Not carried over
  from upstream".

Upstream freezes nothing, for the same reason as v1 and v2: the only `requires_grad` write in the
tree is `Snake.alpha.requires_grad = alpha_trainable`, whose default is `True`, and the only
`.eval()` on a training path belongs to the direct preference optimization recipe's separate
reference model.

## Lineage

The base is `voicestudio/models/cosyvoice_v2/`, mirroring upstream, where `CosyVoice3LM` subclasses
`Qwen2LM` and `CosyVoice3Model` subclasses `CosyVoice2Model`. What is inherited and what is not:

| Class | Base | Why |
|---|---|---|
| `CosyVoiceV3Config` | `CosyVoiceV2Config` | Same flat field set with the v3 geometry, plus an `estimator_config` sub configuration. The flow encoder fields are inherited and unused, since v3 has no flow encoder. |
| `CosyVoiceV3SpeechTokenLM` | `CosyVoiceV2SpeechTokenLM` | `llm_embedding` becomes a read only property returning the speech token table, which is exactly what v3 does, so both training layouts and the decode loop are inherited rather than copied. Only the table widths and the head losing its bias are overridden. |
| `CosyVoiceV3ConditionalCFM` | `CosyVoiceV2ConditionalCFM` | Same objective, same guided Euler solver, same fixed initial noise; only the estimator changes. |
| `CosyVoiceV3ResBlock` | `CosyVoiceV1ResBlock` | Same residual algebra with left padded convolutions. |
| `CosyVoiceV3HiFTGenerator` | `CosyVoiceV1HiFTGenerator` | Same synthesis filter and inverse transform head, rebuilt from causal convolutions. |
| `CosyVoiceV3FeatureExtractor`, `CosyVoiceV3Processor` | the v2 pair | The mel geometry is identical; only the tokenizer's added vocabulary and the speech tokenizer file name change. |
| `CosyVoiceV3GenerationMixin` | `CosyVoiceV2GenerationMixin` | The language model loop is inherited. `token2wav` is written out, because v3 carries no vocoder cache and re-renders the whole accumulated mel each chunk. |

### The estimator comes from f5_tts, and this is where the sibling map paid off

PROJECT.md's sibling inheritance map named `FeedForward` and `TimestepEmbedding` against f5_tts and
gated the check on the CosyVoice migration. v1 rejected both with concrete reasons, because v1's
estimator is a Matcha one dimensional UNet. **v3 is the model the map was pointing at**, and the
rejection is reversed: `cosyvoice/flow/DiT/dit.py` and `modules.py` are the F5-TTS diffusion
transformer, and `F5TTSConfig`'s stock defaults are v3's geometry exactly, `hidden_size` 1024,
`num_hidden_layers` 22, `num_attention_heads` 16, `head_dim` 64, `ff_mult` 2, `dropout` 0.1.

Inherited unchanged: `F5TTSDecoderLayer`, `F5TTSAdaLayerNormFinal`, `F5TTSTimestepEmbedding`,
`F5TTSSinusPositionEmbedding` through it, `F5TTSAttention` and `F5TTSRotaryEmbedding`. The estimator
configuration is an actual `F5TTSConfig`, so the reuse is checkable rather than asserted.

Two classes could not be inherited. `CosyVoiceV3CausalConvPositionEmbedding` is left padded and holds
two separate convolution sequences where `F5TTSConvPositionEmbedding` is centre padded and holds one,
so the parameter paths differ. `CosyVoiceV3InputEmbedding` concatenates a speaker vector that f5_tts
has no equivalent of, in the order noised speech, conditioning speech, encoded tokens, speaker, which
is not f5_tts's order.

**`pe_attn_head=1` is the setting that makes this correct, and it is not obvious.** Upstream calls
x-transformers' `apply_rotary_pos_emb` on the **unreshaped** `(batch, length, 1024)` projection, and
that function rotates only the leading `freqs.shape[-1]` channels, which is `dim_head`, 64. After the
head reshape those 64 channels are the first attention head and nothing else, so the rotary reaches
one head of sixteen. Measured against upstream's own `DiT`, the estimator with the rotary on **every**
head differs by **3.356** where head 0 only differs by 4.702e-03, three orders apart, so the setting
does real work and matching at it is a result rather than a coincidence.

The bridge between the two rotary conventions is exact. x-transformers lays its frequencies out as
interleaved pairs and rotates with `rotate_half` over `(d r)`; llama uses the half split layout.
f5_tts's `deinterleave_head_dim` is precisely the permutation between them, and since query and key
both receive it the attention dot product is invariant. Measured: with the rotary applied by hand
both ways, the fifteen heads that receive none differ by **exactly 0.0** and head 0 by 7.324e-04 on
scores of magnitude 8323, a relative 8.8e-08, which is below one float32 unit in the last place.

Other entries on the sibling map are rejected for v3 for the same reasons as v1 and v2: `Encoder` and
`EncoderLayer` against prompt_tts_pp are moot, since v3 has no encoder; `SourceModule` against
prompt_tts_pp, because v3 uses upstream's interpolating causal generator, which is a third variant
again; and `Snake` against bigvgan, because `BigVGANSnakeActivation` unconditionally resamples
through a Kaiser windowed sinc and `anti_alias_ratio=1` is not an identity.

## The estimator's disagreement with upstream, characterised

The estimator is **not bit exact** against upstream and cannot be, and the reason is entirely in two
library behaviours rather than in the model. Both are places where `transformers` forces float32
inside an otherwise higher precision computation, which is invisible at float32 and visible above it.

- **`eager_attention_forward` computes the softmax in float32.** Its last lines are
  `nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`. Upstream calls
  `F.scaled_dot_product_attention`, which uses the input dtype.
- **`LlamaRotaryEmbedding.forward` computes its frequencies in float32**, under an explicit autocast
  disable, and casts only the result. x-transformers computes them in whatever dtype its buffer
  carries.

Crossing the two in float64, where rounding is about 1e-16 and anything visible is a real difference
in operations:

| attention implementation | rotary | relative difference from upstream |
|---|---|---|
| eager | transformers | 7.710e-05 |
| eager | matched to upstream | 7.921e-05 |
| sdpa | transformers | 1.606e-06 |
| sdpa | matched to upstream | **1.570e-12** |

Five orders of collapse, with the eager softmax as the large term and the rotary as the small one.
The worst single layer falls from 2.678e-04 to 6.834e-15. Neither alone accounts for the difference,
which is why removing only the rotary looked like a failed hypothesis.

**None of this is visible at float32**, which is what the model is run at. Measured against a float64
ground truth, upstream's own float32 sits at 2.745e-04 relative and this implementation at
2.391e-04, an error ratio of **0.871**: neither is the wrong one, and if anything the migrated path
is marginally closer to the truth. The reason a 1e-07 seed reaches 1e-04 is amplification, roughly a
thousandfold through 22 residual layers whose activations grow from 333 to 1436.

There is no bit exactness claim here, and the right statement about the v3 estimator is a relative
agreement of that order.

### A trap worth knowing about

The estimator is a plain `nn.Module`, not a `PreTrainedModel`, so nothing runs the attention
implementation autoselection over its sub configuration and `_attn_implementation` stays `None`,
which dispatches to the eager path and its float32 softmax. `CosyVoiceV3Config` therefore sets it
explicitly. Anyone nesting a `transformers` sub configuration under a non `PreTrainedModel` container
has the same hazard.

## Upstream does not run correctly on transformers 5

Both defects v2 found reproduce on v3, since `CosyVoice3LM` inherits the methods that carry them.
Neither is an omission in this folder and neither is a choice this migration made.

- **`Qwen2Encoder.forward_one_step` mishandles its attention mask.** It passes a padding mask
  covering the current step alone while the cache already holds the prefix. Measured for v2 against
  an uncached full sequence reference, upstream's cached logits are wrong by 9.762 where this
  implementation is at 9.5e-06. On v3, from the same seed and the same text, stock upstream produces
  **52** tokens and this implementation **97**; with the one step mask dropped the two are
  **identical, 97 against 97**, and both end on a stop token. This model deliberately does not
  reproduce upstream's sampled output on transformers 5, because reproducing it would mean
  reproducing the defect.
- **`Qwen2Encoder.__init__` leaves the backbone dtype to `from_pretrained`.** Under upstream's pinned
  4.51.3 that gives float32; under transformers 5 it follows the checkpoint and gives bfloat16 while
  the sibling modules stay float32, so upstream's own module is internally mixed and raises. This
  model is uniformly float32, confirmed by survey.

## Deliberate deviations from upstream

Three, all of which reproduce the computation and change only a side effect or a source of
irreproducibility.

- **The 29 MB `uv` noise tensor is not built.** Upstream's `SourceModuleHnNSF` draws it and returns
  it, and the vocoder discards it with `s, _, _`. It never reaches the output, so not building it
  removes a meta device hazard by construction rather than by fixing it.
- **The fixed sine tensors are seeded.** Upstream draws the per harmonic phase offsets and a 259 MB
  noise tensor from whatever global random state is current when the module is built, which makes its
  v3 vocoder **irreproducible across processes**. This is a property of upstream, not a defect in the
  migration. Here they come from a private generator seeded with `source_noise_seed`, built lazily on
  first use and outside inference mode. One consequence has to be disclosed: a parity comparison
  cannot be made by seeding alone, so the harness points the migrated generator at upstream's two
  tensors, and every vocoder number below is measured under that substitution.
- **`inference` restores the f0 predictor's dtype.** Upstream casts it to float64 and leaves it cast,
  so calling `inference` permanently changes the module. Measured: after upstream's `inference` its
  predictor is float64 and its own training path `forward` raises
  `RuntimeError: Input type (float) and bias type (double) should be the same`; after the migrated
  `inference` the predictor is float32 and `forward` succeeds. The computation is identical, with the
  f0 contour at **exactly 0.0**.

## Verification

Everything below ran on CPU in float32 against the real
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` weights, in the project venv except the two runs that close
this section, which needed `onnxruntime` and ran on a Colab T4 through the `colab` CLI with the model
on CPU.

**Checkpoint coverage.** 951 source tensors convert to 949 over **859,185,455 parameters**,
506,148,480 under `llm`, 332,257,088 under `flow` and 20,779,887 under `hift`, with zero MISSING and
zero UNEXPECTED keys. Two tensors are dropped and each is accounted for rather than assumed:
`llm.model.lm_head.weight`, shown by a bit identity search to equal `llm.model.embed_tokens.weight`,
which is the tie the Qwen2 configuration declares; and
`decoder.estimator.rotary_embed.inv_freq`, x-transformers' persistent frequency buffer, which the
migrated estimator computes for itself. That second one has no bit identical counterpart, so it was
compared directly instead: **0.0 against the frequencies the migrated estimator computes, and 0.0
again on the cosines derived from each**.

Every one of those source tensors is bit identical to the model tensor it lands on, and no model
tensor is left unfilled, so the registered `WeightRenaming` rules account for the checkpoint exactly
rather than merely producing a clean report. `save_pretrained` followed by `from_pretrained` returns
the same tensors with zero missing, unexpected or mismatched keys, which is what keeps the reverse
of the mapping honest.

**Tokenizer.** Upstream's special token list takes the released tokenizer from 151646 to 151924
entries against 151936 embedding rows, so all 278 additions land inside the table with twelve rows to
spare and nothing is resized. Upstream adds the same list, with its own comment that the rows stay
randomly initialised. There is an independent check that the 281 entry transcription is correct in
both content and order: upstream hardcodes 151646 for `<|endofprompt|>` in an assertion, and that id
is only right if this exact list is added in this exact order to this exact tokenizer. It lands on
151646.

**Numeric parity against upstream's own classes.** Reported as `max|diff|` with the relative figure,
since the estimator's activations make an absolute number unreadable on its own:

| Component | Upstream class | difference |
|---|---|---|
| Rotary frequency buffer and its cosines | `RotaryEmbedding` | 0.0 |
| Lookahead layer, with and without context | `PreLookaheadLayer` | 0.0 |
| Fixed initial noise | `CausalConditionalCFM.rand_noise` | 0.0 |
| Flow estimator, full and chunked attention | `DiT` | 9.582e-04 and 9.327e-04 relative |
| Euler solver, full and chunked | `CausalConditionalCFM` | 7.178e-04 and 3.313e-04 relative |
| Flow matching loss, full and chunked | `CausalConditionalCFM.compute_loss` | 1.257e-05 and 1.198e-05 relative |
| Flow inference, all four streaming and finalize settings | `CausalMaskedDiffWithDiT.inference` | 1.024e-04 to 2.914e-04 relative |
| f0 predictor, both finalize settings | `CausalConvRNNF0Predictor` | 0.0 |
| Sine generator waves and voiced mask | `SineGen2` | 0.0 |
| Neural source filter | `SourceModuleHnNSF` | 0.0 |
| Vocoder excitation, both finalize settings | `CausalHiFTGenerator.inference` | 0.0 |
| Vocoder waveform, both finalize settings | `CausalHiFTGenerator.inference` | 1.149e-05 relative |
| Vocoder f0 and waveform, training entry point | `CausalHiFTGenerator.forward` | 0.0 and 1.475e-05 relative |
| Language model text embedding and logits | `Qwen2Encoder` | 0.0 |
| Language model loss, unistream and bistream | `CosyVoice3LM.forward` | 0.0 |

Everything that can be bit exact is. The vocoder's 1.1e-05 is the analysis window floor v1 pinned to
`scipy.signal.get_window("hann", 16, fftbins=True)` against `torch.hann_window(16, periodic=True)`,
and the excitation and f0 being exactly 0.0 places it at the short time Fourier transform and nowhere
earlier. The flow differences are the characterised estimator divergence: `flow inference` at
2.914e-04 relative sits on the 2.391e-04 and 2.745e-04 measured against float64 truth, and the ten
Euler steps **reduce** the estimator's own 9.58e-04 rather than compounding it.

**Why those differences are rounding and not bias.** The flow matching loss is a scalar reduction
over 80 by 120 elements. If the per element differences are independent and zero mean they should
average down by the square root of the count, and a systematic bias would carry through at close to
full size. The prediction and the measurement:

```
9.582e-04 / sqrt(9600) = 9.78e-06        measured 1.257e-05        ratio 1.29
```

Within a factor of 1.3. The lookahead layer at exactly 0.0 pins the divergence to the estimator and
nothing before it, and the fixed initial noise at 0.0 rules out the solver's starting point.

**The autoregressive loop** was checked as a loop. Against upstream with the one step mask corrected,
`CosyVoice3LM.inference` and `generate_speech_tokens` produce the **same 97 speech tokens** from the
same seed, and both end on a stop token rather than the length cap. Against stock upstream they
differ, 52 against 97; see the transformers 5 section. The silence thinning was checked against
upstream's rule on a scripted stream: 19 tokens in, 15 out, a nine token silent run cut to five and a
four token run left alone.

**Training objectives.** Both language model losses match upstream at 0.0 on both sequence layouts.
Gradients from that objective reach 292 parameter tensors, all under `llm`, none under `flow` or
`hift`.

**Round trip through the hub format.** This found a real defect, and it is worth reading as two
separate results rather than one.

*The defect.* `transformers`' rotary embedding computes its frequencies in its constructor and
registers them as two **non persistent** buffers, `inv_freq` and `original_inv_freq`. Under the meta
device initialisation transformers 5 uses, a non persistent buffer is materialised as uninitialised
memory rather than by rerunning the computation that produced it, and `is_meta` is `False`
afterwards, so checking for meta tensors does not catch it. Measured on a reloaded model, the two
buffers came back holding values of **1.77e21** and **5.24e22**, and flow inference differed from the
model it had been saved from by **2.744, a relative 2.5e-01**, which is a materially different output
rather than a precision difference.

The cause is structural and it generalises: `CosyVoiceV1PreTrainedModel._init_weights` enumerates its
own module types and stops, with no `else: super()._init_weights(module)` fallback, and v2 and v3
inherit it. **Any `_init_weights` in this repository that enumerates module types without falling
through to `super()` has the same hole.** It only bites when a module carrying constructor computed
non persistent buffers sits under a plain `nn.Module` rather than a nested `PreTrainedModel`, which
is why v2 is unaffected: its only rotary lives inside `Qwen2Model`, which initialises itself, while
v3's estimator rotary lives inside `CosyVoiceV3ConditionalDecoder`, a plain module reachable only
through `_init_weights`. `CosyVoiceV3PreTrainedModel._init_weights` now rebuilds both buffers from
the configuration, recomputing rather than copying and mirroring the constructor's own selection of
the initialiser including `attention_scaling`. A blanket fallback to `PreTrainedModel._init_weights`
was **not** added, because it would reach v1's implementation, which is inert for unknown types, and
making it reach the base implementation would re-initialise types v1 deliberately leaves alone.
v1 and v2 are one nested plain module away from the same defect and neither is currently reached by a
case that needs it; that is recorded for a human rather than changed here.

*The residual, after the fix.* Every parameter, every buffer including both rotary ones, every tensor
held as a plain attribute, the rotary cosines, the vocoder waveform and the sampled speech token
sequence are all **bit identical**. Flow inference is **not**: it agrees to `1.013e-03`, a relative
**9.314e-05**. This is weaker than v2's `0.000e+00` and should be read as weaker.

The cause is storage layout, demonstrated rather than argued. `from_pretrained` memory maps its
weights and a freshly built model allocates its own, and on layer 19's query projection both are
contiguous with the same stride, dtype and storage offset while the base pointers differ in
alignment, **`data_ptr mod 64` of 0 against 56**. Different alignment leads the matmul to a different
blocking and so a different accumulation order, at roughly float32 epsilon per operation, which the
same thousandfold amplification through 22 residual layers carries to 1e-04. Loading the reloaded
weights into a third, freshly allocated model as contiguous clones, changing the layout and nothing
else, reproduces the built model at **exactly 0.0**, while the memory mapped one stays at 1.577e-03:

| comparison | estimator | flow inference |
|---|---|---|
| built against reloaded | 1.577e-03 | 1.013e-03 |
| built against a contiguous clone of the reloaded weights | **0.000e+00** | **0.000e+00** |
| reloaded against that same clone | 1.577e-03 | |

Two candidate causes were eliminated by measurement rather than by argument. The attention
implementation is `sdpa` on both sides, so the eager float32 softmax path is not involved. Both
models are in eval mode, `training` `False` on the model and on the estimator, so dropout at p=0.1 is
not firing on one side; `same model twice: 0.000e+00` had already excluded that, since dropout in
train mode consumes randomness and two consecutive calls could not have agreed bit for bit.

Three configuration leaves differ after a round trip and none has any effect: `_name_or_path` is `''`
against the save path, and `estimator_config.dtype` and `text_config.dtype` are `None` against
`'float32'`, which is serialisation recording the dtype that was already in force. Both models'
parameters are `torch.float32` throughout, measured. This is the mirror image of the Parler-TTS trap
PROJECT.md records rather than an instance of it: there a sub configuration dtype created a split,
here it records a dtype that is uniform on both sides.

**Generated speech, transcribed back.** Two utterances were synthesized and transcribed with
`facebook/wav2vec2-base-960h`. Model and processor were both built by
`from_pretrained("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")` with no conversion step before it, and the
load report carried no missing, unexpected or mismatched key over 859,185,455 parameters.

| Prompt text | Transcript |
|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` |
| `She sells sea shells by the sea shore.` | `SHE SELLS SEASHELLS BY THE SEASHORE` |

Both are word for word, with the connectionist temporal classification decoder merging the two
compounds of the second. The waveforms are 2.84 s at RMS 0.1430 and 2.44 s at RMS 0.1134.

The conditioning was upstream's sft mode, `sft, speaker embedding only, no reference waveform`,
selected by the geometry test rather than hardcoded: a 192 dimensional campplus speaker vector taken
from the CosyVoice 1 speaker table, no prompt speech tokens and no prompt mel spectrogram. The vector
is in distribution because `campplus.onnx` is byte identical at 28,303,423 bytes across v1, v2 and
v3. The text carried the end of prompt token at the front, which is what upstream's assertion
requires and what an empty instruction prefix produces.

That exercises the language model, the flow matching model and the vocoder end to end, and it is a
meaningful test of all three because the v3 language model does not read the speaker embedding at
all, so every word came out of the text path.

That run predates the ONNX port and used upstream's sft mode only. Zero shot voice cloning from a
reference clip, which is the path that derives speech tokens, a mel spectrogram and a speaker
embedding from a waveform, is exercised by the run recorded under "The ported ONNX components"
below.

**The ported ONNX components, against the graphs they replace.** Three LibriSpeech clips of 2.9, 2.5
and 6.5 seconds went through `onnxruntime` and through the PyTorch port, on the same features. The
speaker embedding, 192 dimensions reaching 2.83 in magnitude, differs by at most 8.106e-06. The
speech tokenizer produced 580 token ids over the three clips and every one is identical; its encoder
output, reaching 8.61 in magnitude, differs by at most 1.627e-05. That residual is float32
reassociation, not a difference in the computation.

**Zero shot voice cloning, end to end.** A 5.86 s LibriSpeech clip, which
`facebook/wav2vec2-base-960h` transcribes as `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND
WE ARE GLAD TO WELCOME HIS GOSPEL`, became 146 speech tokens and a 192 dimensional speaker embedding,
and both were passed to the language model and to the flow matching model together with the prompt
mel spectrogram and the clip's transcript, with the end of prompt marker at the front of the text.

| Asked | Heard back | Waveform |
|---|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` | 4.00 s at 24000 Hz, RMS 0.0787, peak 0.5275 |
| `She sells seashells by the seashore.` | `SHE SELLS SEASHELLS BY THE SEASHORE` | 3.00 s at 24000 Hz, RMS 0.0754, peak 0.5809 |

Both are word for word.

The prompt transcript has to be written the way a caller would write it. Passing the same clip's
LibriSpeech transcript verbatim, upper case and unpunctuated, collapsed the same two calls to a 1.52 s
`LAZY DOG` and to a second waveform at RMS 0.0025 that transcribes as nothing at all. Lower casing it
and adding the full stop is the only difference between that and the table above. v1 and v2 generate
the whole sentence either way, so this sensitivity is v3's.

## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **The vocoder's adversarial objective.** `cosyvoice/hifigan/hifigan.py` is unchanged from v1 and
  v2: `generator_loss` plus 2.0 times feature matching plus 45 times mel reconstruction plus 1.0
  times `tpr_loss` at tau 0.04 plus an L1 on the f0, with the discriminator optimized in an
  alternating turn. `MultipleDiscriminator` and the three losses from `matcha.hifigan.models` are not
  implemented, so the vocoder cannot be trained the way upstream trained it. Same shape as the open
  Vocos item; needs a human.
- **`llm.rl.pt`.** The released v3 directory ships a second language model checkpoint, 2,024,682,701
  bytes, alongside `llm.pt`. It is a reinforcement learning tuned model. Nothing here reads it and no
  decision has been taken about it.
- **Direct preference optimization.** `Qwen2LM.forward_dpo` and `DPOLoss` need a second model
  instance and a preference batch, so they do not fit inside a single `forward`. Still open, with the
  same note as v2 that upstream averages its log probabilities over the positions where the target
  **is** `IGNORE_ID`, which reads like a sign error.
- **The text frontend**, and its connection to the added vocabulary. Upstream's `text_normalize` runs
  `ttsfrd` or `wetext`, expands numbers with `inflect`, splits sentences, and for v3 emits the ARPAbet
  and pinyin markup that the 278 added tokens exist for. None of that is implemented, so those 278
  embedding rows, which upstream notes are randomly initialised in any case, are never reached. Text
  with digits, abbreviations, more than one sentence or phoneme markup does not behave the way
  upstream does.
- **`inference_bistream`.** The interleaved training layout is implemented; the streaming input text
  inference path is not.

## Repository integration

Three things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .cosyvoice_v3 import *` line in its alphabetical
  list, after `cosyvoice_v2`.
- `PROJECT.md`'s status table still records this model as not started, and its sibling inheritance
  map still lists the f5_tts entries as gated on this migration rather than resolved by it.
- `pyproject.toml` still declares an `onnx` extra holding `onnxruntime`, `onnxruntime-gpu` and
  `onnx`. Nothing in these three folders imports any of them any more, and no other model folder does
  either, so that extra and its entry in `all` can go. Otherwise this folder needs nothing: it
  imports `voicestudio.models.f5_tts`, which is already in the repository, and nothing new from
  outside it.
