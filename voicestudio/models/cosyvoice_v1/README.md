# CosyVoice v1

CosyVoice v1 synthesizes speech in three stages. An autoregressive language model turns Whisper
text tokens, an utterance level speaker embedding and an optional speech token prompt into a
sequence of 4096 supervised semantic speech tokens at 50 Hz. A conditional flow matching model
turns those tokens into an 80 bin mel spectrogram, conditioned on the same speaker embedding and
on the prompt's mel spectrogram, with a one dimensional UNet estimator and a fixed step Euler
solver under classifier free guidance. A HiFTNet vocoder turns the mel spectrogram into a 22050 Hz
waveform through a neural source filter and an inverse short time Fourier transform head.

The three networks are trained separately upstream and are released as three separate `.pt` files.

Original model and code: https://github.com/FunAudioLLM/CosyVoice

## Usage

```python
from voicestudio.models.cosyvoice_v1 import CosyVoiceV1ForConditionalGeneration, CosyVoiceV1Processor

model = CosyVoiceV1ForConditionalGeneration.from_pretrained("FunAudioLLM/CosyVoice-300M-SFT")
processor = CosyVoiceV1Processor.from_pretrained("FunAudioLLM/CosyVoice-300M-SFT")

inputs = processor(text="The quick brown fox jumps over the lazy dog.")
prompt = processor.get_speaker("英文女")

waveform = model.generate(
    input_ids=inputs.input_ids,
    speaker_embedding=prompt.speaker_embedding,
    flow_prompt_speech_token_ids=prompt.prompt_speech_token_ids,
    prompt_speech_feat=prompt.speech_feat,
)
```

`get_speaker` reads the `spk2info.pt` at the processor's `speaker_info_path`. `CosyVoice-300M-SFT`
and `CosyVoice-300M-Instruct` ship one holding seven speakers; the base model does not, and the
only table that covers it is the one in the stale `model-scope/CosyVoice-300M` mirror, which
carries a speech token sequence and a prompt mel per speaker rather than an embedding alone.

`prompt_speech_token_ids` conditions the language model and `flow_prompt_speech_token_ids`
conditions the flow matching model. Passing both, together with `prompt_input_ids`, is upstream's
zero shot mode; passing only the flow one is upstream's cross lingual mode; passing
`source_speech_token_ids` bypasses the language model and performs voice conversion.

`processor.normalize_text(text)` is upstream's text front end: it rewrites the string and returns
the sentences upstream synthesizes one at a time. `processor(text=..., normalize=True)` applies the
rewriting to a single sequence without splitting.

`processor.compute_f0(audio, sampling_rate)` returns the `pitch_feat` the vocoder objective
regresses onto, and `model.hift.compute_loss(mel, waveform, pitch_feat)` scores the vocoder against
it. See "Training".

`CosyVoiceV1Tokenizer` is the text tokenizer of the `CosyVoice-300M-25Hz` release, built from that
release's `multilingual_zh_ja_yue_char_del.tiktoken`. The three released 50 Hz checkpoints do not
use it; they use Whisper's vocabulary, which the processor carries as a `WhisperTokenizer`.

The released directories hold one `.pt` file per network rather than a single checkpoint.
`from_pretrained` reads that layout directly: it merges the three files under the name of the
submodule each belongs to into a directory under `HF_HOME`, keyed on the repository and the commit
it resolved to, and the `WeightRenaming` rules registered in `modeling_cosyvoice_v1.py` turn
upstream's module names into this model's as that directory loads. Later loads reuse it, and resolve
nothing but the `cosyvoice.yaml` that names the revision. The processor takes the same repository id
and picks up the speech tokenizer, the speaker encoder and, where the release ships one, the speaker
table. Once the merge is written, the three `.pt` files are dropped from the `huggingface_hub` cache.
Nothing else is: the speech tokenizer graph the processor reads sits in the same directory and stays
there, so the model and the processor can be built in either order.

## Training

`CosyVoiceV1ForConditionalGeneration.forward(labels=...)` returns the language model objective
only, because upstream trains the three networks one at a time with three separate runs of
`cosyvoice/bin/train.py`. The other two objectives are on the submodules:

- **Language model.** `LabelSmoothingLoss` over `speech_vocab_size + 1` classes with
  `smoothing=0.0` and `normalize_length=True`, that is a cross entropy divided by the number of
  unmasked targets. Targets are built by `build_speech_token_labels`: `-1` on the start of
  sequence step, on the speaker embedding step and on every encoded text step, then the utterance's
  speech tokens, then the end of speech token `speech_vocab_size`. The speech tokens are teacher
  forced through `speech_embedding` and come from the speech tokenizer, not from the waveform.
  Upstream also reports a token accuracy, which `CosyVoiceV1Output.accuracy` carries and which is
  not part of the loss.
- **Flow matching.** `CosyVoiceV1FlowModel.forward` returns a single term,
  `CosyVoiceV1ConditionalCFM.compute_loss`: optimal transport conditional flow matching with
  `sigma_min=1e-06`, one timestep per sample drawn from `U(0, 1)`,
  `y = (1 - (1 - sigma_min) t) z + t x1` and target `u = x1 - (1 - sigma_min) z`, scored with
  `mse_loss(pred * mask, u * mask, reduction="sum") / (mask.sum() * n_feats)`. With probability
  `training_cfg_rate` (0.2) per sample the encoder output, the speaker embedding and the
  conditioning mel are all zeroed. The conditioning mel is, with probability one half per sample, a
  prefix of the ground truth mel of a length drawn uniformly from `0` to `0.3 * num_frames`, and
  zeros otherwise. The target mel comes from the waveform.
- **Vocoder.** `CosyVoiceV1HiFTGenerator.compute_loss` returns the two terms of upstream's
  generator turn that need no discriminator. Upstream's `HiFiGan.forward_generator` sums five:
  `generator_loss` over the discriminator outputs, that is `mean((1 - d) ** 2)` per output;
  `feat_match_loss_weight` (2.0) times `feature_loss`, which is itself twice the sum of
  `mean(|f_real - f_generated|)` over every feature map, so four times that sum in the total;
  `multi_mel_spectral_recon_loss_weight` (45) times `mel_loss`, an L1 distance between the log mel
  spectrograms of the real and the generated waveform, summed over the `mel_spec_transform` list,
  which the released recipe fills with a single `matcha.utils.audio.mel_spectrogram` at `n_fft`
  1024, 80 mel bins, hop 256, window 1024, `fmin` 0, `fmax` `null` and `center` `False`;
  `tpr_loss_weight` (1.0) times `tpr_loss` at `tpr_loss_tau` 0.04; and an L1 loss between the
  predicted f0 and `pitch_feat`. The third and the fifth are implemented here, as
  `CosyVoiceV1VocoderOutput.mel_loss` and `.f0_loss`, and their sum is `.loss`. Note that the mel
  the objective is measured over is not the mel the model consumes: this one runs up to the Nyquist
  frequency, the model's stops at 8000 Hz. The other three, and the whole discriminator turn, need
  a discriminator that is not implemented; see "Not carried over from upstream".

Upstream freezes nothing. The only `requires_grad` assignment in upstream's tree is
`Snake.alpha.requires_grad = alpha_trainable` in `cosyvoice/transformer/activation.py`, whose
default is `True` and which the vocoder never overrides, and the only `.eval()` on a training path
is `self.ref_model.eval()` in `Executor`, which belongs to the CosyVoice 2 DPO recipe. Both freeze
bugs the project has hit elsewhere, a frozen module that still runs dropout because nothing called
`.eval()`, and a freeze that does not survive `from_pretrained` because transformers 5 rebuilds
`Parameter` objects, are therefore not reachable here: there is nothing to freeze. What was checked
instead is that each objective reaches only its own network. The language model loss puts gradient
on 401 parameters, all of them under `llm`, and none under `flow` or `hift`; the flow loss puts
gradient on 1185 parameters, all under `flow`, and none under `llm` or `hift`.

## Lineage

CosyVoice v1's flow decoder is `MaskedDiffWithXvec` with a `ConditionalCFM` head over a
`ConditionalDecoder`, which is a one dimensional UNet built from Matcha-TTS blocks. It is not
`CausalMaskedDiffWithDiT`; that class belongs to CosyVoice 3, and it is the reason the DiT lineage
does not apply to this model.

Against the sibling inheritance map:

- **`FeedForward` against f5_tts.** Rejected. `F5TTSFeedForward` is
  `Sequential(Sequential(Linear, GELU(approximate="tanh")), Dropout, Linear)`, while the estimator's
  feed forward is diffusers' `FeedForward(activation_fn="gelu")`, a `ModuleList` of
  `GELU(proj=Linear)` with `approximate="none"`, a dropout and a linear. The activation
  approximation differs, and so do the parameter paths: the checkpoint stores
  `ff.net.0.proj.weight` and `ff.net.2.weight`, which f5_tts's layout cannot express.
- **`TimestepEmbedding` against f5_tts.** Rejected for the same reason.
  `F5TTSTimestepEmbedding` bundles the sinusoidal embedding and a `Sequential(Linear, SiLU, Linear)`
  named `time_mlp` into one module, while the checkpoint stores a separate `time_embeddings` (no
  parameters) and `time_mlp.linear_1` / `time_mlp.linear_2`, which is diffusers' layout. The
  sinusoidal formula itself is identical between `F5TTSSinusPositionEmbedding` and Matcha's
  `SinusoidalPosEmb`, so only that parameter free piece could be inherited, and inheriting a
  parameter free ten line function across model folders buys nothing.
- **`Encoder` and `EncoderLayer` against prompt_tts_pp.** Rejected. prompt_tts_pp inherits
  `FastSpeech2ConformerAttention`, `FastSpeech2ConformerConvolutionModule` and
  `FastSpeech2ConformerMultiLayeredConv1d`, and writes its own `Encoder`/`EncoderLayer` on top.
  CosyVoice's encoders come from WeNet, not ESPnet's FastSpeech2 line, and diverge in three ways
  that reach the weights: the feed forward is two plain `Linear` layers (`w_1`, `w_2`) rather than
  `FastSpeech2ConformerMultiLayeredConv1d`'s two `Conv1d` layers, the checkpoint has neither a
  convolution module nor a macaron branch (`use_cnn_module: False`, `macaron_style: False`), and
  the input is a linear projection with no subsampling rather than a conformer input layer.
  What is left after removing the parts that differ is a plain pre norm block, so there is no
  shared surface worth inheriting.
- **`SourceModule` against prompt_tts_pp.** Rejected. `PromptTTSPPSourceModule` holds only a
  `Linear`, and its sine generation integrates phase with the interpolation trick BigVGAN uses.
  CosyVoice's `SineGen` at 22050 Hz is upstream's `sinegen_type='1'`, which builds harmonics
  directly and wraps with `2 * pi * (cumsum(f0 * i / sr) % 1)`, draws one uniform phase per
  harmonic in `[-pi, pi]` with the fundamental pinned to zero, and applies a `Tanh` module the
  checkpoint does not store but the forward does use. The two produce different waveforms from the
  same f0.
- **`Snake` against bigvgan.** Rejected, now that `voicestudio/models/bigvgan/` exists.
  `CosyVoiceV1Snake` is upstream's `Snake(channels, alpha_logscale=False)`, that is
  `x + (1 / (alpha + 1e-9)) * sin(x * alpha) ** 2` with a per channel `alpha` initialised to ones.
  `BigVGANSnakeActivation` computes that formula in the middle of an anti aliasing sandwich: it
  upsamples by `config.anti_alias_ratio`, applies the nonlinearity and lowpass filters back down
  with a Kaiser windowed sinc it registers as a buffer, and it takes a `BigVGANConfig` and a
  separate `beta`. CosyVoice's activation resamples nothing. Inheriting would mean handing it a
  configuration from another model, carrying a filter buffer this vocoder never uses and overriding
  `forward` to skip the two resampling steps, which is the whole body.
- **`ResBlock` against bigvgan.** Rejected. `BigVGANAmpBlock` holds a `ModuleList` of
  `BigVGANAmpLayer`, each with `conv1` and `conv2`, so its parameters sit at
  `layers.<n>.conv1.weight`. CosyVoice's checkpoint stores `convs1.<n>` and `convs2.<n>`, which
  that layout cannot express, and its layer applies the activation before the convolution while
  BigVGAN's applies it after. It also takes a `BigVGANConfig` for its activation.
- **`ResBlock` against transformers.** Inherited. `CosyVoiceV1ResBlock` now subclasses
  `HifiGanResidualBlock` from `transformers.models.speecht5.modeling_speecht5`, which is the class
  `fastspeech2_conformer`, `seamless_m4t`, `seamless_m4t_v2` and `vits` all carry as a
  `# Copied from`. Its `convs1` and `convs2` are built with the same names, the same
  `nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=(k * d - d) // 2)` and
  the same undilated second convolution, so the checkpoint keys are unchanged. What stays local is
  the pair of `Snake` module lists and the `forward` that puts one in front of each convolution
  where the base class calls `leaky_relu`.

The generator itself is not inherited from any of the four transformers HiFi-GANs.
`SpeechT5HifiGan` and `FastSpeech2ConformerHifiGan` are the same network: `conv_pre`, an
`upsampler` list, `resblocks`, a one channel `conv_post` closed by `tanh`, and `mean`/`scale`
buffers applied when `normalize_before` is set. `SeamlessM4THifiGan` is that network again and
`SeamlessM4TCodeHifiGan` wraps it in a unit embedding, a duration predictor and speaker and
language embeddings. CosyVoice's `HiFTGenerator` names its upsampling list `ups`, adds
`f0_predictor`, `m_source`, `f0_upsamp`, `source_downs`, `source_resblocks` and a
`reflection_pad`, injects the short time Fourier transform of the excitation at every upsampling
stage, and closes on `n_fft + 2` channels read as a magnitude and a phase for an inverse short time
Fourier transform rather than on one channel through `tanh`. Every one of those reaches the
checkpoint, so only the residual block is shared.

The mel spectrogram of the vocoder objective is `mel_spectrogram` and `dynamic_range_compression`
from `voicestudio/models/bigvgan/`, called at `centered=False`. BigVGAN's uncentered framing is
`matcha.utils.audio.mel_spectrogram` term for term: the same reflection padding by
`(n_fft - hop_length) // 2`, the same `torch.sqrt(real ** 2 + imaginary ** 2 + 1e-9)` magnitude,
the same Slaney scaled and Slaney normalized filterbank, and the same `log(clamp(x, min=1e-5))`.

Nothing in transformers-tts covers the missing pieces either: it ships no CosyVoice of any version,
no `SinusoidalPosEmb` / `Block1D` / `ResnetBlock1D` / `BasicTransformerBlock`, no conditional flow
matching wrapper, and no neural source filter.

## Dependencies

Upstream pins more than forty packages. What each turned into:

| Upstream dependency | What replaced it |
|---|---|
| `diffusers==0.29.0` | Removed. Its `GELU`, `Attention` with `AttnProcessor2_0`, `FeedForward` and `TimestepEmbedding` are the only pieces the estimator used, and each is a few dozen lines that now live in `modeling_cosyvoice_v1.py`. |
| `matcha-tts` | Removed. `SinusoidalPosEmb`, `Block1D`, `ResnetBlock1D`, `Downsample1D`, `Upsample1D` and `BASECFM` are inlined the same way. |
| `einops` | Removed. Three call sites, `pack([x, mu], "b * t")`, `rearrange(x, "b c t -> b t c")` and `repeat(spks, "b c -> b c t")`, are a concatenation, a transpose and an expand. |
| `omegaconf`, `hyperpyyaml` | Removed. The released `cosyvoice.yaml` is now `CosyVoiceV1Config`. |
| `torchdiffeq` | Not used by v1; the Euler solver is upstream's own `solve_euler`, kept inline. |
| `deepspeed`, `lightning`, `tensorboard` | Training harness, which `transformers` supplies. |
| `regex` | Removed. Its one call site is `is_only_punctuation`, whose `^[\p{P}\p{S}]*$` is `unicodedata.category(character)[0] in "PS"`. The two agree on all 149251 assigned code points. |
| `tiktoken` | Removed. `get_encoding` only used it to read a `base64 rank` file and to hold the ranks, so `CosyVoiceV1TikTokenConverter` subclasses transformers' own `TikTokenConverter` and reads the file with `base64` from the standard library. |
| `openai-whisper` | Replaced by `WhisperTokenizer` for text and `WhisperFeatureExtractor` for the speech tokenizer's mel. `whisper.tokenizer.Tokenizer`, which upstream's `get_tokenizer` returns, is `CosyVoiceV1Tokenizer`. |
| the English number reading package upstream pins at 7.3.1 | Removed. Upstream calls one method of it, `number_to_words`, to read an English digit run out; `number_to_words` in `processing_cosyvoice_v1.py` is that reading, inlined. See below. |
| `pyworld` | Lazily imported. See below. |
| `wetext`, `ttsfrd` | Text normalizers. See "Not carried over from upstream". |
| `onnxruntime` | Removed. See below. |
| `tensorrt`, `vllm`, `modelscope`, `gradio`, `fastapi`, `grpcio` | Serving paths, not part of the model. |
| `transformers==4.51.3` | Replaced by the `transformers-tts` 5.x fork this repository targets. |

`onnxruntime` was the last one, and it is gone. CosyVoice v1's speech tokenizer
(`speech_tokenizer_v1.onnx`, 522 MB) and speaker encoder (`campplus.onnx`, 28 MB) are the two
components upstream publishes as ONNX graphs, and both are now PyTorch modules.

The speaker encoder is `CosyVoiceV1SpeakerEncoder`, the CAM++ network, and it reads its authors'
own PyTorch release rather than the graph: `campplus_cn_common.bin` on `funasr/campplus`, which is
also on ModelScope as `iic/speech_campplus_sv_zh-cn_16k-common`. The graph is byte identical at
28,303,423 bytes across the v1, v2 and v3 directories, so one port covers all three.

The speech tokenizer has no PyTorch release at all. The section 2.3 search found
`xingchensong/S3Tokenizer` to be a reimplementation that downloads this same ONNX file and converts
it at load time, and the PyTorch tokenizer repositories that do exist (`ResembleAI/s3tokenizer-v2`,
`mlx-community/S3TokenizerV2`, `mlx-community/S3TokenizerV3`) are v2 and v3 only. So
`CosyVoiceV1SpeechTokenizer` reads its weights out of the graph itself, and
`weight_conversion.convert_speech_tokenizer` does that with about a hundred lines of protocol
buffer reading rather than a dependency on `onnx`: three fields of `TensorProto`, three of
`NodeProto`, and the initializer and node lists of `GraphProto`. Only the initializers that the
exporter could not fold kept a readable name; the rest became `onnx::MatMul_1532` and the like. They
are recovered from the graph, because every node that consumes an initializer names its output after
the module it was traced inside, so the module path and the operator together say which parameter an
initializer is. All 96 of v1's initializers map to exactly one parameter each and load with
`strict=True`.

One is imported lazily rather than depended on, and it was not added to `pyproject.toml`.
`CosyVoiceV1Processor.compute_f0` needs `pyworld`, because the target of the vocoder objective's f0
term is the WORLD harvest contour and no other estimator produces the same numbers; substituting one
would change the objective silently. `pyproject.toml` already declares `pyworld>=0.3.5` under the
`eval` extra, so this adds nothing new, but the base install does not carry it. It raises with an
explanation naming what is missing and what to do instead.

The English digit reading is not one of them. `CosyVoiceV1Processor.normalize_text` reads a digit run
out through `CosyVoiceV1NumberSpeller`, whose `number_to_words` is the inlined reading, so nothing has
to be installed for the English branch of the text front end. v2 and v3 inherit the same path, v3
through its own `spell_out_number_outside_markup`, which skips the spans holding a token of its added
vocabulary.

## Verification

Everything below ran in float32 against the real `FunAudioLLM/CosyVoice-300M` weights. The parity
table against upstream's own classes ran in the project venv while the upstream tree was still in
this folder; everything after it ran on a Colab T4 through the `colab` CLI, per CLAUDE.md H14, with
the model on CPU except the language model's generation loop.

**Checkpoint coverage.** 1813 tensors (401 from `llm.pt`, 1185 from `flow.pt`, 227 from `hift.pt`)
convert onto 436,053,808 parameters with zero MISSING and zero UNEXPECTED keys, so no source weight
goes unused. `f0_predictor.condnet` holds its five convolutions at indices 0, 2, 4, 6 and 8, as the
checkpoint confirms.

**Numeric parity against upstream's own classes.** The upstream tree was still in this folder when
these ran, so every migrated module was run side by side with the class it replaces, loaded from
the same weights. Reported as `max|diff|`:

| Component | Upstream class | max abs difference |
|---|---|---|
| Text encoder | `ConformerEncoder` | 0.0 |
| Language model backbone, full sequence | `TransformerEncoder` | 0.0 |
| Language model prefill | `TransformerEncoder.forward_chunk` | 0.0 |
| Language model one cached decode step | `TransformerEncoder.forward_chunk` | 0.0 |
| Flow encoder | `ConformerEncoder` | 0.0 |
| Length regulator, forward and inference | `InterpolateRegulator` | 0.0 |
| Label smoothing loss | `LabelSmoothingLoss` | 0.0 |
| Flow matching estimator | `ConditionalDecoder` | 0.0 |
| Euler solver, 10 steps with guidance 0.7 | `ConditionalCFM.forward` | 0.0 |
| Euler solver streaming cache, and a second chunk from it | `ConditionalCFM.forward` | 0.0 |
| Flow matching loss | `ConditionalCFM.compute_loss` | 0.0 |
| Flow model inference | `MaskedDiffWithXvec.inference` | 0.0 |
| f0 predictor | `ConvRNNF0Predictor` | 0.0 |
| Neural source filter | `SourceModuleHnNSF` | 7.5e-09 |
| Vocoder, from a shared excitation | `HiFTGenerator.decode` | 3.8e-06 |
| Vocoder, inference end to end | `HiFTGenerator.inference` | 3.2e-06 |
| Vocoder, training entry point | `HiFTGenerator.forward` | 1.1e-05 |
| Mel front end | `matcha.utils.audio.mel_spectrogram` | 0.0 |

The vocoder is the only component that is not bit exact. The cause is the analysis window:
upstream builds it with `scipy.signal.get_window("hann", 16, fftbins=True)` and this model with
`torch.hann_window(16, periodic=True)`, which are the same function rounded differently in
float32 and differ by 1.2e-07. That difference enters through the excitation's short time Fourier
transform and grows to 3.8e-06 on a signal whose maximum is 0.99.

The autoregressive loop was checked as a loop, not only as a forward pass: from the same seed,
`TransformerLM.inference` and `generate_speech_tokens` produced the same 480 speech tokens, token
for token.

**Training objectives.** With both random number generators pinned, the language model loss is
16.041545867919922 on upstream and 16.041545867919922 here, and the flow matching loss is
8.12406063079834 on both. Gradients land only where they should: the language model loss reaches
401 parameters, all under `llm`; the flow loss reaches 1185, all under `flow`.

**Generated speech, transcribed back.** Two utterances were synthesized and transcribed with
`facebook/wav2vec2-base-960h`. Model and processor were both built by
`from_pretrained("FunAudioLLM/CosyVoice-300M-SFT")` with no conversion step before it, and the load
report carried no missing, unexpected or mismatched key over 436,053,808 parameters. The prompt is
the `英文女` entry of that release's `spk2info.pt`, which carries a speaker embedding, so no ONNX
graph was involved.

| Prompt text | Transcript |
|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` |
| `She sells sea shells by the sea shore.` | `SHE SELLS SEASHELLS BY THE SEASHORE` |

The first is word for word. The second is too, with the connectionist temporal classification
decoder merging both compounds. The two waveforms are 2.97 s at RMS 0.1221 and 2.28 s at RMS
0.0866.

**Copy synthesis through the vocoder alone.** A transcription passes even when the vocoder is badly
wrong, so the vocoder was measured directly. A 13.7 s recording was resampled to 22050 Hz, turned
into the 80 bin mel the model consumes and vocoded by `CosyVoiceV1HiFTGenerator.inference` alone,
with no language model and no flow matching. The L1 distance between the log mel spectrograms of
the recording and of its resynthesis, over the filter bank the objective uses, is **0.105986**.
Same energy white noise against the same recording scores 3.905116, so the vocoder is 37 times
closer to the signal than a level matched impostor is. Level is preserved too: RMS 0.110275 in,
0.108052 out.

**The vocoder objective on the real checkpoint.** `CosyVoiceV1HiFTGenerator.compute_loss` on that
recording returns loss 12.446939, made of the weighted mel term 4.746673 and the f0 term 7.700266.
Its backward pass puts gradient on 227 parameters, exactly the 227 tensors `hift.pt` carries, all
of them under `hift` and none under `llm` or `flow`.

**`compute_f0` against upstream's own extraction.** `CosyVoiceV1Processor.compute_f0` on the same
recording is bit identical to running upstream's `cosyvoice/dataset/processor.py` lines directly,
`pyworld.harvest` at a frame period of 11.61 ms, `pyworld.stonemask`, and a linear interpolation
onto the 1184 mel frames: `torch.equal` on the float32 cast is `True` and the largest absolute
difference is 0.0. The contour is 995 voiced frames out of 1184 at a mean voiced f0 of 148.57 Hz.
The interpolation has to stay in the double precision `pyworld` returns to get that: doing it in
float32 moves a voicing boundary onto the neighbouring frame and costs 6.6e-03 Hz at the largest
step.

**The mel term against matcha's own function.** Upstream measures it with
`matcha.utils.audio.mel_spectrogram`, which is not vendored here. Reimplemented from Matcha's
source with `librosa.filters.mel` and run on the copy synthesis pair above, it returns 4.80676126
where `CosyVoiceV1HiFTGenerator.mel_loss` returns 4.80676174, a difference of 4.768e-07 on a value
of 4.8. The residual is `librosa.filters.mel` against `torchaudio.functional.melscale_fbanks` in
float32; both are the Slaney scaled, Slaney normalized filter bank.

**Round trip through the hub format.** `from_pretrained` on the released repository id loads
436,053,808 parameters, and `save_pretrained` followed by `from_pretrained` reproduces every
parameter with a largest difference of 0.0 and leaves no meta
parameter or buffer behind. A model reloaded that way computes bit for bit what the model it was
saved from computes: 0.0 on the text encoder, 0.0 on the language model logits and 0.0 on the
vocoder waveform.

Comparing two vocoder runs needs a pinned seed to mean anything. `CosyVoiceV1SineGen` draws one
uniform phase per harmonic and adds noise on every call, so the same mel through the same weights
gives 0.0 twice under one seed and 1.456 under two, on a waveform whose peak is 0.84. The 0.0
above is a seeded comparison.

Getting there took two fixes, both of the same kind. Transformers 5 builds the model on the meta
device before loading weights, and a non-persistent buffer computed in a constructor is then
materialised as uninitialised memory instead of by rerunning that computation. Two of them were
wrong after a reload: the inverse short time Fourier transform window, which read
`[-1.78e-13, 0.0, 0.0, 0.0, ...]` instead of a Hann window, and the relative positional embedding
table shared by all three encoders. Neither raised. The vocoder ran and the reloaded model
generated fluently, but the words were gone: `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` came
back from the reloaded model as `HADNIN YOUR DET SSFUL I SA YO TO DEING HEE`. Both tables are now
built on first use instead of in the constructor. This is the failure PROJECT.md recorded from the
abandoned branch as `torch.istft`'s NOLA check failing with `window overlap add min: 1`; on
transformers 5.16 it does not raise at all, which is worse.

**The text tokenizer of the 25 Hz release.** `CosyVoiceV1Tokenizer` was checked against a
reference implementation of tiktoken's byte level BPE written from its published algorithm, the
`regex` pre tokenizer split followed by rank ordered pair merging over the same 58836 mergeable
ranks. Twelve strings covering English, Chinese, Japanese, Cantonese, mixed script, digits,
contractions, leading and trailing whitespace, tabs and newlines, emoji and a fifty character run
all encode token for token, and every one round trips through `decode`. The vocabulary is 60515
tokens, the size the 25 Hz recipe's `text_token_size` names, with `<|endoftext|>` at 58836 and the
last timestamp `<|30.00|>` at 60514, which is upstream's ordering. `save_pretrained` followed by
`from_pretrained` reproduces every one of those encodings. This ran locally rather than on Colab,
because it touches no checkpoint.

**The two ported ONNX components, against the graphs they replace.** Three LibriSpeech clips of
2.9, 2.5 and 6.5 seconds went through `onnxruntime` and through the PyTorch port, on the same
features. The speaker embedding, 192 dimensions reaching 2.83 in magnitude, differs by at most
8.106e-06. The speech tokenizer produced 1158 token ids over the three clips and every one is
identical; its encoder output, reaching 19.0 in magnitude, differs by at most 4.339e-05. That
residual is float32 reassociation, not a difference in the computation: the graph scales the query
and the key by the fourth root of the head dimension apiece and folds each `Linear` into a `MatMul`
and an `Add`, and a PyTorch `Linear` accumulates its bias differently.

**Zero shot voice cloning, end to end.** The port makes this path runnable for the first time, so it
was run. A 5.86 s LibriSpeech clip, which `facebook/wav2vec2-base-960h` transcribes as
`MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL`, became
293 speech tokens and a 192 dimensional speaker embedding, and both were passed to the language model
and to the flow matching model together with the prompt mel spectrogram and the clip's transcript.

| Asked | Heard back | Waveform |
|---|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` | 2.83 s at 22050 Hz, RMS 0.0693, peak 0.5756 |
| `She sells seashells by the seashore.` | `SHE SELLS SEASHILLS BY THE SEASHORE` | 2.28 s at 22050 Hz, RMS 0.0550, peak 0.3028 |

Every word is there; the second turns the vowel of `SEASHELLS` into an `I`, a decoder edge rather
than a missing word.

**Still unverified.** Whether `WhisperFeatureExtractor(feature_size=128)` with `padding=False`
reproduces `whisper.log_mel_spectrogram(speech, n_mels=128)` frame for frame, and whether
`WhisperTokenizer.from_pretrained("openai/whisper-large-v3")` produces the same ids as
openai-whisper's `get_tokenizer(multilingual=True, num_languages=100).encode(text,
allowed_special="all")`. The generation above exercises the second one indirectly. Both need
`openai-whisper` installed, which is the dependency this migration removed.

`normalize_text` was checked against upstream's `frontend_utils.py`, recovered from this
repository's history and executed side by side, on eleven strings covering both language branches,
the empty string, an SSML marker, a punctuation only string and a paragraph long enough to split.
All eleven agree exactly. `is_only_punctuation`, which drops the `regex` dependency, agrees with
upstream's `^[\p{P}\p{S}]*$` on all 149251 assigned Unicode code points.

**The English digit reading, against the package upstream pins.** `number_to_words` was compared
with the English number reading package upstream's `requirements.txt` pins, at its pinned version
7.3.1, over 41,821 digit strings: every integer from 0 to 10,000, every 97th from 10,000 to
1,000,000, 20,000 strings of 1 to 33 digits drawn from `random.Random(0)`, every one of the numbers
0 to 79 carrying 1 to 20 leading zeros, and the runs of 1 to 13 zeros. **Zero mismatches.** A
further 1,600 strings of 34 to 41 digits, 200 per length from `random.Random(1)`, put the two on the
same side of the largest scale word in every case: 600 agree on a reading and 1,000 raise on both
sides, the oracle with `NumOutOfRangeError` and `number_to_words` with `ValueError`. The oracle was
unpacked into a scratch directory; it is not installed, not imported by this repository and not
declared anywhere.

**The English digit reading, generated and transcribed back.** Zero shot from the 5.86 s LibriSpeech
clip above, `I paid 1234 dollars in 2025 for 7 books.`, three seeds each, on the local GPU, transcribed
with `facebook/wav2vec2-base-960h`, word error rate against the text handed to the model:

| Frontend | WER by seed | Heard back, seed 1 |
|---|---|---|
| on | 0.053 / 0.053 / 0.105 | `I PAID ONE THOUSAND TWO HUNDRED AND THIRTY FOUR DOLLARS IN TWO THOUSAND TWENTY FIVE FOR SEVEN BOOKS` |
| off | 0.778 / 0.667 / 0.889 | `I PAID TWELVE THIRTY FOUR DOLLARS IN EAST IBEAKS FOR SEVEN BOOKS` |

With the front end on, every seed reads all three numbers back and the errors are single words. With
it off, every seed reads `1234` as `TWELVE THIRTY FOUR` and no seed recovers `2025`.

## Not carried over from upstream

Recorded per CLAUDE.md section 2.6.

- **The vocoder's discriminators, and the three objective terms that need them.** Upstream trains
  HiFT as a GAN through `cosyvoice/hifigan/hifigan.py`. Of its five generator side terms, the mel
  reconstruction loss and the f0 loss are implemented here as
  `CosyVoiceV1HiFTGenerator.compute_loss`; see "Training" for all five term by term. The other
  three, `generator_loss`, the feature matching loss and `tpr_loss`, and the whole discriminator
  turn, `discriminator_loss` plus the same weighted `tpr_loss` optimized in an alternating turn
  with its own optimizer, all read discriminator outputs. Those discriminators are
  `MultipleDiscriminator` over `matcha.hifigan.models.MultiPeriodDiscriminator` and
  `cosyvoice.hifigan.discriminator.MultiResSpecDiscriminator`. Leaving them out follows the
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
  half takes no labels. Against that convention, `CosyVoiceV1HiFTGenerator.compute_loss` scoring the
  mel reconstruction and f0 terms goes beyond it rather than falling short of it. The consequence to
  know is that a HiFT trained through those two terms alone would not reproduce a released
  checkpoint. `cosyvoice/hifigan/discriminator.py` was deleted rather than kept as an unmigrated
  file; what went with it is named in "File map". `tpr_loss` in `cosyvoice/utils/losses.py` went
  the same way, since it takes discriminator outputs on both sides and has nothing to score without
  them.
- **`DPOLoss`.** `cosyvoice/utils/losses.py` also held a direct preference optimization loss, used
  by `examples/libritts/cosyvoice2/run_dpo.sh` through `Executor.train_one_epoc_dpo`. It belongs to
  the CosyVoice 2 recipe, not to v1, and this folder deleted it with the rest of that file. Whether
  `voicestudio/models/cosyvoice_v2/` should carry it is that folder's decision, not this one's.
- **Text normalization.** `CosyVoiceV1Processor.normalize_text` is upstream's
  `CosyVoiceFrontEnd.text_normalize` with the two text normalizers left out. Upstream runs the
  input through `ttsfrd` if the resource pack is installed, otherwise `wetext`, otherwise nothing,
  before the rewriting and sentence splitting that this does implement. So the branch this
  reproduces exactly is upstream's own "no frontend is avaliable" path, and numbers, dates and
  abbreviations are not expanded the way an installed normalizer would expand them. English digit
  runs are spelled out by `number_to_words`, which reads them the way the package upstream pins does.
- **`ttsfrd`.** Upstream ships it as a wheel in `FunAudioLLM/CosyVoice-ttsfrd` with a 339 MB
  resource pack. It is not on PyPI and has no source release.
- **Streaming input text.** Upstream's `inference_bistream` accepts a text generator and
  interleaves text and speech tokens. It exists only on `Qwen2LM`, that is CosyVoice 2 and 3, and
  is not part of v1.
- **`n_feats` on the flow matching head.** Upstream's `ConditionalCFM` passes `in_channels` (240)
  to `BASECFM` as `n_feats`, then ignores it and hardcodes 80 everywhere it matters.
  `CosyVoiceV1ConditionalCFM.n_feats` is `estimator_out_channels` (80), which is the value upstream
  actually uses. Identical behaviour for v1; a subclass with a different mel width would diverge.
- **The `CosyVoice-300M-25Hz` release.** Its text tokenizer is migrated,
  `tokenization_cosyvoice_v1.py`, but the release itself is not. `PUBLISHED_CHECKPOINTS` lists only
  the three 50 Hz repositories, `CosyVoiceV1Processor` builds a `WhisperTokenizer` for them, and
  `CosyVoiceV1Config.text_vocab_size` defaults to 51866 rather than the
  60515 the 25 Hz recipe sets. Whether to cover that release is the decision this leaves open. The
  two vocabularies are not interchangeable: it carries 58836 mergeable ranks where
  `openai/whisper-large-v3` carries 51866, 9292 of its tokens are absent from that vocabulary and
  48003 of the rest sit at a different rank, and its special tokens add audio event, emotion and
  TTS vocal markers Whisper has none of. `CosyVoiceV1Tokenizer` is also not registered with
  `AutoTokenizer` under `CosyVoiceV1Config`, because the checkpoints that folder converts today
  tokenize with Whisper's vocabulary and that registration would name the wrong class for them.
- **`num_languages` in upstream's own recipe.** `cosyvoice.yaml` passes `num_languages: 100` to
  `get_tokenizer` while setting `text_token_size: 60515` for the 25 Hz recipe, and those two
  disagree: 100 language tokens give 60510. 60515 is what all 105 entries of `LANGUAGES` give, so
  `CosyVoiceV1Tokenizer` defaults `num_languages` to 105 and reproduces the released vocabulary
  size. The argument stays exposed for anyone who needs upstream's literal 100.

## File map

One counterpart per upstream file, per CLAUDE.md section 2.4. The renames are recorded in
`7e93ecff` and the removals in `86a9fa18` and in the commit that emptied the last of the nested
tree, and `git show` recovers any file named below. Nothing of upstream's directory layout is left:
the folder holds a `README`, an `__init__.py`, five `<kind>_cosyvoice_v1.py` files and
`weight_conversion.py`.

`tokenization_cosyvoice_v1.py` is the one file whose history needs a lowered threshold. Git records
it as a rename of `cosyvoice/tokenizer/tokenizer.py` at 22% similarity, because the migration
rewrote 199 of that file's 327 lines and only the language table survived intact, so
`git log --follow` finds it at `-M20%` and not at the default 50%.

### Transformed into this folder

| Upstream file | Where it went |
|---|---|
| `cosyvoice/llm/llm.py` | `modeling_cosyvoice_v1.py`. `TransformerLM` became `CosyVoiceV1SpeechTokenLM` plus `CosyVoiceV1ForConditionalGeneration`. |
| `cosyvoice/cli/model.py` | `generation_cosyvoice_v1.py`. `CosyVoiceModel.token2wav` and `CosyVoiceModel.tts` became `CosyVoiceV1GenerationMixin`. |
| `cosyvoice/cli/frontend.py` | `processing_cosyvoice_v1.py`. |
| `cosyvoice/cli/cosyvoice.py` | `weight_conversion.py`. Its `load` method became the reading of a released directory, which `from_pretrained` calls when the published layout is what it was given. |
| `cosyvoice.yaml` | `configuration_cosyvoice_v1.py`. |
| `cosyvoice/__init__.py` | `__init__.py`. |
| `cosyvoice/tokenizer/tokenizer.py` | `tokenization_cosyvoice_v1.py`. `get_encoding` and `get_tokenizer` became `CosyVoiceV1Tokenizer` over `CosyVoiceV1TikTokenConverter`. `TO_LANGUAGE_CODE` went, because it only validated the `language` argument of Whisper's decoding API, which the text tokenizer never reaches, and `LANGUAGES` stays because the order of its keys fixes the id of every special token after it. `get_qwen_tokenizer`, `CosyVoice2Tokenizer` and `CosyVoice3Tokenizer` are `SPECIAL_TOKENS` and `CosyVoiceV3Processor.add_special_tokens` in `processing_cosyvoice_v3.py`. |

### Removed, with a counterpart in one of the three folders

| Upstream file | Counterpart |
|---|---|
| `cosyvoice/flow/flow.py` | `MaskedDiffWithXvec` is `CosyVoiceV1FlowModel`, `CausalMaskedDiffWithXvec` is `CosyVoiceV2FlowModel`, `CausalMaskedDiffWithDiT` is `CosyVoiceV3FlowModel`. |
| `cosyvoice/flow/flow_matching.py` | `ConditionalCFM` is `CosyVoiceV1ConditionalCFM`, `CausalConditionalCFM` is `CosyVoiceV2ConditionalCFM` and `CosyVoiceV3ConditionalCFM`. |
| `cosyvoice/flow/decoder.py` | `ConditionalDecoder` is `CosyVoiceV1ConditionalDecoder` and `CausalConditionalDecoder` is `CosyVoiceV2ConditionalDecoder`. `CausalBlock1D`, `CausalResnetBlock1D`, `CausalConv1d` and `Transpose` are the `CosyVoiceV2` classes of the same names. |
| `cosyvoice/flow/length_regulator.py` | `InterpolateRegulator` is `CosyVoiceV1InterpolateRegulator`. |
| `cosyvoice/flow/DiT/dit.py` | `DiT` is `CosyVoiceV3ConditionalDecoder` and `InputEmbedding` is `CosyVoiceV3InputEmbedding`. `TextEmbedding` is F5-TTS carryover that `DiT` never instantiates. |
| `cosyvoice/flow/DiT/modules.py` | `TimestepEmbedding` is `CosyVoiceV3TimestepEmbedding`, `DiTBlock` is `CosyVoiceV3DecoderLayer`, `AdaLayerNormZero_Final` is `CosyVoiceV3AdaLayerNormFinal` and `CausalConvPositionEmbedding` is `CosyVoiceV3CausalConvPositionEmbedding`. `Attention`, `AttnProcessor`, `AdaLayerNormZero`, `FeedForward` and `SinusPositionEmbedding` are inherited from `voicestudio/models/f5_tts/`. `MelSpec`, `GRN`, `ConvNeXtV2Block`, `MMDiTBlock`, `JointAttnProcessor`, `ConvPositionEmbedding`, `precompute_freqs_cis` and `get_pos_embed_indices` are F5-TTS carryover that no CosyVoice configuration reaches. |
| `cosyvoice/hifigan/generator.py` | `HiFTGenerator`, `SineGen`, `SourceModuleHnNSF` and `ResBlock` are the `CosyVoiceV1HiFTGenerator` family, `SineGen2` is `CosyVoiceV2SineGen` and `CausalHiFTGenerator` is `CosyVoiceV3HiFTGenerator`. |
| `cosyvoice/hifigan/f0_predictor.py` | `ConvRNNF0Predictor` is `CosyVoiceV1F0Predictor` and `CausalConvRNNF0Predictor` is `CosyVoiceV3F0Predictor`. |
| `cosyvoice/transformer/encoder.py`, `encoder_layer.py`, `attention.py`, `embedding.py`, `subsampling.py`, `positionwise_feed_forward.py` | Together they are `CosyVoiceV1Encoder` and its parts. The released configurations select one class per file, `ConformerEncoder`, `ConformerEncoderLayer`, `RelPositionMultiHeadedAttention`, `EspnetRelPositionalEncoding`, `LinearNoSubsampling` and `PositionwiseFeedForward`. The others are WeNet variants no CosyVoice configuration selects and no released checkpoint carries weights for. |
| `cosyvoice/transformer/upsample_encoder.py` | `UpsampleConformerEncoder` is `CosyVoiceV2UpsampleEncoder`, `PreLookaheadLayer` is `CosyVoiceV2PreLookaheadLayer` and `Upsample1D` is `CosyVoiceV2Upsample1D`. |
| `cosyvoice/transformer/convolution.py` | `CausalConv1d`, `CausalConv1dUpsample` and `CausalConv1dDownSample` are `CosyVoiceV3CausalConv1d`, `CosyVoiceV3CausalConv1dUpsample` and `CosyVoiceV3CausalConv1dDownsample`. `ConvolutionModule` is never built: every CosyVoice encoder sets `use_cnn_module: False`. |
| `cosyvoice/transformer/activation.py` | `Snake` is `CosyVoiceV1Snake`. `Swish` is a fallback `COSYVOICE_ACTIVATION_CLASSES` reaches only on a torch without `nn.SiLU`. |
| `cosyvoice/transformer/label_smoothing_loss.py` | `LabelSmoothingLoss` is `CosyVoiceV1LabelSmoothingLoss`. |
| `cosyvoice/transformer/decoder.py`, `decoder_layer.py` | None, and none is needed. `TransformerDecoder`, `BiTransformerDecoder` and `DecoderLayer` are WeNet's attention rescoring decoder for speech recognition. Nothing in the upstream tree imported them outside each other, no CosyVoice configuration names them, and the three conversions load `llm.pt`, `flow.pt` and `hift.pt` with zero UNEXPECTED keys, so no released checkpoint carries their weights. |
| `cosyvoice/utils/mask.py` | `make_pad_mask` is `make_pad_mask`, and `add_optional_chunk_mask` is `build_attention_bias` in `modeling_cosyvoice_v1.py` and `build_chunk_mask` in `modeling_cosyvoice_v3.py`. |
| `cosyvoice/utils/common.py` | `ras_sampling`, `nucleus_sampling`, `random_sampling` and `fade_in_out` are in `generation_cosyvoice_v1.py`, `mask_to_bias` is folded into `build_attention_bias`, and `th_accuracy` is `CosyVoiceV1Output.accuracy`. `TrtContextWrapper` is a TensorRT wrapper. |
| `cosyvoice/utils/onnx.py` | `SpeechTokenExtractor` is `CosyVoiceV1Processor.encode_speech_tokens` and `EmbeddingExtractor` is `CosyVoiceV1Processor.encode_speaker`. |
| `cosyvoice/utils/class_utils.py` | `configuration_cosyvoice_v1.py`. It is the string to class dispatch table hyperpyyaml used to build a model from `cosyvoice.yaml`. It had already stopped importing, since it reads `cosyvoice.llm.llm` and `cosyvoice.cli.model`, both moved in `7e93ecff`. |
| `cosyvoice/utils/file_utils.py` | `load_wav` is the processor's load and resample. `read_lists` and `read_json_lists` are manifest readers, and `convert_onnx_to_trt` and `export_cosyvoice2_vllm` are serving exports. |
| `cosyvoice/cli/__init__.py`, `cosyvoice/dataset/__init__.py`, `cosyvoice/transformer/__init__.py`, `cosyvoice/utils/__init__.py` | `__init__.py`. All four are empty. |
| `cosyvoice/hifigan/hifigan.py` | `HiFiGan.forward_generator` is `CosyVoiceV1HiFTGenerator.compute_loss`, without the three terms that read discriminator outputs. `forward_discriminator` has no counterpart. |
| `cosyvoice/utils/losses.py` | `mel_loss` is `CosyVoiceV1HiFTGenerator.mel_loss`, over the mel spectrogram `voicestudio/models/bigvgan/` builds. `tpr_loss` and `DPOLoss` have no counterpart. |
| `cosyvoice/utils/frontend_utils.py` | All seven functions are module level functions of `processing_cosyvoice_v1.py`, and `CosyVoiceFrontEnd.text_normalize`, their only caller, is `CosyVoiceV1Processor.normalize_text`. |
| `cosyvoice/dataset/processor.py` | `compute_f0` is `CosyVoiceV1Processor.compute_f0`. `compute_fbank` is `CosyVoiceV1FeatureExtractor`, `compute_whisper_fbank` is the processor's `WhisperFeatureExtractor`, `tokenize` is the processor's tokenizer, `parse_embedding` is the `speaker_embedding` input, and `parquet_opener` through `padding` is an iterable data pipeline and a collator, which `transformers` supplies. |
| `cosyvoice/tokenizer/assets/multilingual_zh_ja_yue_char_del.tiktoken` | Checkpoint side data rather than library source, the way `f5_tts`'s `vocab.txt` is. `CosyVoiceV1Tokenizer.vocab_files_names` names it, `from_pretrained` resolves it out of the checkpoint directory, and `save_pretrained` writes a `tokenizer.json` that no longer needs it. |

### Removed as out of scope

| Upstream path | Category |
|---|---|
| `runtime/`, 55 files | Serving stacks: a FastAPI server, a gRPC server and a Triton model repository backed by TensorRT-LLM, with their Dockerfiles, clients and conversion scripts. `runtime/triton_trtllm/token2wav*.py` are that stack's own token to waveform paths, which `generation_cosyvoice_v1.py`, `generation_cosyvoice_v2.py` and `generation_cosyvoice_v3.py` cover, and `scripts/convert_cosyvoice3_to_hf.py` merges the speech embedding into the Qwen2 table so `trtllm-build` can read it, which is not the layout this migration reads. |
| `examples/`, 37 files | Training recipes: the LibriTTS and MagicData shell pipelines, DeepSpeed stage configurations, data preparation scripts, and the GRPO recipe under `examples/grpo/cosyvoice2/`, whose reward in `reward_tts.py` is an HTTP call to the Triton server above. `local/prepare_reject_sample.py` generates the rejected half of a preference batch by running `inference_zero_shot`, which `CosyVoiceV2ForConditionalGeneration.generate` covers. |
| `tools/`, 3 files | Data preparation: `extract_embedding.py` and `extract_speech_token.py` run `campplus.onnx` and the speech tokenizer graph over a manifest, which `CosyVoiceV1Processor.encode_speaker` and `encode_speech_tokens` do per utterance, and `make_parquet_list.py` writes the parquet shards. |
| `cosyvoice/vllm/cosyvoice2.py` | Serving: a vLLM plugin registering `CosyVoice2ForCausalLM`. |
| `cosyvoice/bin/`, 4 files | Training and export tooling. `train.py` is upstream's training entry point, read for the objectives above; `average_model.py` averages checkpoints; `export_jit.py` and `export_onnx.py` are export scripts. |
| `cosyvoice/dataset/dataset.py`, `cosyvoice/utils/executor.py`, `train_utils.py`, `scheduler.py` | Training harness: an `IterableDataset` with its distributed sampler, the train and validation loop, distributed and optimizer setup, and the learning rate schedules. `transformers` supplies all four. |
| `webui.py`, `example.py`, `vllm_example.py` | Demo scripts. Their counterpart is the "Usage" section of this README and of the v2 and v3 ones. All three had already stopped importing, since they read `cosyvoice.cli.cosyvoice`, moved in `7e93ecff`. |
| `docker/Dockerfile` | Packaging. |
| `.github/`, `.gitignore`, `.gitmodules` | Continuous integration and repository configuration. |
| `INFO.md`, `FAQ.md`, `CODE_OF_CONDUCT.md` | Upstream documentation. `INFO.md` is upstream's own README. |
| `asset/`, 3 files | Demo assets: the two prompt waveforms `example.py` and `webui.py` read, and a group chat QR code. |
| `third_party/Matcha-TTS` | A gitlink to `shivammehta25/Matcha-TTS`, never checked out here. What CosyVoice imports from it, `SinusoidalPosEmb`, `Block1D`, `ResnetBlock1D`, `Downsample1D`, `Upsample1D`, `BASECFM` and `BasicTransformerBlock`, is inlined in `modeling_cosyvoice_v1.py`. Its `matcha.hifigan.models` losses are not; see "Not carried over from upstream". |

### Removed with nothing to replace them

| Upstream file | Why it went |
|---|---|
| `cosyvoice/hifigan/discriminator.py` | `MultipleDiscriminator`, `MultiResolutionDiscriminator`, `DiscriminatorR`, `MultiResSpecDiscriminator`, `SpecDiscriminator` and `stft`, together with the `matcha.hifigan.models.MultiPeriodDiscriminator` that `MultipleDiscriminator` takes as its `mpd`. None of them is implemented here, and neither are the three objective terms that read their outputs; see "Not carried over from upstream". It was deleted rather than renamed onto the convention, because a file holding unmigrated upstream code under a migrated name is the nested tree with a new name, and a stub under that name would claim a home for something this folder does not have. |
| `LICENSE` | Apache 2.0, identical to the repository root's copy but for the "how to apply" appendix, and carrying no restriction on the weights. The licence lives in the header of `modeling_cosyvoice_v1.py` and at the repository root, per CLAUDE.md section 6. |
| `requirements.txt.bak` | Upstream's dependency list, kept readable until its dependencies were dealt with. "Dependencies" above accounts for every entry, including the ones that could not be removed. |

## Repository integration

Nothing outside this folder was touched. What is worth knowing about the rest of the repository:

- `voicestudio/models/__init__.py` already carries `from .cosyvoice_v1 import *`.
- `pyproject.toml` still declares an `onnx` extra holding `onnxruntime`, `onnxruntime-gpu` and
  `onnx`. Nothing in these three folders imports any of them any more, and no other model folder
  does either, so that extra and its entry in `all` can go. `pyworld` is already declared under the
  `eval` extra; `tiktoken` and `ttsfrd` are deliberately absent and every path that would want one
  imports it lazily and raises with an explanation.
- `voicestudio/models/cosyvoice_v2/` inherits `CosyVoiceV1HiFTGenerator`, so it inherits
  `compute_loss` and `mel_loss` too. It now also sets the constants its own recipe uses, `n_fft`
  1920, hop 480 and window 1920 at 24 kHz, where `CosyVoiceV1Config` defaults to 1024, 256 and 1024.
  The bin count, the two frequency bounds and the weight of 45 are the same in both recipes.
