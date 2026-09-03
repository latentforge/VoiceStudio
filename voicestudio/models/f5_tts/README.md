# F5-TTS

F5-TTS is a non-autoregressive text-to-speech model trained with conditional flow matching over log mel
spectrograms. There is no duration predictor and no alignment model: the character sequence is embedded, encoded by
four ConvNeXt V2 blocks and simply laid alongside the speech frames, and a diffusion transformer regresses the
vector field of an optimal transport path from Gaussian noise to the mel spectrogram, modulated by the flow time
step through adaptive layer norms. Training masks out a random span of the target spectrogram and asks the model to
infill it, which at inference time makes voice cloning a special case: the reference clip occupies the leading
frames, the text to speak follows the reference transcription, and the frames past the reference are the span to
fill in. Generation integrates the vector field with a fixed step solver over a Sway Sampling time grid and decodes
the result with a Vocos vocoder.

The same file also carries E2-TTS, the flat UNet transformer baseline the authors publish alongside F5-TTS. It is
the same flow, the same objective and the same sampler, over a backbone that prepends the time step embedding to
the sequence and joins each second half layer onto its mirrored first half counterpart.

Original model and code: [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)


## Usage

The published `SWivid/F5-TTS` and `SWivid/E2-TTS` checkpoints are bare exponential moving average state dicts with
no config, and the vocoder lives in a repository of its own, so they need a one-time conversion before they load:

```python
from voicestudio.models.f5_tts.weight_conversion import convert

convert("F5TTS_v1_Base", "f5-tts-v1-base-converted")
```

`convert` accepts any key of `PUBLISHED_CHECKPOINTS` (`F5TTS_v1_Base`, `F5TTS_v1_Base_no_zero_init`, `F5TTS_Base`,
`F5TTS_Base_bigvgan`, `E2TTS_Base`), writes the model, the tokenizer and the feature extractor into the output
directory, and downloads [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) into the
model's own `vocoder` sub-model.

```python
import soundfile as sf
import torch

from voicestudio.models.f5_tts import F5TTSForConditionalGeneration, F5TTSProcessor

model_id = "f5-tts-v1-base-converted"

processor = F5TTSProcessor.from_pretrained(model_id)
model = F5TTSForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).to("cuda")
```

```python
ref_audio, sampling_rate = sf.read("reference.wav", dtype="float32")
ref_text = "Some call me nature, others call me mother nature."
gen_text = "I don't really care what you call me. I am mighty and enduring."

inputs = processor(text=gen_text, audio=ref_audio, ref_text=ref_text, sampling_rate=sampling_rate)
generated = model.generate(
    input_ids=inputs["input_ids"].to(model.device),
    conditioning_features=inputs["input_features"].to(model.device),
    attention_mask=inputs["attention_mask"].to(model.device),
    duration=inputs["duration"].to(model.device),
    num_steps=32,
    guidance_scale=2.0,
    sway_sampling_coef=-1.0,
)
waveform = processor.batch_decode(
    generated.mel_spectrogram,
    model.vocoder,
    duration=inputs["duration"],
    reference_length=inputs["reference_length"],
    reference_rms=inputs["reference_rms"],
)
sf.write("output.wav", waveform, processor.feature_extractor.sampling_rate)
```

Long text is split into chunks the reference clip can carry, and the chunk waveforms are cross faded back together
by `batch_decode`:

```python
ref_seconds = len(ref_audio) / sampling_rate
chunks = processor.chunk_text(gen_text, max_chars=processor.max_chunk_chars(ref_text, ref_seconds))
inputs = processor(text=chunks, audio=ref_audio, ref_text=ref_text, sampling_rate=sampling_rate)
```

Passing several chunks at once pads them to a common length. Every released checkpoint was trained with
`attn_mask_enabled=False`, which leaves attention free to read those padding frames, so a batched run only matches
a single one once `model.config.attn_mask_enabled = True` is set; upstream sidesteps this by generating one chunk
at a time, which is also an option here.

`generate` also exposes the rest of the upstream sampler: `ode_method` (`"euler"` or `"midpoint"`), `use_epss` for
the empirically pruned step grid at a low number of function evaluations, `no_ref_audio` to drop the reference
speech, `edit_mask` to keep only part of it, which is what upstream's speech editing script does, `max_duration`,
and `return_trajectory` for the state of the flow at every solver step. A `guidance_scale` below `1e-5` skips the
unconditional branch and halves the compute.


## Training

Training uses the standard `forward`: pass `labels` of shape `(batch_size, sequence_length, mel_dim)` for the
target log mel spectrogram and `attention_mask` for its unpadded frames. Everything else is drawn inside `forward`,
and the returned `loss` is the conditional flow matching loss.

The objective is upstream `src/f5_tts/model/cfm.py`'s `CFM.forward`, term for term. One span mask is drawn per
sample by taking a fraction of the sample's own length uniformly from `frac_lengths_mask`, `(0.7, 1.0)`, placing it
at a uniformly random start, and intersecting it with the padding mask. Gaussian noise `x0` is drawn with the shape
of the target `x1`, one flow time `t` per sample is drawn uniformly from `[0, 1]`, and the model is asked for the
vector field at `(1 - t) * x0 + t * x1`, conditioned on the target with the span zeroed out. The loss is the
element-wise mean squared error against `x1 - x0`, averaged over the frames inside the span only, with no weighting
and no other term. Classifier free guidance is trained in by zeroing the speech conditioning with probability
`audio_drop_prob` (0.3) and, independently, zeroing the speech conditioning and the text together with probability
`cond_drop_prob` (0.2).

Upstream freezes nothing of the flow matching model. `Trainer` hands `model.parameters()` to AdamW in one group,
and no `requires_grad`, `freeze` or `.eval()` call appears anywhere in the upstream `model/` or `train/` trees. The
vocoder is outside that graph entirely: upstream loads it pretrained and calls it only to log samples, never
optimizing it, so `F5TTSForConditionalGeneration.freeze_vocoder` sets `requires_grad = False` on every one of its
parameters on construction and again after `from_pretrained`.


## Lineage

The backbone inherits nothing wholesale, and the report below says why each candidate was rejected.

`F5TTSRotaryEmbedding` is `LlamaRotaryEmbedding` unchanged, and `F5TTSAttention` calls Llama's
`apply_rotary_pos_emb`, `eager_attention_forward` and the `ALL_ATTENTION_FUNCTIONS` interface. Upstream's rotary
embedding is `x_transformers`', whose frequency table is `1 / theta ** (2i / d)` repeated pairwise and whose
`rotate_half` rotates the interleaved pairs `(x0, x1), (x2, x3), ...`, while Llama's splits the head dimension in
half. `deinterleave_head_dim` reorders a head vector into the half split layout before the rotation, which makes
the two identical: the rotation is applied to the same pairs by the same angles, and query and key are reordered
together so the attention scores are unchanged.

`F5TTSAttention` itself is not a Llama-lineage class. Its projections carry biases, its inner dimension is
`num_attention_heads * head_dim` and is independent of `hidden_size`, its output projection is a
`ModuleList([Linear, Dropout])`, `pe_attn_head` restricts the rotary embedding to a prefix of the heads, and the
attention output is zeroed on padding frames. No `LlamaAttention` or `Qwen3Attention` subclass can express that set
without overriding every line of `forward`.

Sibling models under `voicestudio/models/` were checked first. `cosyvoice_v1` is the closest: its
`CosyVoiceV1ConditionalCFM` is also optimal transport conditional flow matching over mel spectrograms with a fixed
step Euler solver and classifier free guidance. It was still not inherited from. Its vector field estimator is a
Conv1d U-Net of ResNet and transformer blocks rather than a diffusion transformer; it conditions on an encoder
output resampled to the mel frame rate plus a speaker embedding rather than on character ids encoded by ConvNeXt V2
and mixed in by a linear projection; its solver steps in place with no trajectory, no interpolation and no
reshaped time grid; and its `forward` is decorated `@torch.inference_mode()`, so it carries no training objective
to reuse. `dia`, `dia2`, `breeze_tts`, `chroma`, `higgs_tts2`, `higgs_tts3` and `qwen3_tts` are autoregressive
codec language models with a backbone plus depth decoder over discrete codebooks, which shares nothing with a
non-autoregressive continuous mel flow. `prompt_tts_pp` decodes with a Gaussian diffusion, not a flow, over a
FastSpeech2 Conformer with a duration predictor.

In `transformers` itself, `fastspeech2_conformer` and `vits` are the two non-autoregressive text-to-speech
lineages, and both are duration-predictor models whose backbones carry no time step conditioning at all, so
neither offers a layer to inherit.

The vocoder is `VocosModel` from `voicestudio/models/vocos`, held as a sub-model the way `ParlerTTSConfig` holds
its `audio_encoder`: `F5TTSConfig.vocoder_config` is a `VocosConfig`, `F5TTSForConditionalGeneration.vocoder` is
the model it builds, and `convert` writes its weights into the same `model.safetensors` under a `vocoder.` prefix.
Vocos is an independently published vocoder with checkpoints of its own, and F5-TTS also ships a BigVGAN variant
of the same slot, so it does not belong inside this folder. `src/transformers/models/` ships `univnet`, the
`speecht5` HiFi-GAN, `vits`, `encodec`, `dac`, `mimi`, `xcodec`, `xcodec2`, `higgs_audio_v2_tokenizer`,
`qwen3_tts_tokenizer_*` and `vibevoice_acoustic_tokenizer`; none of them is a ConvNeXt plus inverse STFT mel
vocoder, so there was nothing to relay to.


## Dependencies

The upstream `pyproject.toml`, kept as `pyproject.toml.bak`, lists 28 runtime dependencies. This migration needs
`torch`, `torchaudio`, `transformers`, `safetensors`, `numpy` and `huggingface_hub`, all of which the repository
already carries. `pyproject.toml` and `uv.lock` need no change, and nothing was added.

- `x_transformers` supplied `RotaryEmbedding`, `apply_rotary_pos_emb` and `RMSNorm`. The first two became
  `LlamaRotaryEmbedding` plus `deinterleave_head_dim` as described above; the third, an L2 normalization scaled by
  `sqrt(dim)` and a learned gain, became `F5TTSUNetRMSNorm`.
- `torchdiffeq` supplied `odeint`. `F5TTSFixedStepODESolver` replaces it: with no `step_size` and no
  `grid_constructor`, `odeint` runs its fixed grid solver on the time points it is handed, takes one Euler or
  midpoint step per interval, and linearly interpolates onto the requested points, which is what the class does in
  forty lines.
- `vocos` supplied the vocoder, now `VocosModel` in `voicestudio/models/vocos`, which F5-TTS composes as a
  sub-model.
- `librosa` and `torchaudio.transforms.MelSpectrogram` supplied the two mel front ends. Both became
  `F5TTSFeatureExtractor`, over `torchaudio.functional.melscale_fbanks`: `norm=None, mel_scale="htk"` reproduces
  the `"vocos"` front end and `norm="slaney", mel_scale="slaney"` reproduces `librosa.filters.mel`'s defaults, which
  is the `"bigvgan"` front end.
- `ema_pytorch`, `accelerate`, `wandb`, `hydra-core`, `bitsandbytes` and `datasets` were the training loop.
  Training goes through `forward` and a `transformers` trainer now.
- `cached_path` became `huggingface_hub.hf_hub_download`, `omegaconf` and `tomli` became the `ARCHITECTURES` table
  in `weight_conversion.py`, and `pydub`, `matplotlib`, `soundfile`, `gradio`, `click`,
  `transformers_stream_generator` and `unidecode` belonged to the CLI, the Gradio apps, the socket server and the
  evaluation scripts, none of which is carried over.
- `rjieba` and `pypinyin` could not be removed and were **not** added to `pyproject.toml`. They are what turns
  Chinese characters into the tone numbered pinyin every released checkpoint's vocabulary is built from, and no
  package in `transformers` does word segmentation plus tone sandhi for Mandarin. `F5TTSTokenizer` imports them
  lazily, so English and other Latin script text works without them and Chinese text raises a named `ImportError`.
  Whether to add them, and whether a Chinese-capable F5-TTS is in scope at all, is left open.


## Verification

`convert` then `from_pretrained` on the real `SWivid/F5-TTS` `F5TTS_v1_Base` and `F5TTS_Base` weights reports no
missing, unexpected or mismatched keys across all 363 tensors and 337096804 parameters each, and on
`SWivid/E2-TTS` `E2TTS_Base` across all 362 tensors and 333470444 parameters. With the Vocos vocoder composed in,
which adds 80 tensors and 13531650 parameters, that is 443 tensors and 350628454 parameters for the two F5-TTS
checkpoints and 442 tensors and 347002094 parameters for `E2TTS_Base`, all three loading clean.

The migrated backbones were checked against the upstream classes themselves, run from the same weights with
`x_transformers` inlined. Over the plain, speech-dropped, text-and-speech-dropped, classifier free guidance
packing and unmasked paths, `F5TTS_v1_Base` agrees to at most 1.2e-05 on outputs whose magnitude is 13.4,
`F5TTS_Base`, which is the `pe_attn_head=1` and `text_mask_padding=False` variant, to at most 9.1e-05 on outputs
whose magnitude is 12.9, and `E2TTS_Base` on the `"unett"` backbone to at most 3.0e-04 on outputs whose magnitude
is 12.5. Layer by layer the disagreement grows smoothly from 5.5e-07 at the first layer at a constant 2e-07 of the
activation scale, and the text embedding and input embedding agree bit for bit, which is float32 accumulation
order and not a structural difference.

The configuration fields no released checkpoint exercises were checked the same way, from weights shared between
the two implementations: `attn_mask_enabled`, `qk_norm="rms_norm"`, `long_skip_connection`, a `pe_attn_head`
between 1 and the head count, `text_mask_padding`, `text_average_upsampling`, `text_conv_layers=0`, and all three
`skip_connect_type` settings of the `"unett"` backbone. Every one agrees to under 1e-06.

`F5TTSFeatureExtractor` agrees with upstream's `MelSpec` to 2.4e-07 on the `"vocos"` front end and 1.2e-05 on the
`"bigvgan"` one. `F5TTSFixedStepODESolver` was checked against a transcription of `torchdiffeq`'s
`FixedGridODESolver.integrate`: on a uniform grid, on the empirically pruned grid and on a sway reshaped grid, in
both `"euler"` and `"midpoint"` mode, the two agree bit for bit, and both match the closed form Euler and midpoint
solutions of `dy/dt = y`.

The training objective was checked the same way: seeded identically, `forward(labels=...)` returns exactly the
loss upstream `CFM.forward` returns, bit for bit, with the same span mask and the same conditioning spectrogram.
On the real `F5TTS_v1_Base` weights all 363 parameters receive a nonzero gradient from `loss.backward()`, and all
362 do on `E2TTS_Base`. Over 24 draws against the demo reference clip the loss is 0.8206 on that clip's own
transcription, 0.9548 on a different sentence and 0.9261 on random character ids, so the objective is reading the
text it is conditioned on.

Generating the upstream demo text in the upstream demo reference voice, at 32 steps with `guidance_scale=2.0` and
`sway_sampling_coef=-1.0`, transcribes back under wav2vec2 as I DON'T REALLY CARE WHAT YOU CALL ME I'VE BEEN A
SILENT SPECTATOR WATCHING SPECIES EVOLVE EMPIRES RISE AND FALL BUT ALWAYS REMEMBER I AM MIGHTY AND ENDURING, word
for word, from the F5-TTS v1 Base, the F5-TTS Base and the E2-TTS Base checkpoints alike. That path runs the
tokenizer, the feature extractor, the sampler and the composed `VocosModel` end to end.

Checkpoint search, per CLAUDE.md section 2.3: the Hugging Face hub holds `SWivid/F5-TTS`, with `F5TTS_Base`,
`F5TTS_Base_bigvgan`, `F5TTS_v1_Base` and `F5TTS_v1_Base_no_zero_init`, and `SWivid/E2-TTS`, with `E2TTS_Base`;
`charactr/vocos-mel-24khz` and `nvidia/bigvgan_v2_24khz_100band_256x` hold the two vocoders. The upstream README
points at those same repositories, and `src/f5_tts/infer/SHARED.md` lists community finetunes of the same
architecture. No Space, Zenodo record or paper appendix was needed. No checkpoint exists for the `MMDiT` backbone
or for `F5TTS_Small`, `F5TTS_v1_Small` and `E2TTS_Small`, whose training configs upstream ships without weights.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **The MMDiT backbone.** `src/f5_tts/model/backbones/mmdit.py` is a third backbone, exported by upstream's
  `f5_tts.model` package and selectable from a training config as `backbone: MMDiT`. It is a joint attention
  MM-DiT: text and speech each get their own q/k/v projections and their own adaptive layer norms, attention runs
  over the concatenated pair, and the last block drops the text branch. `F5TTSConfig.backbone` accepts only
  `"dit"` and `"unett"`, so MMDiT is absent. No configuration in `src/f5_tts/configs/` selects it and no published
  checkpoint uses it, but that is not the same as it not existing, and whether to add it is a scope decision.
- **BigVGAN.** `F5TTS_Base_bigvgan` is a real published checkpoint whose mel front end is the `"bigvgan"` one, and
  `F5TTSFeatureExtractor` reproduces that front end exactly. The vocoder itself is not migrated: upstream pulls
  `nvidia/bigvgan_v2_24khz_100band_256x` in as a git submodule of custom CUDA kernels, anti-aliased Snake
  activations and a multi-band generator, which is a model migration of its own rather than a piece of inlined
  math. `convert` will produce that checkpoint and pair it with the Vocos vocoder, which is the wrong vocoder for
  it. Nothing stops a caller passing their own vocoder to `F5TTSProcessor.batch_decode` instead.
- **Chinese text.** See the `rjieba` and `pypinyin` entry under Dependencies. The tokenizer raises rather than
  silently mistokenizing.
- **Automatic transcription of the reference clip.** `preprocess_ref_audio_text` ran
  `openai/whisper-large-v3-turbo` through a `transformers` pipeline when `ref_text` was empty. `F5TTSProcessor`
  requires `ref_text` instead of transcribing, so a caller has to produce it.
- **Silence based trimming of the reference clip.** The same upstream function used `pydub` to split the clip on
  silence, keep the first segment under twelve seconds, strip leading and trailing silence and append fifty
  milliseconds of it. `F5TTSProcessor` takes the clip as given, so a long or silence-padded reference is passed
  through unchanged where upstream would have shortened it.
- **`duplicate_test` and `t_inter`.** `CFM.sample` could start the flow partway along the path from a duplicated
  conditioning spectrogram, for observing intermediate time steps. `generate` always starts at 0.
- **Streaming, cross-chunk duration allocation and the applications.** `infer_batch_process` could yield fixed
  size chunks for the socket server and could spread a `fix_duration` across chunks by byte weight;
  `F5TTSProcessor` takes one `fix_duration` per entry and returns whole waveforms. `api.py`, `infer_cli.py`,
  `infer_gradio.py`, `speech_edit.py`, `socket_server.py`, `socket_client.py`, `train/finetune_cli.py`,
  `train/finetune_gradio.py`, `train/train.py`, `train/datasets/*`, `eval/*`, `scripts/*` and
  `runtime/triton_trtllm/*` are dropped, along with the Dockerfile, the pre-commit config and the GitHub workflows.
  Their model-behavioural content is accounted for above and in the file map below.
- **`text_max_positions` for the `"unett"` backbone.** Upstream hard codes 8192 sinusoidal text positions in the
  `"dit"` text embedding and 4096 in the `"unett"` one. It is one config field here, defaulting to 8192, and
  `ARCHITECTURES` sets 4096 for the two E2-TTS entries. The table's entries agree with the position clamp only up
  to that length, which no released checkpoint approaches.
- **The reference frame count `infer_batch_process` estimates durations from.** Upstream takes
  `ref_audio_len = num_samples // hop_length` while the mel spectrogram it conditions on has `num_samples //
  hop_length + 1` frames, so it asks for one frame fewer than the reference occupies and leaves the first of those
  frames, about eleven milliseconds of the reference clip, at the head of every generated chunk.
  `F5TTSProcessor.__call__` returns the real frame count as `reference_length`, which shifts the estimated
  duration by up to half a percent and trims the whole reference off. Which of the two to keep is a decision for a
  human, not a bug fixed here.


## File map

One counterpart per upstream file, per CLAUDE.md section 2.4. Eight files were moved onto their transformers names
by `git mv` and edited in place: `dit.py`, `modules.py`, `cfm.py`, `utils.py`, `utils_infer.py`, `trainer.py`,
`F5TTS_v1_Base.yaml` and `model/__init__.py`. The other 87 paths were removed, and every one of them is a row
below.

| Upstream file | Where it went |
|---|---|
| `src/f5_tts/model/backbones/dit.py` | `modeling_f5_tts.py`: `F5TTSModel`, `F5TTSTextEmbedding`, `F5TTSInputEmbedding` |
| `src/f5_tts/model/backbones/unett.py` | `modeling_f5_tts.py`: `F5TTSUNetModel`, `F5TTSUNetTextEmbedding`, `F5TTSUNetLayer`, `F5TTSUNetRMSNorm` |
| `src/f5_tts/model/backbones/mmdit.py` | Not carried over, see above |
| `src/f5_tts/model/modules.py` | `modeling_f5_tts.py` for every block, `feature_extraction_f5_tts.py` for `MelSpec` and its two extractors |
| `src/f5_tts/model/cfm.py` | `generation_f5_tts.py` for `CFM.sample`, `F5TTSForConditionalGeneration.forward` for `CFM.forward` |
| `src/f5_tts/model/utils.py` | `tokenization_f5_tts.py` for the vocabulary and pinyin conversion, `modeling_f5_tts.py` for `lens_to_mask` and `mask_from_frac_lengths`, `generation_f5_tts.py` for `get_epss_timesteps` |
| `src/f5_tts/model/trainer.py` | `weight_conversion.py` for the checkpoint format, `forward` plus a `transformers` trainer for the loop |
| `src/f5_tts/model/dataset.py` | `collate_fn` becomes the `labels` and `attention_mask` arguments of `forward` |
| `src/f5_tts/infer/utils_infer.py` | `processing_f5_tts.py` for `chunk_text`, `preprocess_ref_audio_text`, `infer_process` and the cross fade, `weight_conversion.py` for `load_checkpoint`, `voicestudio/models/vocos` for `load_vocoder` |
| `src/f5_tts/configs/*.yaml` | `ARCHITECTURES` in `weight_conversion.py`, and `F5TTSConfig` |
| `src/f5_tts/infer/examples/vocab.txt`, `data/Emilia_ZH_EN_pinyin/vocab.txt` | The `vocab.txt` each published checkpoint ships, which `convert` downloads |
| `src/f5_tts/infer/speech_edit.py` | The `edit_mask` and `fix_duration` arguments of `generate` and `F5TTSProcessor.__call__`, minus the script that builds the frame level mask |
| `src/third_party/BigVGAN` | Not carried over, see above. The `"bigvgan"` mel front end of `F5TTSFeatureExtractor` is the only part of it that has a counterpart |
| `INFO.md` | This file. It was the upstream `README.md`, renamed when the tree was merged in |
| `src/f5_tts/infer/README.md`, `infer/SHARED.md`, `model/backbones/README.md`, `eval/README.md`, `train/README.md` | This file's Usage, Training and Lineage sections. The community finetune list in `SHARED.md` has no counterpart |
| `ckpts/README.md` | `PUBLISHED_CHECKPOINTS` in `weight_conversion.py`, which names the repository and file of every released checkpoint |
| `src/f5_tts/api.py` | `F5TTSForConditionalGeneration.from_pretrained` plus `F5TTSProcessor`, which is the same three-call surface without the `F5TTS` wrapper class |
| `src/f5_tts/infer/infer_cli.py`, `infer_gradio.py`, `socket_server.py`, `socket_client.py`, `train/finetune_cli.py`, `train/finetune_gradio.py` | Dropped, no counterpart. Applications, not model code |
| `src/f5_tts/train/train.py` | The objective is `forward(labels=...)`. The `accelerate` loop, the EMA shadow copy, the AdamW plus linear warmup and decay schedule and the checkpoint rotation have no counterpart |
| `src/f5_tts/train/datasets/prepare_*.py` | Dropped, no counterpart. Six dataset preparation scripts for Emilia, LibriTTS, LJSpeech, WenetSpeech4TTS and custom csv or wav trees |
| `src/f5_tts/eval/*`, `data/librispeech_pc_test_clean_cross_sentence.lst` | Dropped, no counterpart. The word error rate, speaker similarity and UTMOS harness, its ECAPA-TDNN speaker encoder and its test set metadata |
| `src/f5_tts/scripts/*` | Dropped, no counterpart. Three parameter, FLOP and epoch counting utilities |
| `src/f5_tts/runtime/triton_trtllm/*` | Dropped, no counterpart. Twenty two files of Triton and TensorRT-LLM deployment recipe, including its own ONNX vocoder export and checkpoint converter |
| `src/f5_tts/infer/examples/*` | Dropped, no counterpart. Four sample clips, three toml presets and a story text |
| `Dockerfile`, `ruff.toml`, `.pre-commit-config.yaml`, `.gitignore`, `.gitmodules`, `.github/*` | Dropped, no counterpart. The upstream project's own packaging, linting and CI, which this repository supplies itself |


## Repository integration

Two things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .f5_tts import *` line, and a `from .vocos import *` line for
  the vocoder this model composes.
- `PROJECT.md` needs an F5-TTS status entry carrying the gaps listed above, in particular the MMDiT backbone, the
  BigVGAN vocoder and the `rjieba`/`pypinyin` decision, and a Vocos entry carrying that model's own gaps.
