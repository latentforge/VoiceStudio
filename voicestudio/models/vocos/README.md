# Vocos

Vocos is a neural vocoder that never touches the time domain until the very last step. A ConvNeXt backbone reads a
frame level feature sequence, keeps the frame rate constant through every layer, and a linear head reads its output
as the log magnitude and the phase of a short time Fourier transform, which a single inverse STFT turns into a
waveform. There is no transposed convolution stack and no upsampling: one frame of features becomes one frame of
spectral coefficients, and the overlap add does the rest.

The front end in front of the backbone is swappable, and the two published checkpoints differ only there.
`charactr/vocos-mel-24khz` consumes a 100 channel log mel spectrogram. `charactr/vocos-encodec-24khz` consumes the
sum of the EnCodec codebook embeddings of a frame, read out of one table that holds every codebook of every
bandwidth, and conditions every normalization in the backbone on which bandwidth the codes were encoded at through
an adaptive layer norm with one learned gain and bias per bandwidth.

Original model and code: [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)


## Usage

`from_pretrained` takes either published repository id as it stands. The repositories hold a `config.yaml`
naming three classes and a `pytorch_model.bin` rather than a `config.json` and a `model.safetensors`, so
`VocosModel.from_pretrained` reads that `config.yaml` and builds the configuration itself; no conversion step
comes first.

The mel checkpoint vocodes a log mel spectrogram, and its `VocosFeatureExtractor` computes one from a waveform:

```python
import soundfile as sf

from voicestudio.models.vocos import VocosFeatureExtractor, VocosModel

model_id = "charactr/vocos-mel-24khz"

model = VocosModel.from_pretrained(model_id).eval()
extractor = VocosFeatureExtractor.from_pretrained(model_id)

waveform, sampling_rate = sf.read("reference.wav", dtype="float32")
inputs = extractor(waveform, sampling_rate=sampling_rate)
audio = model(**inputs).audio_values

sf.write("output.wav", audio[0].numpy(), model.config.sampling_rate)
```

The EnCodec checkpoint vocodes codes rather than a spectrogram, so it takes `audio_codes` and the `bandwidth_id`
they were encoded at, which indexes `config.bandwidths`:

```python
import torch
from transformers import EncodecModel

from voicestudio.models.vocos import VocosModel

codec = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
model = VocosModel.from_pretrained("charactr/vocos-encodec-24khz").eval()

with torch.no_grad():
    audio_codes = codec.encode(torch.as_tensor(waveform).reshape(1, 1, -1), bandwidth=6.0).audio_codes[0]
    audio = model(audio_codes=audio_codes, bandwidth_id=torch.tensor([2])).audio_values
```

Two arguments above are load-bearing:

- `bandwidth_id` has to match the bandwidth the codes were encoded at. It selects the adaptive layer norm gain and
  bias the backbone was trained with for that many codebooks, and `config.bandwidths` is the list it indexes, so
  6.0 kbps, which is eight codebooks, is `2`. A wrong id raises nothing and degrades the output.
- `attention_mask`, which `VocosFeatureExtractor` returns for a padded batch, is not a model argument. The
  backbone is fully convolutional and the head's overlap add runs over the whole frame axis, so padded frames
  become padded samples that the caller has to trim.

A first load converts the published layout into a directory under `HF_HOME`, keyed on the repository and the commit it
resolved to, and later loads read that directory through the ordinary loading path instead of converting again.

`weight_conversion.convert` writes that same conversion to a directory of the caller's choosing, for a checkpoint
that is shipped elsewhere or kept outside the cache. It takes either key of `PUBLISHED_CHECKPOINTS` (`"mel"`,
`"encodec"`), a repository id, or a local directory holding the two published files, and both `from_pretrained`
calls above read the directory it writes as readily as the published repository:

```python
from voicestudio.models.vocos.weight_conversion import convert

convert("mel", "vocos-mel-24khz-converted")
```


## Training

Pass the ground truth waveform as `labels` and the standard `forward` returns a loss:

```python
outputs = model(input_features=inputs["input_features"], labels=waveform)
outputs.loss.backward()
```

**What that loss is.** `vocos/loss.py:MelSpecReconstructionLoss` is the L1 distance between the log mel
spectrograms of the target and the generated waveform: a 24 kHz, 1024 point, 256 hop, 100 channel magnitude mel
spectrogram, mapped through `safe_log`, which clips at 1e-7 before taking the logarithm. `VocosMelSpectrogram`
reproduces that resolution, and it is deliberately independent of `config.n_fft` and `config.hop_length`, which
the head synthesizes at: on the EnCodec checkpoint the head runs at 1280 and 320 while the loss stays at 1024 and
256. `VocosModel.forward` returns `config.mel_loss_coeff` times that distance, and `mel_loss_coeff` is 45 in both
released training configs.

**What the full upstream objective is, term by term.** `vocos/experiment.py:VocosExp` is a generative adversarial
network with two optimizers alternating per step, and the reconstruction term above is one of five in the
generator half. `training_step` with `optimizer_idx == 1` sums

- `loss_gen_mp`, the hinge generator loss `mean(relu(1 - D(y_hat)))` over the five sub-discriminators of a
  `MultiPeriodDiscriminator` at periods 2, 3, 5, 7 and 11, divided by their count,
- `mrd_loss_coeff` times the same over the three sub-discriminators of a `MultiResolutionDiscriminator` at FFT
  sizes 2048, 1024 and 512, `mrd_loss_coeff` being 0.1 in `configs/vocos.yaml` and 1.0 in
  `configs/vocos-encodec.yaml`,
- `loss_fm_mp`, the feature matching loss `mean(abs(f_real - f_generated))` summed over every layer of the multi
  period discriminator and divided by its sub-discriminator count,
- `mrd_loss_coeff` times the same over the multi resolution discriminator,
- `mel_loss_coeff` times the mel reconstruction loss, optionally cosine decayed over training when
  `decay_mel_coeff` is set, which neither released config sets.

`optimizer_idx == 0` trains the two discriminators against the hinge discriminator loss
`mean(relu(1 - D(y))) + mean(relu(1 + D(y_hat)))`, again summed over sub-discriminators, divided by their count
and weighted by `mrd_loss_coeff`. `pretrain_mel_steps`, which would run the reconstruction term alone for a while
first, is 0 in both released configs, so the adversarial terms are live from the first step. The encodec variant
draws a `bandwidth_id` uniformly per training step and passes it to the backbone and to both discriminators, which
in that variant carry a conditioning embedding of their own.

**What upstream freezes.** `EncodecFeatures.__init__` sets `requires_grad = False` on every parameter of the
EnCodec model it builds, and calls `self.encodec.eval()` on every forward so that Lightning cannot put it back
into training mode. `codebook_weights`, the table this migration keeps, is created with
`requires_grad=train_codebooks`, and `configs/vocos-encodec.yaml` sets `train_codebooks: false`, so the released
model froze it too. The generator optimizer is given `feature_extractor.parameters()` plus
`backbone.parameters()` plus `head.parameters()`, but the mel front end has no parameters at all and the EnCodec
front end's only one is frozen, so what actually trains is the backbone and the head.


## Lineage

Nothing in `transformers` is a ConvNeXt plus inverse STFT vocoder. `univnet` and the `speecht5` HiFi-GAN are the
two standalone vocoders it ships, and both are time domain generators built from transposed convolutions and
location variable or dilated residual blocks, with no Fourier head to inherit. `vits` decodes with a HiFi-GAN of
its own, and `encodec`, `dac`, `mimi`, `xcodec`, `xcodec2`, `higgs_audio_v2_tokenizer`,
`qwen3_tts_tokenizer_multi_codebook` and `vibevoice_acoustic_tokenizer` are codecs whose decoders are the mirror
of their own encoders. `bark`'s `fine_acoustics` stack is a transformer, not a vocoder.

Under `voicestudio/models/`, `spark_tts` vendored a `vocos.py` of its own, but it is Spark-TTS's own copy of the
ConvNeXt block inside `BiCodec`, not a Vocos vocoder with a Fourier head, and it stays where it is.

The folder layout follows `models/univnet` in `transformers-tts`, the one standalone vocoder there that is a
first-class model with a `configuration_`, a `modeling_` and a `feature_extraction_` file, and
`models/higgs_audio_v2_tokenizer` and `models/qwen3_tts_tokenizer_multi_codebook`, which sit beside the model that
uses them rather than inside it.


## Dependencies

Upstream `requirements.txt` lists `torch`, `torchaudio`, `numpy`, `scipy`, `einops`, `pyyaml`, `huggingface_hub`
and `encodec==0.1.1`, and `requirements-train.txt` adds `pytorch_lightning`, `jsonargparse`, `transformers`,
`matplotlib`, `torchcrepe`, `pesq` and `fairseq`. This migration needs `torch`, `torchaudio`, `transformers`,
`safetensors`, `numpy`, `pyyaml` and `huggingface_hub`, all of which the repository already carries.
`pyproject.toml` and `uv.lock` need no change, and nothing was added.

- `encodec` supplied the EnCodec model the training front end tokenized audio with, and the codebooks it copied
  into `codebook_weights` at construction. `codebook_weights` is a parameter of the published checkpoint, so it is
  loaded rather than rebuilt, and callers that need codes from a waveform use the `EncodecModel` that
  `transformers` already ships.
- `scipy` supplied the cosine window of `MDCT` and `IMDCT`, which are only reachable from the IMDCT heads. Those
  heads are not carried over, so the import goes with them.
- `einops` is used only by the discriminators. It goes with them.
- `pyyaml` stays, and only inside `weight_conversion.py`, because a published Vocos repository's architecture
  lives in a `config.yaml`.
- `pytorch_lightning`, `jsonargparse`, `matplotlib`, `torchcrepe`, `pesq` and `fairseq` were the training loop,
  its command line and its validation metrics. `transformers` was in that list only for
  `get_cosine_schedule_with_warmup`. Training goes through `forward` and a `transformers` trainer now.


## Verification

`from_pretrained` straight onto the published repository ids, with no conversion call before it, reports no
missing, unexpected or mismatched keys on either: all 80 tensors and 13531650 parameters for
`charactr/vocos-mel-24khz`, and all 81 tensors and 10081410 parameters for `charactr/vocos-encodec-24khz`. The
extra tensor of the second is `feature_extractor.codebook_weights`, a 16384 by 128 table, which is 2097152 of
those parameters. It is a real trained parameter and not a buffer, so `_keys_to_ignore_on_load_unexpected` names
only `feature_extractor.mel_spec.` and `head.istft.window`, the mel filterbank, the analysis window and the
inverse STFT window, and never the whole `feature_extractor.` prefix. Read back after a direct load it is a
16384 by 128 tensor with standard deviation 1.64, which is what catches it going missing.

The migrated modules were checked against the upstream classes themselves, run from the same weights.
`VocosBackbone` plus `VocosISTFTHead` agree with upstream's `VocosBackbone` plus `ISTFTHead` to 1.5e-08 on outputs
whose magnitude is 0.097 for the mel checkpoint, and to 1.8e-07 on outputs whose magnitude is 1.2 for the EnCodec
checkpoint, which is the path that exercises the adaptive layer norms and the `"same"` padding overlap add. On the
EnCodec codes VoxInstruct actually generates, `VocosModel` in float32 agrees with upstream bit for bit, at a
maximum absolute difference of 0.0 on a waveform whose magnitude is 0.86. `codes_to_features` agrees with
upstream's `Vocos.codes_to_features` bit for bit. `VocosFeatureExtractor` agrees with upstream's
`MelSpectrogramFeatures` to 2.4e-07 on outputs whose magnitude is 3.6, under both `"center"` and `"same"` padding.

The training objective was checked the same way: `forward(labels=...)` returns exactly 45 times what upstream's
`MelSpecReconstructionLoss` returns for the same pair of waveforms, to 7.6e-06 on a loss of 54.9. On the real mel
weights all 80 parameters receive a nonzero gradient from `loss.backward()`. Against the F5-TTS demo reference
clip the loss is 5.72 on that clip's own waveform, 113.69 on the same clip shifted half a second and 160.15 on
noise of the same standard deviation, so the objective is reading the waveform it is handed.

Copy synthesis of that clip through the mel checkpoint gives a log mel L1 of 0.13700 against 3.193 for
same-energy noise, and `facebook/wav2vec2-base-960h` transcribes it back as SOME CALL ME NATURE OTHERS CALL ME
MOTHER NATURE, word for word with the transcription of the original recording. Through the EnCodec checkpoint at
6 kbps, 8 codebooks, `bandwidth_id=2`, it gives a log mel L1 of 0.5685 against 3.190 for the same noise control,
and transcribes the same. That figure does not reach the mel checkpoint's, and the shortfall is not the vocoder:
running `facebook/encodec_24khz`'s own decoder on the identical codes gives 0.6345, worse than Vocos, so what
separates 0.5685 from 0.13700 is what 6 kbps codes discard rather than anything Vocos loses on top of them.

Checkpoint search, per CLAUDE.md section 2.3: the Hugging Face hub holds `charactr/vocos-mel-24khz` and
`charactr/vocos-encodec-24khz`, which are the two the upstream README points at, and `charactr/vocos-encodec-24khz`
is the one bundled in the released VoxInstruct folder. `BSC-LT/vocos-mel-22khz` and a number of community
retrainings of the same two architectures exist on the hub as well. No Space, Zenodo record or paper appendix was
needed. No checkpoint exists for the `VocosResNetBackbone` or for either IMDCT head, whose training configs
upstream ships without weights.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6.

- **The adversarial half of the training objective.** `forward(labels=...)` returns the mel reconstruction term
  only. `MultiPeriodDiscriminator`, `MultiResolutionDiscriminator`, `GeneratorLoss`, `DiscriminatorLoss` and
  `FeatureMatchingLoss` have no counterpart, so the four remaining terms of the generator loss and the whole
  discriminator loss are absent, and with `pretrain_mel_steps: 0` in both released configs there is no phase of
  upstream training that optimizes the reconstruction term alone. The two discriminators are training-only
  modules that appear in no published checkpoint, and leaving them out follows the `transformers` convention on
  GAN trained vocoders, measured over the 494 `modeling_*.py` files in the 510 model folders of `transformers`
  5.16.1. `Discriminator` appears in two of those files, `electra/modeling_electra.py` and
  `funnel/modeling_funnel.py`, and both are pretraining heads over token logits, `ElectraDiscriminatorPredictions`
  and `FunnelDiscriminatorPredictions` under `FunnelForPreTraining`, not adversaries over a waveform.
  `adversarial` appears in none of them. Every vocoder shipped takes a spectrogram or codes and returns a bare
  tensor, and none takes `labels`:

  | Class | `forward` signature |
  |---|---|
  | `SpeechT5HifiGan` | `(self, spectrogram, **kwargs) -> torch.FloatTensor` |
  | `FastSpeech2ConformerHifiGan` | `(self, spectrogram, **kwargs) -> torch.FloatTensor` |
  | `VitsHifiGan` | `(self, spectrogram, global_conditioning=None) -> torch.FloatTensor` |
  | `SeamlessM4THifiGan`, `SeamlessM4Tv2HifiGan` | `(self, inputs_embeds) -> torch.FloatTensor` |
  | `SeamlessM4TCodeHifiGan` | `(self, input_ids, spkr_id, lang_id, **kwargs) -> tuple[torch.Tensor]` |
  | `SeamlessM4Tv2CodeHifiGan` | `(self, input_ids, speaker_id, lang_id, **kwargs) -> tuple[torch.Tensor]` |
  | `Qwen2_5OmniToken2WavBigVGANModel` | `(self, mel_spectrogram, **kwargs)`, returning a clamped waveform |

  VITS is the case that settles it, because upstream VITS is GAN trained end to end: `VitsModel.forward` declares
  `labels` and its body opens with `raise NotImplementedError("Training of VITS is not supported yet.")`. The one
  shipped speech synthesis model with a real objective, `FastSpeech2ConformerWithHifiGan`, sums L1 mel, duration,
  pitch and energy in `FastSpeech2ConformerLoss` and carries no adversarial term, and its vocoder half takes no
  labels. Against that convention, `VocosModel.forward(labels=...)` returning the mel reconstruction loss goes
  beyond it rather than falling short of it. The consequence to know is that a Vocos trained through `forward`
  alone optimizes the reconstruction term only and would not reproduce a released checkpoint.
- **`VocosResNetBackbone`.** `configs/vocos-resnet.yaml` trains a HiFi-GAN style dilated residual backbone with
  weight normalization in place of the ConvNeXt one. No checkpoint was published for it and neither published
  checkpoint uses it.
- **`IMDCTSymExpHead` and `IMDCTCosHead`, and the `MDCT` and `IMDCT` transforms they need.**
  `configs/vocos-imdct.yaml` trains a head that predicts modified discrete cosine transform coefficients instead
  of STFT coefficients. No checkpoint was published for it.
- **`EncodecFeatures`' own EnCodec model, and `get_encodec_codes`.** The migrated `VocosEncodecFeatures` holds the
  codebook table and nothing else, so it maps codes to features but cannot produce codes from a waveform. That
  path is `transformers`' own `EncodecModel`, which is what the Usage section above and `VoxInstruct` both use,
  but it means `VocosModel` has no single call that reproduces upstream `Vocos.forward` on raw audio for the
  EnCodec variant.
- **The bandwidth draw during training.** `VocosEncodecExp.training_step` draws `bandwidth_id` uniformly from
  `range(len(bandwidths))` for every step and `validation_step` fixes it at 0. `forward` takes the
  `bandwidth_id` it is given, so a caller assembling a training batch has to make that draw.
- **The optimizer, the schedule and the two-optimizer loop.** `configure_optimizers` builds two AdamW optimizers
  at `betas=(0.8, 0.9)` under a cosine schedule with warmup over `trainer.max_steps // 2` steps each, and
  Lightning alternates them. Neither the alternation nor the split parameter groups is what `transformers.Trainer`
  sets up, and reproducing them means a custom training loop.
- **The validation metrics.** `metrics/UTMOS.py` and `metrics/periodicity.py`, and the PESQ scoring in
  `validation_step`, compute UTMOS, PESQ, periodicity, pitch error and voiced/unvoiced F1. They are dropped along
  with `torchcrepe`, `pesq` and the UTMOS checkpoint download.
- **`vocos/dataset.py` and the training entry point.** `VocosDataModule` read a filelist of wav paths and cropped
  random `num_samples` windows, and `train.py` was a `pytorch_lightning` CLI. Those are `Trainer` and data
  collator territory now, and `configs/*.yaml` went with them.


## File map

One counterpart per upstream file, per CLAUDE.md section 2.4. The Vocos port that this folder is built from
already lived inside `voicestudio/models/f5_tts/modeling_f5_tts.py` and
`voicestudio/models/f5_tts/configuration_f5_tts.py` as `F5TTSVocosConvNeXtBlock`, `F5TTSVocosBackbone`,
`F5TTSVocosISTFTHead`, `F5TTSVocosModel` and `F5TTSVocosConfig`, which is a fraction of two files that F5-TTS
still needs, so it could not be moved with `git mv` and was lifted out into the files below instead.

| Upstream file | Where it went |
|---|---|
| `vocos/models.py` | `modeling_vocos.py`: `VocosBackbone`. `VocosResNetBackbone` is not carried over, see above |
| `vocos/heads.py` | `modeling_vocos.py`: `VocosISTFTHead`. The two IMDCT heads are not carried over, see above |
| `vocos/modules.py` | `modeling_vocos.py`: `VocosConvNeXtBlock`, `VocosAdaLayerNorm`, `safe_log`. `ResBlock1` goes with the ResNet backbone, `symlog` and `symexp` with the IMDCT heads |
| `vocos/spectral_ops.py` | `modeling_vocos.py`: the `"center"` and `"same"` branches of `VocosISTFTHead.forward` are `ISTFT`. `MDCT` and `IMDCT` go with the IMDCT heads |
| `vocos/feature_extractors.py` | `feature_extraction_vocos.py` for `MelSpectrogramFeatures`, `modeling_vocos.py`'s `VocosEncodecFeatures` for the codebook table of `EncodecFeatures` |
| `vocos/pretrained.py` | `VocosModel` for the three-module composition and `decode`, `VocosModel.codes_to_features` for `codes_to_features`, `weight_conversion.py` for `from_hparams` and `from_pretrained` |
| `vocos/loss.py` | `modeling_vocos.py`: `VocosMelSpectrogram` plus the `labels` branch of `VocosModel.forward` for `MelSpecReconstructionLoss`. The three adversarial losses are not carried over, see above |
| `vocos/experiment.py` | The `labels` branch of `VocosModel.forward` is the reconstruction term of `training_step`. The optimizers, the schedule, the two-optimizer alternation and the logging have no counterpart |
| `vocos/discriminators.py` | Not carried over, see above |
| `vocos/dataset.py` | Not carried over, see above. `VocosDataModule` is a filelist reader and a random crop |
| `vocos/helpers.py` | Dropped, no counterpart. A spectrogram plotting helper and a gradient norm Lightning callback |
| `configs/vocos.yaml`, `configs/vocos-encodec.yaml` | `VocosConfig`, and `build_config` in `weight_conversion.py`, which reads the same block layout out of a published repository's `config.yaml` |
| `configs/vocos-resnet.yaml`, `configs/vocos-imdct.yaml` | Not carried over, see above |
| `train.py` | The objective is `forward(labels=...)`. The `pytorch_lightning` CLI has no counterpart |
| `metrics/UTMOS.py`, `metrics/periodicity.py` | Not carried over, see above |
| `README.md`, `CONTRIBUTING.md` | This file's Usage, Training and Lineage sections |
| `notebooks/Bark+Vocos.ipynb` | Dropped, no counterpart. A demo notebook that vocodes Bark's EnCodec output |
| `tests/test_heads.py`, `tests/test_spectral_ops.py` | Dropped, no counterpart. They check that the ISTFT and IMDCT round trip, which the Verification section checks against the upstream classes instead |
| `setup.py`, `requirements*.txt`, `.gitignore`, `.github/*` | Dropped, no counterpart. The upstream project's own packaging and CI, which this repository supplies itself |
| `LICENSE` | The header of `modeling_vocos.py`, per CLAUDE.md section 6 |


## Repository integration

Two things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .vocos import *` line, before the `f5_tts` and `vox_instruct`
  lines are reached, though `from ..vocos import ...` inside those two packages already imports it either way.
- `PROJECT.md` needs a Vocos status entry carrying the items listed above.

Nothing in `pyproject.toml` or `uv.lock` changes.
