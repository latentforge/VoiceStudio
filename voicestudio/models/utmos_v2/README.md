# UTMOSv2

UTMOSv2 predicts the mean opinion score a listening panel would give a speech clip, without a reference recording.
It fuses two branches. The first is wav2vec 2.0, whose thirteen hidden states are mixed with learned weights, passed
through a self-attention layer, and pooled as the concatenation of the attended mean and the unattended maximum. The
second treats a mel spectrogram as a photograph: four resolutions of the same excerpt, each rendered as a 512 by 512
three channel image, are read by four separate ImageNet-pretrained EfficientNetV2-S encoders, mixed with learned
weights, concatenated across two excerpts of the clip, pooled along the frequency axis as both an average and a
maximum, and passed through a second self-attention layer. A one-hot vector naming which listening-test corpus the
prediction should imitate is concatenated onto the two pooled halves, and one linear layer reads the result. The
authors train five cross-validation folds and average them, which is the system that won track one of the VoiceMOS
Challenge 2024.

Original model and code: [sarulab-speech/UTMOSv2](https://github.com/sarulab-speech/UTMOSv2)


## Usage

```python
import soundfile as sf
from voicestudio.models.utmos_v2 import UTMOSv2FeatureExtractor, UTMOSv2ForAudioClassification

model_id = "sarulab-speech/UTMOSv2"

extractor = UTMOSv2FeatureExtractor.from_pretrained(model_id)
model = UTMOSv2ForAudioClassification.from_pretrained(model_id).eval().to("cuda")

waveform, sampling_rate = sf.read("speech.wav")
inputs = extractor(waveform, sampling_rate=sampling_rate).to(model.device)
score = model(**inputs).logits.item()
```

Two things about that call are worth knowing.

The features are drawn at random. Both branches read an excerpt from a random position, and each spectrogram is
mixed with a second random excerpt, so calling the extractor twice on one waveform gives two different scores.
That is what the model was trained and evaluated under, and it is not a bug to work around: upstream averages
repeated calls, and so should any number that gets recorded.

```python
import numpy as np
import torch

with torch.no_grad():
    scores = [
        model(**extractor(waveform, sampling_rate=sampling_rate).to(model.device)).logits.item()
        for _ in range(8)
    ]
print(np.mean(scores), np.std(scores))
```

Pass a seeded `generator` to make one call reproducible, and pass `domain=` to name the listening test the score
should imitate. It defaults to `"sarulab"`, which is what upstream's `predict` defaults to; the ten corpora it can
name are in `DOMAINS`.

```python
inputs = extractor(waveform, sampling_rate=16000, domain="bvcc", generator=np.random.default_rng(0))
```

`from_pretrained` on the repository id above works on its own. The published repository holds five loose `.pth`
files and no `config.json`, so the first call converts them into a directory under `HF_HOME` and loads that;
later calls reuse it. `weight_conversion.convert` writes the same directory somewhere of your choosing.


## Training

`forward` takes `labels` and returns the objective upstream optimizes, which is two terms summed:

- a pairwise ranking term at weight `0.7`, `relu(|(s_i - s_j) - (t_i - t_j)| - 0.2)` averaged over every ordered
  pair of the batch and halved, so a batch of one contributes nothing to it, and
- a mean squared error term at weight `0.2`.

The weights do not sum to one; that is upstream's, not a transcription slip. Upstream freezes nothing: the wav2vec
2.0 encoder trains with `freeze=False`, and `fusion_stage3` fine-tunes both branches end to end for two epochs at
`lr=5e-5` after initialising them from separately trained `ssl_only_stage2` and `spec_only` runs.

Set `num_folds=1` to build the single predictor a fold of that training operates on. Training the five-fold
ensemble jointly is not what upstream does, and the loss here is computed on the averaged prediction, so a
`num_folds=5` training run optimizes something upstream never did.

```python
from voicestudio.models.utmos_v2 import UTMOSv2Config, UTMOSv2ForAudioClassification

model = UTMOSv2ForAudioClassification(UTMOSv2Config(num_folds=1))
loss = model(**inputs, labels=torch.tensor([3.8, 2.1])).loss
loss.backward()
```


## Verification

Every number below is on the published `fold0_s42_best_model.pth`, run through the `colab` CLI on an L4.

- **Load report.** `from_pretrained("sarulab-speech/UTMOSv2")` reports no missing and no unexpected keys. All 3343
  keys of an upstream fold checkpoint map onto this model's parameters, with no key and no shape left over in
  either direction, and one predictor holds 203.68M parameters against timm's 21.46M for `tf_efficientnetv2_s`
  less its 1.28M classifier, four times over, plus the wav2vec 2.0 branch.
- **The model against upstream's own class.** Handed identical `input_values`, `input_features` and domain
  vector, `UTMOSv2Model` and upstream's `SSLMultiSpecExtModelV2` built by `utmosv2.create_model` from the same
  weights agree to 4.6e-06 absolute, 0.8902318 against 0.8902364. Running both under `torch.amp.autocast`, which
  is what upstream's own `_predict_impl` does, moves each by 0.001 and does not separate them.
- **The mel front end against librosa.** `mel_spectrogram` agrees with
  `librosa.feature.melspectrogram` followed by `power_to_db(ref=np.max)` and `(x + 80) / 80` to 8.5e-06 on a
  `[0, 1]` scale, at every one of the four window lengths. The resize agrees with
  `torchvision.transforms.Resize((512, 512))` to 7.9e-06, and `remove_silent_sections` returns byte identical
  output to upstream's `remove_silent_section` on real clips, 71432, 58890 and 152383 samples kept out of 93680,
  77040 and 199760.
- **Not yet measured.** An end-to-end paired comparison of this pipeline's score against upstream's `predict` on
  real speech, over enough draws to resolve the difference. A first attempt at sixteen draws per estimate showed
  a gap of up to 0.28 that did not survive a rerun, in which upstream's own estimate for one clip moved from
  3.823 to 3.615, so sixteen draws does not resolve 0.1 MOS and no conclusion should be drawn from it either way.


## Not carried over from upstream

- **Every configuration but `fusion_stage3`.** The upstream repository carries 26 configurations: `spec_only`,
  `ssl_only_stage1`, `ssl_only_stage2`, `fusion_stage2`, the `c_*` variants that select `ssl_multispec_ext`
  (`SSLMultiSpecExtModelV1`) rather than `_v2`, and the `_wo_bvcc` / `_wo_somos` / `_wo_bc` / `_wo_sarulab`
  ablations. Only `fusion_stage3` has published weights, and `utils/_download.py` raises
  `ValueError(f"{cfg_name} is not stored.")` for any other name. The single-branch models
  (`MultiSpecModelV2`, `MultiSpecExtModel`, `SSLExtModel`) and `SSLMultiSpecExtModelV1`, which loads its two
  branches off a hard-coded `outputs/` path, have no counterpart here.
- **The training data pipeline.** `utmosv2/preprocess/_preprocess.py` reads BVCC's `sets/` metadata and the CSV
  files of SOMOS, the four Blizzard corpora and the VoiceMOS 2024 sarulab set off hard-coded `data2/` paths, and
  `utils/_pure/split.py` builds the stratified group split the folds come from. Those are dataset and collator
  territory, and the `.wav`-to-`.npy` clipping pass they depend on went with them.
- **`XYMasking` and mixup between samples.** The training transform masks random rectangles of each spectrogram
  and `run.mixup` blends pairs of samples. Both are augmentations a training loop applies, not preprocessing a
  scorer needs; the mixup *within* one sample, between two excerpts of the same clip, is preprocessing and is
  carried over.
- **`num_repetitions` and the fold loop of `inference.py`.** Upstream's `predict` averages repeated draws itself
  and `inference.py` additionally averages over folds and TTA passes. The fold average is inside this model; the
  repeated draws are the loop shown in the Usage section, because a feature extractor that averaged internally
  could not hand the model a batch.


## File map

Not a `git mv` of the upstream tree. UTMOSv2 was never vendored into this repository: it entered as a pip
dependency in `23433734` and left in `6b407586`, and these files were written against the upstream source read
out of a scratch clone.

| Upstream file | Where it went |
|---|---|
| `utmosv2/model/ssl_multispec.py` | `modeling_utmos_v2.py`: `UTMOSv2Model` is `SSLMultiSpecExtModelV2`. `SSLMultiSpecExtModelV1` is not carried over, see above |
| `utmosv2/model/ssl.py` | `modeling_utmos_v2.py`: the `ssl_encoder`, `ssl_layer_weights` and `ssl_attention` half of `UTMOSv2Model`. `_SSLEncoder`'s `AutoFeatureExtractor` is `UTMOSv2FeatureExtractor`'s normalization, and `get_ssl_output_shape` is `UTMOSv2Config.num_ssl_hidden_states` |
| `utmosv2/model/multi_spec.py` | `modeling_utmos_v2.py`: the `spectrogram_encoders`, `spectrogram_weights` and `spectrogram_attention` half of `UTMOSv2Model`, and `_make_melspec` goes to `feature_extraction_utmos_v2.py` |
| `timm`'s `tf_efficientnetv2_s` | `modeling_utmos_v2.py`: `UTMOSv2SpectrogramEncoder`, `UTMOSv2MBConv`, `UTMOSv2FusedMBConv`, `UTMOSv2ConvBlock`, `UTMOSv2SqueezeExcite`, `UTMOSv2Conv2d`, and `EFFICIENTNET_V2_S_STAGES` for `_gen_efficientnetv2_s`'s `arch_def`. `timm.layers.SelectAdaptivePool2d` at `pool_type="catavgmax"` is the pair of adaptive pools in `_spectrogram_features` |
| `utmosv2/dataset/multi_spec.py`, `utmosv2/dataset/ssl.py`, `utmosv2/dataset/_base.py`, `utmosv2/dataset/_utils.py` | `feature_extraction_utmos_v2.py`. `DataDomainMixin`'s table is `DOMAINS`. The `pd.DataFrame` and `DatasetItem` plumbing has no counterpart |
| `utmosv2/preprocess/_preprocess.py` | `feature_extraction_utmos_v2.py`: `remove_silent_sections`. The rest is not carried over, see above |
| `utmosv2/loss/_losses.py` | `modeling_utmos_v2.py`: the `labels` branch of `UTMOSv2ForAudioClassification.forward`. `CombinedLoss` is the two weighted terms summed there |
| `utmosv2/config/fusion_stage3.py` | `configuration_utmos_v2.py` |
| `utmosv2/config/*.py` (the other 25) | Not carried over, see above |
| `utmosv2/_core/model/_models.py`, `utmosv2/_core/create.py` | `UTMOSv2ForAudioClassification` and its `from_pretrained`. `UTMOSv2Model`'s name-to-class table has no counterpart, since only one of the five is carried over |
| `utmosv2/_core/model/_common.py` | `UTMOSv2ForAudioClassification.forward` for `_predict_impl`'s single pass. `_prepare_data`'s file globbing and `num_repetitions` are not carried over, see above |
| `utmosv2/utils/_download.py`, `utmosv2/utils/_constants.py` | `weight_conversion.py`, which reads the same repository and caches under `HF_HOME` rather than `~/.cache/utmosv2` |
| `utmosv2/transform/_xymasking.py` | Not carried over, see above |
| `utmosv2/_settings/_config.py`, `utmosv2/_import.py` | Dropped, no counterpart. Argument parsing for the upstream CLI, and a lazy importer for optional dependencies |
| `utmosv2/utils/_pure/*`, `utmosv2/utils/_task_dependents/*` | Dropped, no counterpart. The fold split, the training metrics, the checkpoint saving and the logging of the upstream trainer |
| `train.py`, `inference.py` | The objective is `forward(labels=...)`. The CLIs have no counterpart |
| `tests/core_tests/*` | Dropped, no counterpart. They check that `create_model` builds each configuration, which the Verification section checks against the upstream classes instead |
| `docs/*`, `quickstart.ipynb`, `poster.pdf`, `CITATION.cff` | This file |
| `pyproject.toml`, `.github/*`, `.vscode/*` | Dropped, no counterpart. The upstream project's own packaging and CI, which this repository supplies itself |
| `LICENSE` | The header of every file in this folder, per CLAUDE.md section 6 |
