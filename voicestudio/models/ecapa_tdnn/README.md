# ECAPA-TDNN

ECAPA-TDNN turns a speech clip into a 192 dimensional speaker embedding, so that two clips can be compared with a
cosine similarity without either speaker having been seen in training. It is a time delay network with three
additions: each residual block splits its channels into eight groups and convolves them in a chain, giving one
block several receptive field sizes at once; each block ends in a channel gate driven by the utterance average;
and the frame sequence is pooled by an attention that also sees the utterance mean and standard deviation
alongside every frame, into a weighted mean and standard deviation. The outputs of all three residual blocks are
concatenated before that pooling rather than only the last, which is the multi-layer aggregation the name refers
to.

Original model and code: [speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)


## Usage

```python
import soundfile as sf
import torch
from voicestudio.models.ecapa_tdnn import EcapaTdnnFeatureExtractor, EcapaTdnnModel

model_id = "speechbrain/spkrec-ecapa-voxceleb"

extractor = EcapaTdnnFeatureExtractor.from_pretrained(model_id)
model = EcapaTdnnModel.from_pretrained(model_id).eval()

def embed(path):
    waveform, sampling_rate = sf.read(path)
    with torch.no_grad():
        return model(**extractor(waveform, sampling_rate=sampling_rate)).embeddings[0]

similarity = torch.nn.functional.cosine_similarity(embed("a.wav"), embed("b.wav"), dim=0)
```

The embedding is the comparable quantity, and cosine similarity is the comparison the model was trained for. A
threshold on it is what turns the similarity into a verification decision, and that threshold belongs to the
corpus being scored, not to the model.

Pass a batch of clips of different lengths and the extractor pads them and returns the `attention_mask` that
marks the real frames, which the channel gates and the pooling both need. Without it a short clip is scored as
though its padding were silence it actually contained.

```python
inputs = extractor([first, second, third], sampling_rate=16000)
embeddings = model(**inputs).embeddings
```

`EcapaTdnnForXVector` adds the head the authors train the network behind, which scores an embedding against one
learned direction per VoxCeleb speaker. `config.id2label` carries the speaker identifiers.

```python
from voicestudio.models.ecapa_tdnn import EcapaTdnnForXVector

model = EcapaTdnnForXVector.from_pretrained(model_id).eval()
outputs = model(**inputs)
speaker = model.config.id2label[int(outputs.logits[0].argmax())]
```

`from_pretrained` on the repository id above works on its own. The published repository holds SpeechBrain's own
layout, four loose files and a `config.json` naming an interface rather than a model, so the first call converts
it into a directory under `HF_HOME` and loads that; later calls reuse it. `weight_conversion.convert` writes the
same directory somewhere of your choosing.


## Training

`EcapaTdnnForXVector.forward` takes `labels` and returns the objective the VoxCeleb recipe optimizes, which is
`LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.2, scale=30))`: the cosine similarity of the target speaker has
`margin` added to its angle, every similarity is multiplied by `scale`, and the result is read as a softmax over
speakers. Upstream writes the last step as a `KLDivLoss(reduction="sum")` divided by the number of targets, which
for one-hot targets is the cross entropy computed here.

Upstream freezes nothing. `train_speaker_embeddings.py` trains the whole network for ten epochs at a cyclic
learning rate between `1e-8` and `1e-3`, on three second crops, behind four augmentations applied in parallel:
added noise, added reverberation, a dropped frequency band and a dropped time chunk.

```python
loss = model(**inputs, labels=torch.tensor([12, 12, 340])).loss
loss.backward()
```


## Verification

Every number below is on the published `speechbrain/spkrec-ecapa-voxceleb` weights, run through the `colab` CLI
against SpeechBrain 1.1.1's own classes.

- **Load report.** `from_pretrained("speechbrain/spkrec-ecapa-voxceleb")` reports no missing and no unexpected
  keys. All 231 keys of the published `embedding_model.ckpt` map onto this model's parameters, with no key and no
  shape left over in either direction.
- **The network against upstream's own class.** Handed the features upstream's own front end produced,
  `EcapaTdnnModel` and SpeechBrain's `ECAPA_TDNN` agree **bit for bit**, a maximum absolute difference of exactly
  `0.0`, on a full length clip and on a padded one alike.
- **The front end against upstream's own class.** The mean normalized filterbank agrees with
  `Fbank` followed by `InputNormalization(norm_type="sentence", std_norm=False)` to `2.9e-06`, and end to end
  against `EncoderClassifier.encode_batch` the embedding agrees to `3.2e-05`, a cosine similarity of
  `0.99999994`. On a padded batch of two clips of different lengths, where the frame mask has to be right in
  three places, the agreement is `4.2e-05`.
- **The classification head.** Against `classify_batch`, which subtracts the stored training-set mean before the
  cosine similarities, the logits agree to `2.2e-07` and the top scoring speaker is the same on every clip.
- **Speaker discrimination on real speech.** Forty clips of LibriSpeech `validation.clean`, five each from
  speakers 2277, 2035, 2086, 7976, 1988, 777, 84 and 1673, give 80 same-speaker and 700 different-speaker pairs.
  Cosine similarity is `0.7305 +- 0.1426` within a speaker and `0.1317 +- 0.0995` across speakers, an equal
  error rate of `0.0250` on those pairs. Upstream's own embeddings give the same three figures to four decimal
  places, which they must, since the embeddings themselves agree to `3.2e-05` over these clips.

  Note what that last number is and is not. It characterises this checkpoint on eight LibriSpeech speakers,
  not the port, and 780 pairs from eight speakers is far too small a trial list to compare against a published
  VoxCeleb equal error rate. The evidence that the port is faithful is the agreement with upstream above it.


## Not carried over from upstream

- **Every SpeechBrain checkpoint but this one.** `speechbrain/spkrec-xvect-voxceleb` and
  `speechbrain/spkrec-resnet-voxceleb` are different architectures published behind the same interface, and
  `speechbrain/spkrec-ecapa-voxceleb-mel-spec` is this architecture behind a torchaudio front end rather than
  SpeechBrain's own. None of them loads here.
- **`SpeakerRecognition.verify_files` and its threshold.** Upstream ships a wrapper that embeds two files,
  takes their cosine similarity and compares it to a threshold of `0.25`. The similarity is two lines of caller
  code, and a threshold that belongs to one corpus is not something to bake into a model.
- **`Fbank`'s `deltas`, `context`, `param_rand_factor` and the non-triangular filter shapes.** The published
  configuration turns all of them off, and only the triangular filterbank is carried over.
- **The stateful side of `InputNormalization`.** Upstream's is a module that accumulates running statistics
  across batches while training and can be told to stop at an epoch count. Only the two settings this checkpoint
  uses are carried over: the per-utterance mean of the filterbank, which is in the feature extractor, and the
  stored training-set mean of the embeddings, which is a buffer of `EcapaTdnnForXVector`.
- **The four training augmentations.** `AddNoise`, `AddReverb`, `DropFreq` and `DropChunk` and the noise and
  impulse response corpora they download. They are what a training loop applies to a waveform, not preprocessing
  a scorer needs.
- **The VoxCeleb data pipeline and the trainer.** `voxceleb_prepare.py`, the verification trial lists, the
  equal error rate and minimum detection cost scoring in `speaker_verification_cosine.py`, and the PLDA variant
  beside it.


## Lineage

`spark_tts_bicodec` already carries an ECAPA-TDNN as `SparkTTSEcapaTdnn`, and this model does not inherit from
it. Two reasons. It is a different implementation of the paper, the one that names its blocks
`Conv1dReluBn` and `SE_Res2Block`: its aggregation concatenates every block output including the opening
convolution where this one excludes it, its aggregation is a bare convolution and activation where this one is a
convolution, activation and normalization, and its output projection is a `Linear` where this one is a `Conv1d`.
The weights are not interchangeable in either direction. And CLAUDE.md section 2.1 rules out the direction
anyway: Spark-TTS's copy is one consumer's, and a general model that subclassed it would make the general case
depend on that consumer.


## File map

Not a `git mv` of the upstream tree. SpeechBrain is a general purpose toolkit rather than a model repository, it
was never vendored here, and its `eval` extra entry was removed in `6b407586`. These files were written against
the upstream source read out of a scratch clone.

| Upstream file | Where it went |
|---|---|
| `speechbrain/lobes/models/ECAPA_TDNN.py` | `modeling_ecapa_tdnn.py`. `ECAPA_TDNN` is `EcapaTdnnModel`, `TDNNBlock` is `EcapaTdnnTdnnBlock`, `Res2NetBlock` is `EcapaTdnnRes2NetBlock`, `SEBlock` is `EcapaTdnnSqueezeExcite`, `SERes2NetBlock` is `EcapaTdnnSeRes2NetBlock`, `AttentiveStatisticsPooling` is `EcapaTdnnAttentiveStatisticsPooling`, and `Classifier` is the head of `EcapaTdnnForXVector` |
| `speechbrain/nnet/CNN.py` | `modeling_ecapa_tdnn.py`: `EcapaTdnnConv1d` is `Conv1d` at its default `padding="same"` and `padding_mode="reflect"`, which is not what `nn.Conv1d` pads with |
| `speechbrain/nnet/normalization.py` | `nn.BatchNorm1d` directly. `BatchNorm1d` wraps it at the same `eps` and `momentum` and adds a transpose this port does not need |
| `speechbrain/lobes/features.py`, `speechbrain/processing/features.py` | `feature_extraction_ecapa_tdnn.py`. `Fbank` is `filterbank`, `STFT` and `spectral_magnitude` are its `torch.stft` and power, `Filterbank` is `_get_filters` and the decibel conversion, and `InputNormalization` is the mean subtraction at the end of `__call__`. `Deltas`, `ContextWindow`, `DCT` and `MFCC` are not carried over, see above |
| `speechbrain/nnet/losses.py` | `modeling_ecapa_tdnn.py`: the `labels` branch of `EcapaTdnnForXVector.forward` is `LogSoftmaxWrapper` over `AdditiveAngularMargin` |
| `speechbrain/inference/speaker.py`, `speechbrain/inference/classifiers.py` | `EcapaTdnnModel.forward` for `encode_batch` and `EcapaTdnnForXVector.forward` for `classify_batch`. `verify_batch` is not carried over, see above |
| `speechbrain/utils/parameter_transfer.py`, `hyperparams.yaml` | `weight_conversion.py`, which reads the same four files and builds the configuration from the weight shapes rather than from the recipe |
| `speechbrain/dataio/encoder.py`, `label_encoder.txt` | `read_labels` in `weight_conversion.py`, which fills `id2label` and `label2id` |
| `recipes/VoxCeleb/SpeakerRec/*` | The objective is `forward(labels=...)`. The data preparation, the augmentations, the trainer and the scoring have no counterpart, see above |
| `LICENSE` | The header of every file in this folder, per CLAUDE.md section 6 |
