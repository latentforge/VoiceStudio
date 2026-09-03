# VoxInstruct

VoxInstruct turns a free-form natural-language instruction into speech, with the words to be spoken quoted inside the instruction rather than supplied separately. A frozen mT5 encoder reads the instruction and its output is projected into a prefix on the decoder sequence. An autoregressive stage then samples one flat token stream: a language identity token, the HuBERT k-means semantic tokens of the utterance, an end token, and the first EnCodec codebook of the same utterance. A non-autoregressive stage reads the whole codebook grid at once and fills the remaining seven codebooks by confidence-ordered iterative decoding, in the manner of SoundStorm, and a Vocos vocoder over the summed codebook embeddings turns them into a waveform. A speech prompt can be prepended to both spans to carry a voice over.

Original model and code: [thuhcsi/VoxInstruct](https://github.com/thuhcsi/VoxInstruct)


## Usage

`from_pretrained` takes the released checkpoint's repository id as it stands:

```python
import torch
import soundfile as sf

from voicestudio.models.vox_instruct import VoxInstructForConditionalGeneration, VoxInstructProcessor

model_id = "niobures/VoxInstruct"

processor = VoxInstructProcessor.from_pretrained(model_id)
model = VoxInstructForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()

inputs = processor(
    text='A man with a normal voice to say: "Fire a whole platoon, Major."',
    language="en",
).to(model.device)

outputs = model.generate(
    **inputs,
    guidance_scale_semantic_on_text=1.5,
    guidance_scale_acoustic_on_text=3.0,
    guidance_scale_acoustic_on_semantic=1.5,
    num_nar_iterations=8,
)
sf.write("output.wav", outputs.audio_values.float().cpu().numpy().reshape(-1), processor.feature_extractor.sampling_rate)
```

`niobures/VoxInstruct` mirrors the Google Drive folder the upstream README points at; a local copy of that folder
works as `model_id` too. The release is a `models/VoxInstruct/pretrained` tree with no `config.json` or
`model.safetensors` at its root, so `from_pretrained` merges its parts itself: two `torch.save` files of a
bespoke module tree, a `fairseq` HuBERT checkpoint, a scikit-learn k-means codebook, a standalone EnCodec
checkpoint and a Vocos repository. Only the nine files it reads are downloaded, which leaves the mT5 weights and
the 48 kHz EnCodec checkpoint of the release behind.

`language` selects the pronunciation tendency, `"en"` or `"zh"`. `generate` also takes `vocoder`, `"vocos"` by
default, which is what upstream's `infer.sh` runs; `vocoder="encodec"` decodes the same codes with the EnCodec
decoder instead, which is upstream's `--vocoder encodec` alternative. The three guidance scales are independent: one pushes the semantic span away from the branch that drops the instruction, and two push the first codebook away from the branches that drop the instruction and the semantic span. Each scale left at `1.0` saves one forward pass per decoding step. The values above are the ones `infer.sh` uses upstream.

To carry a voice over, pass the reference waveform. The model tokenizes it twice, into semantic tokens that prime the semantic span and into EnCodec codes that prime the acoustic span:

```python
waveform, sampling_rate = sf.read("actor_ref.wav")  # any reference clip; upstream ships examples/actor_ref.wav

inputs = processor(
    text='“下面唱的这是西河大鼓，西河大鼓发源于河北省河间地带。欢迎大家使用 VoxInstruct 模型进行指令到语音生成。”',
    language="zh",
    audio=waveform.T,
    sampling_rate=sampling_rate,
).to(model.device)
```

Two constraints come from the model, not from this code: the transcript of the speech prompt has to appear in the quoted part of the instruction, and `language` has to match the prompt. `generate` decodes one instruction at a time.

A first load converts the published layout into a directory under `HF_HOME`, keyed on the repository and the commit it
resolved to, and later loads read that directory through the ordinary loading path instead of converting again.
The AR and NAR instruction encoders are mT5 encoders whose shared embedding table is one tensor under two names,
and safetensors stores no aliases, so that directory carries both copies and is 5.6 GB against 4.4 GB of
parameters.

`weight_conversion.convert` writes that same conversion to a directory of the caller's choosing, for a checkpoint
that is shipped elsewhere or kept outside the cache, and both `from_pretrained` calls above read it as readily as
the released layout:

```python
from voicestudio.models.vox_instruct.weight_conversion import convert

convert("niobures/VoxInstruct", "voxinstruct-converted")
```


## Training

Pass targets to the processor and the standard `forward` returns a loss:

```python
batch = processor(
    text=['A young man says cheerfully: "Good morning."'],
    language="en",
    semantic_ids=[semantic_ids],   # (num_semantic_frames,), HuBERT k-means ids with runs collapsed
    acoustic_ids=[acoustic_ids],   # (num_acoustic_frames, 8), EnCodec codes
)
outputs = model(**batch)
outputs.loss.backward()
```

`model.semantic_encoder` and `model.audio_encoder` produce those two arrays from a waveform, which is what `utils/extract_hubert.py` and `utils/extract_encodec.py` did offline upstream. `meta_files/textrolspeech/metadata_train.json` is the upstream metadata that pairs each utterance with its transcript, language and instruction list.

The two stages are trained separately upstream, one script each, and they stay separable here: `input_ids` and `labels` drive the autoregressive stage alone, `nar_input_ids` and `nar_labels` the non-autoregressive one. `VoxInstructOutput` reports `ar_loss` and `nar_loss` on their own and `loss` as their sum.

**Autoregressive loss.** `train_ar.py` takes `outputs.logits[:, max_text_len:]`, drops the last position, and scores it against `seqs[:, 1:]` under a mask of `seq_lens - 1` positions, averaging by the number of unmasked targets. That is next-token cross-entropy over the whole flat sequence, the language token and both end tokens included, with padding excluded. `VoxInstructARForCausalLM.forward` computes exactly that through the standard `ForCausalLM` loss, with the processor writing `-100` on the padded positions. Two conditioning dropouts run alongside it, which is what trains the branches classifier-free guidance later contrasts against: with probability `text_free_guidance_ratio` (0.1) the whole instruction encoding is zeroed for a sample, and with probability `semantic_free_guidance_ratio` (0.1) that sample's semantic span is replaced by the padding token.

**Non-autoregressive loss.** `train_nar.py` scores one residual codebook per sample. `model/nar.py` draws `q` uniformly from `1 .. 7`, keeps codebooks below `q` fully visible, keeps an acoustic prompt of `u * acoustic_length` frames visible in every codebook with `u` uniform, and hides a `cos(pi / 2 * v)` fraction of the frames past the prompt with `v` uniform. The head for codebook `q` is scored against the true codebook `q` on the hidden positions only, with the flat acoustic offset subtracted and clamped at zero, again averaged by the number of scored positions. `VoxInstructNARModel.forward` reproduces that draw, mask and target. The same instruction dropout runs, at `text_free_guidance_ratio` 0.25 here, and with probability `acoustic_free_guidance_ratio` (0.3) the prompt length is drawn as zero instead of uniformly, so the stage also learns to work without an acoustic prompt.

**What upstream freezes.** `model/ar.py` and `model/nar.py` set `requires_grad = False` on every mT5 parameter and then wrap the `q` and `v` projections of each encoder block in a rank-16 LoRA adapter, so the instruction path trains through the adapters alone. Everything else in both stages trains. HuBERT, its k-means codebook and EnCodec never appear in the upstream training graph at all, because the semantic and acoustic tokens are extracted to `.npy` files before training starts. `VoxInstructForConditionalGeneration.freeze_encoders` reproduces all of that: it runs on construction and again after `from_pretrained`, freezing both mT5 bodies, the two tokenizers and the Vocos vocoder, and leaving the adapters, both decoders, the embeddings, the segment embeddings, the projections and the residual heads trainable. Upstream runs its `Accelerator` with `find_unused_parameters=True` because a step touches only one of the seven residual heads and never the unused embedding tables inside the two Llama stacks; the same parameters are unused here.


## Verification

`from_pretrained` straight onto `niobures/VoxInstruct`, with no conversion call before it, reports no missing, unexpected or mismatched keys across all 1109 tensors and 1105019876 parameters, of which the composed `VocosModel` is 81 tensors and 10081410 parameters. The extra tensor over a bare backbone and head is `vocoder.feature_extractor.codebook_weights`, the 16384 by 128 table that turns a frame's eight codebook entries into the 128 channel embedding the backbone reads. It is a trained parameter of `charactr/vocos-encodec-24khz` and not a buffer, so it is loaded rather than left at random init.

The four buffers no released file carries, and which therefore have to survive meta-device initialisation, were read back after that load: `semantic_encoder.cluster_centers`, `vocoder.head.window`, `vocoder.mel_spectrogram.window` and `vocoder.mel_spectrogram.filters` are all finite and non-zero. Generating `A man with a normal voice to say: "Fire a whole platoon, Major."` from that load and transcribing the result back with `openai/whisper-small.en` gives `Fire a whole platoon, Major.` word for word.

On the codes this model generates, `VocosModel` in float32 agrees with upstream `vocos`'s own `VocosBackbone` plus `ISTFTHead`, run from the same released weights, bit for bit, at a maximum absolute difference of 0.0 on a waveform whose magnitude is 0.86.

Generating the four English instructions of `examples/example_instructions.txt` at the guidance scales `infer.sh` uses and decoding each one both ways, the two paths transcribe identically under wav2vec2 on three of them, including the 24 word `0006` and `0008` clips, which come back as DELANY HAD READ ONE OR TWO WORKS ON PSYCHIC PHENOMENA AND UNDERSTOOD FROM THEM THAT SPIRIT PROJECTION WAS NOT ONLY QUITE FEASIBLE BUT FAR FROM UNCOMMON word for word. On the 1.6 second `0005` clip the EnCodec decoder gives FIRE A WHOLE PLATOON MAJOR and Vocos gives FIRE A HOLE PLATOON MAJOR. Upstream Vocos, run from its own weights on the same codes, gives the same transcription as this port does, so that word is a property of the vocoder and not of the migration.

Quality does differ, in the direction Vocos is trained for. Encoding a clip with EnCodec at 6 kbps and decoding it both ways, Vocos lands closer to the original in the domain it optimizes, at a mel L1 of 0.41 against the EnCodec decoder's 0.51 on the F5-TTS demo clip and 0.54 against 0.63 on `examples/actor_ref.wav`, and further away in the waveform domain, at 6.6 dB SNR against 7.7 dB and 0.1 dB against 3.8 dB, because it resynthesizes phase from scratch rather than reconstructing the encoded waveform.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **The mT5 tokenizer comes from a different repository than the checkpoint.** The `spiece.model` the checkpoint ships is byte-identical to `google/mt5-base`, but neither that repository nor the checkpoint carries a serialized fast tokenizer, and building one from `spiece.model` needs `sentencepiece` and `protobuf`, which this project does not depend on. `weight_conversion.build_processor` therefore reads `google/mt5-small`'s `onnx/tokenizer.json`, which Google exported from the same vocabulary, through its `tokenizer_id` argument. Whether that indirection is acceptable, or `sentencepiece` should be added instead, is a decision for a human.
- **An off-by-two in the sampling bands is reproduced, not fixed.** The flat vocabulary lays semantic ids out at `1 + num_language_ids ..`, but `inference.py` masks `[semantic_vocab_size + 1 : eos_id]` while sampling the semantic span and `[0 : semantic_vocab_size + 1]` while sampling the first codebook, as if the semantic band began at 1. The last two semantic ids are therefore unsamplable in the semantic span and samplable in the acoustic one. `generate` reproduces the released behaviour exactly.
- **The training-time instruction draw.** `utils/dataset.py` picks one of an utterance's instructions at random and, with probability `description_free_g_ratio` (0.1), replaces it with the raw transcript in quotes, choosing between `"` and `“”` by a coin flip. `VoxInstructProcessor` takes the instruction string it is given, so a caller assembling a training batch has to make that draw.
- **The optimizer and learning rate schedule.** `utils/optimizer.py` builds AdamW with weight decay applied only to parameters of rank two and above, under a Noam-style schedule of `min(step ** -0.5, warmup_steps ** -1.5 * step)` evaluated at `step // num_processes`. Neither the parameter grouping nor that schedule is what `transformers.Trainer` sets up by default, and reproducing them means passing a custom optimizer and `lr_lambda`.
- **Top-k accuracy logging and the mask ratio diagnostic.** `utils/utils.py:compute_loss` also returned top-1 and top-10 accuracy, and the upstream NAR forward returned the realized mask ratio. The two forwards return the loss; `VoxInstructNAROutput.loss_mask` carries the positions the mask ratio was computed from.
- **Corpus-level feature extraction and the training loop.** The `__main__` blocks of `utils/extract_hubert.py` and `utils/extract_encodec.py` walked a folder writing `.npy` codes, and `train_ar.py` and `train_nar.py` wrapped everything in an Accelerate loop with tensorboard logging and periodic checkpointing. Those are `Trainer` and processor territory now, and `configs/accelerate_config.yaml` went with them.
- **Alignment plotting and waveform writing.** `utils/utils.py:plot_alignment` and `save_wav` are dropped along with `matplotlib` and `scipy`.


## Repository integration

Three things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .vox_instruct import *` line, and a `from .vocos import *` line for the vocoder this model composes.
- `PROJECT.md` needs a VoxInstruct status entry carrying the gaps listed above.
- Nothing in `pyproject.toml` or `uv.lock` changes. The migration removes `fairseq`, `encodec`, `vocos`, `peft`, `einops`, `matplotlib`, `scipy`, `accelerate`, `flash-attn`, `sentencepiece` and `protobuf` from what this model needs. `vocos` is gone because the vocoder is now `VocosModel` from `voicestudio/models/vocos`, composed here the way `ParlerTTSConfig` composes its `audio_encoder`: `VoxInstructConfig.vocoder_config` is a `VocosConfig`, `VoxInstructForConditionalGeneration.vocoder` is the model it builds, and `weight_conversion` reads the released `pretrained/vocos-encodec-24khz` folder into it. What remains is `torch`, `torchaudio`, `transformers`, `numpy`, `huggingface_hub`, and, on the path that reads the released layout, `joblib` and `scikit-learn` for the k-means codebook, all of which the project already installs. That path is `weight_conversion`, which `from_pretrained` imports only when the source it is given is the released layout, so loading a converted directory never imports either.

`VoxInstructProcessor` deliberately does not set `feature_extractor_class`. Setting it sends `ProcessorMixin.from_pretrained` down a deprecated lookup that scans `IMAGE_PROCESSOR_MAPPING`, which is a dummy object in an environment without `PIL` and `torchvision`, and the processor then fails to load. Registering `VoxInstructFeatureExtractor` with `AutoFeatureExtractor`, which this package's `__init__.py` does, is the path `transformers` 5 asks for anyway.
