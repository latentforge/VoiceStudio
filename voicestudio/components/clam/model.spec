ClamModel(
  (token_embed): ClamTextTokenEmbed(  # ClapTextModel이었던 것
    (proj): ClapTextEncoder(  # encoder 였던 것, 내용물 숨기기
      (embeddings): ClapTextEmbeddings(
        (word_embeddings): Embedding(50265, 768, padding_idx=1)
        (position_embeddings): Embedding(514, 768, padding_idx=1)
        (token_type_embeddings): Embedding(1, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (layer): ModuleList(
        (0-11): 12 x ClapTextLayer(
          (attention): ClapTextAttention(
            (self): ClapTextSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): ClapTextSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): ClapTextIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ClapTextOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (pooler): ClapTextPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
        (linear1): Linear(in_features=768, out_features=512, bias=True)
        (activation): ReLU()
        (linear2): Linear(in_features=512, out_features=512, bias=True) # split, 혹시 풀링을 해버릴까?
      )
    )
  )
  (patch_embed): ClamAudioPatchEmbed(
    (proj): Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
    (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  )
  (layers): ModuleList(
    (0): ClapAudioStage(
      (blocks): ModuleList(
        (0-1): 2 x ClapAudioLayer(
          (layernorm_before): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (attention): ClapAudioAttention(
            (self): ClapAudioSelfAttention(
              (query): Linear(in_features=128, out_features=128, bias=True)
              (key): Linear(in_features=128, out_features=128, bias=True)
              (value): Linear(in_features=128, out_features=128, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ClapAudioSelfOutput(
              (dense): Linear(in_features=128, out_features=128, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): Identity()
          (layernorm_after): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (intermediate): ClapAudioIntermediate(
            (dense): Linear(in_features=128, out_features=512, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ClapAudioOutput(
            (dense): Linear(in_features=512, out_features=128, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (downsample): ClapAudioPatchMerging(
        (reduction): Linear(in_features=512, out_features=256, bias=False)
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
    (1): ClapAudioStage(
      (blocks): ModuleList(
        (0-1): 2 x ClapAudioLayer(
          (layernorm_before): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (attention): ClapAudioAttention(
            (self): ClapAudioSelfAttention(
              (query): Linear(in_features=256, out_features=256, bias=True)
              (key): Linear(in_features=256, out_features=256, bias=True)
              (value): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ClapAudioSelfOutput(
              (dense): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): Identity()
          (layernorm_after): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (intermediate): ClapAudioIntermediate(
            (dense): Linear(in_features=256, out_features=1024, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ClapAudioOutput(
            (dense): Linear(in_features=1024, out_features=256, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (downsample): ClapAudioPatchMerging(
        (reduction): Linear(in_features=1024, out_features=512, bias=False)
        (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
    (2): ClapAudioStage(
      (blocks): ModuleList(
        (0-11): 12 x ClapAudioLayer(
          (layernorm_before): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (attention): ClapAudioAttention(
            (self): ClapAudioSelfAttention(
              (query): Linear(in_features=512, out_features=512, bias=True)
              (key): Linear(in_features=512, out_features=512, bias=True)
              (value): Linear(in_features=512, out_features=512, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ClapAudioSelfOutput(
              (dense): Linear(in_features=512, out_features=512, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): Identity()
          (layernorm_after): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (intermediate): ClapAudioIntermediate(
            (dense): Linear(in_features=512, out_features=2048, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ClapAudioOutput(
            (dense): Linear(in_features=2048, out_features=512, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (downsample): ClapAudioPatchMerging(
        (reduction): Linear(in_features=2048, out_features=1024, bias=False)
        (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      )
    )
    (3): ClapAudioStage(
      (blocks): ModuleList(
        (0-1): 2 x ClapAudioLayer(
          (layernorm_before): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (attention): ClapAudioAttention(
            (self): ClapAudioSelfAttention(
              (query): Linear(in_features=1024, out_features=1024, bias=True)
              (key): Linear(in_features=1024, out_features=1024, bias=True)
              (value): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ClapAudioSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (drop_path): Identity()
          (layernorm_after): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (intermediate): ClapAudioIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ClapAudioOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (batch_norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  (avgpool): AdaptiveAvgPool1d(output_size=1)
)

(audio_projection): ClapProjectionLayer(
  (linear1): Linear(in_features=1024, out_features=512, bias=True)
  (activation): ReLU()
  (linear2): Linear(in_features=512, out_features=512, bias=True)
)
