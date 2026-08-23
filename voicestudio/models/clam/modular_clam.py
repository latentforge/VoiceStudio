from transformers import PretrainedConfig
from transformers.models.clap import ClapTextModel, ClapAudioModel, ClapTextConfig, ClapAudioConfig
from transformers.models.clap.modeling_clap import ClapAudioEncoder, ClapTextEncoder, ClapAudioPatchEmbed, ClapTextLayer
from transformers.utils import logging

from torch import nn

from typing import Optional


logger = logging.get_logger()


class ClamTextConfig(ClapTextConfig):
    pass


class ClamAudioConfig(ClapAudioConfig):
    pass


class ClamConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClamForSequenceClassification`]. It is used to instantiate an
    ClamForSequenceClassification according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Clam-1B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[Gemma3TextConfig, dict]`, *optional*):
            The config object of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*):
            Custom vision config or dict.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    Example:

    ```python
    >>> from transformers import ClamForSequenceClassification, ClamConfig, ClamAudioConfig, ClamTextConfig

    >>> # Initializing a ClamAudioConfig config
    >>> vision_config = ClamAudioConfig()

    >>> # Initializing a ClamTextConfig config
    >>> text_config = ClamTextConfig()

    >>> # Initializing a ClamModel 1b style configuration
    >>> configuration = ClamConfig(vision_config, text_config)

    >>> # Initializing a model from ClamModel 1b style configuration
    >>> model = ClamForSequenceClassification(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clam"
    sub_configs = {
        "audio_config": ClamAudioConfig,
        "text_config": ClamTextConfig
    }

    def __init__(
        self,
        audio_config: Optional[ClamAudioConfig] = None,
        text_config: Optional[ClamTextConfig] = None,
        initializer_range: float = 0.02,
        **kwargs
    ):
        if text_config is None:
            text_config = ClamTextConfig()
        if audio_config is None:
            audio_config = ClamAudioConfig()

        self.text_config = text_config
        self.audio_config = audio_config
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class ClamTextTokenEmbed(ClapTextModel):
    pass


class ClamAudioPatchEmbed(ClapAudioPatchEmbed):
    pass


from transformers.models.swin.modeling_swin import SwinStage


class ClapMultimodalEncoder(ClapAudioEncoder):
    def __init__(self, config):
        super().__init__()
        self.num_layers = len(config.depths)

        self.config = config
        self.patch_embed = ClapAudioPatchEmbed(config)
        self.enable_fusion = config.enable_fusion
        self.patch_stride = self.patch_embed.patch_stride
        self.spec_size = config.spec_size
        self.freq_ratio = config.spec_size // config.num_mel_bins

        self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))

        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu")]

        grid_size = self.patch_embed.grid_size
        self.input_resolutions = [(grid_size[0] // (2**i), grid_size[1] // (2**i)) for i in range(self.num_layers)]

        self.layers = nn.ModuleList(
            [
                ClapAudioStage(
                    config=config,
                    dim=int(config.patch_embeds_hidden_size * 2**i_layer),
                    input_resolution=self.input_resolutions[i_layer],
                    depth=config.depths[i_layer],
                    num_heads=config.num_attention_heads[i_layer],
                    drop_path=drop_path_rate[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=ClapAudioPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

        self.batch_norm = nn.BatchNorm2d(config.num_mel_bins)
        self.norm = nn.LayerNorm(self.num_features)
        self.depths = config.depths
        self.avgpool = nn.AdaptiveAvgPool1d(1)



from transformers import PreTrainedModel
from transformers.utils import auto_docstring


@auto_docstring
class ClamPreTrainedModel(PreTrainedModel):
    config: ClamConfig
    base_model_prefix = "clam"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SwinStage"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, SwinEmbeddings):
            if module.mask_token is not None:
                module.mask_token.data.zero_()
            if module.position_embeddings is not None:
                module.position_embeddings.data.zero_()
        elif isinstance(module, SwinSelfAttention):
            module.relative_position_bias_table.data.zero_()


        # 아래 부분 정리
        factor = self.config.initializer_factor

        if isinstance(module, ClapTextEmbeddings):
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.token_type_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, ClapModel):
            module.logit_scale_a.data.fill_(math.log(self.config.logit_scale_init_value))
            module.logit_scale_t.data.fill_(math.log(self.config.logit_scale_init_value))
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            in_proj_std = (self.config.hidden_size**-0.5) * ((2 * self.config.num_hidden_layers) ** -0.5) * factor
            nn.init.normal_(module.weight, std=in_proj_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ClapAudioSelfAttention):
            module.relative_position_bias_table.data.zero_()


@auto_docstring
class ClamModel(ClamPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        r"""
        add_pooling_layer (`bool`, *optional*, defaults to `True`):
            Whether or not to apply pooling layer.
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether or not to create and apply mask tokens in the embedding layer.
        """
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = SwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None
    ) -> Union[tuple, SwinModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        return SwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )


class SwinForImageClassification(SwinPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.swin = SwinModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(self.swin.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: bool = False,
            return_dict: Optional[bool] = None,
    ) -> Union[tuple, SwinImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.swin(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=logits, config=self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
