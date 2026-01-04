from typing import Optional, Union
import os

from transformers import PreTrainedModel, AutoModel
from transformers.utils import logging
from torch import nn

from .configuration_selective_tuner import SelectiveTunerConfig
from ...components.style_anchor import DirectStyleAnchorEmbedding, EncoderStyleAnchorEmbedding

logger = logging.get_logger(__name__)


class SelectiveTunerForConditionalGeneration(PreTrainedModel):
    """
    SelectiveTunerForConditionalGeneration model.
    """
    config_class = SelectiveTunerConfig

    def __init__(self, config: SelectiveTunerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._replace_embeddings_with_anchors(self, config)

    @staticmethod
    def _replace_embeddings_with_anchors(instance: PreTrainedModel, config: SelectiveTunerConfig):
        """Replace nn.Embedding layers with StyleAnchorEmbedding."""
        embedding_class = DirectStyleAnchorEmbedding if config.use_direct_anchor else EncoderStyleAnchorEmbedding
        anchor_token_id = config.anchor_token_id

        # Recursively replace embeddings in all modules
        for parent_name, parent_module in instance.named_modules():
            for child_name, child_module in list(parent_module.named_children()):
                # Check if it's a standard nn.Embedding (not already replaced)
                if isinstance(child_module, nn.Embedding) and not isinstance(
                    child_module, (DirectStyleAnchorEmbedding, EncoderStyleAnchorEmbedding)
                ):
                    # Create new anchor embedding with same parameters as original
                    new_embedding = embedding_class(
                        num_embeddings=child_module.num_embeddings,
                        embedding_dim=child_module.embedding_dim,
                        anchor_token_id=anchor_token_id,
                        pretrained_weight=child_module.weight.data.clone(),
                        padding_idx=child_module.padding_idx,
                    )

                    # Replace the module
                    setattr(parent_module, child_name, new_embedding)

                    full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                    logger.info(f"Replaced {full_name} with {embedding_class.__name__}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[SelectiveTunerConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        instance = AutoModel.from_pretrained(
            pretrained_model_name_or_path, *model_args,
            config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download, local_files_only=local_files_only, revision=revision,
            weights_only=weights_only, **kwargs
        )
        cls._replace_embeddings_with_anchors(instance, instance.config)
        return instance
