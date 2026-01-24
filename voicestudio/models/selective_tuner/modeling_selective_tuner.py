from typing import Optional, Union
import types
import os

from transformers import PreTrainedModel, AutoModel
from transformers.utils import logging
from torch import nn
import torch

from .configuration_selective_tuner import SelectiveTunerConfig
from ...components.style_anchor import (
    DirectStyleAnchorEmbedding, EncoderStyleAnchorEmbedding, MixedStyleAnchorEmbedding
)

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
        """Replace nn.Embedding layers with StyleAnchorEmbedding.

        Args:
            instance: PreTrainedModel instance to modify
            config: SelectiveTunerConfig with anchor configuration

        When config.tie_embeddings is True, all replaced embeddings will keep their
        own pretrained weights but share anchor-specific parameters (deltas/bases/encoders).
        """
        embedding_class = DirectStyleAnchorEmbedding if config.use_direct_anchor else EncoderStyleAnchorEmbedding
        if config.use_mixed_anchor:
            embedding_class = MixedStyleAnchorEmbedding
        anchor_token_id = config.anchor_token_id
        tie_embeddings = getattr(config, 'tie_embeddings', True)

        # Get target embedding dimensions from config
        vocab_size = getattr(config, 'vocab_size', None)
        embedding_dim = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', None)

        if vocab_size is None or embedding_dim is None:
            logger.warning(
                f"Cannot determine target embedding size from config. "
                f"Skipping embedding replacement."
            )
            return

        # Store first created embedding's anchor parameters for tying
        shared_anchor_params = None

        # Flatten anchor IDs to check for out-of-bounds access
        all_anchor_ids = []
        if config.use_mixed_anchor:
            d_ids, e_ids = anchor_token_id
            all_anchor_ids.extend(d_ids if isinstance(d_ids, (list, tuple)) else [d_ids])
            all_anchor_ids.extend(e_ids if isinstance(e_ids, (list, tuple)) else [e_ids])
        else:
            all_anchor_ids.extend(anchor_token_id if isinstance(anchor_token_id, (list, tuple)) else [anchor_token_id])

        # Determine strict max ID
        max_anchor_id = max(all_anchor_ids) if all_anchor_ids else -1
        new_num_embeddings = max(vocab_size, max_anchor_id + 1)
        extension_size = new_num_embeddings - vocab_size
        if extension_size > 0:  # Extend vocab if needed
            config.vocab_size = new_num_embeddings
            logger.info(
                f"Extending vocabulary from {vocab_size} to {new_num_embeddings} for anchor support."
                " Top-level generation config's vocab_size is updated, but may still require manual recursive config update."
            )

        # Recursively replace embeddings in all modules
        for parent_name, parent_module in instance.named_modules():
            for child_name, child_module in list(parent_module.named_children()):
                # Check if it's a standard nn.Embedding (not already replaced)
                if isinstance(child_module, nn.Embedding) and not isinstance(
                    child_module, (DirectStyleAnchorEmbedding, EncoderStyleAnchorEmbedding, MixedStyleAnchorEmbedding)
                ):
                    # Only replace if size matches target
                    if child_module.num_embeddings == vocab_size and child_module.embedding_dim == embedding_dim:
                        current_weight = child_module.weight.data.clone()

                        # Extend vocab if needed
                        if extension_size > 0:
                            # Initialize extension with mean of existing embeddings for stability
                            mean_embedding = current_weight.mean(dim=0, keepdim=True)
                            extension_weight = mean_embedding.expand(extension_size, -1).clone()
                            # Add small noise to break symmetry if multiple new tokens
                            extension_weight += torch.randn_like(extension_weight) * 0.01
                            # Concatenate extension to current weight
                            current_weight = torch.cat([current_weight, extension_weight], dim=0)

                        # Create new anchor embedding with potentially extended weights
                        if config.use_mixed_anchor:
                            direct_ids, encoder_ids = anchor_token_id
                            new_embedding = embedding_class(
                                num_embeddings=new_num_embeddings,
                                embedding_dim=child_module.embedding_dim,
                                direct_anchor_token_id=direct_ids,
                                encoder_anchor_token_id=encoder_ids,
                                pretrained_weight=current_weight,
                                padding_idx=child_module.padding_idx,
                            )
                        else:
                            new_embedding = embedding_class(
                                num_embeddings=new_num_embeddings,
                                embedding_dim=child_module.embedding_dim,
                                anchor_token_id=anchor_token_id,
                                pretrained_weight=current_weight,
                                padding_idx=child_module.padding_idx,
                            )

                        # Share anchor parameters if tie_embeddings is True
                        if tie_embeddings:
                            if shared_anchor_params is None:
                                # Store first embedding's anchor parameters
                                if config.use_mixed_anchor:
                                    shared_anchor_params = {
                                        'deltas': new_embedding.anchor_deltas,
                                        'bases': new_embedding.anchor_bases,
                                        'encoders': new_embedding.anchor_encoders
                                    }
                                elif config.use_direct_anchor:
                                    shared_anchor_params = new_embedding.anchor_deltas
                                else:
                                    shared_anchor_params = {
                                        'bases': new_embedding.anchor_bases,
                                        'encoders': new_embedding.anchor_encoders
                                    }
                            else:
                                # Replace anchor parameters with shared ones
                                if config.use_mixed_anchor:
                                    new_embedding.anchor_deltas = shared_anchor_params['deltas']
                                    new_embedding.anchor_bases = shared_anchor_params['bases']
                                    new_embedding.anchor_encoders = shared_anchor_params['encoders']
                                elif config.use_direct_anchor:
                                    new_embedding.anchor_deltas = shared_anchor_params
                                else:
                                    new_embedding.anchor_bases = shared_anchor_params['bases']
                                    new_embedding.anchor_encoders = shared_anchor_params['encoders']

                        # Replace the module
                        setattr(parent_module, child_name, new_embedding)

                        full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                        if tie_embeddings and shared_anchor_params is not None:
                            logger.info(f"Replaced {full_name} with {embedding_class.__name__} (tied anchor params)")
                        else:
                            logger.info(f"Replaced {full_name} with {embedding_class.__name__}")

    def extend_vocabulary(self, tokenizer):
        """
        Extend tokenizer vocabulary with anchor tokens and update config anchor IDs.

        Args:
            tokenizer: Tokenizer instance (e.g., AutoTokenizer)
        """
        config = self.config

        # Determine effective anchor tokens
        tokens_to_add = []
        target_tokens = config.anchor_token

        if config.use_mixed_anchor:
            # Expecting ((directs), (encoders))
            if isinstance(target_tokens, (tuple, list)) and len(target_tokens) == 2:
                for group in target_tokens:
                    if isinstance(group, (tuple, list)):
                        tokens_to_add.extend(group)
                    else:
                        tokens_to_add.append(group)
        else:
             if isinstance(target_tokens, (tuple, list)):
                tokens_to_add.extend(target_tokens)
             else:
                tokens_to_add.append(target_tokens)

        # strings only
        tokens_to_add = [t for t in tokens_to_add if isinstance(t, str)]

        # Add to tokenizer
        num_added = tokenizer.add_tokens(tokens_to_add)
        if num_added > 0:
            logger.info(f"Added {num_added} anchor tokens to tokenizer.")

        # re-resolve IDs
        if config.use_mixed_anchor:
            direct_tokens, encoder_tokens = target_tokens
            direct_ids = tuple([tokenizer.convert_tokens_to_ids(t) for t in direct_tokens])
            encoder_ids = tuple([tokenizer.convert_tokens_to_ids(t) for t in encoder_tokens])
            config.anchor_token_id = (direct_ids, encoder_ids)
        else:
             if isinstance(target_tokens, str):
                 config.anchor_token_id = tokenizer.convert_tokens_to_ids(target_tokens)
             else:
                 config.anchor_token_id = tuple([tokenizer.convert_tokens_to_ids(t) for t in target_tokens])

    def merge_and_unload(self, cast_to_embedding: bool = False):
        """
        Merge anchor deltas into the base weights and reset deltas.
        
        Handles shared weights (tied embeddings) safely:
        1. Merges deltas into each unique weight tensor exactly once.
        2. Resets deltas only after all merges are complete to avoid data loss in tied anchors.
        
        Args:
            cast_to_embedding (bool): If True, replaces StyleAnchorEmbedding modules with 
                                      standard nn.Embedding containing the merged weights.
                                      Useful for optimized inference.
        """
        processed_weight_ids = set()
        anchor_modules = []

        # Collect modules (store name/parent info for replacement if needed)
        # Using instance.named_modules() is better for replacement
        anchor_modules_info = [] # (parent_name, parent_module, child_name, child_module)

        for parent_name, parent_module in self.named_modules():
            for child_name, child_module in list(parent_module.named_children()):
                 if isinstance(child_module, (DirectStyleAnchorEmbedding, EncoderStyleAnchorEmbedding, MixedStyleAnchorEmbedding)):
                     anchor_modules_info.append((parent_name, parent_module, child_name, child_module))
                     anchor_modules.append(child_module)

        # Phase 1: Merge Unique Weights (without resetting)
        for module in anchor_modules:
            weight_id = id(module.weight)
            if weight_id not in processed_weight_ids:
                module.merge_deltas(reset=False)
                processed_weight_ids.add(weight_id)

        # Phase 2: Reset Deltas
        for module in anchor_modules:
            module._reset_deltas()

        logger.info(f"Merged anchor deltas into {len(processed_weight_ids)} unique embedding weights.")

        # Phase 3: Cast to nn.Embedding (Optional)
        if cast_to_embedding:
            # Map original weight ID to NEW nn.Embedding weight parameter
            # This ensures we preserve weight tying in the new nn.Embedding objects
            new_weight_map = {} 

            replaced_count = 0
            for parent_name, parent_module, child_name, child_module in anchor_modules_info:
                weight_id = id(child_module.weight)

                # Create standard embedding
                new_emb = nn.Embedding(
                    num_embeddings=child_module.num_embeddings,
                    embedding_dim=child_module.embedding_dim,
                    padding_idx=child_module.padding_idx,
                    max_norm=child_module.max_norm,
                    norm_type=child_module.norm_type,
                    scale_grad_by_freq=child_module.scale_grad_by_freq,
                    sparse=child_module.sparse
                )

                # Assign Weight
                if weight_id in new_weight_map:
                    # Reuse the same parameter object if weight was tied
                    new_emb.weight = new_weight_map[weight_id]
                else:
                    # Use the data from the merged anchor embedding
                    new_emb.weight = nn.Parameter(child_module.weight.data)
                    new_weight_map[weight_id] = new_emb.weight

                setattr(parent_module, child_name, new_emb)
                replaced_count += 1

            logger.info(f"Converted {replaced_count} StyleAnchorEmbedding modules to standard nn.Embedding.")

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

        # Dynamically bind utility methods to the instance
        instance.extend_vocabulary = types.MethodType(cls.extend_vocabulary, instance)
        instance.merge_and_unload = types.MethodType(cls.merge_and_unload, instance)

        return instance
