"""
Selective Tuner model configuration.

This module defines the configuration for SelectiveTunerForConditionalGeneration.
"""
import os
from copy import deepcopy
from typing import Optional, Union, Any

from transformers.models.auto import AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig, SpecificPretrainedConfigType
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SelectiveTunerConfig(PretrainedConfig):
    """
    Configuration class for SelectiveTunerForConditionalGeneration.

    This configuration enables selective token embedding tuning (BOS tuning) by wrapping
    a base model configuration and adding style anchor parameters. When a base_config is
    provided, it creates a copy and extends it with anchor-specific attributes.

    Args:
        base_config (`PretrainedConfig`, *optional*):
            Base model configuration to extend. If None, creates a minimal SelectiveTunerConfig.
            When provided, deepcopies the base config and adds anchor parameters to it.
        anchor_token (`str` or `tuple[str]`, *optional*, defaults to `"<bos>"`):
            Token(s) to use as style anchor. Can be a single token string or tuple of tokens.
        anchor_token_id (`int` or `tuple[int]`, *optional*):
            Explicit token ID(s) to use as anchor. If provided, overrides token string lookup.
        use_direct_anchor (`bool`, *optional*, defaults to `True`):
            Whether to use direct anchor (DirectStyleAnchor) or encoder-based anchor 
            (EncoderStyleAnchor with 2-layer MLP).
        **kwargs:
            Additional keyword arguments to set as config attributes.

    Example:
        ```python
        from transformers import AutoConfig
        from voicestudio.models.selective_tuner import SelectiveTunerConfig

        # Load base ParlerTTS config
        base_config = AutoConfig.from_pretrained("parler-tts/parler-tts-mini-v1")

        # Create SelectiveTuner config from base config
        config = SelectiveTunerConfig(
            base_config=base_config,
            anchor_token="</s>",
            use_direct_anchor=True,
        )
        # config now has all base_config attributes + anchor_token, anchor_token_id, use_direct_anchor

        # Or create minimal config
        config = SelectiveTunerConfig(
            anchor_token="<bos>",
            use_direct_anchor=True,
        )
        ```
    """
    model_type = "selective_tuner"

    def __new__(
        cls,
        base_config: Optional[PretrainedConfig | str] = None,
        anchor_token: str | tuple[str] = "<bos>",
        anchor_token_id: Optional[int | tuple[int]] = None,
        use_direct_anchor: bool = True,
        **kwargs,
    ):
        """
        Custom __new__ method to enable config inheritance from base models.
        
        When base_config is provided, this method returns a modified copy of that config
        instead of creating a new SelectiveTunerConfig instance. This preserves all
        base model configuration while adding anchor-specific parameters.
        
        Args:
            base_config: Base model config to extend (e.g., ParlerTTS config)
            anchor_token: Token(s) to use as style anchor
            anchor_token_id: Explicit token ID(s) for anchor
            use_direct_anchor: Whether to use direct or encoder-based anchor
            **kwargs: Additional attributes to set on the config
            
        Returns:
            If base_config is None: New SelectiveTunerConfig instance
            If base_config is provided: Deepcopy of base_config with anchor attributes added
        """
        if base_config is None:
            return super().__new__(cls)
        else:  # does not init a new class (use base_config instead)
            # TODO: Fix AutoConfig
            #config = AutoConfig.from_pretrained(base_config) if isinstance(base_config, str) else deepcopy(base_config)
            from ..parler_tts import ParlerTTSConfig
            config = ParlerTTSConfig.from_pretrained(base_config) if isinstance(base_config, str) else deepcopy(base_config)
            config.anchor_token = anchor_token
            if anchor_token_id is None:
                if isinstance(base_config, str):
                    tokenizer = AutoTokenizer.from_pretrained(base_config)
                    if isinstance(anchor_token, str):
                        config.anchor_token_id = tokenizer.convert_tokens_to_ids(anchor_token)
                    else:
                        config.anchor_token_id = tuple([tokenizer.convert_tokens_to_ids(t) for t in anchor_token])
                else:
                    raise AttributeError("`anchor_token_id` cannot be inferred from provided `anchor_token` since there weren't provided any vocab info")
            else:
                config.anchor_token_id = anchor_token_id
            config.use_direct_anchor = use_direct_anchor
            for k, v in kwargs.items():
                setattr(config, k, v)
            return config

    def __init__(
        self,
        base_config: Optional[PretrainedConfig | str] = None,
        anchor_token: str | tuple[str] = "<bos>",
        anchor_token_id: Optional[int | tuple[int]] = None,
        use_direct_anchor: bool = True,
        **kwargs,
    ):
        """
        Initialize SelectiveTunerConfig.

        Note: This is only called when base_config is None (see __new__ method).
        When base_config is provided, __new__ returns the modified base_config directly.

        Args:
            base_config: Not used in __init__ (handled by __new__)
            anchor_token: Token(s) to use as style anchor
            anchor_token_id: Explicit token ID(s) for anchor  
            use_direct_anchor: Whether to use direct or encoder-based anchor
            **kwargs: Additional config parameters
        """
        super().__init__(**kwargs)
        self.anchor_token = anchor_token
        self.anchor_token_id = anchor_token_id
        if anchor_token_id is None:
            raise AttributeError("`anchor_token_id` cannot be inferred from provided `anchor_token` since there weren't provided any vocab info")
        self.use_direct_anchor = use_direct_anchor

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        config_dict, kwargs = super()._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict is None:
            config_dict = {}

        config_dict['base_config'] = pretrained_model_name_or_path
        anchor_token = kwargs.pop('anchor_token', None)
        anchor_token_id = kwargs.pop('anchor_token_id', None)
        use_direct_anchor = kwargs.pop('use_direct_anchor', None)

        for arg, data in (('anchor_token', anchor_token), ('anchor_token_id', anchor_token_id), ('use_direct_anchor', use_direct_anchor)):
            if data is not None:
                config_dict[arg] = data

        return config_dict, kwargs

    @classmethod
    def from_pretrained(
        cls: type[SpecificPretrainedConfigType],
        pretrained_model_name_or_path: Union[str, os.PathLike, PretrainedConfig],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> SpecificPretrainedConfigType:
        if isinstance(pretrained_model_name_or_path, PretrainedConfig):
            kwargs['cache_dir'] = cache_dir
            kwargs['force_download'] = force_download
            kwargs['local_files_only'] = local_files_only
            kwargs['token'] = token
            kwargs['revision'] = revision
            return cls(pretrained_model_name_or_path, **kwargs)

        return super().from_pretrained(
            pretrained_model_name_or_path,
            cache_dir, force_download, local_files_only,
            token, revision
        )
