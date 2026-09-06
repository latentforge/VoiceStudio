# coding=utf-8
# Copyright 2024 The VoxInstruct Authors and the HuggingFace Inc. team. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Semantic tokenizer class for VoxInstruct."""

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.models.hubert.modeling_hubert import HubertModel
from transformers.utils import auto_docstring

from .configuration_vox_instruct import VoxInstructConfig


class VoxInstructSemanticTokenizerModel(PreTrainedModel):
    r"""
    Constructs a VoxInstruct semantic tokenizer, a HuBERT encoder whose intermediate layer output is quantized by a
    k-means codebook into the semantic token stream that conditions both VoxInstruct stages.

    Args:
        config ([`VoxInstructConfig`]):
            Model configuration. `semantic_encoder_config`, `semantic_num_clusters`, `semantic_feature_layer` and
            `semantic_frame_multiple` are read from it.
    """

    config_class = VoxInstructConfig
    main_input_name = "input_values"
    _no_split_modules = ["HubertEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: VoxInstructConfig):
        super().__init__(config)
        self.encoder = HubertModel(config.semantic_encoder_config)
        self.feature_layer = config.semantic_feature_layer
        self.frame_multiple = config.semantic_frame_multiple
        self.register_buffer(
            "cluster_centers",
            torch.zeros(config.semantic_num_clusters, config.semantic_encoder_config.hidden_size),
            persistent=True,
        )
        self.post_init()

    @property
    def codebook_size(self) -> int:
        """Number of k-means centroids."""
        return self.cluster_centers.shape[0]

    def quantize(self, features: torch.Tensor) -> torch.LongTensor:
        r"""
        Assigns every feature frame to its nearest k-means centroid.

        Args:
            features (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Encoder features to quantize.

        Returns:
            `torch.LongTensor` of shape `(batch_size, sequence_length)`: Index of the nearest centroid.
        """
        centers = self.cluster_centers.to(features.dtype)
        # The squared norm of the features is constant across centroids, so it drops out of the argmin.
        scores = features @ centers.transpose(0, 1) - 0.5 * centers.pow(2).sum(dim=-1)
        return scores.argmax(dim=-1)

    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        deduplicate: bool = False,
    ) -> torch.LongTensor:
        r"""
        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Waveform sampled at `config.semantic_sampling_rate`, truncated to a multiple of
                `config.semantic_frame_multiple`.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask over the waveform samples.
            deduplicate (`bool`, *optional*, defaults to `False`):
                Whether to collapse runs of the same token. Requires a batch size of one.

        Returns:
            `torch.LongTensor` of shape `(batch_size, num_frames)`: Semantic token ids in
            `[0, config.semantic_num_clusters)`.
        """
        length = input_values.shape[-1] // self.frame_multiple * self.frame_multiple
        input_values = input_values[..., :length]
        if attention_mask is not None:
            attention_mask = attention_mask[..., :length]

        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        codes = self.quantize(outputs.hidden_states[self.feature_layer])

        if deduplicate:
            if codes.shape[0] != 1:
                raise ValueError("`deduplicate=True` requires a batch size of one.")
            codes = torch.unique_consecutive(codes[0]).unsqueeze(0)
        return codes


__all__ = ["VoxInstructSemanticTokenizerModel"]
