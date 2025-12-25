# Copyright (c) 2025 SparkAudio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch SparkTTS model."""

import math
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchaudio
from torch import Tensor, int32
from torch.amp import autocast
from torch.nn import Module
from torch.nn.utils import weight_norm, remove_weight_norm

from einops import pack, rearrange, reduce, repeat, unpack
from einx import get_at

from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_spark_tts import SparkTTSConfig


logger = logging.get_logger(__name__)


# ============================================================================
# SECTION 1: HELPER FUNCTIONS
# ============================================================================

def exists(val):
    """Check if value exists (is not None)."""
    return val is not None


def default(*args):
    """Return first non-None argument."""
    for arg in args:
        if exists(arg):
            return arg
    return None


def first(l):
    """Return first element of list."""
    return l[0]


def maybe(fn):
    """Decorator to apply function only if input is not None."""
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner


def pack_one(t, pattern):
    """Pack single tensor."""
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    """Unpack single tensor."""
    return unpack(t, ps, pattern)[0]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def round_up_multiple(num, mult):
    """Round up to nearest multiple."""
    return math.ceil(num / mult) * mult


def ema_inplace(moving_avg, new, decay):
    """Exponential moving average in-place update."""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def is_distributed():
    """Check if distributed training is initialized."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_maybe_sync_seed(device, max_size=10_000):
    """Get synchronized random seed across distributed processes."""
    rand_int = torch.randint(0, max_size, (), device=device)
    if is_distributed():
        dist.all_reduce(rand_int)
    return rand_int.item()


def WNConv1d(*args, **kwargs):
    """Weight-normalized Conv1d."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """Weight-normalized ConvTranspose1d."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    """Snake activation function (JIT compiled for speed)."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


def init_weights(m):
    """Initialize weights for Conv1d layers."""
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


# ============================================================================
# SECTION 2: OUTPUT DATACLASSES
# ============================================================================

@dataclass
class BiCodecOutput(ModelOutput):
    """
    Output type of [`BiCodecModel`].

    Args:
        wav_recon (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Reconstructed waveform.
        semantic_tokens (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Semantic tokens from the quantizer.
        global_tokens (`torch.LongTensor` of shape `(batch_size, num_tokens)`):
            Global speaker tokens from the speaker encoder.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Total loss (VQ loss + reconstruction loss) if in training mode.
        vq_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Vector quantization loss.
    """

    wav_recon: Optional[torch.FloatTensor] = None
    semantic_tokens: Optional[torch.LongTensor] = None
    global_tokens: Optional[torch.LongTensor] = None
    loss: Optional[torch.FloatTensor] = None
    vq_loss: Optional[torch.FloatTensor] = None


@dataclass
class SparkTTSOutput(ModelOutput):
    """
    Output type of [`SparkTTSForConditionalGeneration`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Generated waveform.
        semantic_tokens (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Generated semantic tokens from LLM.
        global_tokens (`torch.LongTensor` of shape `(batch_size, num_tokens)`):
            Speaker tokens for voice cloning.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss if labels are provided.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            LLM output logits.
    """

    waveform: Optional[torch.FloatTensor] = None
    semantic_tokens: Optional[torch.LongTensor] = None
    global_tokens: Optional[torch.LongTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


# ============================================================================
# SECTION 3: NEURAL NETWORK MODULES
# ============================================================================

# ----------------------------------------------------------------------------
# 3.1: Basic Building Blocks
# ----------------------------------------------------------------------------

class Snake1d(nn.Module):
    """
    Snake activation function for 1D convolutions.
    
    Args:
        channels (`int`): Number of channels.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    """
    Residual unit with Snake activation and dilated convolutions.
    
    Args:
        dim (`int`, *optional*, defaults to 16): Number of channels.
        dilation (`int`, *optional*, defaults to 1): Dilation factor.
    """
    
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


# ----------------------------------------------------------------------------
# 3.2: Finite Scalar Quantization (FSQ)
# ----------------------------------------------------------------------------

class FSQ(Module):
    """
    Finite Scalar Quantization module.
    Based on: https://arxiv.org/abs/2309.15505
    
    Args:
        levels (`List[int]`): Number of levels for each dimension.
        dim (`int`, *optional*): Feature dimension. If None, uses len(levels).
        num_codebooks (`int`, *optional*, defaults to 1): Number of codebooks.
        keep_num_codebooks_dim (`bool`, *optional*): Whether to keep codebook dimension.
        scale (`float`, *optional*): Scaling factor.
        channel_first (`bool`, *optional*, defaults to False): Whether input is channel-first.
        projection_has_bias (`bool`, *optional*, defaults to True): Whether projections have bias.
        return_indices (`bool`, *optional*, defaults to True): Whether to return indices.
        force_quantization_f32 (`bool`, *optional*, defaults to True): Force FP32 for quantization.
    """
    
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices: bool = True,
        force_quantization_f32: bool = True,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound input tensor to valid range."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        """Scale and shift normalized quantized values."""
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        """Inverse of scale and shift."""
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to codes."""
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Convert codes to indices."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to level indices."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices back to codes."""
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            z (`torch.Tensor`): Input tensor.
            
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: Quantized output and indices.
        """
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        if need_move_channel_last:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            indices = None
            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, "b n c d -> b n (c d)")
            codes = codes.type(orig_dtype)

        out = self.project_out(codes)

        if need_move_channel_last:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")
            indices = maybe(unpack_one)(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, "... 1 -> ...")

        return out, indices



# ----------------------------------------------------------------------------
# 3.3: Residual FSQ
# ----------------------------------------------------------------------------

class ResidualFSQ(Module):
    """
    Residual Finite Scalar Quantization.
    Follows Algorithm 1 in https://arxiv.org/pdf/2107.03312.pdf
    
    Args:
        levels (`List[int]`): Number of levels for each dimension.
        num_quantizers (`int`): Number of quantizers.
        dim (`int`, *optional*): Feature dimension.
        is_channel_first (`bool`, *optional*, defaults to False): Whether input is channel-first.
        quantize_dropout (`bool`, *optional*, defaults to False): Whether to use quantize dropout.
        quantize_dropout_cutoff_index (`int`, *optional*, defaults to 0): Cutoff index for dropout.
        quantize_dropout_multiple_of (`int`, *optional*, defaults to 1): Multiple for dropout.
    """
    
    def __init__(
        self,
        *,
        levels: List[int],
        num_quantizers: int,
        dim: Optional[int] = None,
        is_channel_first: bool = False,
        quantize_dropout: bool = False,
        quantize_dropout_cutoff_index: int = 0,
        quantize_dropout_multiple_of: int = 1,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = len(levels)
        dim = default(dim, codebook_dim)

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)
        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)
            fsq = FSQ(levels=levels, dim=codebook_dim, **kwargs)
            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size
        self.register_buffer("scales", torch.stack(scales), persistent=False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

    @property
    def codebooks(self):
        """Get all codebooks."""
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codes from indices."""
        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        indices, ps = pack([indices], "b * q")

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0.0, "quantize dropout must be greater than 0"
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        mask = indices == -1
        indices = indices.masked_fill(mask, 0)

        all_codes = get_at("q [c] d, b n q -> q b n d", self.codebooks, indices)
        all_codes = all_codes.masked_fill(rearrange(mask, "b n q -> q b n 1"), 0.0)

        scales = rearrange(self.scales, "q d -> q 1 1 d")
        all_codes = all_codes * scales

        (all_codes,) = unpack(all_codes, ps, "q b * d")
        return all_codes

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get output from indices."""
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, "q ... -> ...", "sum")
        return self.project_out(codes_summed)

    def forward(
        self, 
        x: torch.Tensor, 
        return_all_codes: bool = False, 
        rand_quantize_dropout_fixed_seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        num_quant, quant_dropout_multiple_of, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            x.device,
        )

        if self.is_channel_first:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack([x], "b * d")

        x = self.project_in(x)

        quantized_out = 0.0
        residual = x
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        if should_quantize_dropout:
            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)
            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1
                )

            null_indices = torch.full(x.shape[:2], -1.0, device=device, dtype=torch.long)

        with autocast("cuda", enabled=False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):
                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)
                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)

        quantized_out = self.project_out(quantized_out)
        all_indices = torch.stack(all_indices, dim=-1)

        if self.is_channel_first:
            (quantized_out,) = unpack(quantized_out, ps, "b * d")
            (all_indices,) = unpack(all_indices, ps, "b * d")
            quantized_out = rearrange(quantized_out, "b ... d -> b d ...")
            all_indices = rearrange(all_indices, "b ... d -> b d ...")

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        all_codes = self.get_codes_from_indices(all_indices)
        return (*ret, all_codes)


# ----------------------------------------------------------------------------
# 3.4: Factorized Vector Quantization
# ----------------------------------------------------------------------------

class FactorizedVectorQuantize(nn.Module):
    """
    Factorized Vector Quantization module.
    
    Args:
        input_dim (`int`): Input dimension.
        codebook_size (`int`): Size of the codebook.
        codebook_dim (`int`): Dimension of codebook vectors.
        commitment (`float`): Commitment loss weight.
        codebook_loss_weight (`float`, *optional*, defaults to 1.0): Codebook loss weight.
        decay (`float`, *optional*, defaults to 0.99): EMA decay rate.
        threshold_ema_dead_code (`float`, *optional*, defaults to 2): Threshold for dead code expiry.
    """
    
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float,
        codebook_loss_weight: float = 1.0,
        decay: float = 0.99,
        threshold_ema_dead_code: float = 2,
        momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.momentum = momentum

        if input_dim != self.codebook_dim:
            self.in_project = WNConv1d(input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(self.codebook_dim, input_dim, kernel_size=1)
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))

    def forward(self, z: torch.Tensor) -> Dict[str, Any]:
        """Quantize input using codebook."""
        z_e = self.in_project(z)
        z_q, indices, dists = self.decode_latents(z_e)

        embed_onehot = F.one_hot(indices, self.codebook_size).type(z_e.dtype)
        avg_probs = torch.mean(embed_onehot.reshape(-1, self.codebook_size), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        active_num = (embed_onehot.sum(0).sum(0) > 0).sum()
        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0).sum(0), self.decay)
            active_num = sum(self.cluster_size > self.threshold_ema_dead_code)

        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2]) * self.commitment
            )
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2]) * self.codebook_loss_weight
            )
        else:
            commit_loss = torch.zeros(0, device=z.device)
            codebook_loss = torch.zeros(0, device=z.device)

        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_project(z_q)

        vq_loss = (commit_loss + codebook_loss).mean()

        return {
            "z_q": z_q,
            "indices": indices,
            "dists": dists,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "active_num": active_num.float(),
        }

    def tokenize(self, z: torch.Tensor) -> torch.Tensor:
        """Tokenize input."""
        z_e = self.in_project(z)
        _, indices, _ = self.decode_latents(z_e)
        return indices

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """Detokenize indices."""
        z_q = self.decode_code(indices)
        z_q = self.out_project(z_q)
        return z_q

    def get_emb(self):
        """Get codebook embeddings."""
        return self.codebook.weight

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Embed code indices."""
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Decode code indices."""
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latents to quantized codes."""
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)

        return z_q, indices, dist


# ----------------------------------------------------------------------------
# 3.5: Vocos Building Blocks
# ----------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with learnable embeddings.
    
    Args:
        condition_dim (`int`): Dimension of the condition.
        embedding_dim (`int`): Dimension of the embeddings.
        eps (`float`, *optional*, defaults to 1e-6): Epsilon for numerical stability.
    """
    
    def __init__(self, condition_dim: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Linear(condition_dim, embedding_dim)
        self.shift = nn.Linear(condition_dim, embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding)
        shift = self.shift(cond_embedding)
        x = F.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block adapted for 1D audio signal.
    Based on: https://github.com/facebookresearch/ConvNeXt
    
    Args:
        dim (`int`): Number of input channels.
        intermediate_dim (`int`): Dimensionality of the intermediate layer.
        layer_scale_init_value (`float`): Initial value for the layer scale.
        condition_dim (`int`, *optional*): Dimension for AdaLayerNorm conditioning.
    """
    
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        x = residual + x
        return x


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 with dilated 1D convolutions.
    
    Args:
        dim (`int`): Number of input channels.
        kernel_size (`int`, *optional*, defaults to 3): Size of the convolutional kernel.
        dilation (`Tuple[int, int, int]`, *optional*, defaults to (1, 3, 5)): Dilation factors.
        lrelu_slope (`float`, *optional*, defaults to 0.1): Negative slope of LeakyReLU.
        layer_scale_init_value (`float`, *optional*): Initial value for layer scale.
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=dilation[0],
                                 padding=self.get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=dilation[1],
                                 padding=self.get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=dilation[2],
                                 padding=self.get_padding(kernel_size, dilation[2]))),
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1,
                                 padding=self.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1,
                                 padding=self.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1,
                                 padding=self.get_padding(kernel_size, 1))),
        ])

        self.gamma = nn.ParameterList([
            (nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
             if layer_scale_init_value is not None else None),
            (nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
             if layer_scale_init_value is not None else None),
            (nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
             if layer_scale_init_value is not None else None),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = F.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)


class VocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks.
    
    Args:
        input_channels (`int`): Number of input features channels.
        dim (`int`): Hidden dimension of the model.
        intermediate_dim (`int`): Intermediate dimension used in ConvNeXtBlock.
        num_layers (`int`): Number of ConvNeXtBlock layers.
        layer_scale_init_value (`float`, *optional*): Initial value for layer scaling.
        condition_dim (`int`, *optional*): Dimension for conditioning.
    """
    
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
                condition_dim=condition_dim,
            )
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert condition is not None
            x = self.norm(x.transpose(1, 2), condition)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, condition)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


# ----------------------------------------------------------------------------
# 3.6: Sampling Block
# ----------------------------------------------------------------------------

class SamplingBlock(nn.Module):
    """
    Sampling block for upsampling or downsampling.
    
    Args:
        dim (`int`): Input dimension.
        groups (`int`, *optional*, defaults to 1): Number of groups.
        upsample_scale (`int`, *optional*, defaults to 1): Upsampling scale.
        downsample_scale (`int`, *optional*, defaults to 1): Downsampling scale.
    """
    
    def __init__(
        self,
        dim: int,
        groups: int = 1,
        upsample_scale: int = 1,
        downsample_scale: int = 1,
    ):
        super().__init__()
        self.upsample_scale = upsample_scale
        self.downsample_scale = downsample_scale

        if self.upsample_scale > 1:
            self.de_conv_upsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(
                    dim, dim,
                    kernel_size=upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    groups=groups,
                ),
            )

        if self.downsample_scale > 1:
            self.conv_downsampler = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(
                    dim, dim,
                    kernel_size=2 * downsample_scale,
                    stride=downsample_scale,
                    padding=downsample_scale // 2 + downsample_scale % 2,
                    groups=groups,
                ),
            )

    @staticmethod
    def repeat_upsampler(x, upsample_scale):
        return x.repeat_interleave(upsample_scale, dim=2)

    @staticmethod
    def skip_downsampler(x, downsample_scale):
        return F.avg_pool1d(x, kernel_size=downsample_scale, stride=downsample_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        if self.upsample_scale > 1:
            repeat_res = self.repeat_upsampler(x, self.upsample_scale)
            deconv_res = self.de_conv_upsampler(x)
            upmerge_res = repeat_res + deconv_res
        else:
            upmerge_res = x
            repeat_res = x

        if self.downsample_scale > 1:
            conv_res = self.conv_downsampler(upmerge_res)
            skip2_res = self.skip_downsampler(upmerge_res, self.downsample_scale)
            skip1_res = self.skip_downsampler(repeat_res, self.downsample_scale)
        else:
            conv_res = upmerge_res
            skip2_res = upmerge_res
            skip1_res = repeat_res

        final_res = conv_res + skip1_res + skip2_res
        return final_res


# ----------------------------------------------------------------------------
# 3.7: Encoder and Decoder
# ----------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Encoder module with VocosBackbone and downsampling blocks.
    
    Args:
        input_channels (`int`): Number of input channels.
        vocos_dim (`int`): Vocos hidden dimension.
        vocos_intermediate_dim (`int`): Vocos intermediate dimension.
        vocos_num_layers (`int`): Number of Vocos layers.
        out_channels (`int`): Output channels.
        sample_ratios (`List[int]`, *optional*, defaults to [1, 1]): Downsampling ratios.
    """
    
    def __init__(
        self,
        input_channels: int,
        vocos_dim: int,
        vocos_intermediate_dim: int,
        vocos_num_layers: int,
        out_channels: int,
        sample_ratios: List[int] = [1, 1],
    ):
        super().__init__()
        self.encoder = VocosBackbone(
            input_channels=input_channels,
            dim=vocos_dim,
            intermediate_dim=vocos_intermediate_dim,
            num_layers=vocos_num_layers,
            condition_dim=None,
        )

        modules = [
            nn.Sequential(
                SamplingBlock(dim=vocos_dim, groups=vocos_dim, downsample_scale=ratio),
                VocosBackbone(
                    input_channels=vocos_dim,
                    dim=vocos_dim,
                    intermediate_dim=vocos_intermediate_dim,
                    num_layers=2,
                    condition_dim=None,
                ),
            )
            for ratio in sample_ratios
        ]

        self.downsample = nn.Sequential(*modules)
        self.project = nn.Linear(vocos_dim, out_channels)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """
        Args:
            x (`torch.Tensor` of shape `(batch_size, input_channels, length)`): Input tensor.
            
        Returns:
            `torch.Tensor` of shape `(batch_size, length, out_channels)`: Encoded features.
        """
        x = self.encoder(x)
        x = self.downsample(x)
        x = self.project(x)
        return x.transpose(1, 2)


class DecoderBlock(nn.Module):
    """
    Decoder block with transposed convolution and residual units.
    
    Args:
        input_dim (`int`, *optional*, defaults to 16): Input dimension.
        output_dim (`int`, *optional*, defaults to 8): Output dimension.
        kernel_size (`int`, *optional*, defaults to 2): Kernel size.
        stride (`int`, *optional*, defaults to 1): Stride.
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        kernel_size: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim, output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveGenerator(nn.Module):
    """
    Wave generator (decoder) module.
    
    Args:
        input_channel (`int`): Input channels.
        channels (`int`): Base number of channels.
        rates (`List[int]`): Upsampling rates.
        kernel_sizes (`List[int]`): Kernel sizes for each upsampling layer.
        d_out (`int`, *optional*, defaults to 1): Output dimension (waveform channels).
    """
    
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        kernel_sizes: List[int],
        d_out: int = 1,
    ):
        super().__init__()

        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, kernel_size, stride)]

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ----------------------------------------------------------------------------
# 3.8: Speaker Modules
# ----------------------------------------------------------------------------

# Pooling layers for speaker embedding

class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling.
    
    Args:
        in_dim (`int`): Input dimension.
        bottleneck_dim (`int`, *optional*, defaults to 128): Bottleneck dimension.
        global_context_att (`bool`, *optional*, defaults to False): Whether to use global context attention.
    """
    
    def __init__(self, in_dim: int, bottleneck_dim: int = 128, global_context_att: bool = False):
        super().__init__()
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self_attention_dim = self.linear2.out_channels
        return self_attention_dim * 2


# Alias for convenience
ASTP = AttentiveStatisticsPooling


# ECAPA-TDNN components

class Res2Conv1dReluBn(nn.Module):
    """Res2Conv1d with ReLU and BatchNorm."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        scale: int = 4,
    ):
        super().__init__()
        assert channels % scale == 0, f"{channels} % {scale} != 0"
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    """Conv1d + ReLU + BatchNorm."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """Squeeze-and-Excitation connection."""
    
    def __init__(self, channels: int, se_bottleneck_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SE_Res2Block(nn.Module):
    """SE-Res2Block of the ECAPA-TDNN architecture."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        scale: int,
    ):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SE_Connect(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se_res2block(x)


class ECAPA_TDNN_GLOB_c512(nn.Module):
    """
    ECAPA-TDNN with global context attention for speaker embedding.
    
    Args:
        feat_dim (`int`): Input feature dimension.
        embed_dim (`int`): Output embedding dimension.
    """
    
    def __init__(self, feat_dim: int, embed_dim: int):
        super().__init__()
        channels = 512

        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = ASTP(in_dim=out_channels, global_context_att=True)
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x (`torch.Tensor` of shape `(batch_size, time, feat_dim)`): Input features.
            return_latent (`bool`, *optional*, defaults to False): Whether to return latent features.
            
        Returns:
            `torch.Tensor` or `Tuple[torch.Tensor, torch.Tensor]`: Speaker embedding (and latent if requested).
        """
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        latent = F.relu(self.conv(out))
        out = self.bn(self.pool(latent))
        out = self.linear(out)

        if return_latent:
            return out, latent
        return out


# Perceiver Resampler components

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, scale: bool = True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = default(self.gamma, 1)
        return F.normalize(x, dim=-1) * self.scale * gamma


class GEGLU(nn.Module):
    """Gated GLU activation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class Attention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(
        self,
        dim: int,
        dim_context: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        cross_attn_include_queries: bool = False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.heads
        has_context = exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        dim_inner = int(dim * mult * 2 / 3)
        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner * 2),
            GEGLU(),
            nn.Linear(dim_inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for speaker embedding.
    
    Args:
        dim (`int`): Latent dimension.
        depth (`int`, *optional*, defaults to 2): Number of layers.
        dim_context (`int`, *optional*): Context dimension.
        num_latents (`int`, *optional*, defaults to 32): Number of latent tokens.
        dim_head (`int`, *optional*, defaults to 64): Dimension per attention head.
        heads (`int`, *optional*, defaults to 8): Number of attention heads.
        ff_mult (`int`, *optional*, defaults to 4): Feed-forward multiplier.
    """
    
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 2,
        dim_context: Optional[int] = None,
        num_latents: int = 32,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        cross_attn_include_queries=True,
                    ),
                    FeedForward(dim=dim, mult=ff_mult),
                ])
            )

        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = x.shape[0]
        x = self.proj_context(x)
        latents = repeat(self.latents, "n d -> b n d", b=batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder that extracts x-vector and d-vector from mel spectrograms.
    
    Args:
        input_dim (`int`, *optional*, defaults to 100): Acoustic feature dimension.
        out_dim (`int`, *optional*, defaults to 512): Output dimension of x-vector and d-vector.
        latent_dim (`int`, *optional*, defaults to 128): Latent dimension before quantization.
        token_num (`int`, *optional*, defaults to 32): Sequence length of speaker tokens.
        fsq_levels (`List[int]`, *optional*, defaults to [4, 4, 4, 4, 4, 4]): FSQ levels.
        fsq_num_quantizers (`int`, *optional*, defaults to 1): Number of FSQ quantizers.
    """
    
    def __init__(
        self,
        input_dim: int = 100,
        out_dim: int = 512,
        latent_dim: int = 128,
        token_num: int = 32,
        fsq_levels: List[int] = [4, 4, 4, 4, 4, 4],
        fsq_num_quantizers: int = 1,
    ):
        super().__init__()

        self.speaker_encoder = ECAPA_TDNN_GLOB_c512(feat_dim=input_dim, embed_dim=out_dim)
        self.perceiver_sampler = PerceiverResampler(
            dim=latent_dim, dim_context=512 * 3, num_latents=token_num
        )
        self.quantizer = ResidualFSQ(
            levels=fsq_levels,
            num_quantizers=fsq_num_quantizers,
            dim=latent_dim,
            is_channel_first=True,
            quantize_dropout=False,
        )
        self.project = nn.Linear(latent_dim * token_num, out_dim)

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codes from indices."""
        zq = self.quantizer.get_codes_from_indices(indices.transpose(1, 2))
        return zq.transpose(1, 2)

    def get_indices(self, mels: torch.Tensor) -> torch.Tensor:
        """Get indices from mel spectrograms."""
        mels = mels.transpose(1, 2)
        x = self.perceiver_sampler(mels).transpose(1, 2)
        zq, indices = self.quantizer(x)
        return indices

    def forward(self, mels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mels (`torch.Tensor` of shape `(batch_size, mel_dim, time)`): Mel spectrograms.
            
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: x-vector and d-vector.
        """
        x_vector, features = self.speaker_encoder(mels, True)
        x = self.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)
        zq, indices = self.quantizer(x)
        x = zq.reshape(zq.shape[0], -1)
        d_vector = self.project(x)
        return x_vector, d_vector

    def tokenize(self, mels: torch.Tensor) -> torch.Tensor:
        """Tokenize mel spectrograms to speaker tokens."""
        _, features = self.speaker_encoder(mels, True)
        x = self.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)
        zq, indices = self.quantizer(x)
        return indices

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """Detokenize speaker tokens to d-vector."""
        zq = self.quantizer.get_output_from_indices(indices.transpose(1, 2)).transpose(1, 2)
        x = zq.reshape(zq.shape[0], -1)
        d_vector = self.project(x)
        return d_vector


# ============================================================================
# SECTION 4: BICODEC MODEL
# ============================================================================

class BiCodecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SparkTTSConfig
    base_model_prefix = "bicodec"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BiCodecModel(BiCodecPreTrainedModel):
    """
    BiCodec model for audio tokenization and detokenization.
    
    This model integrates an encoder, decoder, quantizer, speaker encoder, prenet, and postnet
    for high-quality speech synthesis.
    
    Args:
        config (`SparkTTSConfig`): Model configuration class.
    """
    
    def __init__(self, config: SparkTTSConfig):
        super().__init__(config)
        self.config = config
        
        # Get audio tokenizer config
        atc = config.audio_tokenizer_config
        
        # Initialize mel spectrogram transform
        mel_params = atc["mel_params"]
        self.mel_spec = nn.Sequential(
            nn.ReflectionPad1d((mel_params["n_fft"] - mel_params["hop_length"]) // 2),
        )
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_params["sample_rate"],
            n_fft=mel_params["n_fft"],
            win_length=mel_params["win_length"],
            hop_length=mel_params["hop_length"],
            f_min=mel_params["mel_fmin"],
            f_max=mel_params["mel_fmax"],
            n_mels=mel_params["num_mels"],
            power=1,
            center=False,
        )
        
        # Initialize encoder
        self.encoder = Encoder(**atc["encoder"])
        
        # Initialize quantizer
        self.quantizer = FactorizedVectorQuantize(**atc["quantizer"])
        
        # Initialize speaker encoder
        self.speaker_encoder = SpeakerEncoder(**atc["speaker_encoder"])
        
        # Initialize prenet and postnet
        self.prenet = VocosBackbone(**atc["prenet"])
        self.postnet = VocosBackbone(**atc["postnet"])
        
        # Initialize decoder (wave generator)
        self.decoder = WaveGenerator(**atc["decoder"])
        
        # Post-initialization
        self.post_init()
    
    def get_mel_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from waveform."""
        wav = self.mel_spec(wav)
        mel = self.mel_transform(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel
    
    def remove_weight_norm(self):
        """Remove weight normalization from decoder."""
        for module in self.decoder.modules():
            if hasattr(module, 'remove_weight_norm'):
                module.remove_weight_norm()
    
    def tokenize(self, feat: torch.Tensor, ref_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize audio features into semantic and global tokens.
        
        Args:
            feat (`torch.Tensor` of shape `(batch_size, feat_dim, time)`): Input features.
            ref_wav (`torch.Tensor` of shape `(batch_size, time)`): Reference waveform for speaker embedding.
            
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: Semantic tokens and global speaker tokens.
        """
        mel = self.get_mel_spectrogram(ref_wav.unsqueeze(1)).squeeze(1)
        
        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))
        
        return semantic_tokens, global_tokens
    
    def detokenize(
        self, 
        semantic_tokens: torch.Tensor, 
        global_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Detokenize semantic and global tokens into waveform.
        
        Args:
            semantic_tokens (`torch.Tensor` of shape `(batch_size, time)`): Semantic tokens.
            global_tokens (`torch.Tensor` of shape `(batch_size, num_tokens)`): Global speaker tokens.
            
        Returns:
            `torch.Tensor` of shape `(batch_size, time)`: Reconstructed waveform.
        """
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        
        x = self.prenet(z_q, d_vector)
        x = self.postnet(x)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x).squeeze(1)
        
        return wav_recon
    
    def forward(
        self,
        input_features: torch.Tensor,
        reference_waveform: torch.Tensor,
        target_waveform: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BiCodecOutput]:
        """
        Forward pass through BiCodec model.
        
        Args:
            input_features (`torch.Tensor` of shape `(batch_size, feat_dim, time)`): Input features.
            reference_waveform (`torch.Tensor` of shape `(batch_size, time)`): Reference waveform.
            target_waveform (`torch.Tensor` of shape `(batch_size, time)`, *optional*): Target waveform for training.
            return_dict (`bool`, *optional*): Whether to return a `BiCodecOutput` instead of tuple.
            
        Returns:
            `BiCodecOutput` or `Tuple`: Model outputs.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get mel spectrogram
        mel = self.get_mel_spectrogram(reference_waveform.unsqueeze(1)).squeeze(1)
        
        # Encode
        z = self.encoder(input_features.transpose(1, 2))
        vq_outputs = self.quantizer(z)
        
        # Speaker encoding
        x_vector, d_vector = self.speaker_encoder(mel.transpose(1, 2))
        
        # Decode
        x = self.prenet(vq_outputs["z_q"], d_vector)
        pred_feat = self.postnet(x)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x).squeeze(1)
        
        # Compute loss if target is provided
        loss = None
        if target_waveform is not None and self.training:
            loss = vq_outputs["vq_loss"]
        
        if not return_dict:
            output = (wav_recon, vq_outputs["indices"], self.speaker_encoder.get_indices(mel.transpose(1, 2)))
            return ((loss,) + output) if loss is not None else output
        
        return BiCodecOutput(
            wav_recon=wav_recon,
            semantic_tokens=vq_outputs["indices"],
            global_tokens=self.speaker_encoder.get_indices(mel.transpose(1, 2)),
            loss=loss,
            vq_loss=vq_outputs["vq_loss"],
        )


# ============================================================================
# SECTION 5: SPARKTTS MODEL
# ============================================================================

class SparkTTSPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SparkTTSConfig
    base_model_prefix = "spark_tts"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        std = self.config.init_std if hasattr(self.config, 'init_std') else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SparkTTSForConditionalGeneration(SparkTTSPreTrainedModel):
    """
    SparkTTS model for text-to-speech generation.
    
    This model combines a language model (LLM), BiCodec audio tokenizer, and Wav2Vec2 feature extractor
    for high-quality zero-shot voice cloning and controllable speech generation.
    
    Args:
        config (`SparkTTSConfig`): Model configuration class.
    """
    
    def __init__(self, config: SparkTTSConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize BiCodec
        self.bicodec = BiCodecModel(config)
        
        # LLM and Wav2Vec2 will be loaded separately via from_pretrained
        self.llm = None
        self.wav2vec2 = None
        self.wav2vec2_processor = None
        
        # Post-initialization
        self.post_init()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load pretrained SparkTTS model from HuggingFace Hub or local path.
        
        This method handles loading the composite model structure with BiCodec, LLM, and Wav2Vec2.
        """
        # Load config
        config = kwargs.pop("config", None)
        if config is None:
            config = SparkTTSConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Initialize model
        model = cls(config)
        
        # Load LLM from subdirectory
        from transformers import AutoModelForCausalLM, AutoTokenizer
        llm_path = Path(pretrained_model_name_or_path) / config.llm_path
        model.llm = AutoModelForCausalLM.from_pretrained(llm_path, **kwargs)
        model.tokenizer = AutoTokenizer.from_pretrained(llm_path, **kwargs)
        
        # Load Wav2Vec2 from subdirectory
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        wav2vec2_path = Path(pretrained_model_name_or_path) / config.wav2vec2_path
        model.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_path, **kwargs)
        model.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_path, **kwargs)
        
        # Load BiCodec weights from subdirectory
        from safetensors.torch import load_file
        bicodec_path = Path(pretrained_model_name_or_path) / config.bicodec_path
        bicodec_weights = load_file(bicodec_path / "model.safetensors")
        model.bicodec.load_state_dict(bicodec_weights, strict=False)
        
        return model
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        reference_waveform: Optional[torch.Tensor] = None,
        reference_audio_path: Optional[str] = None,
        gender: Optional[str] = None,
        pitch: Optional[str] = None,
        speed: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate speech from text input with optional voice cloning or controllable generation.
        
        This method supports two modes:
        1. **Voice Cloning**: Provide `reference_waveform` or `reference_audio_path` to clone a voice
        2. **Controllable Generation**: Provide `gender`, `pitch`, `speed` for controlled synthesis
        
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*): 
                Input token IDs. Either this or `text` must be provided.
            text (`str`, *optional*): 
                Input text to synthesize. Will be tokenized if provided.
            reference_waveform (`torch.Tensor` of shape `(batch_size, time)`, *optional*): 
                Reference audio waveform for voice cloning.
            reference_audio_path (`str`, *optional*): 
                Path to reference audio file for voice cloning.
            gender (`str`, *optional*): 
                Gender for controllable generation. Options: "female", "male".
            pitch (`str`, *optional*): 
                Pitch level. Options: "very_low", "low", "moderate", "high", "very_high".
            speed (`str`, *optional*): 
                Speed level. Options: "very_low", "low", "moderate", "high", "very_high".
            max_new_tokens (`int`, *optional*): 
                Maximum number of tokens to generate.
            temperature (`float`, *optional*): 
                Sampling temperature.
            top_k (`int`, *optional*): 
                Top-k sampling parameter.
            top_p (`float`, *optional*): 
                Top-p (nucleus) sampling parameter.
            do_sample (`bool`, *optional*, defaults to True): 
                Whether to use sampling or greedy decoding.
            
        Returns:
            `torch.Tensor` of shape `(batch_size, time)`: Generated waveform.
            
        Example:
        
        ```python
        # Voice cloning
        waveform = model.generate(
            text="Hello, how are you?",
            reference_audio_path="reference.wav"
        )
        
        # Controllable generation
        waveform = model.generate(
            text="Hello, how are you?",
            gender="female",
            pitch="moderate",
            speed="moderate"
        )
        ```
        """
        if self.llm is None or self.bicodec is None or self.wav2vec2 is None:
            raise RuntimeError(
                "Model components not loaded. Use `from_pretrained` to load the complete model."
            )
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        
        # Prepare input_ids from text if needed
        if input_ids is None:
            if text is None:
                raise ValueError("Either `input_ids` or `text` must be provided")
            
            # Tokenize text
            if not hasattr(self, 'tokenizer'):
                raise RuntimeError("Tokenizer not loaded. Use `from_pretrained` to load the model.")
            
            # Prepare prompt based on mode
            if reference_waveform is not None or reference_audio_path is not None:
                # Voice cloning mode
                prompt = self._prepare_voice_cloning_prompt(text, reference_waveform, reference_audio_path)
            elif gender is not None or pitch is not None or speed is not None:
                # Controllable generation mode
                prompt = self._prepare_controllable_prompt(text, gender, pitch, speed)
            else:
                raise ValueError(
                    "Must provide either reference audio (for voice cloning) or "
                    "control parameters (gender/pitch/speed) for controllable generation"
                )
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(self.llm.device)
            attention_mask = inputs.attention_mask.to(self.llm.device)
        else:
            attention_mask = kwargs.get('attention_mask', None)
        
        # Generate semantic tokens using LLM
        with torch.no_grad():
            generated_ids = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract semantic tokens from generated sequence
        semantic_tokens, global_tokens_from_llm = self._extract_semantic_tokens(generated_ids, input_ids)
        
        # Extract global speaker tokens from reference audio or use LLM-generated ones
        if reference_waveform is not None or reference_audio_path is not None:
            global_tokens = self._extract_global_tokens(reference_waveform, reference_audio_path)
        elif global_tokens_from_llm is not None:
            # For controllable generation, use tokens generated by LLM
            global_tokens = global_tokens_from_llm
        else:
            raise ValueError("No global tokens available")
        
        # Detokenize to waveform using BiCodec
        with torch.no_grad():
            waveform = self.bicodec.detokenize(
                semantic_tokens=semantic_tokens.to(self.bicodec.device),
                global_tokens=global_tokens.to(self.bicodec.device)
            )
        
        return waveform
    
    def _prepare_voice_cloning_prompt(
        self, 
        text: str, 
        reference_waveform: Optional[torch.Tensor],
        reference_audio_path: Optional[str]
    ) -> str:
        """Prepare prompt for voice cloning mode."""
        # Load reference audio if path provided
        if reference_audio_path is not None:
            import torchaudio
            reference_waveform, sr = torchaudio.load(reference_audio_path)
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                reference_waveform = resampler(reference_waveform)
        
        # Extract global tokens from reference
        # This will be used in _extract_global_tokens
        self._cached_reference_waveform = reference_waveform
        
        # Build prompt with special tokens
        prompt = (
            "<|voice_cloning|>"
            "<|start_content|>"
            f"{text}"
            "<|end_content|>"
            "<|start_global_token|>"
            # Global tokens will be inserted during generation
            "<|end_global_token|>"
        )
        return prompt
    
    def _prepare_controllable_prompt(
        self,
        text: str,
        gender: Optional[str],
        pitch: Optional[str],
        speed: Optional[str]
    ) -> str:
        """Prepare prompt for controllable generation mode."""
        # Default values
        gender = gender or "female"
        pitch = pitch or "moderate"
        speed = speed or "moderate"
        
        # Map to token IDs (based on original Spark-TTS)
        gender_map = {"female": 0, "male": 1}
        levels_map = {"very_low": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4}
        
        gender_id = gender_map.get(gender, 0)
        pitch_id = levels_map.get(pitch, 2)
        speed_id = levels_map.get(speed, 2)
        
        # Build prompt with control tokens
        prompt = (
            "<|controllable_tts|>"
            "<|start_content|>"
            f"{text}"
            "<|end_content|>"
            "<|start_style_label|>"
            f"<|gender_{gender_id}|>"
            f"<|pitch_label_{pitch_id}|>"
            f"<|speed_label_{speed_id}|>"
            "<|end_style_label|>"
        )
        return prompt
    
    def _extract_semantic_tokens(
        self, 
        generated_ids: torch.Tensor,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract semantic and global tokens from LLM generated sequence.
        
        Returns:
            Tuple of (semantic_tokens, global_tokens). global_tokens is None for voice cloning mode.
        """
        import re
        
        # Trim input tokens from generated sequence
        trimmed_ids = [
            output_ids[len(input_ids[i]):] 
            for i, output_ids in enumerate(generated_ids)
        ]
        
        # Decode generated tokens to text
        predicts = self.tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        
        # Extract semantic token IDs using regex (matching original Spark-TTS)
        semantic_token_ids = [
            int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)
        ]
        
        if not semantic_token_ids:
            raise ValueError(
                "No semantic tokens found in generated sequence. "
                "Make sure the LLM is properly trained to generate bicodec_semantic_* tokens."
            )
        
        # Convert to tensor
        semantic_tokens = torch.tensor(semantic_token_ids, dtype=torch.long).unsqueeze(0)
        
        # Extract global tokens if present (for controllable generation)
        global_token_ids = [
            int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)
        ]
        
        global_tokens = None
        if global_token_ids:
            global_tokens = torch.tensor(global_token_ids, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        
        return semantic_tokens, global_tokens
    
    def _extract_global_tokens(
        self,
        reference_waveform: Optional[torch.Tensor],
        reference_audio_path: Optional[str]
    ) -> torch.Tensor:
        """Extract global speaker tokens from reference audio."""
        # Load reference audio if path provided
        if reference_audio_path is not None:
            import torchaudio
            reference_waveform, sr = torchaudio.load(reference_audio_path)
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                reference_waveform = resampler(reference_waveform)
        elif hasattr(self, '_cached_reference_waveform'):
            reference_waveform = self._cached_reference_waveform
        
        if reference_waveform is None:
            raise ValueError("No reference audio provided")
        
        # Extract Wav2Vec2 features
        with torch.no_grad():
            # Ensure correct shape
            if reference_waveform.dim() == 1:
                reference_waveform = reference_waveform.unsqueeze(0)
            
            # Extract features using processor
            if hasattr(self, 'wav2vec2_processor'):
                inputs = self.wav2vec2_processor(
                    reference_waveform.squeeze().cpu().numpy(),
                    sampling_rate=self.config.sample_rate,
                    return_tensors="pt"
                )
                features = self.wav2vec2(inputs.input_values.to(self.wav2vec2.device))
                # Mix hidden states (layers 11, 14, 16)
                features_mix = (
                    features.hidden_states[11] + 
                    features.hidden_states[14] + 
                    features.hidden_states[16]
                ) / 3
            else:
                # Fallback: use BiCodec's tokenization
                _, global_tokens = self.bicodec.tokenize(
                    feat=reference_waveform.unsqueeze(0),
                    ref_wav=reference_waveform.squeeze()
                )
                return global_tokens
        
        # Tokenize using BiCodec speaker encoder
        mel = self.bicodec.get_mel_spectrogram(reference_waveform.unsqueeze(1)).squeeze(1)
        global_tokens = self.bicodec.speaker_encoder.tokenize(mel.transpose(1, 2))
        
        return global_tokens
    
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SparkTTSOutput]:
        """
        Forward pass for training.
        
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`): Input token IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*): Attention mask.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*): Labels for language modeling.
            return_dict (`bool`, *optional*): Whether to return a `SparkTTSOutput`.
            
        Returns:
            `SparkTTSOutput` or `Tuple`: Model outputs.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through LLM
        if self.llm is not None:
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            loss = llm_outputs.loss if labels is not None else None
            logits = llm_outputs.logits
        else:
            loss = None
            logits = None
        
        if not return_dict:
            output = (None, None, None, loss, logits)
            return output
        
        return SparkTTSOutput(
            waveform=None,
            semantic_tokens=None,
            global_tokens=None,
            loss=loss,
            logits=logits,
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SparkTTSConfig",
    "BiCodecModel",
    "BiCodecPreTrainedModel",
    "SparkTTSForConditionalGeneration",
    "SparkTTSPreTrainedModel",
    "BiCodecOutput",
    "SparkTTSOutput",
]

