# coding=utf-8
# Copyright 2026 LatentForge and the HuggingFace Inc. team. All rights reserved.
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
"""NVFP4 quantization aware training, applied as a checkpoint loads."""

import logging
from typing import Any

import torch
from torch import nn
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    _addmm_nvfp4_dispatch,
    per_tensor_amax_to_scale,
)
from transformers.quantizers import HfQuantizer
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.utils.quantization_config import QuantizationConfigMixin


logger = logging.getLogger(__name__)

NVFP4_BLOCK_SIZE = 16
"""Elements per NVFP4 block scale. The packed layout admits no other value."""


def _hadamard_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Builds a normalized Hadamard matrix by Sylvester's construction.

    Args:
        size (`int`):
            Order of the matrix. Must be a power of two.
        dtype (`torch.dtype`):
            Precision the matrix is returned in.
        device (`torch.device`):
            Device the matrix is built on.

    Returns:
        `torch.Tensor`: An orthogonal `(size, size)` matrix, scaled so that `H @ H.T` is the
        identity rather than `size` times it.

    Raises:
        ValueError: If `size` is not a positive power of two.
    """
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {size}")
    matrix = torch.ones(1, 1, dtype=torch.float32, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0
        )
    return (matrix * size**-0.5).to(dtype)


def _rotate(tensor: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Applies a block diagonal rotation along the last dimension.

    Args:
        tensor (`torch.Tensor`):
            The tensor to rotate, whose last dimension is a multiple of the transform's order.
        transform (`torch.Tensor`):
            The `(group, group)` rotation applied to each group of the last dimension.

    Returns:
        `torch.Tensor`: The rotated tensor, in the shape it came in.
    """
    group = transform.shape[0]
    shape = tensor.shape
    return (tensor.reshape(-1, group) @ transform).reshape(shape)


def _to_nvfp4(tensor: torch.Tensor):
    """Quantizes one tensor to NVFP4 under a dynamic per-tensor scale.

    Args:
        tensor (`torch.Tensor`):
            A two dimensional tensor whose last dimension is a multiple of [`NVFP4_BLOCK_SIZE`].

    Returns:
        `NVFP4Tensor`: The quantized tensor, carrying its block scales and its global scale.
    """
    scale = per_tensor_amax_to_scale(torch.max(torch.abs(tensor)))
    return NVFP4Tensor.to_nvfp4(tensor, block_size=NVFP4_BLOCK_SIZE, per_tensor_scale=scale)


class _HadamardNVFP4Matmul(torch.autograd.Function):
    r"""NVFP4 matmul on rotated operands, differentiated at the precision it was rotated in.

    A Hadamard rotation is orthogonal, so rotating both operands along the reduction axis leaves
    the product alone: `(x H) (W H)^T` is `x W^T`. What it does change is the distribution each
    operand is quantized from, since the rotation spreads an outlier across its whole group
    instead of forcing the group's scale to cover it alone.

    The forward runs on the FP4 tensor cores. The backward reads the same operands back at their
    original precision and runs in it, which is the straight through estimator the rotation is
    differentiated under.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        rotated_hidden = _rotate(hidden, transform)
        rotated_weight = _rotate(weight, transform)
        quantized_hidden = _to_nvfp4(rotated_hidden)
        quantized_weight = _to_nvfp4(rotated_weight)
        ctx.save_for_backward(quantized_hidden, quantized_weight, transform)
        return _addmm_nvfp4_dispatch(quantized_hidden, quantized_weight.t(), None, bias)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        quantized_hidden, quantized_weight, transform = ctx.saved_tensors
        rotated_hidden = quantized_hidden.dequantize(quantized_hidden.orig_dtype)
        rotated_weight = quantized_weight.dequantize(quantized_weight.orig_dtype)

        # The rotation sits between the parameter and the matmul, so each gradient comes back in
        # rotated coordinates and is carried out of them by the transpose.
        grad_hidden = _rotate(grad_output @ rotated_weight, transform.t())
        grad_weight = _rotate(grad_output.t() @ rotated_hidden, transform.t())
        grad_bias = grad_output.sum(0) if ctx.needs_input_grad[2] else None
        return grad_hidden, grad_weight, grad_bias, None


class HadamardNVFP4Linear(nn.Linear):
    r"""A linear whose forward runs in NVFP4 on Hadamard rotated operands.

    The weight stays a full precision `Parameter`, which is what the optimizer accumulates into.
    Quantization happens on every forward from that master weight, so the run trains a model that
    already sees the error its four bit deployment will have.

    Args:
        in_features (`int`):
            Size of the reduction axis. Must be a multiple of `hadamard_group_size`.
        out_features (`int`):
            Size of the output axis.
        bias (`bool`, *optional*, defaults to `True`):
            Whether the layer carries a bias. The bias is added outside the quantized matmul.
        hadamard_group_size (`int`, *optional*, defaults to 16):
            Order of the rotation applied to each group of the reduction axis. Matching the NVFP4
            block size keeps a rotation group and a scale group the same set of elements.
        device (`torch.device`, *optional*):
            Device the layer is built on.
        dtype (`torch.dtype`, *optional*):
            Precision of the master weight.

    Raises:
        ValueError: If `in_features` is not a multiple of `hadamard_group_size`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hadamard_group_size: int = NVFP4_BLOCK_SIZE,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        if in_features % hadamard_group_size:
            raise ValueError(
                f"in_features {in_features} is not a multiple of the rotation group "
                f"{hadamard_group_size}"
            )
        self.hadamard_group_size = hadamard_group_size
        # Not persistent: Sylvester's construction is deterministic, so a rebuilt basis is the
        # basis the interrupted run used, and a saved one would go MISSING against any checkpoint
        # that predates the rewrite.
        self.register_buffer(
            "forward_transform",
            _hadamard_matrix(hadamard_group_size, self.weight.dtype, self.weight.device),
            persistent=False,
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, hadamard_group_size: int = NVFP4_BLOCK_SIZE
    ) -> "HadamardNVFP4Linear":
        """Wraps an existing linear, keeping its weights.

        Args:
            linear (`nn.Linear`):
                The layer to replace.
            hadamard_group_size (`int`, *optional*, defaults to 16):
                Order of the rotation.

        Returns:
            `HadamardNVFP4Linear`: A layer holding `linear`'s weight and bias.
        """
        replacement = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            hadamard_group_size=hadamard_group_size,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        replacement.weight = linear.weight
        replacement.bias = linear.bias
        return replacement

    def to_linear(self) -> nn.Linear:
        """Returns the plain linear this layer trained, at full precision.

        Returns:
            `nn.Linear`: A layer holding the master weight and bias, with no rotation and no
            quantization on its forward.
        """
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        linear.weight = self.weight
        linear.bias = self.bias
        return linear

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Runs the rotated NVFP4 matmul.

        Args:
            hidden (`torch.Tensor`):
                Activations, of any rank whose last dimension is `in_features`.

        Returns:
            `torch.Tensor`: The result, in `hidden`'s shape and dtype.
        """
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        transform = self.forward_transform.to(flat.dtype)
        out = _HadamardNVFP4Matmul.apply(flat, self.weight.to(flat.dtype), self.bias, transform)
        return out.reshape(*shape[:-1], out.shape[-1])


def _replace_linears(
    module: nn.Module,
    skip: tuple[str, ...],
    hadamard_group_size: int,
    prefix: str,
    replaced: list[str],
) -> None:
    """Swaps every eligible `nn.Linear` under `module` in place.

    Args:
        module (`nn.Module`):
            The subtree to walk.
        skip (`tuple[str, ...]`):
            Name fragments whose layers are left alone.
        hadamard_group_size (`int`):
            Order of the rotation.
        prefix (`str`):
            Qualified name of `module`, used to build the names reported back.
        replaced (`list[str]`):
            Accumulator the qualified names of replaced layers are appended to.
    """
    for name, child in module.named_children():
        qualified = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and not isinstance(child, HadamardNVFP4Linear):
            if any(fragment in qualified for fragment in skip):
                continue
            if child.in_features % hadamard_group_size:
                logger.warning(
                    "%s has in_features %d, not a multiple of %d, so it stays unquantized",
                    qualified,
                    child.in_features,
                    hadamard_group_size,
                )
                continue
            setattr(module, name, HadamardNVFP4Linear.from_linear(child, hadamard_group_size))
            replaced.append(qualified)
        else:
            _replace_linears(child, skip, hadamard_group_size, qualified, replaced)


def _prepare_nvfp4_qat(
    model: nn.Module,
    skip: tuple[str, ...] = ("lm_head",),
    hadamard_group_size: int = NVFP4_BLOCK_SIZE,
) -> list[str]:
    """Replaces the model's linear projections with their rotated NVFP4 form.

    Args:
        model (`nn.Module`):
            The model to convert, already on its device.
        skip (`tuple[str, ...]`, *optional*, defaults to `("lm_head",)`):
            Name fragments left at their loaded precision. An output head quantized to four bits
            is the usual source of a broken run.
        hadamard_group_size (`int`, *optional*, defaults to 16):
            Order of the rotation applied to each group of the reduction axis.

    Returns:
        `list[str]`: Qualified names of the replaced layers, in traversal order.
    """
    replaced: list[str] = []
    _replace_linears(model, tuple(skip), hadamard_group_size, "", replaced)
    logger.info("%d linear projections now run in NVFP4", len(replaced))
    return replaced


def convert_nvfp4_qat(model: nn.Module) -> list[str]:
    """Returns the model to plain linears holding the trained master weights.

    Args:
        model (`nn.Module`):
            A model [`prepare_nvfp4_qat`] converted.

    Returns:
        `list[str]`: Qualified names of the layers returned to `nn.Linear`, in traversal order.
    """
    restored: list[str] = []

    def walk(module: nn.Module, prefix: str) -> None:
        for name, child in module.named_children():
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, HadamardNVFP4Linear):
                setattr(module, name, child.to_linear())
                restored.append(qualified)
            else:
                walk(child, qualified)

    walk(model, "")
    return restored


QUANTIZATION_METHOD = "nvfp4"


@register_quantization_config(QUANTIZATION_METHOD)
class NVFP4Config(QuantizationConfigMixin):
    r"""Runs a model's linear projections in NVFP4 on Hadamard rotated operands.

    The weight stays a full precision `Parameter`, which is what the optimizer accumulates into.
    Quantization happens on every forward from that master weight, so a run trains a model that
    already sees the error its four bit deployment will have, and a checkpoint it writes holds
    ordinary weights that load anywhere.

    Args:
        hadamard_group_size (`int`, *optional*, defaults to 16):
            Order of the rotation applied to each group of the reduction axis. Matching the NVFP4
            block size keeps a rotation group and a scale group the same set of elements.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `["lm_head"]`):
            Name fragments left at their loaded precision. An output head quantized to four bits
            is the usual source of a broken run.

    Raises:
        ValueError: If `hadamard_group_size` is not a positive power of two.
    """

    def __init__(
        self,
        hadamard_group_size: int = NVFP4_BLOCK_SIZE,
        modules_to_not_convert: list[str] | None = None,
        **kwargs: Any,
    ):
        if hadamard_group_size <= 0 or hadamard_group_size & (hadamard_group_size - 1):
            raise ValueError(
                f"hadamard_group_size must be a positive power of two, got {hadamard_group_size}"
            )
        self.quant_method = QUANTIZATION_METHOD
        self.hadamard_group_size = hadamard_group_size
        self.modules_to_not_convert = (
            ["lm_head"] if modules_to_not_convert is None else list(modules_to_not_convert)
        )
        super().__init__(**kwargs)


@register_quantizer(QUANTIZATION_METHOD)
class NVFP4HfQuantizer(HfQuantizer):
    r"""Applies [`NVFP4Config`] as a checkpoint loads.

    The rewrite happens before the weights arrive, and [`HadamardNVFP4Linear`] holds the same
    `weight` and `bias` a plain linear does, so the checkpoint loads into it with no renaming and
    no conversion.
    """

    requires_calibration = False
    required_packages = ["torchao"]
    quantization_config: NVFP4Config

    def validate_environment(self, *args: Any, **kwargs: Any) -> None:
        """Refuses a device the FP4 tensor cores are not on.

        Raises:
            RuntimeError: If no CUDA device is available, or it predates Blackwell.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("NVFP4 quantization needs a CUDA device.")
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            raise RuntimeError(
                "NVFP4 quantization needs a Blackwell GPU or newer, which is the first to carry "
                f"the FP4 tensor cores; this device reports compute capability {major}.x."
            )

    def _process_model_before_weight_loading(self, model: nn.Module, **kwargs: Any) -> None:
        """Replaces the model's linear projections before the checkpoint is read into them.

        Args:
            model (`nn.Module`):
                The model being loaded.
        """
        _prepare_nvfp4_qat(
            model,
            skip=tuple(self.quantization_config.modules_to_not_convert),
            hadamard_group_size=self.quantization_config.hadamard_group_size,
        )

    @property
    def is_trainable(self) -> bool:
        """Whether a model quantized this way may be handed to a [`Trainer`].

        Returns:
            `bool`: Always `True`. The master weight is what trains; the four bit form is built
            from it on every forward and is never optimized.
        """
        return True

    @property
    def is_serializable(self) -> bool:
        """Whether such a model can be saved.

        Returns:
            `bool`: Always `True`. What a checkpoint holds is the master weight, in the precision
            it was loaded in.
        """
        return True


__all__ = [
    "HadamardNVFP4Linear",
    "NVFP4Config",
    "NVFP4HfQuantizer",
    "convert_nvfp4_qat",
]
